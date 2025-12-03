import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.transforms import functional as TF
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import pandas as pd

st.set_page_config(
    page_title="FlagExplorer",
    layout="wide",
    page_icon="🏳️",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .block-container { padding-top: 1rem; padding-bottom: 2rem; }
        
        /* Sidebar Width */
        [data-testid="stSidebar"] {
            min-width: 350px;
            max-width: 450px;
        }
        
        /* Font sizing for small match cards */
        .small-font {
            font-size: 11px !important;
            line-height: 1.2;
            text-align: center;
            margin-top: 5px;
        }
    </style>
""", unsafe_allow_html=True)

YOLO_MODEL_PATH = 'best.pt' 
ARCFACE_MODEL_PATH = 'v9_arcface_squarepad_model-train-val-new-dataset-no-pad.pth'
GALLERY_PATH = 'flag_gallery_index.pth'
CSV_PATH = 'country_codes_processed.csv'
TEST_IMAGES_DIR = 'test_images' 
GALLERY_ROOT_DIR = 'output-flags'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        p_left, p_top = (max_wh - w) // 2, (max_wh - h) // 2
        p_right, p_bottom = max_wh - w - p_left, max_wh - h - p_top
        padding = (p_left, p_top, p_right, p_bottom)
        return TF.pad(image, padding, padding_mode='reflect')

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1.0 / self.p)

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
    def forward(self, input, label):
        return input

class FlagRecognitionModel(nn.Module):
    def __init__(self, num_classes, embedding_dim=512):
        super(FlagRecognitionModel, self).__init__()
        self.backbone = models.resnet50(weights=None)
        self.backbone.avgpool = nn.Identity()
        self.backbone.fc = nn.Identity()
        self.gem = GeM()
        fc_inputs = 2048 
        self.bn1 = nn.BatchNorm1d(fc_inputs)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(fc_inputs, embedding_dim)
        self.bn2 = nn.BatchNorm1d(embedding_dim)
        self.arcface = ArcMarginProduct(embedding_dim, num_classes)

    def forward(self, x, labels=None):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        features = self.gem(x).flatten(1)
        features = self.bn1(features)
        features = self.dropout(features)
        features = self.fc(features)
        embeddings = self.bn2(features)
        return F.normalize(embeddings, p=2, dim=1)


@st.cache_resource
def load_yolo_model(path):
    return YOLO(path)

@st.cache_resource
def load_gallery_and_metadata(gallery_path, csv_path):
    if not os.path.exists(gallery_path):
        return None, None, None, None, None

    #  Load Gallery
    data = torch.load(gallery_path, map_location=DEVICE)
    gallery_feats = data['feats'].to(DEVICE)
    gallery_labels = data['labels']
    gallery_paths = data['paths']
    
    raw_idx_to_class = data['idx_to_class']
    idx_to_class = {int(k): v for k, v in raw_idx_to_class.items()}

    # Load Names CSV (Robust Method)
    code_to_name = {}
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()
            
            if 'Alpha-2 code' in df.columns and 'Country' in df.columns:
                
                df['Alpha-2 code'] = df['Alpha-2 code'].astype(str).str.lower().str.strip()
                df['Country'] = df['Country'].astype(str).str.strip()
                
                code_to_name = dict(zip(df['Alpha-2 code'], df['Country']))
            else:
                st.error(f"CSV Header Error. Found: {df.columns.tolist()}")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

    
    manual_fixes = {
        'gb-eng': 'England', 'gb-sct': 'Scotland', 'gb-wls': 'Wales', 'gb-nir': 'Northern Ireland',
        'xk': 'Kosovo', 'eu': 'European Union', 'un': 'United Nations', 'us': 'United States'
    }
    code_to_name.update(manual_fixes)

    return gallery_feats, gallery_labels, gallery_paths, idx_to_class, code_to_name

@st.cache_resource
def load_classification_model(model_path, num_classes):
    model = FlagRecognitionModel(num_classes=num_classes)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


val_transforms = transforms.Compose([
    SquarePad(),
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def resize_for_display(image, fixed_width=400):
    aspect_ratio = image.height / image.width
    new_height = int(fixed_width * aspect_ratio)
    return image.resize((fixed_width, new_height))

st.title("🏳️ FlagExplorer")
st.markdown("**Created by: Biraj Koirala (st124371)**")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Configuration")
    conf_thres = st.slider("YOLO Confidence", 0.1, 1.0, 0.25, 0.05)
    st.divider()
    
    input_mode = st.radio("Input Source", ["Sample Gallery", "Upload Image"])
    input_image = None
    selected_file_name = None
    
    if input_mode == "Upload Image":
        uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
        if uploaded_file:
            input_image = Image.open(uploaded_file).convert('RGB')
    else:
        st.subheader("Sample Gallery")
        if os.path.exists(TEST_IMAGES_DIR):
            all_files = sorted([f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith(('png', 'jpg', 'jpeg'))])
            
            selected_file = st.selectbox("Select an image:", all_files)
            
            if selected_file:
                selected_path = os.path.join(TEST_IMAGES_DIR, selected_file)
                if os.path.exists(selected_path):
                    input_image = Image.open(selected_path).convert('RGB')
                    st.image(input_image, caption="Selected Image", use_container_width=True)
        else:
            st.error("Test images directory not found.")

gallery_feats, gallery_labels, gallery_paths, idx_to_class, code_to_name = load_gallery_and_metadata(GALLERY_PATH, CSV_PATH)

if gallery_feats is not None:
    num_classes = len(idx_to_class)
    classify_model = load_classification_model(ARCFACE_MODEL_PATH, num_classes)
    yolo_model = load_yolo_model(YOLO_MODEL_PATH)

    if input_image:
        
        # SECTION 1: OBJECT DETECTION
        st.header("1. Object Detection")
        
        with st.spinner("Running YOLOv8..."):
            results = yolo_model.predict(source=input_image, conf=conf_thres, save=False)
            result = results[0]
            boxes = result.boxes
            
            res_plotted = result.plot() 
            res_plotted_rgb = res_plotted[..., ::-1]
            detected_pil = Image.fromarray(res_plotted_rgb)

        c_det1, c_det2 = st.columns(2)
        with c_det1:
            st.image(input_image, caption="Original Image", width=400)
        with c_det2:
            st.image(detected_pil, caption=f"YOLO Output ({len(boxes)} detected)", width=400)

        # SECTION 2: CLASSIFICATION
        if len(boxes) > 0:
            st.divider()
            st.header(f"2. Flag Classification ({len(boxes)} Flags)")
            
            for i, box in enumerate(boxes):
                st.markdown(f"**Detection #{i+1}**")
                
                # Crop
                coords = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = coords
                cropped_flag = input_image.crop((x1, y1, x2, y2))
                
                # Inference
                img_tensor = val_transforms(cropped_flag).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    query_emb = classify_model(img_tensor).cpu()
                
                # Similarity
                sims = torch.mm(query_emb, gallery_feats.t())
                scores, indices = torch.topk(sims, k=5, dim=1)
                scores = scores[0].numpy()
                indices = indices[0].numpy()

            
                # Layout: [ Query (Medium) | Match1 (Small) | Match2 (Small) ... ]
                # Ratios: 2 : 1 : 1 : 1 : 1 : 1
                grid = st.columns([1.5, 0.8, 0.8, 0.8, 0.8, 0.8])
                
                # ... existing code for creating columns ...
                # Column 0: The Query (Restricted width)
                with grid[0]:
                    st.image(cropped_flag, caption="Query", width=200)
                
                # Columns 1-5: The Matches
                for j in range(5):
                    idx = indices[j]
                    score = scores[j]
                    
                    # 1. Get Class and Name Info
                    match_label_idx = gallery_labels[idx].item()
                    match_code = idx_to_class[match_label_idx] # This is the folder name (e.g., 'np', 'us')
                    display_code = match_code.lower().strip()
                    match_name = code_to_name.get(display_code, display_code.upper())
                    
                    # 2. FIX: Reconstruct Path Dynamically
                    # Get just the filename from the stored path (e.g., 'aug_0_123.jpg')
                    stored_path = gallery_paths[idx]
                    filename = os.path.basename(stored_path)
                    
                    # Build the correct local path: output-flags/country_code/filename
                    full_img_path = os.path.join(GALLERY_ROOT_DIR, match_code, filename)
                    
                    with grid[j+1]:
                        # 3. Check and Display
                        if os.path.exists(full_img_path):
                            m_img = Image.open(full_img_path).convert('RGB')
                            st.image(m_img, width=100)
                        else:
                            # Debug info if image is still missing
                            st.warning("Img Not Found") 
                            # st.caption(f"{match_code}/{filename}") # Uncomment to debug paths
                        
                        # Info
                        score_color = "green" if score > 0.6 else "red"
                        st.markdown(f"""
                        <div class="small-font">
                            <b>{match_name}</b><br>
                            <span style='color:{score_color}'>Sim: {score:.2f}</span>
                        </div>
                        """, unsafe_allow_html=True)
                # Column 0: The Query (Restricted width)
                
                st.divider()
        else:
            st.info("No flags detected in this image.")

else:
    st.error(f"Gallery Index not found at {GALLERY_PATH}. Please run the indexing script first.")