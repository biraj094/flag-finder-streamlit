import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.transforms import functional as TF
from ultralytics import YOLO
from PIL import Image, ImageOps
import numpy as np
import os
import pandas as pd

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Flag Detective",
    layout="wide",
    page_icon="🏳️",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR BETTER UI ---
st.markdown("""
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        h1 { margin-bottom: 0px; }
        .stDataFrame { font-size: 0.85rem; }
        /* Style for the result cards */
        .result-card {
            background-color: #f0f2f6;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# --- CONFIGURATION PATHS ---
YOLO_MODEL_PATH = 'best.pt' 
ARCFACE_MODEL_PATH = 'v9_arcface_squarepad_model-train-val-new-dataset.pth'
GALLERY_PATH = 'flag_gallery_index.pth'
CSV_PATH = 'country_codes_processed.csv'
TEST_IMAGES_DIR = 'test_images' 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MODEL CLASSES (MUST MATCH TRAINING) ---
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

# --- CACHED LOADERS ---

@st.cache_resource
def load_yolo_model(path):
    return YOLO(path)

@st.cache_resource
def load_gallery_and_metadata(gallery_path, csv_path):
    if not os.path.exists(gallery_path):
        return None, None, None, None, None

    # 1. Load Gallery
    data = torch.load(gallery_path, map_location=DEVICE)
    gallery_feats = data['feats'].to(DEVICE)
    gallery_labels = data['labels']
    gallery_paths = data['paths']
    
    # Robustly handle keys (ensure they are integers)
    raw_idx_to_class = data['idx_to_class']
    idx_to_class = {int(k): v for k, v in raw_idx_to_class.items()}

    # 2. Load Names CSV
    code_to_name = {}
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            df['Alpha-2 code'] = df['Alpha-2 code'].astype(str).str.lower().str.strip()
            df['Country'] = df['Country'].str.strip()
            code_to_name = dict(zip(df['Alpha-2 code'], df['Country']))
        except:
            pass

    # 3. Manual Patch for Dataset Specifics
    manual_fixes = {
        'gb-eng': 'England', 'gb-sct': 'Scotland', 'gb-wls': 'Wales', 'gb-nir': 'Northern Ireland',
        'xk': 'Kosovo', 'eu': 'European Union', 'un': 'United Nations'
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

# --- HELPERS ---
val_transforms = transforms.Compose([
    SquarePad(),
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def resize_for_display(image, fixed_height=400):
    aspect_ratio = image.width / image.height
    new_width = int(fixed_height * aspect_ratio)
    return image.resize((new_width, fixed_height))

# --- MAIN APP ---

st.title("🏳️ Flag Detective")
st.markdown("Upload an image to detect flags and identify the country.")

# SIDEBAR
with st.sidebar:
    st.header("Configuration")
    conf_thres = st.slider("Detection Confidence", 0.1, 1.0, 0.25, 0.05)
    
    st.divider()
    st.subheader("Input Source")
    input_option = st.radio("Choose Source:", ["📂 Sample Gallery", "⬆️ Upload Image"])

    input_image = None
    
    if input_option == "⬆️ Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
        if uploaded_file:
            input_image = Image.open(uploaded_file).convert('RGB')
    else:
        if os.path.exists(TEST_IMAGES_DIR):
            sample_files = sorted([f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith(('png', 'jpg', 'jpeg'))])
            selected_sample = st.selectbox("Select a sample image:", sample_files)
            if selected_sample:
                input_image = Image.open(os.path.join(TEST_IMAGES_DIR, selected_sample)).convert('RGB')
        else:
            st.error("Test images directory not found.")

# LOAD RESOURCES
gallery_feats, gallery_labels, gallery_paths, idx_to_class, code_to_name = load_gallery_and_metadata(GALLERY_PATH, CSV_PATH)

if not gallery_feats is None:
    num_classes = len(idx_to_class)
    classify_model = load_classification_model(ARCFACE_MODEL_PATH, num_classes)
    yolo_model = load_yolo_model(YOLO_MODEL_PATH)

    # MAIN LOGIC
    if input_image:
        # STEP 1: DETECTION
        with st.spinner("Analyzing image..."):
            # Run YOLO
            results = yolo_model.predict(source=input_image, conf=conf_thres, save=False)
            result = results[0]
            boxes = result.boxes
            
            # Create Plot
            res_plotted = result.plot() # BGR numpy array
            res_plotted_rgb = res_plotted[..., ::-1] # Convert to RGB
            detected_pil = Image.fromarray(res_plotted_rgb)

        # UI: DISPLAY STEP 1 (Side by Side)
        st.subheader("1. Detection Results")
        col_orig, col_det = st.columns(2)
        
        # Resize for consistent UI
        disp_orig = resize_for_display(input_image)
        disp_det = resize_for_display(detected_pil)

        with col_orig:
            st.image(disp_orig, caption="Original Image", use_container_width=False)
        with col_det:
            st.image(disp_det, caption=f"YOLO Detection ({len(boxes)} found)", use_container_width=False)

        # STEP 2: CLASSIFICATION
        if len(boxes) > 0:
            st.divider()
            st.subheader(f"2. Classification ({len(boxes)} Flags Found)")
            
            # Grid Layout for Results
            for i, box in enumerate(boxes):
                # UI: Use a container with border for each flag
                with st.container(border=True):
                    # Logic: Crop
                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = coords
                    cropped_flag = input_image.crop((x1, y1, x2, y2))
                    
                    # Logic: Inference
                    img_tensor = val_transforms(cropped_flag).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        query_emb = classify_model(img_tensor).cpu()
                    
                    # Logic: Search
                    sims = torch.mm(query_emb, gallery_feats.t())
                    scores, indices = torch.topk(sims, k=5, dim=1)
                    scores = scores[0].numpy()
                    indices = indices[0].numpy()

                    # UI: Split into Thumbnail and Data
                    c_thumb, c_table = st.columns([1, 4])
                    
                    with c_thumb:
                        # Center the image vertically in the column if possible, or just show it
                        st.image(cropped_flag, width=150, caption=f"Flag #{i+1}")
                    
                    with c_table:
                        # Prepare Dataframe
                        results_data = []
                        for j, idx in enumerate(indices):
                            match_label_idx = gallery_labels[idx].item()
                            match_code = idx_to_class[match_label_idx] # Integer lookup
                            match_name = code_to_name.get(match_code, match_code.upper())
                            similarity = scores[j]
                            
                            results_data.append({
                                "Rank": j+1,
                                "Country": match_name,
                                "Code": match_code,
                                "Confidence": similarity
                            })
                        
                        df_res = pd.DataFrame(results_data)
                        
                        # Use Streamlit's column configuration for a prettier table
                        st.dataframe(
                            df_res,
                            column_config={
                                "Rank": st.column_config.NumberColumn("Rank", format="#%d"),
                                "Country": st.column_config.TextColumn("Country", width="medium"),
                                "Code": st.column_config.TextColumn("Code"),
                                "Confidence": st.column_config.ProgressColumn(
                                    "Confidence", min_value=0, max_value=1, format="%.2f"
                                ),
                            },
                            hide_index=True,
                            use_container_width=True
                        )
        else:
            st.info("No flags were detected. Try lowering the confidence threshold in the sidebar.")

else:
    st.error("Gallery index not loaded. Please ensure 'flag_gallery_index.pth' is in the directory.")