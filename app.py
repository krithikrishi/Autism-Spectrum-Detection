import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import nibabel as nib
import cv2
import tempfile
import os
import joblib
import shutil
import gzip
from io import BytesIO

# ==============================
# CONFIG
# ==============================
BASE_DIR = r"C:\Users\Krithik Rishi\OneDrive\Desktop\ABIDE\Combined Data"
MODEL_PATH = os.path.join(BASE_DIR, "final_resnet_pca16_model.joblib")
PCA_PATH = os.path.join(BASE_DIR, "pca_transform.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "scaler_transform.joblib")

st.set_page_config(page_title="Autism Detection", page_icon="   ")

# ==============================
# LOAD MODELS
# ==============================
@st.cache_resource
def load_models():
    model = joblib.load(MODEL_PATH)
    pca = joblib.load(PCA_PATH)
    scaler = joblib.load(SCALER_PATH)

    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet.fc = torch.nn.Identity()
    resnet.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet.to(device)
    return model, pca, scaler, resnet, device

model, pca, scaler, resnet, device = load_models()

# ==============================
# TRANSFORMS (ResNet18)
# ==============================
def resnet_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

# ==============================
# FEATURE EXTRACTION
# ==============================
def extract_resnet_features(img_tensor):
    with torch.no_grad():
        feats = resnet(img_tensor.to(device))
    return feats.cpu().numpy()

# ==============================
# UTILS
# ==============================
def is_gzip_bytes(b: bytes) -> bool:
    # GZIP magic bytes: 1f 8b
    return len(b) >= 2 and b[0] == 0x1F and b[1] == 0x8B

# ==============================
# NII HANDLER (robust)
# ==============================
def process_nii(uploaded_file):
    """
    Robustly reads NIfTI:
      - Detects gzip (handles .nii.gz uploaded as .nii)
      - Writes to a proper temp path with the correct suffix
      - Reads with nibabel
      - Returns a single 2D slice as a 3-channel tensor with ImageNet normalization
    """
    tmp_path = None
    try:
        raw = uploaded_file.getvalue()

        # Choose suffix based on gzip magic bytes
        suffix = ".nii.gz" if is_gzip_bytes(raw) else ".nii"

        # Write to a temp file with correct suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            tmp.write(raw)

        # If gzip, just let nibabel load .nii.gz; else .nii
        nii_obj = nib.load(tmp_path)
        nii_data = nii_obj.get_fdata().astype(np.float32)

        # Accept 2D or 3D (some ABIDE files are 2D packed in NIfTI)
        if nii_data.ndim == 3:
            # middle axial slice
            z = nii_data.shape[2] // 2
            slice2d = nii_data[:, :, z]
        elif nii_data.ndim == 2:
            slice2d = nii_data
        else:
            st.error(f"Unsupported NII shape: {nii_data.shape}")
            return None

        # Normalize intensities safely
        slice2d = np.nan_to_num(slice2d)
        hi = np.percentile(slice2d, 99)
        if hi > 0:
            slice2d = np.clip(slice2d, 0, hi) / hi
        else:
            # fallback to min-max if 99th percentile is zero
            mn, mx = float(slice2d.min()), float(slice2d.max())
            if mx > mn:
                slice2d = (slice2d - mn) / (mx - mn)
            else:
                slice2d = np.zeros_like(slice2d, dtype=np.float32)

        # Resize → 224x224 then convert to 3-channel PIL and apply ImageNet normalization
        slice2d = cv2.resize(slice2d, (224, 224), interpolation=cv2.INTER_AREA)
        # to RGB PIL
        pil_img = Image.fromarray((slice2d * 255).astype(np.uint8)).convert("RGB")
        tensor = resnet_transform()(pil_img).unsqueeze(0)  # [1,3,224,224]
        return tensor

    except Exception as e:
        st.error(f"Error reading NII file: {e}")
        return None
    finally:
        # Clean up temp file if it was created
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

# ==============================
# PNG HANDLER (ResNet18-consistent)
# ==============================
def process_png(uploaded_file):
    try:
        img = Image.open(uploaded_file).convert("RGB")
        tensor = resnet_transform()(img).unsqueeze(0)
        return tensor
    except Exception as e:
        st.error(f"Error processing PNG: {e}")
        return None

# ==============================
# PREDICT
# ==============================
def predict_from_feats(feats_512):
    # feats_512: numpy [1,512]
    feats_scaled = scaler.transform(feats_512)
    feats_pca = pca.transform(feats_scaled)  # [1,16]
    pred = model.predict(feats_pca)[0]
    # Optional: show class probabilities if model supports it
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(feats_pca)[0]  # [p_control, p_autistic]
    return pred, proba

# ==============================
# UI
# ==============================
st.title(" Autism Detection (ResNet18 + PCA + RF/XGB/SVC)")
st.caption("Upload a NIfTI (.nii / .nii.gz) or PNG image to predict **Autistic** vs **Typical Control**.")

file_type = st.radio("Select file type:", ("NII (.nii / .nii.gz)", "PNG (.png)"))
if file_type.startswith("NII"):
    uploaded = st.file_uploader("Upload a NIfTI file", type=["nii", "nii.gz"])
else:
    uploaded = st.file_uploader("Upload a PNG image", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    st.success(f"File received: {uploaded.name}")

    if file_type.startswith("NII"):
        tensor = process_nii(uploaded)
        preview_note = "Processed a NIfTI slice (center axial)."
    else:
        tensor = process_png(uploaded)
        preview_note = None

    if tensor is not None:
        # Extract 512-dim ResNet18 features
        feats = extract_resnet_features(tensor)  # [1,512]
        pred, proba = predict_from_feats(feats)

        # Mapping: 0 = Control, 1 = Autistic
        label = "**Typical Control**" if pred == 0 else "**Autistic**"
        st.markdown(f"### Prediction: {label}")

        if proba is not None:
            st.write(f"**Probabilities** → Control: {proba[0]:.3f} | Autistic: {proba[1]:.3f}")

        if file_type.startswith("NII"):
            st.info(preview_note)
        else:
            st.image(uploaded, caption="Uploaded Image", use_container_width=True)

