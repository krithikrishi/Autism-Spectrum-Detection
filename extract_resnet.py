import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

# ==============================
# CONFIG
# ==============================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE_DIR = r"C:\Users\Krithik Rishi\OneDrive\Desktop\dataset\ABIDE\Combined Data"

AUTISTIC_DIR = os.path.join(BASE_DIR, "Autistic")
CONTROL_DIR = os.path.join(BASE_DIR, "Typical_Control")
OUTPUT_FILE = os.path.join(BASE_DIR, "nii_resnet_features_multislice.csv")

IMG_SIZE = 224
SLICES_PER_VOLUME = 4  # Keep as 4 for consistency with the scholar
BATCH_SIZE = 16

# ==============================
# MODEL SETUP (ResNet18)
# ==============================
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
resnet.fc = nn.Identity()  # Remove classifier
resnet = resnet.to(DEVICE)
resnet.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ==============================
# FUNCTION: Extract slices (2D-safe)
# ==============================
def extract_slices(nii_path, num_slices=SLICES_PER_VOLUME):
    """Handles both 2D and 3D .nii files safely."""
    nii = nib.load(nii_path).get_fdata()
    nii = np.squeeze(nii)  # removes any extra dimensions

    # If 2D, just duplicate it as "slices" for consistency
    if nii.ndim == 2:
        slices = [nii] * num_slices
    elif nii.ndim == 3:
        depth = nii.shape[2]
        indices = np.linspace(0, depth - 1, num_slices, dtype=int)
        slices = [nii[:, :, i] for i in indices]
    else:
        return []

    imgs = []
    for img_2d in slices:
        img_2d = (img_2d - np.min(img_2d)) / (np.ptp(img_2d) + 1e-8)
        img_2d = np.uint8(img_2d * 255)
        img = Image.fromarray(img_2d).convert("RGB")
        imgs.append(transform(img))
    return imgs

# ==============================
# MAIN EXTRACTION LOOP
# ==============================
def process_folder(folder, label):
    all_features = []
    nii_files = [f for f in os.listdir(folder) if f.endswith(".nii")]
    print(f"\nProcessing {os.path.basename(folder)}: {len(nii_files)} files")

    with torch.no_grad():
        for nii_file in tqdm(nii_files):
            nii_path = os.path.join(folder, nii_file)
            slices = extract_slices(nii_path)
            if not slices:
                continue

            imgs = torch.stack(slices).to(DEVICE)
            feats = resnet(imgs).cpu().numpy()

            for i, f in enumerate(feats):
                record = {"File": f"{nii_file}_slice{i}", "Label": label}
                for j, val in enumerate(f):
                    record[f"feat_{j}"] = val
                all_features.append(record)
    return all_features

# ==============================
# EXECUTE
# ==============================
print(f"✅ Using device: {DEVICE}")
autistic_feats = process_folder(AUTISTIC_DIR, 1)
control_feats = process_folder(CONTROL_DIR, 0)

# Combine and save
df = pd.DataFrame(autistic_feats + control_feats)

if not df.empty:
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Saved multi-slice ResNet18 features → {OUTPUT_FILE}")
    print(f"Shape: {df.shape}")
    print(df["Label"].value_counts())
else:
    print("⚠️ No features extracted — check if .nii files were readable.")
