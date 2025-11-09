import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

# ==============================
# CONFIG
# ==============================
BASE_DIR = r"C:\Users\Krithik Rishi\OneDrive\Desktop\dataset\ABIDE\Combined Data"
INPUT_FILE = os.path.join(BASE_DIR, "nii_resnet_features_multislice.csv")
OUTPUT_PCA_FILE = os.path.join(BASE_DIR, "nii_resnet_features_multislice_pca16.csv")
MODEL_FILE = os.path.join(BASE_DIR, "final_resnet_pca16_model.joblib")
PCA_MODEL_FILE = os.path.join(BASE_DIR, "pca_transform.joblib")
SCALER_FILE = os.path.join(BASE_DIR, "scaler_transform.joblib")
SEED = 42

# ==============================
# LOAD AND CLEAN DATA
# ==============================
print("📁 Loading slice-level feature dataset...")
df = pd.read_csv(INPUT_FILE)

if 'SUB_ID' not in df.columns:
    df.rename(columns={df.columns[0]: 'SUB_ID'}, inplace=True)

feature_cols = [c for c in df.columns if c not in ['SUB_ID', 'Label']]
df = df.dropna(subset=feature_cols + ['Label'])

print(f"✅ Cleaned dataset: {len(df)} slices, {len(feature_cols)} numeric features")
print(f"Label distribution: {df['Label'].value_counts().to_dict()}")

X = df[feature_cols].values
y = df['Label'].values
groups = df['SUB_ID'].values

# ==============================
# SCALING + PCA REDUCTION
# ==============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=16, random_state=SEED)
X_pca = pca.fit_transform(X_scaled)

# save the scaler + PCA for inference in Streamlit
joblib.dump(scaler, SCALER_FILE)
joblib.dump(pca, PCA_MODEL_FILE)

print(f"💾 Scaler saved to: {SCALER_FILE}")
print(f"💾 PCA transformer saved to: {PCA_MODEL_FILE}")

# save the PCA-reduced dataset for inspection
df_pca = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(16)])
df_pca['Label'] = y
df_pca['SUB_ID'] = groups
df_pca.to_csv(OUTPUT_PCA_FILE, index=False)
print(f"💾 Saved PCA(16) slice dataset → {OUTPUT_PCA_FILE}  (shape={df_pca.shape})")

# ==============================
# SUBJECT-LEVEL SPLIT
# ==============================
splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=SEED)
train_idx, test_idx = next(splitter.split(X_pca, y, groups))
X_train, X_test = X_pca[train_idx], X_pca[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
groups_train, groups_test = groups[train_idx], groups[test_idx]

print(f"🧠 Subjects (train/test): {len(np.unique(groups_train))}/{len(np.unique(groups_test))}")

# ==============================
# MODEL TRAINING & EVALUATION
# ==============================
def evaluate_model(model, name):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    print(f"{name} → Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}")
    return acc, model, preds

print("\n🚀 Training models with PCA(16) features...")

svc = SVC(kernel='rbf', C=2, probability=True, random_state=SEED)
svc_acc, svc_model, svc_preds = evaluate_model(svc, "SVC")

xgb = XGBClassifier(max_depth=8, learning_rate=0.05, eval_metric='logloss', use_label_encoder=False)
xgb_acc, xgb_model, xgb_preds = evaluate_model(xgb, "XGBoost")

rf = RandomForestClassifier(n_estimators=400, max_depth=None, random_state=SEED)
rf_acc, rf_model, rf_preds = evaluate_model(rf, "RandomForest")

ensemble = VotingClassifier(estimators=[
    ('svc', svc_model),
    ('xgb', xgb_model),
    ('rf', rf_model)
], voting='soft')

ensemble.fit(X_train, y_train)
ensemble_preds = ensemble.predict(X_test)
ensemble_acc = accuracy_score(y_test, ensemble_preds)
ensemble_f1 = f1_score(y_test, ensemble_preds)
print(f"\n🤖 Ensemble → Accuracy: {ensemble_acc:.3f} | F1: {ensemble_f1:.3f}")

# ==============================
# SUBJECT-LEVEL AGGREGATION
# ==============================
subject_df = pd.DataFrame({'SUB_ID': groups_test, 'True': y_test, 'Pred': ensemble_preds})
subject_majority = subject_df.groupby('SUB_ID').agg(lambda x: round(x.mean())).reset_index()

subj_acc = accuracy_score(subject_majority['True'], subject_majority['Pred'])
subj_prec = precision_score(subject_majority['True'], subject_majority['Pred'])
subj_rec = recall_score(subject_majority['True'], subject_majority['Pred'])
subj_f1 = f1_score(subject_majority['True'], subject_majority['Pred'])

print(f"\n🧠 Subject-level (majority vote over slices):")
print(f"Accuracy: {subj_acc:.3f} | Precision: {subj_prec:.3f} | Recall: {subj_rec:.3f} | F1: {subj_f1:.3f}")

# ==============================
# SAVE BEST MODEL
# ==============================
best_model = max([(svc_acc, svc_model, "SVC"),
                  (xgb_acc, xgb_model, "XGBoost"),
                  (rf_acc, rf_model, "RandomForest"),
                  (ensemble_acc, ensemble, "Ensemble")],
                 key=lambda x: x[0])

joblib.dump(best_model[1], MODEL_FILE)
print(f"\n🏆 Best Model: {best_model[2]} | Accuracy: {best_model[0]:.3f}")
print(f"💾 Model saved to: {MODEL_FILE}")
