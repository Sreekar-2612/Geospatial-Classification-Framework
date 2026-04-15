"""
Quick script to retrain Random Forest with 180-feature extraction (fixed HOG).
Run this to update rf_baseline.joblib and rf_scaler.joblib
"""
import sys
from pathlib import Path
import cv2
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Setup paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "2750"
MODELS_DIR = PROJECT_ROOT / "models"

# Import feature extraction
sys.path.append(str(PROJECT_ROOT))
from src.features import extract_lulc_features

# Find classes
all_classes = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
print(f"✓ Found {len(all_classes)} classes: {all_classes}")

# Load data
X, y = [], []
for class_idx, class_name in enumerate(all_classes):
    class_dir = DATA_DIR / class_name
    images = list(class_dir.glob("*.jpg"))
    print(f"  Loading {class_name}: {len(images)} images")
    
    for img_path in tqdm(images, desc=f"  {class_name}", leave=False):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Extract features
        feat = extract_lulc_features(img)
        X.append(feat)
        y.append(class_idx)

X = np.array(X)
y = np.array(y)

print(f"\n✓ Loaded data: X shape {X.shape}, y shape {y.shape}")
print(f"  Features per image: {X.shape[1]} (expected 180)")
if X.shape[1] != 180:
    print(f"  ⚠️ ERROR: Expected 180 features, got {X.shape[1]}")
    sys.exit(1)

# Train RF with 180 features
print("\n🔄 Training Random Forest (300 estimators)...")
rf_model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, verbose=1)
rf_model.fit(X, y)

print(f"✓ Train accuracy: {rf_model.score(X, y):.4f}")

# Train scaler
print("\n🔄 Training StandardScaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
rf_model_scaled = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf_model_scaled.fit(X_scaled, y)
print(f"✓ Scaled train accuracy: {rf_model_scaled.score(X_scaled, y):.4f}")

# Save models
print("\n💾 Saving models...")
joblib.dump(rf_model, MODELS_DIR / "rf_baseline.joblib")
joblib.dump(scaler, MODELS_DIR / "rf_scaler.joblib")
print(f"✓ Saved: {MODELS_DIR / 'rf_baseline.joblib'}")
print(f"✓ Saved: {MODELS_DIR / 'rf_scaler.joblib'}")

print("\n✅ Done! Restart Streamlit to load new models.")
