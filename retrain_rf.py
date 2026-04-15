import cv2
import numpy as np
import joblib
from pathlib import Path
import sys

sys.path.insert(0, str(Path().cwd()))
from src.features import extract_lulc_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")

classes = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
data, labels = [], []

for i, cls in enumerate(classes):
    img_paths = list((DATA_DIR / cls).glob("*.jpg"))
    print(f"Extracting features for {cls} ({len(img_paths)} images)...")
    for img_path in tqdm(img_paths[:500]): # Limit to 500 per class for rapid retrain to save time
        img = cv2.imread(str(img_path))
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            data.append(extract_lulc_features(img_rgb))
            labels.append(i)

X, y = np.array(data), np.array(labels)
print(f"Feature matrix shape: {X.shape}")

scaler = StandardScaler()
X = scaler.fit_transform(X)

print("Training RandomForest model...")
rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
rf.fit(X, y)

MODELS_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(rf, MODELS_DIR / "rf_baseline.joblib")
joblib.dump(scaler, MODELS_DIR / "rf_scaler.joblib")
print("Saved retrained RF model and scaler!")
