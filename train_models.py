"""
Train and save models locally for the Streamlit LULC dashboard.

This script downloads EuroSAT data (if not present), maps the 10 EuroSAT classes
to the 5 dashboard classes (Agriculture, Buildings, Forest, Roads, Water),
trains both the Random Forest and CNN (ResNet-18) models, and saves them
to the models/ directory.

Usage:
    python train_models.py
"""

import os
import ssl
import sys
import shutil
import zipfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

DASHBOARD_CLASSES = ["Agriculture", "Buildings", "Forest", "Roads", "Water"]

EUROSAT_TO_DASHBOARD = {
    "AnnualCrop": "Agriculture",
    "PermanentCrop": "Agriculture",
    "Pasture": "Agriculture",
    "HerbaceousVegetation": "Agriculture",
    "Forest": "Forest",
    "Residential": "Buildings",
    "Industrial": "Buildings",
    "Highway": "Roads",
    "River": "Water",
    "SeaLake": "Water",
}


def download_eurosat():
    """Download and extract EuroSAT if not already present."""
    if DATA_DIR.exists() and any(DATA_DIR.iterdir()):
        print(f"Data already exists at {DATA_DIR}")
        return

    print("Downloading EuroSAT RGB dataset (~90 MB)...")
    zip_path = PROJECT_ROOT / "EuroSAT.zip"
    url = "https://madm.dfki.de/files/sentinel/EuroSAT.zip"

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    opener = urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=ctx)
    )
    urllib.request.install_opener(opener)

    urllib.request.urlretrieve(url, str(zip_path))

    print("Extracting...")
    raw_dir = PROJECT_ROOT / "EuroSAT_raw"
    with zipfile.ZipFile(str(zip_path), "r") as z:
        z.extractall(str(raw_dir))

    raw_root = raw_dir / "2750"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for cls_dir in raw_root.iterdir():
        if cls_dir.is_dir():
            shutil.copytree(str(cls_dir), str(DATA_DIR / cls_dir.name))

    zip_path.unlink(missing_ok=True)
    shutil.rmtree(str(raw_dir), ignore_errors=True)

    classes = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
    print(f"Done! Classes found: {classes}")


def build_mapped_dataset():
    """
    Build a mapped dataset directory where the 10 EuroSAT classes are merged
    into the 5 dashboard classes.
    """
    mapped_dir = PROJECT_ROOT / "data" / "mapped"
    if mapped_dir.exists() and any(mapped_dir.iterdir()):
        classes = sorted([d.name for d in mapped_dir.iterdir() if d.is_dir()])
        print(f"Mapped dataset already exists: {classes}")
        return mapped_dir

    print("Mapping EuroSAT classes to dashboard classes...")
    mapped_dir.mkdir(parents=True, exist_ok=True)

    for dash_cls in DASHBOARD_CLASSES:
        (mapped_dir / dash_cls).mkdir(exist_ok=True)

    for eurosat_cls, dash_cls in EUROSAT_TO_DASHBOARD.items():
        src = DATA_DIR / eurosat_cls
        dst = mapped_dir / dash_cls
        if not src.exists():
            print(f"  Warning: {src} not found, skipping")
            continue
        for img_file in src.glob("*.jpg"):
            target = dst / f"{eurosat_cls}_{img_file.name}"
            if not target.exists():
                shutil.copy2(str(img_file), str(target))

    classes = sorted([d.name for d in mapped_dir.iterdir() if d.is_dir()])
    for c in classes:
        count = len(list((mapped_dir / c).glob("*.jpg")))
        print(f"  {c}: {count} images")
    return mapped_dir


def train_rf(mapped_dir):
    """Train Random Forest model using hand-crafted features."""
    print("\n" + "=" * 60)
    print("Training Random Forest Model")
    print("=" * 60)

    sys.path.insert(0, str(PROJECT_ROOT))
    from src.features import extract_lulc_features

    data = []
    labels = []
    classes = sorted([d.name for d in mapped_dir.iterdir() if d.is_dir()])

    print(f"Feature extraction for classes: {classes}")
    for i, cls in enumerate(classes):
        paths = list((mapped_dir / cls).glob("*.jpg"))[:500]
        for img_path in tqdm(paths, desc=f"  {cls}"):
            img = cv2.imread(str(img_path))
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                feat = extract_lulc_features(img_rgb)
                data.append(feat)
                labels.append(i)

    X = np.array(data)
    y = np.array(labels)
    print(f"Feature matrix shape: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training Random Forest (n_estimators=100)...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    acc = accuracy_score(y_test, rf.predict(X_test))
    print(f"\nRF Test Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, rf.predict(X_test), target_names=classes))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    rf_path = MODELS_DIR / "rf_baseline.joblib"
    joblib.dump(rf, rf_path)
    print(f"Random Forest saved to {rf_path}")

    return acc


def train_cnn(mapped_dir):
    """Train CNN (ResNet-18) model."""
    print("\n" + "=" * 60)
    print("Training CNN (ResNet-18) Model")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = datasets.ImageFolder(str(mapped_dir), transform=transform)
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"Classes: {class_names} ({num_classes} classes)")

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    if device.type == "cpu":
        num_epochs = 3
        batch_sz = 16
        print(f"CPU mode: training for {num_epochs} epochs with batch size {batch_sz}")
    else:
        num_epochs = 5
        batch_sz = 32
        print(f"GPU mode: training for {num_epochs} epochs with batch size {batch_sz}")

    train_loader = DataLoader(train_dataset, batch_size=batch_sz, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_sz, shuffle=False, num_workers=0)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_acc = 0.0
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "cnn_final.pth"

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        pbar = tqdm(train_loader, desc=f"  Epoch {epoch + 1}/{num_epochs}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_loss = running_loss / train_size
        epoch_acc = running_corrects.double() / train_size
        print(f"  Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), model_path)
            print(f"  -> Saved best model (acc={best_acc:.4f})")

    print(f"\nEvaluating on test set...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    test_acc = accuracy_score(all_labels, all_preds)
    print(f"\nCNN Test Accuracy: {test_acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    print(f"CNN model saved to {model_path}")

    return test_acc


def update_metrics(rf_acc, cnn_acc):
    """Update report/metrics.csv with the trained model accuracies."""
    import pandas as pd

    metrics_path = PROJECT_ROOT / "report" / "metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
    else:
        df = pd.DataFrame(columns=["Model", "Accuracy"])

    if "Model" not in df.columns:
        df = pd.DataFrame(columns=["Model", "Accuracy"])

    for name, acc in [("DIP Features + RF", rf_acc), ("ResNet-18 (CNN)", cnn_acc)]:
        if name in df["Model"].values:
            df.loc[df["Model"] == name, "Accuracy"] = acc
        else:
            new_row = pd.DataFrame({"Model": [name], "Accuracy": [acc]})
            df = pd.concat([df, new_row], ignore_index=True)

    df.to_csv(metrics_path, index=False)
    print(f"\nMetrics updated at {metrics_path}")


def main():
    print("=" * 60)
    print("LULC Model Training Pipeline")
    print("=" * 60)

    download_eurosat()
    mapped_dir = build_mapped_dataset()

    rf_acc = train_rf(mapped_dir)
    cnn_acc = train_cnn(mapped_dir)

    update_metrics(rf_acc, cnn_acc)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"  RF model:  {MODELS_DIR / 'rf_baseline.joblib'}")
    print(f"  CNN model: {MODELS_DIR / 'cnn_final.pth'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
