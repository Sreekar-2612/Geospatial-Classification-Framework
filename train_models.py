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
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from collections import Counter

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

    zip_path = PROJECT_ROOT / "EuroSAT.zip"

    if not zip_path.exists():
        print("Downloading EuroSAT RGB dataset (~90 MB)...")
        url = "https://madm.dfki.de/files/sentinel/EuroSAT.zip"

        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        opener = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=ctx)
        )
        urllib.request.install_opener(opener)

        urllib.request.urlretrieve(url, str(zip_path))
    else:
        print(f"Using existing {zip_path}")

    print("Extracting...")
    raw_dir = PROJECT_ROOT / "EuroSAT_raw"
    with zipfile.ZipFile(str(zip_path), "r") as z:
        z.extractall(str(raw_dir))

    raw_root = raw_dir / "2750"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for cls_dir in raw_root.iterdir():
        if cls_dir.is_dir():
            shutil.copytree(str(cls_dir), str(DATA_DIR / cls_dir.name))

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
    """Train Random Forest model using hand-crafted features with StandardScaler."""
    print("\n" + "=" * 60)
    print("Training Random Forest Model")
    print("=" * 60)

    sys.path.insert(0, str(PROJECT_ROOT))
    from src.features import extract_lulc_features

    # Filter to classes with actual data
    all_classes = sorted([d.name for d in mapped_dir.iterdir() if d.is_dir()])
    classes_with_data = [cls for cls in all_classes if any((mapped_dir / cls).glob("*.jpg"))]
    
    if not classes_with_data:
        print("No images found in mapped dataset!")
        return 0.0
    
    print(f"Found {len(classes_with_data)} classes with data: {classes_with_data}")

    data = []
    labels = []
    class_to_idx = {cls: i for i, cls in enumerate(classes_with_data)}

    print(f"Feature extraction for classes: {classes_with_data}")
    for i, cls in enumerate(classes_with_data):
        paths = list((mapped_dir / cls).glob("*.jpg"))
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

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Training Random Forest (n_estimators=300, balanced)...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    acc = accuracy_score(y_test, rf.predict(X_test))
    print(f"\nRF Test Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, rf.predict(X_test), target_names=classes_with_data))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    rf_path = MODELS_DIR / "rf_baseline.joblib"
    scaler_path = MODELS_DIR / "rf_scaler.joblib"
    joblib.dump(rf, rf_path)
    joblib.dump(scaler, scaler_path)
    print(f"Random Forest saved to {rf_path}")
    print(f"Scaler saved to {scaler_path}")

    return acc


def train_cnn(mapped_dir):
    """Train CNN (EfficientNet-B1) with class-balanced sampling, strong augmentation,
    warmup, cosine LR, and a two-phase freeze/unfreeze strategy."""
    print("\n" + "=" * 60)
    print("Training CNN (ResNet-18) Model — Tuned")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Filter to classes with actual images
    all_classes = sorted([d.name for d in mapped_dir.iterdir() if d.is_dir()])
    classes_with_data = [cls for cls in all_classes if any((mapped_dir / cls).glob("*.jpg"))]
    
    if not classes_with_data:
        print("No images found for CNN training!")
        return 0.0
    
    # Create a temporary dataset directory with only classes that have data
    temp_dataset_dir = PROJECT_ROOT / "data" / "temp_dataset"
    temp_dataset_dir.mkdir(parents=True, exist_ok=True)
    for cls in classes_with_data:
        (temp_dataset_dir / cls).mkdir(exist_ok=True)
        for img in (mapped_dir / cls).glob("*.jpg"):
            import shutil as sh
            target = temp_dataset_dir / cls / img.name
            if not target.exists():
                sh.copy2(str(img), str(target))
    
    full_dataset = datasets.ImageFolder(str(temp_dataset_dir), transform=None)
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"Classes: {class_names} ({num_classes} classes)")

    total = len(full_dataset)
    train_size = int(0.7 * total)
    val_size = int(0.1 * total)
    test_size = total - train_size - val_size

    gen = torch.Generator().manual_seed(42)
    train_subset, val_subset, test_subset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size], generator=gen,
    )

    # --- Class-balanced WeightedRandomSampler for training ---
    train_labels = [full_dataset.targets[i] for i in train_subset.indices]
    class_counts = Counter(train_labels)
    print(f"Train class distribution: {dict(sorted(class_counts.items()))}")
    weight_per_class = {c: 1.0 / count for c, count in class_counts.items()}
    sample_weights = [weight_per_class[label] for label in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=torch.Generator().manual_seed(42),
    )

    # Class-weighted loss as secondary balance signal
    class_weight_tensor = torch.tensor(
        [1.0 / class_counts.get(i, 1) for i in range(num_classes)], dtype=torch.float32
    )
    class_weight_tensor = class_weight_tensor / class_weight_tensor.sum() * num_classes
    class_weight_tensor = class_weight_tensor.to(device)

    train_dataset = _TransformSubset(train_subset, train_transform)
    val_dataset = _TransformSubset(val_subset, eval_transform)
    test_dataset = _TransformSubset(test_subset, eval_transform)

    if device.type == "cpu":
        num_epochs = 20
        batch_sz = 32
    else:
        num_epochs = 50
        batch_sz = 128
    freeze_epochs = 3
    print(f"Training for {num_epochs} epochs (backbone frozen for first {freeze_epochs}), batch size {batch_sz}")

    train_loader = DataLoader(train_dataset, batch_size=batch_sz, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_sz, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_sz, shuffle=False, num_workers=0)

    # --- Model setup with two-phase freeze/unfreeze ---
    model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    model = model.to(device)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)

    optimizer = optim.AdamW(model.classifier.parameters(), lr=5e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)

    best_val_acc = 0.0
    patience_counter = 0
    early_stop_patience = 12
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "cnn_final.pth"

    for epoch in range(num_epochs):
        # Phase 2: unfreeze backbone after freeze_epochs
        if epoch == freeze_epochs:
            print(f"\n  >>> Unfreezing backbone at epoch {epoch + 1}")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW([
                {"params": model.classifier.parameters(), "lr": 2e-3},
                {"params": (p for n, p in model.named_parameters()
                            if "classifier" not in n and p.requires_grad), "lr": 1e-4},
            ], weight_decay=1e-4)
            scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - freeze_epochs, eta_min=1e-7)

        model.train()
        running_loss = 0.0
        running_corrects = 0
        n_samples = 0

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
            running_corrects += torch.sum(preds == labels.data).item()
            n_samples += inputs.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()

        epoch_loss = running_loss / n_samples
        epoch_acc = running_corrects / n_samples

        model.eval()
        val_corrects = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels).item()
                val_total += labels.size(0)
        val_acc = val_corrects / val_total

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch + 1}/{num_epochs} — Loss: {epoch_loss:.4f}, "
              f"Train Acc: {epoch_acc:.4f}, Val Acc: {val_acc:.4f}, LR: {lr_now:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            print(f"  -> Saved best model (val_acc={best_val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"  Early stopping at epoch {epoch + 1} (no improvement for {early_stop_patience} epochs)")
                break

    print(f"\nEvaluating on test set...")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
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


class _TransformSubset(torch.utils.data.Dataset):
    """Wraps a Subset to apply a specific transform to PIL images."""

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


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
