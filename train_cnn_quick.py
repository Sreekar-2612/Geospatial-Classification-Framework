"""
Quick script to train and save EfficientNet-B1 CNN model
Run this from project root: python train_cnn_quick.py
"""
import os
import sys
import shutil
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ========== CONFIG ==========
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
IMG_SIZE = 128
NUM_EPOCHS = 15 if torch.cuda.is_available() else 5
BATCH_SIZE = 32 if torch.cuda.is_available() else 16
CLASSES = ["Agriculture", "Buildings", "Forest", "Roads", "Water"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ========== SETUP ==========
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Find available classes
available_classes = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
available_classes = [c for c in available_classes if any((DATA_DIR / c).glob("*.jpg"))]
num_classes = len(available_classes)

print(f"Available classes: {available_classes}")
print(f"Classes to train: {num_classes}")

if num_classes == 0:
    print("⚠ No training data found!")
    sys.exit(1)

# ========== DATASET PREP ==========
print("\nPreparing temporary dataset...")
temp_dir = PROJECT_ROOT / "temp_cnn_dataset"
if temp_dir.exists():
    shutil.rmtree(temp_dir)
temp_dir.mkdir()

total_images = 0
for cls in available_classes:
    (temp_dir / cls).mkdir()
    src_dir = DATA_DIR / cls
    for img_path in tqdm(list(src_dir.glob("*.jpg"))[:500], desc=f"  {cls}", leave=False):
        shutil.copy2(img_path, temp_dir / cls / img_path.name)
        total_images += 1

print(f"Total images for training: {total_images}")

# ========== TRANSFORMS ==========
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(45),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ========== LOAD DATASET ==========
dataset = datasets.ImageFolder(str(temp_dir), transform=None)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_subset, val_subset = torch.utils.data.random_split(
    dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
)

class TransformDataset(torch.utils.data.Dataset):
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

train_dataset = TransformDataset(train_subset, train_transform)
val_dataset = TransformDataset(val_subset, eval_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")

# ========== MODEL SETUP ==========
print("\nLoading EfficientNet-B1...")
model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

best_val_acc = 0.0
patience = 5
patience_counter = 0

# ========== TRAINING ==========
print(f"\nTraining for {NUM_EPOCHS} epochs...")
for epoch in range(NUM_EPOCHS):
    # Train
    model.train()
    train_loss = 0.0
    train_correct = 0
    
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=True):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * images.size(0)
        train_correct += (torch.argmax(outputs, 1) == labels).sum().item()
    
    train_loss /= len(train_dataset)
    train_acc = train_correct / len(train_dataset)
    
    # Validate
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_correct += (torch.argmax(outputs, 1) == labels).sum().item()
    
    val_acc = val_correct / len(val_dataset)
    scheduler.step()
    
    print(f"  Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        model_path = MODELS_DIR / "cnn_final.pth"
        torch.save(model.state_dict(), model_path)
        print(f"  ✓ Model saved (val_acc={val_acc:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

print(f"\n✓ Training complete! Best model saved to {MODELS_DIR / 'cnn_final.pth'}")

# Cleanup
if temp_dir.exists():
    shutil.rmtree(temp_dir)
    print("✓ Cleaned up temporary dataset")
