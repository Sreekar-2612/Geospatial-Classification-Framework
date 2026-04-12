import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from pathlib import Path

DATA_DIR = Path('data/processed')
MODEL_PATH = Path('models/cnn_final.pth')

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])
full_dataset = datasets.ImageFolder(str(DATA_DIR), transform=transform)

class_names = full_dataset.classes
print("Detected classes:", class_names)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
model = model.to(device)

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)
print("Saved proper 6-class ResNet architecture to cnn_final.pth!")
