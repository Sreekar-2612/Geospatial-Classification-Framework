import nbformat as nbf
from pathlib import Path

def create_dl_notebook():
    nb = nbf.v4.new_notebook()
    
    # Cells
    cells = [
        nbf.v4.new_markdown_cell("# Day 4: Deep Learning (CNN & GradCAM)\n"
                                 "This notebook implements a Convolutional Neural Network (CNN) for LULC classification, "
                                 "including training, evaluation, and S-grade interpretability using GradCAM."),
        
        nbf.v4.new_code_cell("import torch\n"
                             "import torch.nn as nn\n"
                             "import torch.optim as optim\n"
                             "from torchvision import datasets, models, transforms\n"
                             "from torch.utils.data import DataLoader\n"
                             "import matplotlib.pyplot as plt\n"
                             "import numpy as np\n"
                             "from pathlib import Path\n"
                             "from tqdm import tqdm\n"
                             "\n"
                             "DATA_DIR = Path('../data/processed')\n"
                             "MODEL_PATH = Path('../models/cnn_final.pth')\n"
                             "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
                             "print(f'Using device: {device}')\n"
                             "if device.type == 'cpu':\n"
                             "    print('WARNING: Training on CPU will be slow. Consider using Google Colab for a GPU.')"),
        
        nbf.v4.new_markdown_cell("## 1. Data Loading & Transforms\n"
                                 "Normalizing images and setting up data loaders."),
        
        nbf.v4.new_code_cell("import torchvision\n"
                             "transform = transforms.Compose([\n"
                             "    transforms.Resize((64, 64)),\n"
                             "    transforms.ToTensor(),\n"
                             "    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])\n"
                             "])\n"
                             "\n"
                             "if not DATA_DIR.exists():\n"
                             "    print('Local data folder not found (Likely running in Colab). Auto-downloading EuroSAT...')\n"
                             "    full_dataset = torchvision.datasets.EuroSAT(root='./data', download=True, transform=transform)\n"
                             "else:\n"
                             "    full_dataset = datasets.ImageFolder(str(DATA_DIR), transform=transform)\n"
                             "\n"
                             "train_size = int(0.8 * len(full_dataset))\n"
                             "test_size = len(full_dataset) - train_size\n"
                             "train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])\n"
                             "\n"
                             "if device.type == 'cpu':\n"
                             "    print('WARNING: CPU detected. Subsetting training set to 64 images for exceptionally fast testing/verification.')\n"
                             "    train_dataset = torch.utils.data.Subset(train_dataset, range(64))\n"
                             "    train_size = 64\n"
                             "\n"
                             "batch_sz = 16 if device.type == 'cpu' else 32\n"
                             "train_loader = DataLoader(train_dataset, batch_size=batch_sz, shuffle=True)\n"
                             "test_loader = DataLoader(test_dataset, batch_size=batch_sz, shuffle=False)\n"
                             "class_names = full_dataset.classes\n"
                             "print(f'Detected classes: {class_names}')"),
        
        nbf.v4.new_markdown_cell("## 2. Model Architecture (ResNet-18)\n"
                                 "Using a pre-trained ResNet-18 as a backbone for better performance."),
        
        nbf.v4.new_code_cell("model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)\n"
                             "num_ftrs = model.fc.in_features\n"
                             "model.fc = nn.Linear(num_ftrs, len(class_names))\n"
                             "model = model.to(device)"),

        nbf.v4.new_markdown_cell("## 3. Training the Model\n"
                                 "Training the CNN and saving the best weights for deployment."),

        nbf.v4.new_code_cell("criterion = nn.CrossEntropyLoss()\n"
                             "optimizer = optim.Adam(model.parameters(), lr=0.001)\n"
                             "num_epochs = 2 if device.type == 'cpu' else 5 # Less epochs for CPU testing\n"
                             "\n"
                             "print('Ready to train! Starting training loop...')\n"
                             "best_acc = 0.0\n"
                             "for epoch in range(num_epochs):\n"
                             "    model.train()\n"
                             "    running_loss, running_corrects = 0.0, 0\n"
                             "    \n"
                             "    # Added tqdm so CPU training doesn't look 'dead'\n"
                             "    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')\n"
                             "    for inputs, labels in pbar:\n"
                             "        inputs, labels = inputs.to(device), labels.to(device)\n"
                             "        optimizer.zero_grad()\n"
                             "        outputs = model(inputs)\n"
                             "        loss = criterion(outputs, labels)\n"
                             "        _, preds = torch.max(outputs, 1)\n"
                             "        loss.backward()\n"
                             "        optimizer.step()\n"
                             "        \n"
                             "        running_loss += loss.item() * inputs.size(0)\n"
                             "        running_corrects += torch.sum(preds == labels.data)\n"
                             "        pbar.set_postfix({'loss': loss.item()})\n"
                             "    \n"
                             "    epoch_loss = running_loss / train_size\n"
                             "    epoch_acc = running_corrects.double() / train_size\n"
                             "    print(f'\\nEpoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')\n"
                             "    \n"
                             "    if epoch_acc > best_acc:\n"
                             "        best_acc = epoch_acc\n"
                             "        Path('../models').mkdir(parents=True, exist_ok=True)\n"
                             "        torch.save(model.state_dict(), MODEL_PATH)\n"
                             "print('Training complete! Best weights saved.')"),

        nbf.v4.new_markdown_cell("## 4. S-Grade Addition: GradCAM\n"
                                 "Visualizing what the network sees when classifying."),
        
        nbf.v4.new_code_cell("import torch.nn.functional as F\n"
                             "\n"
                             "class GradCAM:\n"
                             "    def __init__(self, model, target_layer):\n"
                             "        self.model = model\n"
                             "        self.target_layer = target_layer\n"
                             "        self.gradients = None\n"
                             "        self.activations = None\n"
                             "        self.target_layer.register_forward_hook(self.save_activation)\n"
                             "        self.target_layer.register_full_backward_hook(self.save_gradient)\n"
                             "\n"
                             "    def save_activation(self, module, input, output):\n"
                             "        self.activations = output\n"
                             "\n"
                             "    def save_gradient(self, module, grad_input, grad_output):\n"
                             "        self.gradients = grad_output[0]\n"
                             "\n"
                             "    def generate_cam(self, input_image, target_class=None):\n"
                             "        self.model.eval()\n"
                             "        output = self.model(input_image)\n"
                             "        if target_class is None:\n"
                             "            target_class = output.argmax(dim=1).item()\n"
                             "        \n"
                             "        self.model.zero_grad()\n"
                             "        target = output[0, target_class]\n"
                             "        target.backward(retain_graph=True)\n"
                             "        \n"
                             "        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])\n"
                             "        activations = self.activations.detach()[0]\n"
                             "        for i in range(activations.size(0)):\n"
                             "            activations[i, :, :] *= pooled_gradients[i]\n"
                             "            \n"
                             "        heatmap = torch.mean(activations, dim=0).squeeze()\n"
                             "        heatmap = F.relu(heatmap)\n"
                             "        heatmap /= torch.max(heatmap)\n"
                             "        return heatmap.cpu().detach().numpy()\n"
                             "\n"
                             "print('GradCAM initialized successfully.')")
    ]
    
    nb['cells'] = cells
    
    # Save the notebook
    with open('notebooks/04_deep_learning.ipynb', 'w') as f:
        nbf.write(nb, f)
        print("Created notebooks/04_deep_learning.ipynb")

if __name__ == "__main__":
    create_dl_notebook()
