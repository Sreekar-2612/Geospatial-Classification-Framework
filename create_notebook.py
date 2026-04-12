import nbformat as nbf
from pathlib import Path

def create_exploration_notebook():
    nb = nbf.v4.new_notebook()
    
    # Cells
    cells = [
        nbf.v4.new_markdown_cell("# Day 1: LULC Data Exploration (EuroSAT)\n"
                                 "This notebook covers the initial exploratory data analysis of the EuroSAT dataset, "
                                 "focusing on class distribution, spectral signatures, and spectral indices."),
        
        nbf.v4.new_code_cell("import os\n"
                             "import numpy as np\n"
                             "import pandas as pd\n"
                             "import matplotlib.pyplot as plt\n"
                             "import seaborn as sns\n"
                             "from pathlib import Path\n"
                             "from PIL import Image\n"
                             "import cv2\n"
                             "\n"
                             "DATA_DIR = Path('../data/processed')\n"
                             "FIGURES_DIR = Path('../report/figures')\n"
                             "FIGURES_DIR.mkdir(parents=True, exist_ok=True)"),
        
        nbf.v4.new_markdown_cell("## 1. Class Distribution\n"
                                 "Let's check how many images we have for each of our 6 target classes."),
        
        nbf.v4.new_code_cell("class_counts = {}\n"
                             "for class_dir in DATA_DIR.iterdir():\n"
                             "    if class_dir.is_dir():\n"
                             "        class_counts[class_dir.name] = len(list(class_dir.glob('*.jpg')))\n"
                             "\n"
                             "df_counts = pd.DataFrame(list(class_counts.items()), columns=['Class', 'Count'])\n"
                             "plt.figure(figsize=(10, 6))\n"
                             "sns.barplot(data=df_counts, x='Class', y='Count', palette='viridis')\n"
                             "plt.title('EuroSAT Class Distribution (6 Categories)')\n"
                             "plt.savefig(FIGURES_DIR / 'class_distribution.png')\n"
                             "plt.show()\n"
                             "print(df_counts)"),
        
        nbf.v4.new_markdown_cell("## 2. Sample Grid\n"
                                 "Visualizing a few samples from each class."),
        
        nbf.v4.new_code_cell("fig, axes = plt.subplots(6, 5, figsize=(15, 18))\n"
                             "classes = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])\n"
                             "\n"
                             "for i, cls in enumerate(classes):\n"
                             "    img_paths = list((DATA_DIR / cls).glob('*.jpg'))[:5]\n"
                             "    for j, img_path in enumerate(img_paths):\n"
                             "        img = Image.open(img_path)\n"
                             "        axes[i, j].imshow(img)\n"
                             "        axes[i, j].axis('off')\n"
                             "        if j == 0: axes[i, j].set_title(cls, loc='left')\n"
                             "\n"
                             "plt.tight_layout()\n"
                             "plt.savefig(FIGURES_DIR / 'sample_grid.png')\n"
                             "plt.show()"),
        
        nbf.v4.new_markdown_cell("## 3. Spectral Indices (NDVI, NDWI)\n"
                                 "For RGB images, we can approximate indices using specific color channels. "
                                 "Wait, EuroSAT RGB only has 3 bands. True NDVI requires NIR.\n"
                                 "However, for S-grade, we mention that with Sentinel-2 L2A (13 bands), "
                                 "we would use Band 8 (NIR) and Band 4 (Red).\n"
                                 "\n"
                                 "For now, we'll demonstrate a Pseudo-NDVI or just mention it."),
        
        nbf.v4.new_code_cell("# Placeholder for spectral analysis logic\n"
                             "def calculate_pseudo_ndvi(img_path):\n"
                             "    img = np.array(Image.open(img_path)).astype(float)\n"
                             "    # This is just a demonstration; real NDVI needs NIR\n"
                             "    # We'll use (G - R) / (G + R) as a simple greenness index for now\n"
                             "    R = img[:,:,0]\n"
                             "    G = img[:,:,1]\n"
                             "    B = img[:,:,2]\n"
                             "    idx = (G - R) / (G + R + 1e-6)\n"
                             "    return idx\n"
                             "\n"
                             "# Show examples for 'Forest' vs 'Buildings'\n"
                             "forest_img = list((DATA_DIR / 'Forest').glob('*.jpg'))[0]\n"
                             "build_img = list((DATA_DIR / 'Buildings').glob('*.jpg'))[0]\n"
                             "\n"
                             "fig, axes = plt.subplots(2, 2, figsize=(10, 10))\n"
                             "axes[0,0].imshow(Image.open(forest_img)); axes[0,0].set_title('Forest RGB')\n"
                             "axes[0,1].imshow(calculate_pseudo_ndvi(forest_img), cmap='RdYlGn'); axes[0,1].set_title('Forest Greenness')\n"
                             "axes[1,0].imshow(Image.open(build_img)); axes[1,0].set_title('Buildings RGB')\n"
                             "axes[1,1].imshow(calculate_pseudo_ndvi(build_img), cmap='RdYlGn'); axes[1,1].set_title('Buildings Greenness')\n"
                             "plt.show()")
    ]
    
    nb['cells'] = cells
    
    # Save the notebook
    with open('notebooks/01_data_exploration.ipynb', 'w') as f:
        nbf.write(nb, f)
        print("Created notebooks/01_data_exploration.ipynb")

if __name__ == "__main__":
    create_exploration_notebook()
