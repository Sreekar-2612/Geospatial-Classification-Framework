import nbformat as nbf
from pathlib import Path

def create_dip_notebook():
    nb = nbf.v4.new_notebook()
    
    # Cells
    cells = [
        nbf.v4.new_markdown_cell("# Day 2: Advanced Digital Image Processing (DIP) Techniques\n"
                                 "This notebook implements advanced DIP methods for LULC classification, "
                                 "focusing on texture (GLCM, LBP) and shape (HOG) analysis."),
        
        nbf.v4.new_code_cell("import cv2\n"
                             "import numpy as np\n"
                             "import matplotlib.pyplot as plt\n"
                             "from pathlib import Path\n"
                             "from PIL import Image\n"
                             "from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog\n"
                             "from skimage import exposure\n"
                             "\n"
                             "DATA_DIR = Path('../data/processed')\n"
                             "FIGURES_DIR = Path('../report/figures')"),
        
        nbf.v4.new_markdown_cell("## 1. Texture Analysis: GLCM\n"
                                 "Gray-Level Co-occurrence Matrix (GLCM) extracts statistical texture features."),
        
        nbf.v4.new_code_cell("def extract_glcm_features(img_path):\n"
                             "    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)\n"
                             "    glcm = graycomatrix(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)\n"
                             "    \n"
                             "    contrast = graycoprops(glcm, 'contrast')[0, 0]\n"
                             "    energy = graycoprops(glcm, 'energy')[0, 0]\n"
                             "    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]\n"
                             "    correlation = graycoprops(glcm, 'correlation')[0, 0]\n"
                             "    \n"
                             "    return {'contrast': contrast, 'energy': energy, 'homogeneity': homogeneity, 'correlation': correlation}\n"
                             "\n"
                             "test_img = list(DATA_DIR.glob('*/*.jpg'))[0]\n"
                             "print(f'GLCM Features for {test_img.parent.name}: {extract_glcm_features(test_img)}')"),
        
        nbf.v4.new_markdown_cell("## 2. Texture Analysis: LBP\n"
                                 "Local Binary Patterns (LBP) are powerful for describing local spatial patterns."),
        
        nbf.v4.new_code_cell("def plot_lbp(img_path):\n"
                             "    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)\n"
                             "    radius = 3\n"
                             "    n_points = 8 * radius\n"
                             "    lbp = local_binary_pattern(img, n_points, radius, method='uniform')\n"
                             "    \n"
                             "    plt.figure(figsize=(10, 5))\n"
                             "    plt.subplot(1, 2, 1); plt.imshow(img, cmap='gray'); plt.title('Original Gray')\n"
                             "    plt.subplot(1, 2, 2); plt.imshow(lbp, cmap='gray'); plt.title('LBP Map')\n"
                             "    plt.show()\n"
                             "\n"
                             "plot_lbp(test_img)"),
        
        nbf.v4.new_markdown_cell("## 3. Shape Analysis: HOG\n"
                                 "Histogram of Oriented Gradients (HOG) captures edge directions and local shapes."),
        
        nbf.v4.new_code_cell("def plot_hog(img_path):\n"
                             "    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)\n"
                             "    # channel_axis=None is used for grayscale images in newer skimage versions\n"
                             "    fd, hog_image = hog(img, orientations=8, pixels_per_cell=(8, 8),\n"
                             "                    cells_per_block=(1, 1), visualize=True)\n"
                             "    \n"
                             "    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))\n"
                             "    \n"
                             "    plt.figure(figsize=(10, 5))\n"
                             "    plt.subplot(1, 2, 1); plt.imshow(img, cmap='gray'); plt.title('Original Gray')\n"
                             "    plt.subplot(1, 2, 2); plt.imshow(hog_image_rescaled, cmap='gray'); plt.title('HOG Visualization')\n"
                             "    plt.show()\n"
                             "\n"
                             "plot_hog(test_img)")
    ]
    
    nb['cells'] = cells
    
    # Save the notebook
    with open('notebooks/02_dip_techniques.ipynb', 'w') as f:
        nbf.write(nb, f)
        print("Updated notebooks/02_dip_techniques.ipynb")

if __name__ == "__main__":
    create_dip_notebook()
