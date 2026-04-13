import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog

def extract_lulc_features(img_np):
    """
    Standardized feature extraction for LULC classification.
    Expects RGB image (np.array).
    """
    # Ensure 64x64 or resize for consistency if needed (EuroSAT is 64x64)
    if img_np.shape[0] != 64 or img_np.shape[1] != 64:
        img_np = cv2.resize(img_np, (64, 64))
        
    img_rgb = img_np
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # 1. Color Features (Means & Stds) - 6 features
    color_features = np.hstack([np.mean(img_rgb, axis=(0, 1)), np.std(img_rgb, axis=(0, 1))])
    
    # 2. GLCM Texture - 4 features
    glcm = graycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
    glcm_features = [graycoprops(glcm, p)[0, 0] for p in ['contrast', 'energy', 'homogeneity', 'correlation']]
    
    # 3. LBP Texture (Uniform) - 26 features
    lbp = local_binary_pattern(gray, 24, 3, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
    lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-6)
    
    # 4. HOG Shape ( orientations=8, 16x16 cells) - 128 features (8 * (4*4))
    # Corrected for scikit-image API
    hog_features = hog(gray, orientations=8, pixels_per_cell=(16, 16),
                      cells_per_block=(1, 1), visualize=False)
    
    # 5. Advanced Spectral Indices (RGB-based surrogates) - 4 features
    # Helps distinguish Water (high relative Blue) from Shadows (neutral low intensity)
    R, G, B = img_rgb[:,:,0].astype(float), img_rgb[:,:,1].astype(float), img_rgb[:,:,2].astype(float)
    total = R + G + B + 1e-6
    
    # Excess Blue Index (targets Water)
    exb = np.mean((2*B - R - G) / total)
    # Blue/Red Ratio (Water usually has higher B/R than shadows)
    br_ratio = np.mean(B / (R + 1e-6))
    # Excess Green Index (targets Vegetation vs shadows)
    exg = np.mean((2*G - R - B) / total)
    # Overall Intensity (Shadows are darker than most water)
    intensity = np.mean(gray) / 255.0
    
    spectral_features = [exb, br_ratio, exg, intensity]
    
    return np.hstack([color_features, glcm_features, lbp_hist, hog_features, spectral_features])
