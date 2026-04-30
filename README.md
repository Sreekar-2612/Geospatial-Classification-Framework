# 🛰️ Land Use & Land Cover (LULC) Classification Intelligence Dashboard

## 📌 Project Overview
This project is an advanced, high-performance Land Use and Land Cover (LULC) classification system focused on **Satellite Imagery Analysis**. Engineered for a Digital Image Processing (DIP) course, it is designed with an **S-Grade** target, combining traditional Computer Vision methods with state-of-the-art Deep Learning models. 

The primary goal is the semantic segmentation and systematic classification of remote sensing data (EuroSAT RGB dataset) into **6 fundamental topographic categories**:
1.  **Water Bodies** (Rivers, Lakes, Seas)
2.  **Forest/Parks** (Natural vegetation, Dense clusters)
3.  **Buildings/Urban** (Residential, Industrial, Infrastructure)
4.  **Roads/Pavement** (Highways, Major pathways)
5.  **Agriculture/Grass** (Crops, Pastures, Farmland)
6.  **Shadows/Unknown / Desert/Barren** (Depending on model classification confidence)

---

## ✅ What Has Been Accomplished (Completed Workflow)

### 1. 🏗️ Robust Project Architecture
A highly modular, research-ready, scalable architecture has been implemented to standardize the AI lifecycle:
- **`data/`**: Manages the automated ingestion and cataloging of processed $64 \times 64$ satellite patches.
- **`notebooks/`**: A sequential 5-part Jupyter Research Workflow detailing data exploration, feature extraction, ML training, and DL implementation.
- **`src/`**: Houses reusable scripts (`features.py` for extraction, `utils.py` for metrics coordination), ensuring zero duplicate logic.
- **`models/`**: Secure persistence layer for trained artifacts (e.g., `rf_baseline.joblib` for ML, `cnn_final.pth` for Deep Learning).
- **`app/`**: A fully functional, production-ready **Streamlit web application** showcasing live inference.
- **`report/`**: Directory for capturing automated qualitative figures and CSV metrics (`metrics.csv`) output by the model testing.

### 2. 🐍 Environment Hardening & Validation
- **`requirements.txt`**: Locked, minimal dependencies for rapid replication across different hardware environments. Includes computer vision (OpenCV, skimage), Machine Learning (Scikit-Learn), Deep Learning (PyTorch), and UX (Streamlit, Plotly).
- **`setup_check.py`**: A diagnostic script built to verify system capabilities (Python version, CUDA acceleration, library integrity) before pipeline execution.

### 3. 📔 Sequential Machine Intelligence Workflow
The core "Brain" spans 5 meticulously structured notebooks:
1. **`01_data_exploration.ipynb`**: Execution of Exploratory Data Analysis (EDA). Provides statistical class distributions, spectral signature visualizations, and an initial introduction to spectral index generation (Pseudo-NDVI).
2. **`02_dip_techniques.ipynb`**: Implementation of Advanced DIP. Applies feature extraction techniques such as Gray-Level Co-occurrence Matrix (**GLCM**) for statistical textures, Local Binary Patterns (**LBP**) for spatial patterns, and Histogram of Oriented Gradients (**HOG**) to map edges and structural geometries.
3. **`03_ml_baseline.ipynb`**: Deployment of algorithmic baselines. Uses the consolidated `src.features` pipeline to train powerful classifiers including **Random Forest (RF)**, Support Vector Machines (SVM), K-Nearest Neighbors (KNN), and Gradient Boosting (XGBoost logic). 
4. **`04_deep_learning.ipynb`**: Implemented a complete **ResNet-18 PyTorch training loop**, alongside **GradCAM feature visualization** to build advanced spatial interpretability by explaining exactly where the network looks for a classification.
5. **`05_advanced_analysis.ipynb`**: Constructed an automated **Change Detection simulation** to precisely map urban expansions (new structures), and built a **Pseudo-NDBI** spectral index calculator extracting built-up infrastructures purely from RGB imagery using geometric channel differences.

### 4. 🚀 The Interactive Streamlit Dashboard (`app/main.py`)
A flagship deployment that blends real-time AI classification with qualitative computer vision tools for a high-impact demo:
- **Spatial Smoothing**: Applies morphological closing and median blurring (via user toggle) to reduce salt-and-pepper noise originating from satellite atmospheric interference.
- **Infrastructure Detection (Edge + Morphology)**: Applies Canny Edge estimation followed by structural dilation to visibly isolate Road/Building infrastructure elements.
- **Vegetative Health (Pseudo-NDVI)**: Utilizes a custom mathematical transformation `(G - R) / (G + R)` paired with an optical heat scale mapping to isolate high-chlorophyll regions.
- **Semantic Classification (Dynamic K-Means)**: Iterates a vectorized $K=8$ Means clustering in the RGB space. Dynamically measures Euclidean distances against pre-calibrated, haze-adjusted semantic RGB centroids to auto-label distinct satellite terrains.
- **Intelligence Report**: Executes realtime inference via the pre-trained `rf_baseline.joblib` and evaluates using DL models.
- **Live Benchmarking**: Ingests `report/metrics.csv` immediately rendering an interactive comparative Plotly visualization spanning DIP feature outputs and algorithmic efficacies.

---

## 🛠️ Technical Assumptions & Implementation Logic

1. **Dataset Dynamics**: 
   - **EuroSAT RGB**: Leveraging the 3-band version strictly to provide seamless visualization across standard screens and to maximize real-time edge processing speeds without requiring specialized TIF handlers.
2. **Feature Extraction Focus**:
   - The analysis inherently assumes standardizing images into $64 \times 64$ patches. Contextual topology (how buildings interact with roads) proves much more effective than solitary pixel matching.
3. **Feature Engineering**:
   - Includes rigorous ablation validation—proving explicitly why certain models outperform others by cross-examining classical DIP vs ML vs Deep Learning topologies within the same unified testing structure.

---

## 🚀 Execution Guide

### Phase 1: Environment Bootstrapping
1. **Initialize Requirements**: Execute `pip install -r requirements.txt` within your environment.
2. **Execute Diagnostic Setup Check**: Run `python setup_check.py` to confirm system integrity and GPU availability.
3. **Data Prepping**: Run `python utils/download_data.py` (if script exists) or align your local EuroSAT data to exactly match the target 6 folders inside `data/processed/`.

### Phase 2: Pipeline Execution
1. Sequentially run the Jupyter notebooks in `notebooks/` prefix order `01` to `05`.
2. Ensure you've completed `03_ml_baseline.ipynb` to serialize `rf_baseline.joblib` and `04_deep_learning.ipynb` to output the `cnn_final.pth` needed for Live deployment. Key charts will persist into `report/`.

### Phase 3: Launching the Dashboard Demo
Deploy the intelligence hub via Streamlit from your master project root:
```bash
streamlit run app/main.py
```
Open a browser session and upload a sample `.jpg` from your `data/processed/` repository to test real-time classification.

---

## 📈 Quality Benchmarks
To guarantee **quality output** , the following performance indicators are persistently tracked and heavily scrutinized:
- **CNN Classification Accuracy**: Aiming exceptionally high (>95%).
- **Geometric Intersection (IoU)**: Evaluates semantic map overlap.
- **Interpretability Checks**: Verifying whether GradCAM heatmaps align physically with what a human observer registers as the dominant geographical entity.
