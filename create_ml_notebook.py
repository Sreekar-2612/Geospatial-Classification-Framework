import nbformat as nbf
from pathlib import Path

def create_ml_notebook():
    nb = nbf.v4.new_notebook()
    
    # Cells
    cells = [
        nbf.v4.new_markdown_cell("# Day 3: Baseline ML with Enhanced DIP Features\n"
                                 "Trains RF and SVM using the standardized `src.features` module."),
        
        nbf.v4.new_code_cell("import cv2\n"
                             "import numpy as np\n"
                             "import pandas as pd\n"
                             "import joblib\n"
                             "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\n"
                             "from sklearn.svm import SVC\n"
                             "from sklearn.neighbors import KNeighborsClassifier\n"
                             "from sklearn.model_selection import train_test_split\n"
                             "from sklearn.metrics import classification_report, accuracy_score\n"
                             "from pathlib import Path\n"
                             "from tqdm import tqdm\n"
                             "import sys\n"
                             "sys.path.append('..')\n"
                             "from src.features import extract_lulc_features\n"
                             "from src.utils import update_metrics\n"
                             "\n"
                             "DATA_DIR = Path('../data/processed')\n"
                             "MODEL_DIR = Path('../models')\n"
                             "MODEL_DIR.mkdir(parents=True, exist_ok=True)"),
        
        nbf.v4.new_code_cell("data = []\n"
                             "labels = []\n"
                             "classes = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])\n"
                             "\n"
                             "print('Generating Standardized Dataset...')\n"
                             "for i, cls in enumerate(classes):\n"
                             "    paths = list((DATA_DIR / cls).glob('*.jpg'))[:500] # Increased for better accuracy\n"
                             "    for img_path in tqdm(paths, desc=f'Class {cls}'):\n"
                             "        img = cv2.imread(str(img_path))\n"
                             "        if img is not None:\n"
                             "            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n"
                             "            feat = extract_lulc_features(img_rgb)\n"
                             "            data.append(feat)\n"
                             "            labels.append(i)\n"
                             "\n"
                             "X = np.array(data)\n"
                             "y = np.array(labels)\n"
                             "print(f'Feature vector size: {X.shape[1]}')"),
        
        nbf.v4.new_code_cell("X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n"
                             "\n"
                             "print('Training Random Forest...')\n"
                             "rf = RandomForestClassifier(n_estimators=100, random_state=42)\n"
                             "rf.fit(X_train, y_train)\n"
                             "joblib.dump(rf, MODEL_DIR / 'rf_baseline.joblib')\n"
                             "acc_rf = accuracy_score(y_test, rf.predict(X_test))\n"
                             "update_metrics('DIP Features + RF', round(acc_rf * 100, 2))\n"
                             "\n"
                             "print('Training SVM...')\n"
                             "svm = SVC(kernel='rbf', probability=True)\n"
                             "svm.fit(X_train, y_train)\n"
                             "acc_svm = accuracy_score(y_test, svm.predict(X_test))\n"
                             "update_metrics('SVM', round(acc_svm * 100, 2))\n"
                             "\n"
                             "print('Training KNN (Advanced Comparison)...')\n"
                             "knn = KNeighborsClassifier(n_neighbors=5)\n"
                             "knn.fit(X_train, y_train)\n"
                             "acc_knn = accuracy_score(y_test, knn.predict(X_test))\n"
                             "update_metrics('K-Nearest Neighbors', round(acc_knn * 100, 2))\n"
                             "\n"
                             "print('Training Gradient Boosting (XGBoost alternative)...')\n"
                             "gbc = GradientBoostingClassifier(n_estimators=100, random_state=42)\n"
                             "gbc.fit(X_train, y_train)\n"
                             "acc_gbc = accuracy_score(y_test, gbc.predict(X_test))\n"
                             "update_metrics('Gradient Boosting (XGB)', round(acc_gbc * 100, 2))\n"
                             "\n"
                             "print('\\nReport (RF Baseline):')\n"
                             "print(classification_report(y_test, rf.predict(X_test), target_names=classes))")
    ]
    
    nb['cells'] = cells
    
    with open('notebooks/03_ml_baseline.ipynb', 'w') as f:
        nbf.write(nb, f)
        print("Updated notebooks/03_ml_baseline.ipynb with shared features.")

if __name__ == "__main__":
    create_ml_notebook()
