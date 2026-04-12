import os
import requests
import zipfile
import shutil
from pathlib import Path

DATA_URL = "https://madm.dfki.de/files/sentinel/EuroSAT.zip" # Official EuroSAT RGB download
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Mapping 10 EuroSAT classes to 6 target classes
CLASS_MAPPING = {
    "River": "Water",
    "SeaLake": "Water",
    "Forest": "Forest",
    "Residential": "Buildings",
    "Industrial": "Buildings",
    "Highway": "Roads",
    "AnnualCrop": "Agriculture",
    "HerbaceousVegetation": "Agriculture",
    "Pasture": "Agriculture",
    "PermanentCrop": "Agriculture"
}

def download_data():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "EuroSAT.zip"
    
    if not zip_path.exists():
        print(f"Downloading EuroSAT dataset from {DATA_URL}...")
        response = requests.get(DATA_URL, stream=True, verify=False)
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    else:
        print("Dataset already downloaded.")

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(RAW_DIR)
    print("Extraction complete.")

def organize_classes():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Locate the extracted folder (usually "2750" or similar)
    extracted_path = RAW_DIR / "2750"
    if not extracted_path.exists():
        # Sometimes it extracts into a folder named EuroSAT or something else
        for folder in RAW_DIR.iterdir():
            if folder.is_dir() and "2750" in folder.name:
                extracted_path = folder
                break
    
    if not extracted_path.exists():
        print("Could not find extracted image folder. Please check organization.")
        return

    print("Organizing classes into target categories...")
    for old_class, new_class in CLASS_MAPPING.items():
        src_path = extracted_path / old_class
        dest_path = PROCESSED_DIR / new_class
        
        if src_path.exists():
            dest_path.mkdir(parents=True, exist_ok=True)
            for img_file in src_path.glob("*.jpg"):
                # Copy instead of move for safety
                shutil.copy(img_file, dest_path / f"{old_class}_{img_file.name}")
        else:
            print(f"Warning: {old_class} directory not found.")
    
    print(f"Dataset organized in {PROCESSED_DIR}")

if __name__ == "__main__":
    download_data()
    organize_classes()
