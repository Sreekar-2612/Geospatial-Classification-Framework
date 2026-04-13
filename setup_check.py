import sys
import torch
import cv2
import numpy as np
import sklearn
import streamlit as st

def check_setup():
    print("--- Environment Setup Check ---")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"Scikit-learn version: {sklearn.__version__}")
    print(f"Streamlit version: {st.__version__}")
    print("-------------------------------")
    print("All core libraries are installed successfully!")

if __name__ == "__main__":
    check_setup()
