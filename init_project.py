import os
from pathlib import Path

dirs = [
    "data/raw",
    "data/processed",
    "notebooks",
    "src",
    "models",
    "app",
    "report/figures"
]

def init_dirs():
    for d in dirs:
        p = Path(d)
        p.mkdir(parents=True, exist_ok=True)
        print(f"Created: {p}")

if __name__ == "__main__":
    init_dirs()
