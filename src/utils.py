import os
import pandas as pd
from pathlib import Path

METRICS_PATH = Path("report/metrics.csv")

def update_metrics(method_name, accuracy):
    """
    Updates or adds a metric for a given method in the comparison table.
    """
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    if METRICS_PATH.exists():
        df = pd.read_csv(METRICS_PATH)
    else:
        df = pd.DataFrame(columns=["Method", "Accuracy (%)"])

    # Update or add
    if method_name in df["Method"].values:
        df.loc[df["Method"] == method_name, "Accuracy (%)"] = accuracy
    else:
        new_row = pd.DataFrame({"Method": [method_name], "Accuracy (%)": [accuracy]})
        df = pd.concat([df, new_row], ignore_index=True)
    
    df.to_csv(METRICS_PATH, index=False)
    print(f"Updated {method_name} with {accuracy}% accuracy.")

if __name__ == "__main__":
    # Test update
    update_metrics("K-Means (DIP)", 62.5)
    update_metrics("DIP Features + RF", 84.8)
