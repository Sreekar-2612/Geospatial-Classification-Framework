import os
import pandas as pd
from pathlib import Path

METRICS_PATH = Path("report/metrics.csv")

def update_metrics(method_name, accuracy):
    """
    Updates or adds a metric for a given method in the comparison table.
    Stores accuracy as a 0-1 float in the unified 'Model' / 'Accuracy' schema.
    """
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    acc_normalised = accuracy / 100.0 if accuracy > 1 else accuracy

    if METRICS_PATH.exists():
        df = pd.read_csv(METRICS_PATH)
    else:
        df = pd.DataFrame(columns=["Model", "Accuracy"])

    if "Model" not in df.columns:
        df.rename(columns={"Method": "Model", "Accuracy (%)": "Accuracy"}, inplace=True)
        if "Accuracy" in df.columns:
            df["Accuracy"] = pd.to_numeric(df["Accuracy"], errors="coerce")
            mask = df["Accuracy"] > 1
            df.loc[mask, "Accuracy"] = df.loc[mask, "Accuracy"] / 100.0

    if method_name in df["Model"].values:
        df.loc[df["Model"] == method_name, "Accuracy"] = acc_normalised
    else:
        new_row = pd.DataFrame({"Model": [method_name], "Accuracy": [acc_normalised]})
        df = pd.concat([df, new_row], ignore_index=True)

    df.to_csv(METRICS_PATH, index=False)
    print(f"Updated {method_name} with {accuracy}% accuracy.")

if __name__ == "__main__":
    # Test update
    update_metrics("K-Means (DIP)", 62.5)
    update_metrics("DIP Features + RF", 84.8)
