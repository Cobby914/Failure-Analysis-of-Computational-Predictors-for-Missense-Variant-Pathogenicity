import pandas as pd
import numpy as np

INPUT = "data/humsavar_dbnsfp53_cleaned.csv"
OUTPUT = "data/humsavar_dbnsfp53_sqlready.csv"

NUMERIC_COLS = [
    "SIFT_score",
    "Polyphen2_HDIV_score",
    "CADD_raw",
    "CADD_phred",
    "REVEL_score"
]

def clean_value(x):
    if pd.isna(x):
        return np.nan
    # Split on pipes, keep the first valid value
    parts = str(x).split("|")
    for p in parts:
        p = p.strip()
        if p == "." or p == "":
            continue
        try:
            return float(p)
        except:
            continue
    return np.nan

print("[INFO] Loading CSV...")
df = pd.read_csv(INPUT, dtype=str)

print("[INFO] Cleaning numeric predictor fields...")

for col in NUMERIC_COLS:
    df[col] = df[col].apply(clean_value)

print("[INFO] Converting all numeric fields to floats...")
for col in NUMERIC_COLS:
    df[col] = df[col].astype(float)

df.to_csv(OUTPUT, index=False)
print("\n[SAVED]", OUTPUT)
print("[DONE] SQL-ready dataset created.")
