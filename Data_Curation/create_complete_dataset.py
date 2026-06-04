import pandas as pd
import numpy as np

INPUT = "data/humsavar_dbnsfp53_cleaned.csv"
OUTPUT = "data/humsavar_dbnsfp53_complete.csv"

# Columns that must be present for a row to be kept
REQUIRED_COLUMNS = [
    "Gene", "Entry", "FTId", "AA_change", "Category", "dbSNP", "Disease", "Label",
    "chr", "pos", "ref", "alt", "rs_dbSNP",
    "SIFT_score", "SIFT_pred",
    "Polyphen2_HDIV_score", "Polyphen2_HDIV_pred",
    "CADD_raw", "CADD_phred",
    "REVEL_score"
]

# Numeric columns
NUMERIC_COLS = [
    "SIFT_score",
    "Polyphen2_HDIV_score",
    "CADD_raw",
    "CADD_phred",
    "REVEL_score"
]

print("[INFO] Loading dataset...")
df = pd.read_csv(INPUT, dtype=str)

#####################################################
# Helper to clean numeric fields
#####################################################

def clean_numeric(x):
    if pd.isna(x):
        return np.nan

    # Some rows contain "0.994|.|.|0.994|."
    parts = str(x).split("|")
    for part in parts:
        part = part.strip()
        if part == "." or part == "":
            continue
        try:
            return float(part)
        except:
            continue
    return np.nan

#####################################################
# Clean numeric predictor columns
#####################################################
print("[INFO] Cleaning numeric fields...")

for col in NUMERIC_COLS:
    df[col] = df[col].apply(clean_numeric)

#####################################################
# Clean categorical predictor fields
#####################################################
print("[INFO] Cleaning categorical fields...")

# PolyPhen & SIFT sometimes have things like "D|D"
CAT_COLS = ["SIFT_pred", "Polyphen2_HDIV_pred"]

def clean_categorical(x):
    if pd.isna(x):
        return np.nan
    parts = str(x).split("|")
    for p in parts:
        p = p.strip()
        if p in ["D", "T", "P", "B"]:
            return p
    return np.nan

for col in CAT_COLS:
    df[col] = df[col].apply(clean_categorical)

#####################################################
# Drop rows with ANY missing required fields
#####################################################
print("[INFO] Removing incomplete rows...")
cleaned_df = df.dropna(subset=REQUIRED_COLUMNS)

print(f"[INFO] Original rows: {len(df):,}")
print(f"[INFO] Fully complete rows: {len(cleaned_df):,}")
print(f"[INFO] Rows removed: {len(df) - len(cleaned_df):,}")

#####################################################
# Save the fully-complete dataset
#####################################################
cleaned_df.to_csv(OUTPUT, index=False)

print(f"[SAVED] {OUTPUT}")
print("[DONE] Fully complete dataset created.")
