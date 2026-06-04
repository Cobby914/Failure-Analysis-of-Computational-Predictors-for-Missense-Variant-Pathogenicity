import pandas as pd
import os

print("SCRIPT STARTED")   # <-- WILL ALWAYS PRINT

INPUT_FILE  = "data/humsavar_dbnsfp53_merged_fast.csv"
OUTPUT_FILE = "data/humsavar_dbnsfp53_cleaned.csv"

# Check if file exists
if not os.path.exists(INPUT_FILE):
    print(f"[ERROR] Input file not found: {INPUT_FILE}")
    print("Current directory:", os.getcwd())
    print("Files in current directory:", os.listdir())
    exit()

print("[INFO] Loading file...")
df = pd.read_csv(INPUT_FILE, dtype=str)
print(f"[INFO] Loaded {len(df):,} rows")

required_cols = ["chr", "pos", "ref", "alt", "rs_dbSNP"]

print("[INFO] Cleaning...")
clean_df = df.dropna(subset=required_cols)

removed = len(df) - len(clean_df)
print(f"[INFO] Removed {removed:,} rows")
print(f"[INFO] Remaining {len(clean_df):,} rows")

clean_df.to_csv(OUTPUT_FILE, index=False)
print(f"[SAVED] {OUTPUT_FILE}")
