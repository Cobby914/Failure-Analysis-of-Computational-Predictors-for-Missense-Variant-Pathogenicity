import pandas as pd

# Load parsed Humsavar
df = pd.read_csv("data/humsavar.csv")

# Normalize category text
df["Category"] = df["Category"].str.strip().str.upper()

# Mapping to text labels
label_map = {
    "LP/P": "Pathogenic",
    "P": "Pathogenic",
    "PATHOGENIC": "Pathogenic",
    "LB/B": "Benign",
    "B": "Benign",
    "BENIGN": "Benign"
}

# Apply labels
df["Label"] = df["Category"].map(label_map)

# Keep only rows with valid labels (drop US)
labeled_df = df[df["Label"].notnull()].copy()

# Save
labeled_df.to_csv("data/humsavar_labeled.csv", index=False)

print("Original rows:", len(df))
print("Labeled rows:", len(labeled_df))
print("Saved to data/humsavar_labeled.csv")
 