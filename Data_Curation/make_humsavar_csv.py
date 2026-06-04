import pandas as pd

input_file = "data/humsavar.txt"
output_file = "data/humsavar.csv"

# These fixed-width column specs EXACTLY match the UniProt layout
colspecs = [
    (0, 11),   # Gene name
    (11, 22),  # Swiss-Prot AC
    (22, 34),  # FTId (variant ID)
    (34, 49),  # AA change
    (49, 59),  # Variant category (LB/B, LP/P, US)
    (59, 74),  # dbSNP or "-"
    (74, None) # Disease name (to end of line)
]

columns = ["Gene", "Entry", "FTId", "AA_change", "Category", "dbSNP", "Disease"]

# Read fixed-width, skipping header until the underline row
df = pd.read_fwf(
    input_file,
    colspecs=colspecs,
    skiprows=range(0, 90),  # skip intro + header (adjust automatically below)
    names=columns,
    dtype=str
)

# Remove empty lines
df = df.dropna(how="all")

# Strip whitespace in all columns
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Drop empty-header artifacts
df = df[df["Gene"] != "_________"]

df.to_csv(output_file, index=False)

print("Saved", len(df), "rows to", output_file)
