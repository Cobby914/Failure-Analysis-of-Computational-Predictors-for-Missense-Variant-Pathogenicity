"""
Export missing failure-analysis CSVs, build AlphaFold variant cohorts,
download structures, merge pLDDT, and write summary tables.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "Data"
STRUCTURES_DIR = DATA_DIR / "alphafold_structures"

FEATURES = [
    "SIFT_score_clean",
    "PolyPhen_score_clean",
    "CADD_raw_clean",
    "CADD_phred_clean",
    "REVEL_score_clean",
]
FAILURE_COLS = ["SIFT_failed", "PolyPhen_failed", "CADD_failed", "REVEL_failed"]


def build_failure_df() -> pd.DataFrame:
    model_df = pd.read_csv(DATA_DIR / "model_df.csv", na_values="?", low_memory=False)
    X = model_df[FEATURES]
    y = model_df["y"]
    _, _, _, _, _, test_idx = train_test_split(
        X, y, model_df.index, test_size=0.2, random_state=42, stratify=y
    )
    failure_df = model_df.loc[test_idx].copy()

    failure_df["SIFT_pred_binary"] = (failure_df["SIFT_score_clean"] < 0.05).astype(int)
    failure_df["PolyPhen_pred_binary"] = (failure_df["PolyPhen_score_clean"] > 0.909).astype(int)
    failure_df["CADD_pred_binary"] = (failure_df["CADD_phred_clean"] > 20).astype(int)
    failure_df["REVEL_pred_binary"] = (failure_df["REVEL_score_clean"] > 0.5).astype(int)

    for pred, col in [
        ("SIFT", "SIFT_pred_binary"),
        ("PolyPhen", "PolyPhen_pred_binary"),
        ("CADD", "CADD_pred_binary"),
        ("REVEL", "REVEL_pred_binary"),
    ]:
        failure_df[f"{pred}_failed"] = failure_df[col] != failure_df["y"]

    failure_df["num_predictors_failed"] = failure_df[FAILURE_COLS].sum(axis=1)
    failure_df["any_failure"] = failure_df[FAILURE_COLS].any(axis=1)
    return failure_df


def export_missing_summaries(failure_df: pd.DataFrame) -> None:
    for pred, col, thr in [
        ("SIFT", "SIFT_score_clean", 0.05),
        ("PolyPhen", "PolyPhen_score_clean", 0.909),
        ("CADD", "CADD_phred_clean", 20),
        ("REVEL", "REVEL_score_clean", 0.5),
    ]:
        failure_df[f"{pred}_boundary_distance"] = (failure_df[col] - thr).abs()

    boundary_comparison = pd.DataFrame(
        {
            "Predictor": ["SIFT", "PolyPhen-2", "CADD PHRED", "REVEL"],
            "Mean Distance When Correct": [
                failure_df.loc[failure_df[f"{p}_failed"] == False, f"{p}_boundary_distance"].mean()
                for p in ["SIFT", "PolyPhen", "CADD", "REVEL"]
            ],
            "Mean Distance When Incorrect": [
                failure_df.loc[failure_df[f"{p}_failed"] == True, f"{p}_boundary_distance"].mean()
                for p in ["SIFT", "PolyPhen", "CADD", "REVEL"]
            ],
        }
    )
    boundary_comparison.to_csv(DATA_DIR / "confidence_boundary_analysis.csv", index=False)

    prediction_cols = [
        "SIFT_pred_binary",
        "PolyPhen_pred_binary",
        "CADD_pred_binary",
        "REVEL_pred_binary",
    ]
    failure_df["num_pathogenic_votes"] = failure_df[prediction_cols].sum(axis=1)
    failure_df["predictor_disagreement"] = failure_df["num_pathogenic_votes"].between(1, 3)

    summary_results = {
        "Total variants in test-set failure_df": len(failure_df),
        "Variants with any predictor failure": int(failure_df["any_failure"].sum()),
        "Any failure rate": failure_df["any_failure"].mean(),
        "Mean predictors failed per variant": failure_df["num_predictors_failed"].mean(),
        "Predictor disagreement rate": failure_df["predictor_disagreement"].mean(),
        "Failure rate among disagreement cases": failure_df.loc[
            failure_df["predictor_disagreement"], "any_failure"
        ].mean(),
        "Failure rate among agreement cases": failure_df.loc[
            ~failure_df["predictor_disagreement"], "any_failure"
        ].mean(),
    }
    summary_df = pd.DataFrame(list(summary_results.items()), columns=["Metric", "Value"])
    summary_df.to_csv(DATA_DIR / "phase_4_5_6_summary.csv", index=False)
    print("Saved confidence_boundary_analysis.csv and phase_4_5_6_summary.csv")


def export_alphafold_cohorts(failure_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    gene_failure = (
        failure_df.groupby("Gene")
        .agg(
            total_variants=("Gene", "count"),
            avg_predictors_failed=("num_predictors_failed", "mean"),
            variants_with_any_failure=("num_predictors_failed", lambda x: (x > 0).sum()),
        )
        .reset_index()
    )
    gene_failure["any_failure_rate"] = (
        gene_failure["variants_with_any_failure"] / gene_failure["total_variants"]
    )

    top_genes_min20 = (
        gene_failure[gene_failure["total_variants"] >= 20]
        .sort_values(["any_failure_rate", "avg_predictors_failed", "total_variants"], ascending=False)
        .head(10)
    )
    top_genes_min20.to_csv(DATA_DIR / "top_genes_min20.csv", index=False)

    min20_gene_names = top_genes_min20["Gene"].tolist()
    alphafold_min20 = failure_df[failure_df["Gene"].isin(min20_gene_names)].copy()
    alphafold_min20.to_csv(DATA_DIR / "alphafold_variants_min20.csv", index=False)
    print(f"Saved top_genes_min20.csv and alphafold_variants_min20.csv ({len(alphafold_min20)} variants)")
    return alphafold_min20, top_genes_min20


# --- AlphaFold helpers (from Models/AlphaFold.ipynb) ---


def clean_uniprot_id(entry):
    if pd.isna(entry):
        return None
    entry = str(entry).strip()
    if "|" in entry:
        parts = entry.split("|")
        if len(parts) >= 2:
            entry = parts[1]
    return entry


def get_alphafold_metadata(uniprot_id):
    uniprot_id = clean_uniprot_id(uniprot_id)
    for uid in [uniprot_id, uniprot_id.split("-")[0]]:
        url = f"https://alphafold.ebi.ac.uk/api/prediction/{uid}"
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200:
                data = r.json()
                if len(data) > 0:
                    return data[0]
        except Exception as e:
            print(f"API error for {uid}: {e}")
    return None


def download_from_alphafold_api(uniprot_id, out_dir: Path) -> str | None:
    meta = get_alphafold_metadata(uniprot_id)
    if meta is None:
        print(f"No AlphaFold API result for {uniprot_id}")
        return None

    download_urls = []
    if "cifUrl" in meta:
        download_urls.append(meta["cifUrl"])
    if "pdbUrl" in meta:
        download_urls.append(meta["pdbUrl"])

    for url in download_urls:
        filename = url.split("/")[-1]
        out_path = out_dir / filename
        if out_path.exists() and out_path.stat().st_size > 1000:
            print(f"Already downloaded: {filename}")
            return str(out_path)
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200 and len(r.text) > 1000:
                out_path.write_text(r.text, encoding="utf-8")
                print(f"Downloaded {uniprot_id}: {filename}")
                return str(out_path)
        except Exception as e:
            print(f"Download error for {uniprot_id}: {e}")
    print(f"Metadata found but download failed for {uniprot_id}")
    return None


def parse_plddt_from_pdb(pdb_path):
    residue_scores = {}
    with open(pdb_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("ATOM"):
                try:
                    residue_num = int(line[22:26].strip())
                    b_factor = float(line[60:66].strip())
                except ValueError:
                    continue
                residue_scores.setdefault(residue_num, []).append(b_factor)
    return pd.DataFrame(
        [{"protein_position": n, "pLDDT": np.mean(s)} for n, s in residue_scores.items()]
    )


def parse_plddt_from_cif(cif_path):
    rows = []
    in_atom_loop = False
    headers = []
    with open(cif_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "loop_":
                in_atom_loop = True
                headers = []
                continue
            if in_atom_loop and line.startswith("_atom_site."):
                headers.append(line)
                continue
            if in_atom_loop and headers and not line.startswith("_"):
                if line.startswith("#") or line == "":
                    in_atom_loop = False
                    continue
                parts = line.split()
                if len(parts) < len(headers):
                    continue
                header_map = {h: i for i, h in enumerate(headers)}
                try:
                    seq_idx = header_map.get("_atom_site.label_seq_id")
                    b_idx = header_map.get("_atom_site.B_iso_or_equiv")
                    if seq_idx is None or b_idx is None:
                        continue
                    rows.append(
                        {
                            "protein_position": int(parts[seq_idx]),
                            "pLDDT_atom": float(parts[b_idx]),
                        }
                    )
                except (ValueError, IndexError):
                    continue
    if not rows:
        return pd.DataFrame(columns=["protein_position", "pLDDT"])
    atom_df = pd.DataFrame(rows)
    return (
        atom_df.groupby("protein_position")["pLDDT_atom"]
        .mean()
        .reset_index()
        .rename(columns={"pLDDT_atom": "pLDDT"})
    )


def parse_plddt_from_structure(path: str) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=["protein_position", "pLDDT"])
    if path.lower().endswith(".pdb"):
        return parse_plddt_from_pdb(path)
    if path.lower().endswith(".cif"):
        return parse_plddt_from_cif(path)
    return pd.DataFrame(columns=["protein_position", "pLDDT"])


def extract_position(aa_change):
    if pd.isna(aa_change):
        return np.nan
    match = re.search(r"(\d+)", str(aa_change))
    return int(match.group(1)) if match else np.nan


def plddt_category(score):
    if pd.isna(score):
        return "Missing"
    if score >= 90:
        return "Very high confidence"
    if score >= 70:
        return "Confident"
    if score >= 50:
        return "Low confidence"
    return "Very low confidence"


def load_all_plddt() -> pd.DataFrame:
    plddt_tables = []
    for path in STRUCTURES_DIR.glob("AF-*"):
        if path.suffix.lower() not in (".cif", ".pdb"):
            continue
        entry = path.name.split("-")[1]
        plddt_df = parse_plddt_from_structure(str(path))
        plddt_df["Entry"] = entry
        plddt_tables.append(plddt_df)
    return pd.concat(plddt_tables, ignore_index=True) if plddt_tables else pd.DataFrame()


def merge_existing_plddt(variants: pd.DataFrame, label: str) -> pd.DataFrame:
    all_plddt = load_all_plddt()
    out = variants.copy()
    out["protein_position"] = out["AA_change"].apply(extract_position)
    out = out.merge(all_plddt, on=["Entry", "protein_position"], how="left")
    out["pLDDT_category"] = out["pLDDT"].apply(plddt_category)
    out["any_predictor_failed"] = out["num_predictors_failed"] > 0
    out["cohort"] = label
    return out


def attach_plddt(variants: pd.DataFrame, label: str) -> pd.DataFrame:
    STRUCTURES_DIR.mkdir(parents=True, exist_ok=True)
    unique_entries = variants["Entry"].dropna().unique()
    plddt_tables = []
    for entry in unique_entries:
        path = download_from_alphafold_api(entry, STRUCTURES_DIR)
        plddt_df = parse_plddt_from_structure(path)
        plddt_df["Entry"] = entry
        plddt_tables.append(plddt_df)

    all_plddt = pd.concat(plddt_tables, ignore_index=True) if plddt_tables else pd.DataFrame()
    out = variants.copy()
    out["protein_position"] = out["AA_change"].apply(extract_position)
    out = out.merge(all_plddt, on=["Entry", "protein_position"], how="left")
    out["pLDDT_category"] = out["pLDDT"].apply(plddt_category)
    out["any_predictor_failed"] = out["num_predictors_failed"] > 0
    out["cohort"] = label
    return out


def alphafold_stats(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    plot_df = df.dropna(subset=["pLDDT", "num_predictors_failed"]).copy()
    if plot_df.empty:
        return pd.DataFrame()

    failed = plot_df.loc[plot_df["any_predictor_failed"], "pLDDT"].astype(float)
    nonfailed = plot_df.loc[~plot_df["any_predictor_failed"], "pLDDT"].astype(float)

    if len(failed) >= 1 and len(nonfailed) >= 1:
        _, mw_p = mannwhitneyu(failed, nonfailed, alternative="two-sided")
    else:
        mw_p = np.nan

    if len(plot_df) >= 3:
        rho, spearman_p = spearmanr(plot_df["pLDDT"], plot_df["num_predictors_failed"])
    else:
        rho, spearman_p = np.nan, np.nan

    stat_df = plot_df.dropna(subset=["pLDDT", "any_predictor_failed"]).copy()
    stat_df["low_confidence"] = stat_df["pLDDT"] < 70
    contingency = pd.crosstab(stat_df["low_confidence"], stat_df["any_predictor_failed"])
    if contingency.shape == (2, 2):
        odds_ratio, fisher_p = fisher_exact(contingency)
    else:
        odds_ratio, fisher_p = np.nan, np.nan

    stat_df["very_low_confidence"] = stat_df["pLDDT"] < 50
    contingency_vl = pd.crosstab(stat_df["very_low_confidence"], stat_df["any_predictor_failed"])
    if contingency_vl.shape == (2, 2):
        odds_ratio_vl, fisher_p_vl = fisher_exact(contingency_vl)
    else:
        odds_ratio_vl, fisher_p_vl = np.nan, np.nan

    return pd.DataFrame(
        {
            "cohort": [prefix] * 8,
            "metric": [
                "n_variants_with_plddt",
                "failed_median_plddt",
                "nonfailed_median_plddt",
                "mannwhitney_p",
                "spearman_rho",
                "spearman_p",
                "fisher_low_confidence_odds_ratio",
                "fisher_low_confidence_p",
            ],
            "value": [
                len(plot_df),
                failed.median(),
                nonfailed.median(),
                mw_p,
                rho,
                spearman_p,
                odds_ratio,
                fisher_p,
            ],
        }
    )


def run_alphafold_pipeline(*, download: bool = True) -> None:
    top10_path = DATA_DIR / "alphafold_variants.csv"
    if not top10_path.exists():
        print("Warning: alphafold_variants.csv not found; skipping top-10 cohort.")
        top10 = pd.DataFrame()
    else:
        top10 = pd.read_csv(top10_path, low_memory=False)
        if "num_predictors_failed" not in top10.columns:
            top10 = attach_failure_cols(top10)

    min20 = pd.read_csv(DATA_DIR / "alphafold_variants_min20.csv", low_memory=False)

    if download:
        top10_plddt = attach_plddt(top10, "top10_failure_genes") if len(top10) else pd.DataFrame()
        min20_plddt = attach_plddt(min20, "min20_high_failure_genes")
    else:
        top10_plddt = merge_existing_plddt(top10, "top10_failure_genes") if len(top10) else pd.DataFrame()
        min20_plddt = merge_existing_plddt(min20, "min20_high_failure_genes")

    combined = pd.concat([top10_plddt, min20_plddt], ignore_index=True)
    combined.to_csv(DATA_DIR / "alphafold_variants_with_plddt.csv", index=False)
    print(f"Saved alphafold_variants_with_plddt.csv ({len(combined)} rows)")

    stats_parts = []
    if len(top10_plddt):
        stats_parts.append(alphafold_stats(top10_plddt, "top10_failure_genes"))
    stats_parts.append(alphafold_stats(min20_plddt, "min20_high_failure_genes"))
    stats = pd.concat(stats_parts, ignore_index=True)
    stats.to_csv(DATA_DIR / "alphafold_stat_summary.csv", index=False)
    print("Saved alphafold_stat_summary.csv")

    gene_summary = (
        min20_plddt.dropna(subset=["pLDDT"])
        .groupby("Gene")
        .agg(
            total_variants=("Gene", "count"),
            failure_rate=("any_predictor_failed", "mean"),
            mean_pLDDT=("pLDDT", "mean"),
            median_pLDDT=("pLDDT", "median"),
            low_confidence_rate=("pLDDT", lambda x: (x < 70).mean()),
        )
        .reset_index()
        .sort_values("failure_rate", ascending=False)
    )
    gene_summary.to_csv(DATA_DIR / "alphafold_gene_summary_min20.csv", index=False)
    print("Saved alphafold_gene_summary_min20.csv")


def attach_failure_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SIFT_pred_binary"] = (df["SIFT_score_clean"] < 0.05).astype(int)
    df["PolyPhen_pred_binary"] = (df["PolyPhen_score_clean"] > 0.909).astype(int)
    df["CADD_pred_binary"] = (df["CADD_phred_clean"] > 20).astype(int)
    df["REVEL_pred_binary"] = (df["REVEL_score_clean"] > 0.5).astype(int)
    for pred, col in [
        ("SIFT", "SIFT_pred_binary"),
        ("PolyPhen", "PolyPhen_pred_binary"),
        ("CADD", "CADD_pred_binary"),
        ("REVEL", "REVEL_pred_binary"),
    ]:
        df[f"{pred}_failed"] = df[col] != df["y"]
    df["num_predictors_failed"] = df[FAILURE_COLS].sum(axis=1)
    return df


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Skip downloads; rebuild pLDDT merges and stat tables from cached structures.",
    )
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    print("Building failure_df from model_df.csv ...")
    failure_df = build_failure_df()

    print("Exporting missing summary CSVs ...")
    export_missing_summaries(failure_df)

    print("Exporting AlphaFold min20 cohort ...")
    export_alphafold_cohorts(failure_df)

    if args.stats_only:
        print("Rebuilding pLDDT merges from cached structures ...")
        run_alphafold_pipeline(download=False)
    else:
        print("Running AlphaFold download + pLDDT merge (network required) ...")
        run_alphafold_pipeline(download=True)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
