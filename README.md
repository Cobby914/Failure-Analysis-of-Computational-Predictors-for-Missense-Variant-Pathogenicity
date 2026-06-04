# Failure Analysis of Missense Variant Pathogenicity Predictors

This project analyzes where computational predictors fail when classifying missense variants as benign or pathogenic. Using a merged Humsavar/dbNSFP variant annotation dataset, the project benchmarks several pathogenicity predictors and investigates patterns in their misclassifications.

## Project Overview

Missense variants can alter amino acid sequences and potentially affect protein function. Tools such as SIFT, PolyPhen-2, CADD, and REVEL are commonly used to predict whether these variants are likely benign or pathogenic. However, these predictors do not always agree and can fail on certain genes, amino acid substitutions, or biochemical changes.

The goal of this project is to identify what types of missense variants are consistently misclassified by state-of-the-art pathogenicity predictors.

## Research Questions

This project focuses on four main hypotheses:

- Certain genes have higher predictor failure rates than others.
- Specific amino acid substitutions are more difficult to classify.
- Predictor disagreement is associated with higher misclassification rates.
- Many failures occur near predictor decision boundaries where confidence is low.

## Dataset

The dataset is a merged variant annotation dataset based on UniProt/Humsavar and dbNSFP annotations. Each row represents a single amino-acid-changing genetic variant. Variants are labeled as benign or pathogenic and include predictor scores from tools such as SIFT, PolyPhen-2, CADD, and REVEL. The project presentation describes the dataset as being based on Humsavar and dbNSFP annotations, with each row corresponding to one amino-acid-changing variant. :contentReference[oaicite:0]{index=0}

## Methods

The analysis includes:

- Cleaning and filtering missense variant data
- Converting variant labels into benign and pathogenic classes
- Benchmarking SIFT, PolyPhen-2, CADD, REVEL, logistic regression, and random forest
- Evaluating models using accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices
- Identifying failed predictions where model output disagrees with known labels
- Grouping failures by gene, amino acid substitution, biochemical category, and disease annotation
- Using AlphaFold confidence scores to explore whether failed variants occur in low-confidence or high-confidence structural regions

## Main Findings

The analysis found that predictor failures are not random. Certain genes showed much higher failure rates, and specific amino acid substitutions and biochemical changes were enriched among misclassified variants. Predictor disagreement was the strongest indicator of failure. AlphaFold analysis suggested that failures are not explained only by low structural confidence, since many failed variants occurred in high-confidence regions.

## Tools and Libraries

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib
- seaborn
- AlphaFold structure/confidence data

## Repository Structure

```text
.
├── Data/                  # Input datasets and analysis exports (CSVs)
├── Data_Curation/         # Scripts to build the merged dataset
├── DataExploration/       # Exploratory notebooks
├── Models/                # Benchmarking and AlphaFold notebooks
├── scripts/               # export_and_alphafold.py (batch exports + pLDDT)
└── README.md
```

Key exports in `Data/` include `phase_4_5_6_summary.csv`, `confidence_boundary_analysis.csv`, `alphafold_variants_min20.csv`, and `alphafold_variants_with_plddt.csv`. Re-run exports with:

```bash
python scripts/export_and_alphafold.py
python scripts/export_and_alphafold.py --stats-only  # reuse downloaded structures
```