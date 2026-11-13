# CDC Diabetes Health Indicators: Benchmark Repository

Phase 1 of the Machine Learning final project (ESILV, group DIA3).

This repository focuses on data ingestion, exploratory data analysis, and baseline modeling on the 253,680 survey records from the CDC Diabetes Health Indicators dataset (UCI dataset 891).

## Repository Structure

```
cdc-benchmark/
├── data/
│   ├── raw/                 # Downloaded dataset (excluded from git via .gitignore)
│   └── processed/           # Processed features and splits
├── notebooks/
│   └── 01_cdc_diabetes_benchmark.ipynb  # Main end-to-end benchmark notebook
├── src/
│   ├── data_loader.py       # Programmatic download and checksum verification
│   ├── preprocessing.py     # Feature groupings and train/test splits
│   ├── models.py            # Simple candidate model pipelines
│   └── evaluate.py          # Probabilistic evaluation and calibration metrics
├── .gitignore
├── requirements.txt
└── README.md
```

## Dataset Specifications

- Source: UCI Machine Learning Repository (Dataset ID: 891)
- URL: https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators
- Rows: 253,680 respondents
- Predictors: 21 discrete/ordinal lifestyle and clinical features
- Target: `Diabetes_binary` (0 = no diabetes, 1 = prediabetes or diabetes)
- Class balance: 218,334 negative (86.07%), 35,346 positive (13.93%)
- SHA-256 Checksum: `9f71fda9d4ae5f4878c99b9233b6a16accfa9a17c194116a6b78100540934964`

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Download data:
   ```bash
   python src/data_loader.py
   ```
3. Run the benchmark notebook:
   Open `notebooks/01_cdc_diabetes_benchmark.ipynb` in VS Code or JupyterLab.
