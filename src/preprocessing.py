"""Feature definitions and a single, deterministic development/test split."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

LIFESTYLE_VARS = ["PhysActivity", "Fruits", "Veggies", "Smoker", "HvyAlcoholConsump"]
DEMOGRAPHIC_VARS = ["Sex", "Age", "Education", "Income"]
HEALTH_VARS = [
    "BMI",
    "HighBP",
    "HighChol",
    "CholCheck",
    "Stroke",
    "HeartDiseaseorAttack",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
]
ACCESS_VARS = ["AnyHealthcare", "NoDocbcCost"]
LIFESTYLE_AND_DEMO = LIFESTYLE_VARS + DEMOGRAPHIC_VARS
ALL_PREDICTORS = LIFESTYLE_AND_DEMO + HEALTH_VARS + ACCESS_VARS
TARGET = "Diabetes_binary"
ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT / "data/raw/cdc_diabetes_raw.csv"


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Retain source indices to audit split membership without using ID as input."""
    return train_test_split(
        df[ALL_PREDICTORS].copy(),
        df[TARGET].astype(int),
        stratify=df[TARGET],
        test_size=test_size,
        random_state=random_state,
    )


def load_preprocessed_splits(filepath=RAW_DATA_PATH, test_size=0.2, random_state=42):
    from .data_loader import verify_dataset

    df, _ = verify_dataset(filepath)
    return split_data(df, test_size, random_state)
