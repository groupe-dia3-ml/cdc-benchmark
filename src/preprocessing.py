import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "cdc_diabetes_raw.csv")

# Feature grouping as defined in project specifications
LIFESTYLE_VARS = ["PhysActivity", "Fruits", "Veggies", "Smoker", "HvyAlcoholConsump"]
DEMOGRAPHIC_VARS = ["Sex", "Age", "Education", "Income"]
HEALTH_VARS = [
    "BMI", "HighBP", "HighChol", "CholCheck", "Stroke", 
    "HeartDiseaseorAttack", "GenHlth", "MentHlth", "PhysHlth", "DiffWalk"
]
ACCESS_VARS = ["AnyHealthcare", "NoDocbcCost"]

ALL_PREDICTORS = LIFESTYLE_VARS + DEMOGRAPHIC_VARS + HEALTH_VARS + ACCESS_VARS
LIFESTYLE_AND_DEMO = LIFESTYLE_VARS + DEMOGRAPHIC_VARS

def load_preprocessed_splits(filepath=RAW_DATA_PATH, test_size=0.20, random_state=42):
    """
    Loads raw CSV, excludes 'ID', extracts features and target,
    and returns stratified train/test splits.
    """
    df = pd.read_csv(filepath)
    
    # Exclude ID and target from predictors
    X = df[ALL_PREDICTORS].copy()
    y = df["Diabetes_binary"].values
    
    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    print(f"Dataset split: Train={X_train.shape[0]} samples, Test={X_test.shape[0]} samples")
    print(f"Positive class prevalence in Train: {y_train.mean():.4f}, in Test: {y_test.mean():.4f}")
    return X_train, X_test, y_train, y_test
