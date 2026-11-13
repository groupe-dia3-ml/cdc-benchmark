import os
import urllib.request
import hashlib
import pandas as pd

UCI_DATA_URL = "https://archive.ics.uci.edu/static/public/891/data.csv"
RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "cdc_diabetes_raw.csv")

def download_dataset(url=UCI_DATA_URL, destination=RAW_DATA_PATH, force=False):
    """
    Downloads the CDC Diabetes Health Indicators dataset from UCI repository
    if not already present locally.
    """
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    if os.path.exists(destination) and not force:
        print(f"Dataset already exists at: {destination}")
        return destination

    print(f"Downloading CDC Diabetes dataset from: {url} ...")
    urllib.request.urlretrieve(url, destination)
    print(f"Download complete: {destination}")
    return destination

def verify_dataset(filepath=RAW_DATA_PATH):
    """
    Verifies row counts, column counts, and computes SHA-256 checksum.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    checksum = sha256.hexdigest()

    df = pd.read_csv(filepath)
    print("--- Dataset Verification ---")
    print(f"Rows: {df.shape[0]} (Expected: 253,680)")
    print(f"Columns: {df.shape[1]}")
    print(f"SHA-256: {checksum}")
    print("Columns list:", list(df.columns))
    print(f"Target counts (Diabetes_binary):\n{df['Diabetes_binary'].value_counts(dropna=False)}")
    return df, checksum

if __name__ == "__main__":
    path = download_dataset()
    df, csum = verify_dataset(path)
