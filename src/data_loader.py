"""Acquire the exact UCI export; fail closed on content or schema drift."""

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .preprocessing import ALL_PREDICTORS, RAW_DATA_PATH, TARGET

UCI_DATA_URL = "https://archive.ics.uci.edu/static/public/891/data.csv"
EXPECTED_SHA256 = "9f71fda9d4ae5f4878c99b9233b6a16accfa9a17c194116a6b78100540934964"
RANGES = {name: (0, 1) for name in ALL_PREDICTORS}
RANGES.update(
    BMI=(12, 98),
    Age=(1, 13),
    Education=(1, 6),
    Income=(1, 8),
    GenHlth=(1, 5),
    MentHlth=(0, 30),
    PhysHlth=(0, 30),
)


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_frame(df: pd.DataFrame, exact_export: bool = True) -> None:
    if set(df.columns) != set(ALL_PREDICTORS + ["ID", TARGET]):
        raise ValueError("Unexpected dataset columns")
    if df.isna().any().any():
        raise ValueError("Missing values in the selected export")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in df.dtypes):
        raise ValueError("Non-numeric survey data")
    if not np.isfinite(df.to_numpy()).all():
        raise ValueError("Non-finite survey data")
    if not df["ID"].is_unique or not (df["ID"] % 1 == 0).all():
        raise ValueError("Invalid respondent identifiers")
    for name, (low, high) in {**RANGES, TARGET: (0, 1)}.items():
        if not df[name].between(low, high).all() or not (df[name] % 1 == 0).all():
            raise ValueError(f"Invalid values for {name}")
    if exact_export and (len(df) != 253680 or df[TARGET].value_counts().to_dict() != {0: 218334, 1: 35346}):
        raise ValueError("Unexpected row or target counts")


def verify_dataset(filepath: str | Path = RAW_DATA_PATH):
    checksum = file_sha256(filepath)
    if checksum != EXPECTED_SHA256:
        raise ValueError(f"UCI checksum mismatch: {checksum}")
    df = pd.read_csv(filepath)
    validate_frame(df)
    return df, checksum


def download_dataset(url=UCI_DATA_URL, destination=RAW_DATA_PATH, force=False) -> Path:
    destination = Path(destination)
    if destination.exists() and not force:
        verify_dataset(destination)
        return destination
    if url != UCI_DATA_URL:
        raise ValueError("Only the documented UCI binary export is accepted")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".download")
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            temporary.write_bytes(response.read())
        _, checksum = verify_dataset(temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    destination.with_suffix(".provenance.json").write_text(
        json.dumps(
            {
                "url": url,
                "sha256": checksum,
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
        + "\n"
    )
    return destination


if __name__ == "__main__":
    path = download_dataset()
    frame, checksum = verify_dataset(path)
    print(f"Verified {frame.shape}: {checksum}")
