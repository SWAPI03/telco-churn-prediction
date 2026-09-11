"""Data loading utilities for the Telco churn project."""

from pathlib import Path

import pandas as pd

# data/raw/telco_churn.csv, resolved relative to the project root
RAW_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "telco_churn.csv"


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw Telco churn CSV into a DataFrame.

    Parameters
    ----------
    path : Path
        Location of the raw CSV file. Defaults to ``data/raw/telco_churn.csv``.

    Returns
    -------
    pd.DataFrame
        The dataset exactly as read from disk (no cleaning yet).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find the dataset at {path}.\n"
            "Download it from Kaggle (see the README), rename it to "
            "'telco_churn.csv', and save it under data/raw/."
        )
    return pd.read_csv(path)
