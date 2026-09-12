"""Preprocessing for the Telco churn dataset: cleaning, encoding, splitting."""

import pandas as pd
from sklearn.model_selection import train_test_split


def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the text ``TotalCharges`` column into a proper numeric column.

    On Day 1 you found that ``TotalCharges`` is stored as text, and that 11 rows
    hold a blank space instead of a number (which ``isna()`` did NOT flag,
    because a space is a valid string). Fix that here.
    """
    df = df.copy()
    # ---- YOUR TASK (Day 2, part 1) ----
    # 1. Convert df["TotalCharges"] to numbers. A blank space cannot be parsed,
    #    so tell pandas to turn anything unparseable into NaN instead of raising
    #    an error. Look at the `errors=` option of pd.to_numeric.
    # 2. The rows that became NaN are all customers with tenure == 0 (brand new,
    #    never billed a running total yet). Fill those NaNs with 0.
    # Delete the line below once you have written both steps.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    return df


def prepare_data(df: pd.DataFrame):
    """Full preprocessing pipeline. Returns ``(X, y)`` ready for modeling."""
    df = df.copy()
    df = df.drop(columns=["customerID"])          # an ID is not a predictor
    df = clean_total_charges(df)
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})  # target -> numbers

    # ---- YOUR TASK (Day 2, part 2) ----
    # Every remaining text column (gender, Contract, PaymentMethod, ...) must
    # become numeric before a model can use it. One-hot encode them with
    # pd.get_dummies(...), passing drop_first=True (which drops one redundant
    # column per feature). Assign the result back to `df`.
    # Delete the line below once you have written it.
    df = pd.get_dummies(df, drop_first=True)

    y = df["Churn"]
    X = df.drop(columns=["Churn"])
    return X, y


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """Stratified train/test split.

    ``stratify=y`` keeps the same churn ratio (~27%) in both the train and test
    sets. That matters a lot when the classes are imbalanced: without it, a split
    could accidentally put too few churners in the test set.
    """
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
