"""Day 3: a full preprocessing + classifier pipeline that accepts RAW data.

Unlike Day 2 (manual ``get_dummies``), here the encoding and scaling live INSIDE
the sklearn pipeline. That means the saved model can take a raw customer record
and predict directly, with no separate preprocessing step. That is what makes
the model clean to deploy in the Streamlit app.
"""

from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "churn_pipeline.joblib"

# The numeric columns in the cleaned frame; everything else is categorical text.
NUMERIC_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]


def build_pipeline(categorical_cols, numeric_cols=NUMERIC_COLS):
    """Preprocessing (scale numeric + one-hot categorical) plus Logistic Regression.

    ``class_weight="balanced"`` up-weights the rare churn class so the model
    stops ignoring churners, directly targeting the low recall you saw on Day 2.
    ``handle_unknown="ignore"`` keeps the model from crashing if the app ever
    sends a category it did not see in training.
    """
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )
    return Pipeline(
        [
            ("prep", preprocess),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def save_model(pipeline, path: Path = MODEL_PATH) -> None:
    """Persist a fitted pipeline to disk with joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_model(path: Path = MODEL_PATH):
    """Load a previously saved pipeline."""
    return joblib.load(path)
