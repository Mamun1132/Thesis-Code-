"""Data cleaning, model definition and training for the maternal risk screening tool.

The cleaning steps and model settings follow Chapter 3 of the FYDP report:
rows without a label, exact duplicates and ages above 60 are removed, a BMI of
zero is set to missing, and a Random Forest with 400 trees (seed 42) is trained
behind a median imputer.
"""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

FEATURES = [
    "Age",
    "Systolic BP",
    "Diastolic",
    "BS",
    "Body Temp",
    "BMI",
    "Previous Complications",
    "Preexisting Diabetes",
    "Gestational Diabetes",
    "Mental Health",
    "Heart Rate",
]
TARGET = "Risk Level"
SEED = 42
THRESHOLD = 0.3  # screening threshold chosen in Section 4.3 of the report

ROOT = Path(__file__).resolve().parent.parent
# Inside the thesis repository the app reads the dataset from ../data/; a standalone
# copy of the app can keep its own data/kurigram_maternal_health.csv instead.
_CANDIDATES = [
    ROOT / "data" / "kurigram_maternal_health.csv",
    ROOT.parent / "data" / "maternal_health_risk_kurigram.csv",
]
DATA_PATH = next((p for p in _CANDIDATES if p.exists()), _CANDIDATES[-1])
MODEL_PATH = ROOT / "models" / "rf_model.joblib"
META_PATH = ROOT / "models" / "model_meta.json"


def load_clean(path: Path | str = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Return the cleaned feature table and binary label (High = 1)."""
    df = pd.read_csv(path)
    df = df.dropna(subset=[TARGET]).drop_duplicates()
    df = df[~(df["Age"] > 60)].copy()
    df.loc[df["BMI"] == 0, "BMI"] = np.nan
    X = df[FEATURES].reset_index(drop=True)
    y = (df[TARGET] == "High").astype(int).reset_index(drop=True)
    return X, y


def make_model(seed: int = SEED) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("rf", RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=1)),
        ]
    )


def train_and_save(data_path: Path | str = DATA_PATH) -> dict:
    X, y = load_clean(data_path)
    model = make_model().fit(X, y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    meta = {
        "model": "RandomForestClassifier(n_estimators=400, random_state=42) with median imputation",
        "threshold": THRESHOLD,
        "features": FEATURES,
        "training_records": int(len(y)),
        "high_risk_records": int(y.sum()),
        "medians": {f: float(v) for f, v in zip(FEATURES, model.named_steps["impute"].statistics_)},
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    return meta
