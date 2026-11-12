"""Loading, auditing and cleaning the Kurigram maternal health dataset (Section 3.2, Table 3.3)."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "maternal_health_risk_kurigram.csv"
TARGET = "Risk Level"
NUMERIC = ["Age", "Systolic BP", "Diastolic", "BS", "Body Temp", "BMI", "Heart Rate"]
BINARY = ["Previous Complications", "Preexisting Diabetes", "Gestational Diabetes", "Mental Health"]
FEATURES = ["Age", "Systolic BP", "Diastolic", "BS", "Body Temp", "BMI",
            "Previous Complications", "Preexisting Diabetes", "Gestational Diabetes", "Mental Health", "Heart Rate"]


def load_raw(path=DATA_PATH):
    return pd.read_csv(path)


def clean(df, min_age=None):
    """Apply the cleaning steps of Table 3.3 and return (X, y, audit).

    1. drop rows without a risk label
    2. drop exact duplicate rows
    3. drop records with an age above 60
    4. treat BMI = 0 as missing (imputed later inside each training fold)
    5. encode the label as High = 1, Low = 0
    min_age: optional lower age limit used only for the sensitivity analysis.
    """
    audit = [("Raw dataset loaded", None, len(df))]
    df = df.dropna(subset=[TARGET])
    audit.append(("Rows with missing risk label removed", audit[-1][2] - len(df), len(df)))
    df = df.drop_duplicates()
    audit.append(("Exact duplicate rows removed", audit[-1][2] - len(df), len(df)))
    df = df[df["Age"] <= 60]
    audit.append(("Records with age above 60 removed", audit[-1][2] - len(df), len(df)))
    df = df.copy()
    n_zero = int((df["BMI"] == 0).sum())
    df.loc[df["BMI"] == 0, "BMI"] = np.nan
    audit.append(("BMI of zero set to missing", n_zero, len(df)))
    if min_age is not None:
        n_before = len(df)
        df = df[df["Age"] >= min_age]
        audit.append((f"Records with age below {min_age} removed", n_before - len(df), len(df)))
    y = (df[TARGET] == "High").astype(int).to_numpy()
    X = df[FEATURES].reset_index(drop=True)
    audit.append(("Risk label encoded (High = 1, Low = 0)", None, len(df)))
    audit = pd.DataFrame(audit, columns=["Action", "Records affected", "Records remaining"])
    return X, y, audit
