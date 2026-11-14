"""Input validation, prediction and a simple per-patient explanation."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd

from .pipeline import FEATURES, META_PATH, MODEL_PATH, THRESHOLD

# Accepted ranges follow the value ranges of the cleaned Kurigram data (Table 3.2).
# Values outside them are rejected, since the model was never trained on them.
NUMERIC_RANGES = {
    "Age": (10, 60, "years"),
    "Systolic BP": (70, 200, "mmHg"),
    "Diastolic": (40, 140, "mmHg"),
    "BS": (3.0, 19.0, "mmol/L"),
    "Body Temp": (97, 103, "°F"),
    "BMI": (15, 37, "kg/m²"),
    "Heart Rate": (58, 92, "bpm"),
}
BINARY = ["Previous Complications", "Preexisting Diabetes", "Gestational Diabetes", "Mental Health"]
OPTIONAL = {"BMI"}  # BMI was the most often missing value; it is imputed with the training median

LABELS = {
    "Age": "Age",
    "Systolic BP": "Systolic blood pressure",
    "Diastolic": "Diastolic blood pressure",
    "BS": "Blood sugar",
    "Body Temp": "Body temperature",
    "BMI": "Body mass index",
    "Previous Complications": "Previous complications",
    "Preexisting Diabetes": "Pre-existing diabetes",
    "Gestational Diabetes": "Gestational diabetes",
    "Mental Health": "Mental health issue",
    "Heart Rate": "Heart rate",
}


class ValidationError(ValueError):
    def __init__(self, errors: dict[str, str]):
        super().__init__("; ".join(f"{k}: {v}" for k, v in errors.items()))
        self.errors = errors


def validate(raw: dict) -> tuple[dict, list[str]]:
    """Check and convert one patient's inputs. Returns (clean values, warnings)."""
    errors: dict[str, str] = {}
    warnings: list[str] = []
    clean: dict[str, float] = {}
    for name in FEATURES:
        value = raw.get(name)
        if value is None or (isinstance(value, str) and value.strip() == ""):
            if name in OPTIONAL:
                clean[name] = np.nan
                warnings.append(f"{LABELS[name]} was not entered and was filled with the training median.")
            else:
                errors[name] = "is required"
            continue
        try:
            num = float(value)
        except (TypeError, ValueError):
            errors[name] = "must be a number"
            continue
        if not np.isfinite(num):
            errors[name] = "must be a finite number"
            continue
        if name in BINARY:
            if num not in (0.0, 1.0):
                errors[name] = "must be 0 (no) or 1 (yes)"
                continue
        else:
            lo, hi, unit = NUMERIC_RANGES[name]
            if not lo <= num <= hi:
                errors[name] = f"must be between {lo} and {hi} {unit}"
                continue
        clean[name] = num
    if not errors and clean["Diastolic"] >= clean["Systolic BP"]:
        errors["Diastolic"] = "must be lower than systolic pressure"
    if errors:
        raise ValidationError(errors)
    return clean, warnings


@dataclass
class Prediction:
    probability: float
    risk: str
    threshold: float
    factors: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "probability": round(self.probability, 4),
            "risk": self.risk,
            "threshold": self.threshold,
            "factors": self.factors,
            "warnings": self.warnings,
        }


class Screener:
    def __init__(self, model_path=MODEL_PATH, meta_path=META_PATH, threshold: float = THRESHOLD):
        if not model_path.exists():
            # The trained model is not stored in git; build it on first use.
            from .pipeline import train_and_save

            train_and_save()
        self.model = joblib.load(model_path)
        self.meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self.medians = self.meta.get("medians") or dict(
            zip(FEATURES, self.model.named_steps["impute"].statistics_)
        )
        self.threshold = threshold

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[FEATURES])[:, 1]

    def predict(self, raw: dict) -> Prediction:
        clean, warnings = validate(raw)
        row = pd.DataFrame([clean], columns=FEATURES)
        # Explanation: start from a typical patient (training medians) and give her
        # one of this patient's values at a time; the rise in predicted risk shows
        # how much that value alone pushes towards high risk. One batch call.
        base = pd.DataFrame([{f: self.medians[f] for f in FEATURES}], columns=FEATURES)
        variants = [row, base]
        for name in FEATURES:
            alt = base.copy()
            if not np.isnan(clean[name]):
                alt[name] = clean[name]
            variants.append(alt)
        probs = self.predict_proba(pd.concat(variants, ignore_index=True))
        p, p_base = float(probs[0]), float(probs[1])
        effects = sorted(
            ((FEATURES[i], float(probs[i + 2]) - p_base) for i in range(len(FEATURES))),
            key=lambda t: t[1],
            reverse=True,
        )
        factors = [
            {"feature": LABELS[name], "value": None if np.isnan(clean[name]) else clean[name], "effect": round(d, 3)}
            for name, d in effects
            if d >= 0.05
        ][:3]
        risk = "High" if p >= self.threshold else "Low"
        return Prediction(probability=p, risk=risk, threshold=self.threshold, factors=factors, warnings=warnings)
