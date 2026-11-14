import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from screening.pipeline import FEATURES, THRESHOLD, load_clean, make_model
from screening.predictor import ValidationError


def test_cleaned_data_matches_report():
    X, y = load_clean()
    assert len(y) == 1166 and int(y.sum()) == 471
    assert int(X.isna().sum().sum()) == 25


def test_saved_model_equals_fresh_training(screener):
    """Train-serve consistency: the deployed model gives the same probabilities as retraining."""
    X, y = load_clean()
    fresh = make_model().fit(X, y).predict_proba(X)[:, 1]
    deployed = screener.predict_proba(X)
    assert np.array_equal(fresh, deployed)


def test_api_matches_model_for_every_valid_record(client, screener):
    """End-to-end: JSON API probabilities equal direct model output for all records the form accepts."""
    X, _ = load_clean()
    checked = 0
    for _, row in X.iterrows():
        payload = {k: (None if pd.isna(v) else float(v)) for k, v in row.items()}
        r = client.post("/api/predict", json=payload)
        if r.status_code == 422:
            continue
        assert r.status_code == 200
        direct = float(screener.predict_proba(pd.DataFrame([row], columns=FEATURES))[0])
        assert abs(r.get_json()["probability"] - round(direct, 4)) < 1e-9
        checked += 1
    assert checked >= 1150


def test_threshold_rule(screener, valid):
    pred = screener.predict(valid)
    assert (pred.risk == "High") == (pred.probability >= THRESHOLD)


def test_screening_threshold_regression():
    """Out-of-fold check that the 0.3 threshold still misses no high-risk record (Table 4.7)."""
    X, y = load_clean()
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    oof = cross_val_predict(make_model(), X, y, cv=cv, method="predict_proba")[:, 1]
    tn, fp, fn, tp = confusion_matrix(y, (oof >= THRESHOLD).astype(int)).ravel()
    assert fn == 0
    assert fp <= 30


def test_prediction_is_deterministic(screener, valid):
    assert screener.predict(valid).probability == screener.predict(valid).probability


def test_invalid_input_never_reaches_model(screener, valid):
    try:
        screener.predict(dict(valid, BS=50))
    except ValidationError:
        return
    raise AssertionError("out-of-range input was not rejected")
