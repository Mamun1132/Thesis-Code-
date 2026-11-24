"""All experiments reported in Chapter 4 of the FYDP report."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict, cross_validate

from .data import BINARY, NUMERIC
from .models import SEED, make_pipeline

SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def repeated_cv():
    return RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)


# ----------------------------------------------------------------- Section 4.1
def compare_groups(X, y):
    """Table 4.1: medians (IQR) or percentages by risk level with Mann-Whitney U or chi-square tests."""
    rows = []
    hi, lo = X[y == 1], X[y == 0]
    for c in NUMERIC:
        a, b = hi[c].dropna(), lo[c].dropna()
        p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
        fmt = lambda s: f"{s.median():g} ({s.quantile(.25):g}-{s.quantile(.75):g})"
        rows.append((c, fmt(a), fmt(b), "Mann-Whitney U", p, len(a), len(b)))
    for c in BINARY:
        a, b = hi[c].dropna(), lo[c].dropna()
        table = pd.crosstab(pd.concat([a, b]), np.r_[np.ones(len(a)), np.zeros(len(b))])
        p = stats.chi2_contingency(table, correction=True)[1]
        rows.append((c, f"{100 * a.mean():.1f}%", f"{100 * b.mean():.1f}%", "Chi-square", p, len(a), len(b)))
    return pd.DataFrame(rows, columns=["Feature", "High risk", "Low risk", "Test", "p-value", "n high", "n low"])


# ----------------------------------------------------------------- Section 4.2
def cross_validated_comparison(models, X, y):
    """Table 4.2: mean and standard deviation of each metric over the 25 folds; also returns fold scores."""
    folds, rows = {}, []
    for name, model in models.items():
        r = cross_validate(model, X, y, cv=repeated_cv(), scoring=SCORING, n_jobs=1)
        folds[name] = {m: r[f"test_{m}"] for m in SCORING}
        row = {"Model": name}
        for m in SCORING:
            row[f"{m} mean"] = 100 * r[f"test_{m}"].mean()
            row[f"{m} SD"] = 100 * r[f"test_{m}"].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows), folds


def corrected_t_test(a, b, n_train, n_test, k=25):
    """Corrected resampled t-test (Nadeau and Bengio) for repeated cross-validation.

    The variance of the fold differences is multiplied by (1/k + n_test/n_train) instead of 1/k.
    For 5 x 5 cross-validation this factor is 1/25 + 1/4 = 0.29 and the test has k - 1 = 24 degrees of freedom.
    """
    d = np.asarray(a) - np.asarray(b)
    var = d.var(ddof=1)
    t = d.mean() / np.sqrt((1.0 / k + n_test / n_train) * var) if var > 0 else np.inf
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return t, p


def holm(pvalues):
    """Holm step-down adjusted p-values."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def significance_tests(folds, best, n):
    n_test = n / 5
    n_train = n - n_test
    rows = []
    for name in folds:
        if name == best:
            continue
        t, p = corrected_t_test(folds[best]["accuracy"], folds[name]["accuracy"], n_train, n_test)
        rows.append((name, t, p))
    res = pd.DataFrame(rows, columns=["Compared with", "t", "p (unadjusted)"])
    res["p (Holm-adjusted)"] = holm(res["p (unadjusted)"])
    res.insert(0, "Best model", best)
    return res


# ----------------------------------------------------------------- out-of-fold evaluation
def oof_probabilities(model, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    return cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]


def summarize_predictions(y, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {"Accuracy": 100 * accuracy_score(y, pred), "Precision": 100 * precision_score(y, pred),
            "Recall": 100 * recall_score(y, pred), "F1-score": 100 * f1_score(y, pred),
            "ROC-AUC": 100 * roc_auc_score(y, prob), "Brier score": brier_score_loss(y, prob),
            "Missed high-risk": int(fn), "False alarms": int(fp), "Errors": int(fn + fp),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}


def wilson_interval(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (centre - half), 100 * (centre + half)


# ----------------------------------------------------------------- Section 4.3
def rule_baselines(X, y):
    """Table 4.5: three clinical rules applied without any fitting."""
    rules = {
        "Rule: pre-existing or gestational diabetes": (X["Preexisting Diabetes"] == 1) | (X["Gestational Diabetes"] == 1),
        "Rule: any of the four history indicators": X[["Previous Complications", "Preexisting Diabetes",
                                                       "Gestational Diabetes", "Mental Health"]].eq(1).any(axis=1),
        "Rule: blood pressure >= 140/90 mmHg": (X["Systolic BP"] >= 140) | (X["Diastolic"] >= 90),
    }
    rows = []
    for name, flag in rules.items():
        pred = flag.astype(int).to_numpy()
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        rows.append({"Method": name, "Accuracy": 100 * accuracy_score(y, pred), "F1-score": 100 * f1_score(y, pred),
                     "Recall": 100 * recall_score(y, pred), "Missed high-risk": int(fn), "False alarms": int(fp)})
    return pd.DataFrame(rows)


def rf():
    return make_pipeline(RandomForestClassifier(n_estimators=400, random_state=SEED, n_jobs=-1), False)


def feature_ablation(X, y):
    """Table 4.6: Random Forest retrained on feature subsets under the same repeated cross-validation."""
    history = ["Previous Complications", "Preexisting Diabetes", "Gestational Diabetes", "Mental Health"]
    vitals = ["Age", "Systolic BP", "Diastolic", "BS", "Body Temp", "BMI", "Heart Rate"]
    subsets = {
        "All 11 features": list(X.columns),
        "Without diabetes indicators": [c for c in X.columns if c not in ("Preexisting Diabetes", "Gestational Diabetes")],
        "Without BMI and heart rate": [c for c in X.columns if c not in ("BMI", "Heart Rate")],
        "Vital signs only": vitals,
        "History indicators only": history,
        "Blood pressure, blood sugar and age": ["Systolic BP", "Diastolic", "BS", "Age"],
    }
    rows = []
    for name, cols in subsets.items():
        r = cross_validate(rf(), X[cols], y, cv=repeated_cv(), scoring=SCORING)
        rows.append({"Feature set": name, **{m: 100 * r[f"test_{m}"].mean() for m in SCORING}})
    return pd.DataFrame(rows)


def single_feature_auc(X, y):
    rows = []
    for c in X.columns:
        r = cross_validate(rf(), X[[c]], y, cv=repeated_cv(), scoring=["roc_auc"])
        rows.append((c, r["test_roc_auc"].mean()))
    return pd.DataFrame(rows, columns=["Feature", "ROC-AUC"]).sort_values("ROC-AUC", ascending=False)


def permutation_importances(X, y):
    """Drop in ROC-AUC on the test part of each of five folds, 20 shuffles per feature."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    imp = []
    for tr, te in cv.split(X, y):
        m = clone(rf()).fit(X.iloc[tr], y[tr])
        r = permutation_importance(m, X.iloc[te], y[te], scoring="roc_auc", n_repeats=20, random_state=SEED)
        imp.append(r.importances_mean)
    imp = np.array(imp)
    return pd.DataFrame({"Feature": X.columns, "Mean drop in ROC-AUC": imp.mean(axis=0),
                         "SD across folds": imp.std(axis=0, ddof=1)}).sort_values("Mean drop in ROC-AUC", ascending=False)


def learning_curve_folds(models, X, y, names=None):
    """Per-fold training and validation accuracy at 10 training sizes (10% to 100% of each training part)."""
    rows = []
    fractions = np.linspace(0.1, 1.0, 10)
    for name in names or list(models):
        for frac in fractions:
            for k, (tr, te) in enumerate(repeated_cv().split(X, y)):
                rng = np.random.RandomState(SEED)
                n_sub = int(round(frac * len(tr)))
                sub = rng.choice(tr, n_sub, replace=False) if n_sub < len(tr) else tr
                m = clone(models[name]).fit(X.iloc[sub], y[sub])
                rows.append({"Model": name, "Fraction": round(frac, 1), "Fold": k + 1, "Training patients": n_sub,
                             "Training accuracy": 100 * accuracy_score(y[sub], m.predict(X.iloc[sub])),
                             "Validation accuracy": 100 * accuracy_score(y[te], m.predict(X.iloc[te]))})
    return pd.DataFrame(rows)


def learning_curves(folds):
    """Figures 4.11, 4.14, 4.17 and 4.23: mean over the 25 folds at each training size."""
    g = folds.groupby(["Model", "Fraction"], sort=False)
    return pd.DataFrame({
        "Training patients": g["Training patients"].mean().round().astype(int),
        "Training accuracy": g["Training accuracy"].mean(),
        "Validation accuracy": g["Validation accuracy"].mean(),
        "Validation SD": g["Validation accuracy"].std(ddof=1),
    }).reset_index().drop(columns="Fraction")


def paired_learning_difference(folds, a="Gradient Boosting", b="Random Forest"):
    """Figure 4.24b: validation accuracy of model a minus model b in each fold, mean and SD at each training size."""
    w = folds[folds["Model"].isin([a, b])].pivot_table(index=["Fraction", "Fold"], columns="Model",
                                                       values="Validation accuracy")
    d = (w[a] - w[b]).groupby(level="Fraction")
    sizes = folds.groupby("Fraction")["Training patients"].mean().round().astype(int)
    return pd.DataFrame({"Training patients": sizes, f"{a} minus {b} (mean)": d.mean(),
                         "SD": d.std(ddof=1)}).reset_index(drop=True)


def threshold_table(y, prob, thresholds=(0.5, 0.4, 0.3, 0.2)):
    rows = []
    for t in thresholds:
        s = summarize_predictions(y, prob, t)
        rows.append({"Threshold": t, "Recall": s["Recall"], "Precision": s["Precision"], "Accuracy": s["Accuracy"],
                     "Missed high-risk": s["Missed high-risk"], "False alarms": s["False alarms"]})
    return pd.DataFrame(rows)


def calibration(y, prob, bins=10):
    frac_pos, mean_pred = calibration_curve(y, prob, n_bins=bins)
    return pd.DataFrame({"Mean predicted probability": mean_pred, "Observed share of high risk": frac_pos})
