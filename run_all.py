"""Reproduce the experiments of the FYDP report and write all result tables to results/.

Usage:  python run_all.py            (full run, a few minutes on a laptop CPU)
"""
from pathlib import Path

import numpy as np
import pandas as pd

from maternal_risk import analysis as A
from maternal_risk.data import clean, load_raw
from maternal_risk.models import get_models

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)


def save(df, name):
    df.to_csv(OUT / f"{name}.csv", index=False, float_format="%.4f")
    print(f"\n=== {name} ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def main():
    raw = load_raw()
    X, y = clean(raw)[:2]
    _, _, audit = clean(raw)
    save(audit, "table_3_3_preprocessing")
    missing = X.isna().sum()
    save(missing[missing > 0].rename("Missing values").reset_index().rename(columns={"index": "Feature"}),
         "missing_values_after_cleaning")
    print(f"\nRecords: {len(y)}, high risk: {y.sum()} ({100 * y.mean():.1f}%)")

    save(A.compare_groups(X, y), "table_4_1_group_comparison")

    models = get_models()
    table, folds = A.cross_validated_comparison(models, X, y)
    save(table, "table_4_2_cross_validation_mean_sd")
    fold_rows = [{"Model": m, "Fold": i + 1, **{k: v[i] for k, v in f.items()}} for m, f in folds.items() for i in range(25)]
    save(pd.DataFrame(fold_rows), "fold_scores_25_folds")
    best = table.sort_values("f1 mean", ascending=False)["Model"].iloc[0]
    save(A.significance_tests(folds, best, len(y)), "corrected_t_test_holm")

    oof, rows = {}, []
    for name, model in models.items():
        oof[name] = A.oof_probabilities(model, X, y)
        rows.append({"Model": name, **A.summarize_predictions(y, oof[name])})
    oof_table = pd.DataFrame(rows)
    save(oof_table, "tables_4_3_4_4_out_of_fold")
    pd.DataFrame({"y": y, **oof}).to_csv(OUT / "out_of_fold_probabilities.csv", index=False)
    gb = oof_table.set_index("Model").loc["Gradient Boosting"]
    lo, hi = A.wilson_interval(int(round(gb["Accuracy"] / 100 * len(y))), len(y))
    print(f"\nGradient Boosting pooled accuracy 95% Wilson CI: {lo:.1f}% to {hi:.1f}%")

    save(A.rule_baselines(X, y), "table_4_5_rule_baselines")
    save(A.feature_ablation(X, y), "table_4_6_feature_ablation")
    save(A.single_feature_auc(X, y), "figure_4_20_single_feature_auc")
    save(A.permutation_importances(X, y), "figure_4_21_permutation_importance")
    save(A.threshold_table(y, oof["Random Forest"]), "table_4_7_threshold_random_forest")
    save(A.calibration(y, oof["Random Forest"]), "figure_4_22_calibration_random_forest")
    no_skill = np.mean((y - y.mean()) ** 2)
    print(f"\nBrier score of a model that always predicts the prevalence: {no_skill:.3f}")
    save(A.learning_curves(models, X, y), "learning_curves")

    Xs, ys, _ = clean(raw, min_age=15)
    sens, _ = A.cross_validated_comparison({"Gradient Boosting": models["Gradient Boosting"]}, Xs, ys)
    save(sens, "sensitivity_without_ages_below_15")


if __name__ == "__main__":
    main()
