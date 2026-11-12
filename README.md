# Machine Learning Model for Accurate Prediction of Maternal Health Risk

Code for the final year design project (FYDP) report of **Md. Abdulla Al Mamun** and **Rakib Miah**,
Department of Computer Science and Engineering, Daffodil International University
(supervisor: Ms. Shayla Sharmin, co-supervisor: Mr. Shahadat Hossain).

The code reproduces every table and supporting analysis of Chapters 3 and 4 of the report from the raw
Kurigram maternal health dataset, using the leakage-controlled protocol described there:

- cleaning of the 1,205 raw records (missing labels, exact duplicates, ages above 60, BMI of zero set to missing), leaving 1,166 records with 471 high-risk patients;
- median imputation and standard scaling fitted inside each training fold only;
- eight classical classifiers compared with stratified five-fold cross-validation repeated five times (seed 42);
- corrected resampled t-test (variance factor 1/25 + 1/4 = 0.29, 24 degrees of freedom) with Holm correction;
- pooled out-of-fold predictions, three clinical rule baselines, feature ablation, single-feature AUC, permutation importance, learning curves, threshold tuning, calibration and a sensitivity analysis.

## How to run

```bash
pip install -r requirements.txt
python run_all.py
```

The full run takes about 15 minutes on a laptop CPU. All tables are written to `results/` as CSV files.

| File in `results/` | Report |
|---|---|
| `table_3_3_preprocessing.csv` | Table 3.3 |
| `table_4_1_group_comparison.csv` | Table 4.1 |
| `table_4_2_cross_validation_mean_sd.csv`, `fold_scores_25_folds.csv` | Table 4.2 (mean and SD), Figure 4.8 |
| `corrected_t_test_holm.csv` | Section 4.2, statistical comparison |
| `tables_4_3_4_4_out_of_fold.csv`, `out_of_fold_probabilities.csv` | Tables 4.3 and 4.4, Figures 4.9 to 4.19 |
| `table_4_5_rule_baselines.csv` | Table 4.5 |
| `table_4_6_feature_ablation.csv` | Table 4.6 |
| `figure_4_20_single_feature_auc.csv`, `figure_4_21_permutation_importance.csv` | Figures 4.20 and 4.21 |
| `table_4_7_threshold_random_forest.csv`, `figure_4_22_calibration_random_forest.csv` | Table 4.7, Figure 4.22 |
| `learning_curves.csv` | Figures 4.11, 4.14 and 4.17 |
| `sensitivity_without_ages_below_15.csv` | Section 4.3, sensitivity analysis |

## Agreement with the report

Results were produced with Python 3.13, scikit-learn 1.9.1, pandas 3.0.5, NumPy 2.5.3 and SciPy 1.18.1.

- Cleaning, Table 4.1, the rule baselines, the feature ablation, single-feature AUC and permutation importance match the report exactly.
- Cross-validated means (Table 4.2) agree with the report within 0.2 percentage points. Gradient Boosting reaches 98.8% accuracy (SD 0.6), an F1-score of 98.5% and an ROC-AUC of 99.95%.
- Pooled out-of-fold errors match for six of the eight models. Extra Trees has 22 errors (report: 21) and Decision Tree splits its 41 errors as 21 missed and 20 false alarms (report: 22 and 19).
- At a threshold of 0.3, Random Forest misses no high-risk patient with 28 false alarms (report: 29).
- The corrected t-test gives p = 0.55 for Gradient Boosting against Random Forest (report: 0.49) and p = 0.0498 against Extra Trees (report: 0.04). Neither is significant after Holm correction, as in the report.

Small differences come from library versions and from the order in which random numbers are drawn.

## Data

`data/maternal_health_risk_kurigram.csv` is the Kurigram General Hospital maternal health risk dataset of
Mojumdar et al., released on Mendeley Data under the CC BY 4.0 licence. See `data/README.md` for the citation.

## Repository layout

```
data/                    raw dataset and its source
maternal_risk/data.py    loading, auditing and cleaning
maternal_risk/models.py  the eight fold-wise pipelines (Table 3.4)
maternal_risk/analysis.py  all experiments of Chapter 4
run_all.py               runs everything and writes results/
results/                 output tables
```
