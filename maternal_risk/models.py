"""The eight classifiers of Table 3.4, each wrapped in a fold-wise preprocessing pipeline (Figure 3.2)."""
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 42


def make_pipeline(model, scale):
    """Median imputation (and scaling for scale-dependent models) fitted on the training part of each fold only."""
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def get_models():
    return {
        "Gradient Boosting": make_pipeline(HistGradientBoostingClassifier(random_state=SEED), False),
        "Random Forest": make_pipeline(RandomForestClassifier(n_estimators=400, random_state=SEED, n_jobs=-1), False),
        "Extra Trees": make_pipeline(ExtraTreesClassifier(n_estimators=400, random_state=SEED, n_jobs=-1), False),
        "SVM": make_pipeline(SVC(kernel="rbf", probability=True, random_state=SEED), True),
        "KNN": make_pipeline(KNeighborsClassifier(n_neighbors=7), True),
        "Logistic Regression": make_pipeline(LogisticRegression(max_iter=3000), True),
        "Decision Tree": make_pipeline(DecisionTreeClassifier(max_depth=6, random_state=SEED), False),
        "Naive Bayes": make_pipeline(GaussianNB(), True),
    }
