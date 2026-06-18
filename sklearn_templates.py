"""
============================================================================
 scikit-learn TEMPLATES  —  copy-paste building blocks for ML projects
============================================================================
A reference library of the standard scikit-learn workflows. Each SECTION is
self-contained: copy the block you need into your own notebook/script and
swap in your data where you see  # >>> REPLACE.

This whole file also RUNS top-to-bottom on built-in datasets, so you can
execute it once to see every template work:   python sklearn_templates.py

Sections
  1. Setup / imports
  2. Quick classification (minimal end-to-end)
  3. Quick regression (minimal end-to-end)
  4. Full preprocessing Pipeline for MIXED data (numbers + text + missing)
  5. Cross-validation
  6. Hyperparameter tuning (GridSearchCV + RandomizedSearchCV, with Pipeline)
  7. FULL evaluation  (accuracy, precision, recall, F1, ROC-AUC, MCC ... )
  8. PREDICTION on new / unseen data (predict + probabilities + confidence)
  9. Save / load a model
 10. *** COMPLETE END-TO-END PIPELINE ***  (train -> tune -> evaluate ->
        save -> reload -> predict on a brand-new dataset)
 11. Model "swap" cheat-sheet (drop-in estimators)
============================================================================
"""

# ===========================================================================
# 1. SETUP / IMPORTS  (put at the top of every project)
# ===========================================================================
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import (train_test_split, cross_val_score,
                                     StratifiedKFold, KFold,
                                     GridSearchCV, RandomizedSearchCV)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

RANDOM_STATE = 42                       # fix randomness -> reproducible runs
np.random.seed(RANDOM_STATE)


# ===========================================================================
# 2. QUICK CLASSIFICATION  (the 6-line core: split -> fit -> predict -> score)
# ===========================================================================
def template_quick_classification():
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target        # >>> REPLACE with your own X (features), y (label)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    model = RandomForestClassifier(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)                       # train
    preds = model.predict(X_test)                     # predict
    print("[2] accuracy:", round(accuracy_score(y_test, preds), 4))
    return model


# ===========================================================================
# 3. QUICK REGRESSION  (same shape, numeric target)
# ===========================================================================
def template_quick_regression():
    from sklearn.datasets import load_diabetes
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, r2_score

    data = load_diabetes(as_frame=True)
    X, y = data.data, data.target        # >>> REPLACE with your own X, y (y is a number)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)

    model = RandomForestRegressor(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"[3] MAE: {mean_absolute_error(y_test, preds):.2f} | "
          f"R2: {r2_score(y_test, preds):.4f}")
    return model


# ===========================================================================
# 4. FULL PREPROCESSING PIPELINE for MIXED data
#    Handles: missing values + categorical encoding + numeric scaling,
#    all leak-free inside one object you can CV / tune / save / reuse.
# ===========================================================================
def make_preprocessor(numeric_features, categorical_features):
    """ColumnTransformer that imputes+scales numbers and
    imputes+one-hot-encodes categories. Reuse in any pipeline."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric,     numeric_features),
        ("cat", categorical, categorical_features),
    ])


def make_demo_mixed_data(n=500, seed=RANDOM_STATE):
    """Build a small MIXED dataset (numbers + text + missing values).
    Stand-in for:  df = pd.read_csv('your.csv')  """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age":    rng.integers(18, 80, n).astype(float),
        "income": rng.normal(50000, 15000, n),
        "city":   rng.choice(["NY", "SF", "LA"], n),
        "plan":   rng.choice(["free", "pro"], n),
    })
    df["target"] = ((df["income"] > 50000) ^ (df["plan"] == "pro")).astype(int)
    for c in ["age", "income", "city"]:                       # punch holes
        df.loc[rng.choice(n, n // 12, replace=False), c] = np.nan
    return df


def template_full_pipeline():
    from sklearn.ensemble import RandomForestClassifier

    df = make_demo_mixed_data()
    df = df.dropna(subset=["target"])                         # never impute the label

    numeric_features     = ["age", "income"]                 # >>> REPLACE numeric cols
    categorical_features = ["city", "plan"]                   # >>> REPLACE text cols

    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    clf = Pipeline([
        ("prep",  make_preprocessor(numeric_features, categorical_features)),
        ("model", RandomForestClassifier(random_state=RANDOM_STATE)),
    ])
    clf.fit(X_train, y_train)
    print("[4] mixed-data pipeline fitted. Test accuracy:",
          round(clf.score(X_test, y_test), 4))
    return clf, X_train, X_test, y_train, y_test


# ===========================================================================
# 5. CROSS-VALIDATION  (a trustworthy score, not one lucky split)
# ===========================================================================
def template_cross_validation():
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier

    X, y = load_breast_cancer(return_X_y=True)                # >>> REPLACE with your X, y
    model = RandomForestClassifier(random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    # scoring: "f1","roc_auc","r2","neg_mean_absolute_error", ...
    print(f"[5] 5-fold CV: {scores.mean():.4f} +/- {scores.std():.4f}")
    return scores


# ===========================================================================
# 6. HYPERPARAMETER TUNING  (Grid + Random, inside a Pipeline)
#    KEY: to tune a step inside a pipeline use  "stepname__param".
# ===========================================================================
def template_grid_search(clf, X_train, y_train, X_test, y_test):
    param_grid = {
        "model__n_estimators":          [100, 300],
        "model__max_depth":             [None, 5, 10],
        "model__min_samples_leaf":      [1, 2, 4],
        "prep__num__imputer__strategy": ["mean", "median"],   # tune preprocessing too
    }
    grid = GridSearchCV(clf, param_grid, cv=5, scoring="f1", n_jobs=-1)
    grid.fit(X_train, y_train)
    print("[6a] GridSearch best params:", grid.best_params_)
    print("[6a] GridSearch best CV f1 :", round(grid.best_score_, 4))
    return grid.best_estimator_                               # the refit, tuned pipeline


def template_random_search(clf, X_train, y_train):
    from scipy.stats import randint
    param_dist = {
        "model__n_estimators":     randint(100, 500),
        "model__max_depth":        [None, 5, 10, 20],
        "model__min_samples_leaf": randint(1, 8),
        "model__max_features":     ["sqrt", "log2", None],
    }
    rand = RandomizedSearchCV(clf, param_dist, n_iter=20, cv=5,
                              scoring="f1", random_state=RANDOM_STATE, n_jobs=-1)
    rand.fit(X_train, y_train)
    print("[6b] RandomSearch best params:", rand.best_params_)
    return rand.best_estimator_


# ===========================================================================
# 7. FULL EVALUATION  (every common metric; pick what matches your goal)
# ===========================================================================
def evaluate_classification(model, X_test, y_test):
    """accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, MCC,
    confusion matrix, per-class report. Works binary AND multi-class."""
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 precision_score, recall_score, f1_score,
                                 roc_auc_score, matthews_corrcoef,
                                 confusion_matrix, classification_report)
    pred = model.predict(X_test)
    classes = np.unique(y_test)
    avg = "binary" if len(classes) == 2 else "weighted"

    print("  Accuracy          :", round(accuracy_score(y_test, pred), 4))
    print("  Balanced accuracy :", round(balanced_accuracy_score(y_test, pred), 4))
    print("  Precision         :", round(precision_score(y_test, pred, average=avg, zero_division=0), 4))
    print("  Recall            :", round(recall_score(y_test, pred, average=avg, zero_division=0), 4))
    print("  F1 score          :", round(f1_score(y_test, pred, average=avg, zero_division=0), 4))
    print("  MCC               :", round(matthews_corrcoef(y_test, pred), 4))

    if hasattr(model, "predict_proba"):                       # ROC-AUC needs probabilities
        proba = model.predict_proba(X_test)
        if len(classes) == 2:
            auc = roc_auc_score(y_test, proba[:, 1])
        else:
            auc = roc_auc_score(y_test, proba, multi_class="ovr", average="weighted")
        print("  ROC-AUC           :", round(auc, 4))

    print("  Confusion matrix  :\n", confusion_matrix(y_test, pred))
    print("  Per-class report  :\n", classification_report(y_test, pred, zero_division=0))


def evaluate_regression(model, X_test, y_test):
    """MAE, MSE, RMSE, R2, explained variance, MAPE."""
    from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                                 r2_score, explained_variance_score,
                                 mean_absolute_percentage_error)
    pred = model.predict(X_test)
    print("  MAE                :", round(mean_absolute_error(y_test, pred), 4))
    print("  MSE                :", round(mean_squared_error(y_test, pred), 4))
    print("  RMSE               :", round(mean_squared_error(y_test, pred) ** 0.5, 4))
    print("  R2                 :", round(r2_score(y_test, pred), 4))
    print("  Explained variance :", round(explained_variance_score(y_test, pred), 4))
    print("  MAPE               :", round(mean_absolute_percentage_error(y_test, pred), 4))


# ===========================================================================
# 8. PREDICTION on NEW / UNSEEN data
#    The pipeline applies the SAME preprocessing (impute/encode/scale)
#    to new rows automatically -- even if they contain missing values.
# ===========================================================================
def predict_on_new_data(model, X_new):
    """Return X_new with a 'prediction' column (+ 'confidence' if available)."""
    preds = model.predict(X_new)                  # the actual prediction step
    out = X_new.copy()
    out["prediction"] = preds
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_new)
        out["confidence"] = proba.max(axis=1).round(3)   # P of the chosen class
    return out


# ===========================================================================
# 9. SAVE / LOAD a trained model (the WHOLE pipeline, so prep travels with it)
# ===========================================================================
def save_model(model, path):
    joblib.dump(model, path)
    print(f"[9] saved -> {path}")

def load_model(path):
    model = joblib.load(path)
    print(f"[9] loaded <- {path}")
    return model


# ===========================================================================
# 10. *** COMPLETE END-TO-END PIPELINE ***
#     The full lifecycle in one place:
#     data -> split -> pipeline -> fit -> tune -> FULL evaluation
#          -> save -> reload -> predict on a BRAND-NEW dataset.
#     This is the template to copy for a real project.
# ===========================================================================
def complete_pipeline(model_path="/tmp/final_model.joblib"):
    from sklearn.ensemble import RandomForestClassifier

    print("\n========== COMPLETE PIPELINE ==========")

    # --- STEP 1: get data ---------------------------------------------------
    df = make_demo_mixed_data()                  # >>> REPLACE: df = pd.read_csv("your.csv")
    df = df.dropna(subset=["target"])            # never impute the label
    numeric_features     = ["age", "income"]     # >>> REPLACE
    categorical_features = ["city", "plan"]      # >>> REPLACE
    target               = "target"              # >>> REPLACE

    X = df.drop(target, axis=1)
    y = df[target]
    print("STEP 1  data:", X.shape, "| classes:", np.unique(y))

    # --- STEP 2: train / test split (keep test set sacred) ------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    print("STEP 2  split -> train:", X_train.shape, "test:", X_test.shape)

    # --- STEP 3: build the pipeline (preprocess + model) --------------------
    pipe = Pipeline([
        ("prep",  make_preprocessor(numeric_features, categorical_features)),
        ("model", RandomForestClassifier(random_state=RANDOM_STATE)),
    ])
    print("STEP 3  pipeline built")

    # --- STEP 4: tune hyperparameters via CV (on training data only) --------
    param_grid = {
        "model__n_estimators":     [100, 300],
        "model__max_depth":        [None, 5, 10],
        "model__min_samples_leaf": [1, 2],
    }
    search = GridSearchCV(pipe, param_grid, cv=5, scoring="f1", n_jobs=-1)
    search.fit(X_train, y_train)
    best = search.best_estimator_
    print("STEP 4  tuned. best params:", search.best_params_)

    # --- STEP 5: FULL evaluation on the untouched test set ------------------
    print("STEP 5  evaluation on test set:")
    evaluate_classification(best, X_test, y_test)

    # --- STEP 6: save the trained pipeline ----------------------------------
    save_model(best, model_path)

    # --- STEP 7: reload it (simulating a fresh session / another machine) ---
    reloaded = load_model(model_path)

    # --- STEP 8: predict on a BRAND-NEW dataset -----------------------------
    # New incoming rows, SAME columns as training, NO labels.
    # Note: includes missing values (NaN) -> the pipeline handles them.
    new_data = pd.DataFrame({
        "age":    [25, 60, np.nan, 41],
        "income": [42000, 88000, 55000, np.nan],
        "city":   ["NY", "LA", "SF", "NY"],
        "plan":   ["free", "pro", "pro", "free"],
    })
    print("STEP 8  predicting on 4 brand-new rows:")
    result = predict_on_new_data(reloaded, new_data)
    print(result.to_string(index=False))
    print("========== PIPELINE COMPLETE ==========\n")
    return best


# ===========================================================================
# 11. MODEL "SWAP" CHEAT-SHEET  (same pipeline, change the final estimator)
# ===========================================================================
def model_zoo_classification():
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import (RandomForestClassifier,
                                  HistGradientBoostingClassifier)
    from sklearn.naive_bayes import GaussianNB
    return {
        "LogReg":     LogisticRegression(max_iter=5000),
        "KNN":        KNeighborsClassifier(),
        "SVM":        SVC(probability=True),
        "Tree":       DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Forest":     RandomForestClassifier(random_state=RANDOM_STATE),
        "GBoost":     HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        "NaiveBayes": GaussianNB(),
    }


def model_zoo_regression():
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.svm import SVR
    from sklearn.ensemble import (RandomForestRegressor,
                                  HistGradientBoostingRegressor)
    return {
        "Linear":  LinearRegression(), "Ridge": Ridge(), "Lasso": Lasso(),
        "Elastic": ElasticNet(), "SVR": SVR(),
        "Forest":  RandomForestRegressor(random_state=RANDOM_STATE),
        "GBoost":  HistGradientBoostingRegressor(random_state=RANDOM_STATE),
    }


def template_compare_models():
    from sklearn.datasets import load_breast_cancer
    X, y = load_breast_cancer(return_X_y=True)                # >>> REPLACE with your X, y
    print("[11] comparing classifiers (5-fold CV accuracy):")
    for name, est in model_zoo_classification().items():
        pipe = Pipeline([("scaler", StandardScaler()), ("model", est)])
        score = cross_val_score(pipe, X, y, cv=5, scoring="accuracy").mean()
        print(f"     {name:12s}: {score:.4f}")


# ===========================================================================
# RUN EVERYTHING (confirm all templates work)
# ===========================================================================
if __name__ == "__main__":
    print("=" * 60)
    template_quick_classification()
    template_quick_regression()
    print("-" * 60)
    clf, X_tr, X_te, y_tr, y_te = template_full_pipeline()
    template_cross_validation()
    print("-" * 60)
    tuned = template_grid_search(clf, X_tr, y_tr, X_te, y_te)
    print("-" * 60)
    print("[7] FULL evaluation of the tuned pipeline:")
    evaluate_classification(tuned, X_te, y_te)
    print("-" * 60)
    template_compare_models()

    # The capstone: full lifecycle incl. reload + predict on new data
    complete_pipeline()

    print("All templates ran successfully.")
