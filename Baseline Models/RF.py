#RF
import os
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.sparse import issparse
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.ensemble import RandomForestClassifier

# Safe SMOTE via pipeline
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    USE_SMOTE = True
except ImportError:
    print("imblearn not installed — SMOTE disabled.")
    USE_SMOTE = False


# CONFIGURATION
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "smallgraphs"
LABEL_FILE = BASE_DIR / "data" / "metainfo.csv"

N_COMPONENTS = 100
N_JOBS = -1


# ----------------------------
# LOAD GRAPH DATA
# ----------------------------
X, subject_ids = [], []

for fname in os.listdir(DATA_DIR):
    if fname.endswith(".mat"):
        fpath = os.path.join(DATA_DIR, fname)
        mat = sio.loadmat(fpath)

        matrix = None
        for key in ["fibergraph", "fiber", "connectivity"]:
            if key in mat:
                matrix = mat[key]
                break

        if matrix is None:
            continue

        if issparse(matrix):
            matrix = matrix.toarray()

        # symmetrize
        matrix = (matrix + matrix.T) / 2.0

        # upper triangular flatten → vectorized graph features
        vec = matrix[np.triu_indices_from(matrix, k=1)]
        X.append(vec)

        subject_id = os.path.splitext(fname)[0].split("_")[0]
        subject_ids.append(subject_id)

X = np.array(X)
subject_ids = np.array(subject_ids)

print(f"Loaded {len(X)} subjects, feature vector size: {X.shape[1]}")


# ----------------------------
# LOAD LABELS
# ----------------------------
df = pd.read_csv(LABEL_FILE)

df.columns = df.columns.astype(str)
df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace("\ufeff", "", regex=False)

if "URSI" not in df.columns:
    first_col = df.columns[0]
    print(f"Renaming first column {first_col!r} → URSI")
    df = df.rename(columns={first_col: "URSI"})

df["URSI"] = df["URSI"].astype(str).str.strip()
df = df[df["URSI"].isin(subject_ids)]

valid_ids = df["URSI"].values
mask = np.isin(subject_ids, valid_ids)
X = X[mask]

print(f"Matched {len(df)} subjects with metadata")


# ----------------------------
# SEX CLASSIFICATION (SAFE)
# ----------------------------

def run_rf_classification_sex(X, y):
    print("\n=== Random Forest: SEX CLASSIFICATION ===")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=50
    )

    pipeline = ImbPipeline(steps=[
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=min(N_COMPONENTS, X.shape[1]))),
        ("rf", RandomForestClassifier(random_state=50, n_jobs=N_JOBS))
    ])

    param_grid = {
        "rf__n_estimators": [100, 300, 500],
        "rf__max_depth": [None, 10, 20, 40],
        "rf__max_features": ["sqrt", "log2"],
        "rf__min_samples_split": [2, 5],
        "rf__min_samples_leaf": [1, 2]
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=N_JOBS,
    )

    grid.fit(X_train, y_train)

    print("Best RF params:", grid.best_params_)

    y_pred = grid.predict(X_test)

    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    cv_acc = cross_val_score(grid.best_estimator_, X, y, cv=5, scoring="accuracy").mean()
    print(f"Cross-validation Accuracy: {cv_acc:.3f}")


y_sex = df["Sex"].values
run_rf_classification_sex(X, y_sex)


# ----------------------------
# SUBJECT TYPE CLASSIFICATION (SAFE — NO LEAKAGE)
# ----------------------------

def run_rf_subject_type_safe(X, y):
    print("\n=== SAFE Random Forest: SUBJECT TYPE CLASSIFICATION ===")

    # Split FIRST — no transforms before this
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=50
    )

    # Build leakage-free pipeline
    steps = [
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=min(N_COMPONENTS, X.shape[1]))),
    ]

    if USE_SMOTE:
        print("Using SMOTE (safe, train-only)…")
        steps.append(("smote", SMOTE(k_neighbors=3)))

    steps.append(("rf", RandomForestClassifier(random_state=50, n_jobs=N_JOBS)))

    pipeline = ImbPipeline(steps=steps)

    param_grid = {
        "rf__n_estimators": [100, 300, 500],
        "rf__max_depth": [None, 10, 20, 40],
        "rf__max_features": ["sqrt", "log2"],
        "rf__min_samples_split": [2, 5],
        "rf__min_samples_leaf": [1, 2]
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=N_JOBS,
    )

    grid.fit(X_train, y_train)

    print("Best RF params:", grid.best_params_)

    y_pred = grid.predict(X_test)

    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    cv_acc = cross_val_score(grid.best_estimator_, X, y, cv=5, scoring="accuracy").mean()
    print(f"Cross-validation Accuracy: {cv_acc:.3f}")


y_subject = df["Subject_type"].values
run_rf_subject_type_safe(X, y_subject)
