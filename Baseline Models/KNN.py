#KNN

import os
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#SMOTE for balancing
try:
    from imblearn.over_sampling import SMOTE
    USE_SMOTE = True
except ImportError:
    print("imblearn not installed, skipping SMOTE balancing.")
    USE_SMOTE = False

# CONFIGURATION
DATA_DIR = r"/Users/ishani/SML Data/brainnetworks/smallgraphs"
LABEL_FILE = r"/Users/ishani/SML Data/brainnetworks/metainfo.csv"
N_COMPONENTS = 100  # PCA components
N_JOBS = -1         

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

        matrix = (matrix + matrix.T) / 2  # symmetrize
        X.append(matrix[np.triu_indices_from(matrix, k=1)])
        subject_ids.append(os.path.splitext(fname)[0].split("_")[0])

X = np.array(X)
print(f"Loaded {len(X)} subjects, feature vector size: {X.shape[1]}")

# LOAD LABELS
df = pd.read_csv(LABEL_FILE)
df["URSI"] = df["URSI"].astype(str).str.strip()
df = df[df["URSI"].isin(subject_ids)]

valid_ids = df["URSI"].values
mask = np.isin(subject_ids, valid_ids)
X = X[mask]
print(f"Matched {len(df)} subjects with metadata")

# SCALE + PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

max_components = min(N_COMPONENTS, min(X_scaled.shape))
pca = PCA(n_components=max_components)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA reduced to {max_components} components "
      f"({np.sum(pca.explained_variance_ratio_)*100:.2f}% variance explained)")


def run_knn_classification(X, y, task_name):
    print(f"\n=== Task: {task_name} ===")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=50
    )

    param_grid = {
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights": ["uniform", "distance"],
        "p": [1, 2]  # 1=Manhattan, 2=Euclidean
    }

    grid = GridSearchCV(
        KNeighborsClassifier(),
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=N_JOBS
    )
    grid.fit(X_train, y_train)
    best_knn = grid.best_estimator_

    print(f"Best KNN params: {grid.best_params_}")
    y_pred = best_knn.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.3f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    cv_acc = cross_val_score(best_knn, X, y, cv=5, scoring="accuracy").mean()
    print(f"Cross-validation Accuracy: {cv_acc:.3f}")

#SEX CLASSIFICATION
y_sex = df["Sex"].values
run_knn_classification(X_pca, y_sex, "Sex (Male=0, Female=1)")

#SUBJECT TYPE CLASSIFICATION
y_subject = df["Subject_type"].values
X_subject = X_pca.copy()


if USE_SMOTE:
    print("\nBalancing Subject_type classes with SMOTE...")
    sm = SMOTE(random_state=50, k_neighbors=3)
    X_subject, y_subject = sm.fit_resample(X_subject, y_subject)
    print("Balanced Subject_type distribution:", np.bincount(y_subject))

run_knn_classification(X_subject, y_subject, "Subject Type (0=Normal, 1=Math, 2=Creative)")
