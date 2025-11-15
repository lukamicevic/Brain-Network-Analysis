#SVM

import os
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

# CONFIGURATION
BASE_DIR = Path(__file__).resolve().parent.parent  
DATA_DIR = BASE_DIR / "data" / "smallgraphs"
LABEL_FILE = BASE_DIR / "data" / "metainfo.csv"

X = []
subject_ids = []

# FIX 
files = [f for f in os.listdir(DATA_DIR) if f.endswith(".mat")]
files = sorted(files, key=lambda x: x.split("_")[0])   # sort by URSI ID


for fname in files:
    if fname.endswith('.mat'):
        fpath = os.path.join(DATA_DIR, fname)
        mat = sio.loadmat(fpath)
        
        
        if 'fibergraph' in mat:
            matrix = mat['fibergraph']
        
        else:
            print(f"Skipping {fname}, no valid key found.")
            continue
        
        # Convert sparse to dense
        if issparse(matrix):
            matrix = matrix.toarray()
        
        
        if matrix.shape[0] != matrix.shape[1]:
            print(f"Skipping {fname}, not a square matrix: {matrix.shape}")
            continue
        
        # Flatten upper triangle
        upper_tri = matrix[np.triu_indices_from(matrix, k=1)]
        X.append(upper_tri)
        subject_ids.append(os.path.splitext(fname)[0].split('_')[0]) 

X = np.array(X)
print(f"Loaded {len(X)} subjects, feature vector size: {X.shape[1]}")

# LOAD LABELS
labels_df = pd.read_csv(LABEL_FILE)
labels_df['URSI'] = labels_df['URSI'].astype(str).str.strip()


labels_df = labels_df[labels_df['URSI'].isin(subject_ids)]
print(f"Matched {len(labels_df)} subjects with metadata")


y_gender = np.array(labels_df['Sex'])
y_math = np.array(labels_df['Subject_type'])

tasks = {
    'Gender (Male=0, Female=1)': y_gender,
    #'Math capability (Normal=0, High=1)': y_math
}
print("Math class distribution:", np.bincount(labels_df['Subject_type']))


def train_and_evaluate(X, y, task_name):
    print(f" Task: {task_name}")
    

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=50, stratify=y)

    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train SVM
    svm = SVC(kernel='rbf', C=1.0, gamma='scale')
    svm.fit(X_train, y_train)

 
    y_pred = svm.predict(X_test)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Cross-validation
    scores = cross_val_score(svm, X, y, cv=5)
    print(f"Cross-Validation Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")


for name, y in tasks.items():
    train_and_evaluate(X, y, name)
