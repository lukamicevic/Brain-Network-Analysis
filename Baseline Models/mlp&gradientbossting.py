Python 3.11.8 (v3.11.8:db85d51d3e, Feb  6 2024, 18:02:37) [Clang 13.0.0 (clang-1300.0.29.30)] on darwin
Type "help", "copyright", "credits" or "license()" for more information.
"""

This script implements:
  1) An MLP baseline on graph-level features derived from node features
  2) Gradient Boosting baselines on the same features


import argparse
from typing import Dict, List

import numpy as np
from scipy.stats import spearmanr

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    mean_absolute_error,
    r2_score,
)

from data_loader import BrainDataLoader


def build_node_stats_features(feat_list: List[np.ndarray]) -> np.ndarray:
    """
    Build subject-level features by aggregating node features.

    For each subject (graph), we compute:
        - mean over nodes
        - standard deviation over nodes
        - max over nodes

    and concatenate them into a single feature vector.

    Args:
        feat_list: list of [N_nodes, F] arrays

    Returns:
        X: [N_subjects, 3F] numpy array
    """
    features = []
    for X_nodes in feat_list:
        X_nodes = np.asarray(X_nodes)
        assert X_nodes.ndim == 2, "Each entry in feat_list must be 2D [N_nodes, F]"

        mean = X_nodes.mean(axis=0)
        std = X_nodes.std(axis=0)
        max_ = X_nodes.max(axis=0)

        subj_feat = np.concatenate([mean, std, max_], axis=0)
        features.append(subj_feat)

    return np.vstack(features)


def evaluate_baselines(
    X: np.ndarray,
    labels: Dict[str, np.ndarray],
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Run k-fold cross-validation for extra baselines.

    We use the sex label for stratification and evaluate:
      - MLPClassifier and GradientBoostingClassifier for sex classification
      - MLPRegressor and GradientBoostingRegressor for math & creativity regression

    Metrics:
      - sex: accuracy, ROC-AUC
      - math/creativity: MAE, R^2, Spearman rho

    Args:
        X: [N_subjects, D] feature matrix
        labels: dict with keys 'sex', 'math', 'creativity'
        n_splits: number of CV folds
        random_state: random seed

    Returns:
        Dictionary of aggregated metrics for each model/task.
    """
    y_sex = labels["sex"].astype(int)
    y_math = labels["math"].astype(float)
    y_creativity = labels["creativity"].astype(float)

    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    results: Dict[str, Dict[str, List[float]]] = {
        "mlp_sex": {"acc": [], "roc_auc": []},
        "gb_sex": {"acc": [], "roc_auc": []},
        "mlp_math": {"mae": [], "r2": [], "rho": []},
        "gb_math": {"mae": [], "r2": [], "rho": []},
        "mlp_creativity": {"mae": [], "r2": [], "rho": []},
        "gb_creativity": {"mae": [], "r2": [], "rho": []},
    }

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_sex), start=1):
        print(f"\n=== Fold {fold}/{n_splits} ===")

        X_train, X_test = X[train_idx], X[test_idx]
        y_sex_train, y_sex_test = y_sex[train_idx], y_sex[test_idx]
        y_math_train, y_math_test = y_math[train_idx], y_math[test_idx]
        y_cre_train, y_cre_test = y_creativity[train_idx], y_creativity[test_idx]

        # Standardize features for MLP
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # -------------------------
        # 1) Sex classification
        # -------------------------
        # MLP classifier
        mlp_clf = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            max_iter=1000,
            random_state=random_state,
        )
        mlp_clf.fit(X_train_scaled, y_sex_train)
        sex_probs_mlp = mlp_clf.predict_proba(X_test_scaled)[:, 1]
        sex_pred_mlp = (sex_probs_mlp >= 0.5).astype(int)

        acc_mlp = accuracy_score(y_sex_test, sex_pred_mlp)
        roc_mlp = roc_auc_score(y_sex_test, sex_probs_mlp)
        results["mlp_sex"]["acc"].append(acc_mlp)
        results["mlp_sex"]["roc_auc"].append(roc_mlp)

        print(f"MLP sex - acc: {acc_mlp:.3f}, roc_auc: {roc_mlp:.3f}")

        gb_clf = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
        )
        gb_clf.fit(X_train, y_sex_train)
        sex_probs_gb = gb_clf.predict_proba(X_test)[:, 1]
        sex_pred_gb = (sex_probs_gb >= 0.5).astype(int)

        acc_gb = accuracy_score(y_sex_test, sex_pred_gb)
        roc_gb = roc_auc_score(y_sex_test, sex_probs_gb)
        results["gb_sex"]["acc"].append(acc_gb)
        results["gb_sex"]["roc_auc"].append(roc_gb)

        print(f"GB sex  - acc: {acc_gb:.3f}, roc_auc: {roc_gb:.3f}")

        mlp_reg_math = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            max_iter=1000,
            random_state=random_state,
        )
        mlp_reg_math.fit(X_train_scaled, y_math_train)
        math_pred_mlp = mlp_reg_math.predict(X_test_scaled)

        mae_mlp_math = mean_absolute_error(y_math_test, math_pred_mlp)
        r2_mlp_math = r2_score(y_math_test, math_pred_mlp)
        rho_mlp_math, _ = spearmanr(y_math_test, math_pred_mlp)

        results["mlp_math"]["mae"].append(mae_mlp_math)
        results["mlp_math"]["r2"].append(r2_mlp_math)
        results["mlp_math"]["rho"].append(rho_mlp_math)

        print(
            f"MLP math - MAE: {mae_mlp_math:.3f}, R2: {r2_mlp_math:.3f}, "
            f"Spearman rho: {rho_mlp_math:.3f}"
        )

        # Gradient Boosting regressor
        gb_reg_math = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
        )
        gb_reg_math.fit(X_train, y_math_train)
        math_pred_gb = gb_reg_math.predict(X_test)

        mae_gb_math = mean_absolute_error(y_math_test, math_pred_gb)
        r2_gb_math = r2_score(y_math_test, math_pred_gb)
        rho_gb_math, _ = spearmanr(y_math_test, math_pred_gb)

        results["gb_math"]["mae"].append(mae_gb_math)
        results["gb_math"]["r2"].append(r2_gb_math)
        results["gb_math"]["rho"].append(rho_gb_math)

        print(
            f"GB math  - MAE: {mae_gb_math:.3f}, R2: {r2_gb_math:.3f}, "
            f"Spearman rho: {rho_gb_math:.3f}"
        )

        mlp_reg_cre = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            max_iter=1000,
            random_state=random_state,
        )
        mlp_reg_cre.fit(X_train_scaled, y_cre_train)
        cre_pred_mlp = mlp_reg_cre.predict(X_test_scaled)

        mae_mlp_cre = mean_absolute_error(y_cre_test, cre_pred_mlp)
        r2_mlp_cre = r2_score(y_cre_test, cre_pred_mlp)
        rho_mlp_cre, _ = spearmanr(y_cre_test, cre_pred_mlp)

        results["mlp_creativity"]["mae"].append(mae_mlp_cre)
        results["mlp_creativity"]["r2"].append(r2_mlp_cre)
        results["mlp_creativity"]["rho"].append(rho_mlp_cre)

        print(
            f"MLP creativity - MAE: {mae_mlp_cre:.3f}, R2: {r2_mlp_cre:.3f}, "
            f"Spearman rho: {rho_mlp_cre:.3f}"
        )

        # Gradient Boosting regressor
        gb_reg_cre = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
        )
        gb_reg_cre.fit(X_train, y_cre_train)
        cre_pred_gb = gb_reg_cre.predict(X_test)

        mae_gb_cre = mean_absolute_error(y_cre_test, cre_pred_gb)
        r2_gb_cre = r2_score(y_cre_test, cre_pred_gb)
        rho_gb_cre, _ = spearmanr(y_cre_test, cre_pred_gb)

        results["gb_creativity"]["mae"].append(mae_gb_cre)
        results["gb_creativity"]["r2"].append(r2_gb_cre)
        results["gb_creativity"]["rho"].append(rho_gb_cre)

        print(
            f"GB creativity  - MAE: {mae_gb_cre:.3f}, R2: {r2_gb_cre:.3f}, "
            f"Spearman rho: {rho_gb_cre:.3f}"
        )

    aggregated: Dict[str, Dict[str, float]] = {}
    for model_name, metrics in results.items():
        aggregated[model_name] = {
            metric_name: float(np.mean(values))
            for metric_name, values in metrics.items()
        }

    return aggregated


def main():
    parser = argparse.ArgumentParser(
        description="Extra baseline models (MLP + Gradient Boosting) for brain connectivity project"
...     )
...     parser.add_argument(
...         "--data_dir",
...         type=str,
...         default="../data",
...         help="Path to the data directory (where .mat / metadata live)",
...     )
...     parser.add_argument(
...         "--n_splits",
...         type=int,
...         default=5,
...         help="Number of folds for StratifiedKFold CV",
...     )
...     args = parser.parse_args()
... 
...     # Load data
...     loader = BrainDataLoader(args.data_dir)
...     adj_list, feat_list, labels = loader.load_all_subjects()
... 
...     print(
...         f"Loaded {len(adj_list)} subjects, "
...         f"graph size: {adj_list[0].shape}, "
...         f"node features: {feat_list[0].shape}"
...     )
... 
...     X = build_node_stats_features(feat_list)
...     print(f"Subject-level feature matrix shape: {X.shape}")
... 
... 
...     aggregated_results = evaluate_baselines(
...         X, labels, n_splits=args.n_splits
...     )
... 
...     print("\n=== Aggregated results over folds ===")
...     for model_name, metrics in aggregated_results.items():
...         print(f"\n{model_name}:")
...         for metric_name, value in metrics.items():
...             print(f"  {metric_name}: {value:.3f}")
... 
... 
if __name__ == "__main__":
    main()
