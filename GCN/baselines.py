"""
Baseline models for comparison.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, r2_score
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
import networkx as nx
from typing import Dict, List, Tuple
from tqdm import tqdm


class GraphFeatureExtractor:
    """Extract hand-crafted features from brain graphs."""

    def __init__(self):
        pass

    def extract_features(self, A: np.ndarray) -> np.ndarray:
        """
        Extract graph-level features.

        Features:
        - Node strength statistics (mean, std, max, min)
        - Global efficiency
        - Clustering coefficient (mean, std)
        - Betweenness centrality (mean, std)
        - Assortativity
        - Modularity
        - Small-worldness metrics
        - Top-k edge weights

        Args:
            A: Adjacency matrix (70x70)

        Returns:
            Feature vector
        """
        features = []

        # Node strength features
        node_strength = A.sum(axis=0) + A.sum(axis=1)
        features.extend([
            node_strength.mean(),
            node_strength.std(),
            node_strength.max(),
            node_strength.min(),
            np.percentile(node_strength, 25),
            np.percentile(node_strength, 75)
        ])

        # Create NetworkX graph
        G = nx.from_numpy_array(A, create_using=nx.Graph)

        # Global efficiency
        try:
            global_eff = nx.global_efficiency(G)
            features.append(global_eff)
        except:
            features.append(0)

        # Clustering coefficient
        clustering_coeffs = list(nx.clustering(G).values())
        features.extend([
            np.mean(clustering_coeffs),
            np.std(clustering_coeffs),
            np.max(clustering_coeffs)
        ])

        # Betweenness centrality
        try:
            betweenness = nx.betweenness_centrality(G, k=20)
            betweenness_vals = list(betweenness.values())
            features.extend([
                np.mean(betweenness_vals),
                np.std(betweenness_vals),
                np.max(betweenness_vals)
            ])
        except:
            features.extend([0, 0, 0])

        # Degree statistics
        degrees = [d for n, d in G.degree(weight='weight')]
        features.extend([
            np.mean(degrees),
            np.std(degrees),
            np.max(degrees)
        ])

        # Assortativity
        try:
            assortativity = nx.degree_assortativity_coefficient(G, weight='weight')
            features.append(assortativity)
        except:
            features.append(0)

        # Transitivity (global clustering coefficient)
        try:
            transitivity = nx.transitivity(G)
            features.append(transitivity)
        except:
            features.append(0)

        # Density
        density = nx.density(G)
        features.append(density)

        # Average shortest path length (on largest component)
        try:
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            avg_path_length = nx.average_shortest_path_length(subgraph, weight='weight')
            features.append(avg_path_length)
        except:
            features.append(0)

        # Top-k edge weights (k=10)
        edge_weights = []
        for u, v, w in G.edges(data='weight'):
            if w is not None:
                edge_weights.append(w)

        if len(edge_weights) > 0:
            edge_weights_sorted = sorted(edge_weights, reverse=True)
            top_k = edge_weights_sorted[:10]
            # Pad if necessary
            while len(top_k) < 10:
                top_k.append(0)
            features.extend(top_k)
        else:
            features.extend([0] * 10)

        # Edge weight statistics
        if len(edge_weights) > 0:
            features.extend([
                np.mean(edge_weights),
                np.std(edge_weights),
                np.median(edge_weights),
                np.percentile(edge_weights, 90)
            ])
        else:
            features.extend([0, 0, 0, 0])

        return np.array(features)


class BaselineModels:
    """Baseline models using hand-crafted features."""

    def __init__(self):
        self.feature_extractor = GraphFeatureExtractor()

        # Classification models
        self.sex_models = {
            'logistic': LogisticRegression(max_iter=1000, random_state=42),
            'rf': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': SVC(kernel='rbf', probability=True, random_state=42)
        }

        # Regression models
        self.math_models = {
            'ridge': Ridge(alpha=1.0),
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'svr': SVR(kernel='rbf')
        }

        self.creativity_models = {
            'ridge': Ridge(alpha=1.0),
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'svr': SVR(kernel='rbf')
        }

    def extract_all_features(self,
                            adjacency_matrices: List[np.ndarray]) -> np.ndarray:
        """Extract features for all subjects."""
        features = []
        for A in tqdm(adjacency_matrices, desc='Extracting features'):
            feat = self.feature_extractor.extract_features(A)
            features.append(feat)
        return np.array(features)

    def evaluate_baseline(self,
                         adjacency_matrices: List[np.ndarray],
                         labels: Dict[str, np.ndarray],
                         k_folds: int = 5) -> Dict:
        """
        Evaluate baseline models with k-fold CV.

        Returns:
            Dictionary with results for each model and task
        """
        print("Extracting graph features...")
        X = self.extract_all_features(adjacency_matrices)

        print(f"Feature matrix shape: {X.shape}")

        # Stratified k-fold
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

        results = {
            'sex': {name: [] for name in self.sex_models.keys()},
            'math': {name: [] for name in self.math_models.keys()},
            'creativity': {name: [] for name in self.creativity_models.keys()}
        }

        for fold, (train_val_idx, test_idx) in enumerate(skf.split(X, labels['sex'])):
            print(f"\nFold {fold + 1}/{k_folds}")

            # Split train/val
            train_size = int(0.85 * len(train_val_idx))
            train_idx = train_val_idx[:train_size]
            val_idx = train_val_idx[train_size:]

            # Prepare data
            X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]

            # Standardize features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)

            # Sex classification
            y_sex_train = labels['sex'][train_idx]
            y_sex_test = labels['sex'][test_idx]

            for name, model in self.sex_models.items():
                model.fit(X_train, y_sex_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]

                acc = accuracy_score(y_sex_test, y_pred)
                auroc = roc_auc_score(y_sex_test, y_proba)

                results['sex'][name].append({
                    'accuracy': acc,
                    'auroc': auroc
                })

            # Math regression
            y_math_train = labels['math'][train_idx]
            y_math_test = labels['math'][test_idx]

            # Remove NaN values
            valid_train = ~np.isnan(y_math_train)
            valid_test = ~np.isnan(y_math_test)

            for name, model in self.math_models.items():
                if valid_train.sum() > 0 and valid_test.sum() > 0:
                    model.fit(X_train[valid_train], y_math_train[valid_train])
                    y_pred = model.predict(X_test[valid_test])

                    mae = mean_absolute_error(y_math_test[valid_test], y_pred)
                    r2 = r2_score(y_math_test[valid_test], y_pred)
                    spearman, _ = spearmanr(y_math_test[valid_test], y_pred)

                    results['math'][name].append({
                        'mae': mae,
                        'r2': r2,
                        'spearman': spearman
                    })

            # Creativity regression
            y_creativity_train = labels['creativity'][train_idx]
            y_creativity_test = labels['creativity'][test_idx]

            valid_train = ~np.isnan(y_creativity_train)
            valid_test = ~np.isnan(y_creativity_test)

            for name, model in self.creativity_models.items():
                if valid_train.sum() > 0 and valid_test.sum() > 0:
                    model.fit(X_train[valid_train], y_creativity_train[valid_train])
                    y_pred = model.predict(X_test[valid_test])

                    mae = mean_absolute_error(y_creativity_test[valid_test], y_pred)
                    r2 = r2_score(y_creativity_test[valid_test], y_pred)
                    spearman, _ = spearmanr(y_creativity_test[valid_test], y_pred)

                    results['creativity'][name].append({
                        'mae': mae,
                        'r2': r2,
                        'spearman': spearman
                    })

        # Aggregate results
        aggregated = {}

        for task in ['sex', 'math', 'creativity']:
            aggregated[task] = {}
            for model_name, fold_results in results[task].items():
                aggregated[task][model_name] = {}

                if len(fold_results) > 0:
                    # Get all metric keys
                    metric_keys = fold_results[0].keys()

                    for metric in metric_keys:
                        values = [r[metric] for r in fold_results]
                        aggregated[task][model_name][f'{metric}_mean'] = np.mean(values)
                        aggregated[task][model_name][f'{metric}_std'] = np.std(values)

        # Print results
        print("\n" + "="*70)
        print("BASELINE RESULTS (Mean ± Std)")
        print("="*70)

        print("\nSex Classification:")
        for model_name, metrics in aggregated['sex'].items():
            print(f"  {model_name}:")
            print(f"    Accuracy: {metrics['accuracy_mean']:.4f} ± {metrics['accuracy_std']:.4f}")
            print(f"    AUROC: {metrics['auroc_mean']:.4f} ± {metrics['auroc_std']:.4f}")

        print("\nMath Regression:")
        for model_name, metrics in aggregated['math'].items():
            print(f"  {model_name}:")
            print(f"    MAE: {metrics['mae_mean']:.4f} ± {metrics['mae_std']:.4f}")
            print(f"    R²: {metrics['r2_mean']:.4f} ± {metrics['r2_std']:.4f}")
            print(f"    Spearman: {metrics['spearman_mean']:.4f} ± {metrics['spearman_std']:.4f}")

        print("\nCreativity Regression:")
        for model_name, metrics in aggregated['creativity'].items():
            print(f"  {model_name}:")
            print(f"    MAE: {metrics['mae_mean']:.4f} ± {metrics['mae_std']:.4f}")
            print(f"    R²: {metrics['r2_mean']:.4f} ± {metrics['r2_std']:.4f}")
            print(f"    Spearman: {metrics['spearman_mean']:.4f} ± {metrics['spearman_std']:.4f}")

        return {
            'results': results,
            'aggregated': aggregated
        }


if __name__ == '__main__':
    from data_loader import BrainDataLoader

    # Load data
    loader = BrainDataLoader('../data')
    adj_list, feat_list, labels = loader.load_all_subjects()

    # Run baselines
    baseline = BaselineModels()
    results = baseline.evaluate_baseline(adj_list, labels, k_folds=5)

    # Save results
    import pickle
    with open('results_baselines.pkl', 'wb') as f:
        pickle.dump(results, f)

    print("\nResults saved to results_baselines.pkl")
