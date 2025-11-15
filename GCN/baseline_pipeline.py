"""
Clean baseline ML pipeline for structural brain connectome analysis.
Built from scratch with comprehensive graph biomarkers and metadata fusion.
"""
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, r2_score
from scipy.stats import spearmanr
from scipy import sparse
from scipy.linalg import eigh

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("WARNING: XGBoost not available. Install with: pip install xgboost")

from tqdm import tqdm


@dataclass
class SubjectData:
    """Container for single subject's data."""
    adjacency: np.ndarray
    metadata: Dict[str, float]
    ursi: str


class ConnectomePreprocessor:
    """Preprocess structural connectivity matrices."""
    
    def __init__(self):
        pass
    
    def preprocess(self, A: np.ndarray) -> np.ndarray:
        """
        Preprocess adjacency matrix.
        
        Steps:
        1. Enforce symmetry: (A + A^T) / 2
        2. Zero diagonal
        3. Apply log1p transform
        4. Normalize by total weight
        
        Args:
            A: Raw 70×70 connectivity matrix
            
        Returns:
            Preprocessed matrix
        """
        # Enforce symmetry
        A_sym = (A + A.T) / 2.0
        
        # Zero diagonal
        np.fill_diagonal(A_sym, 0)
        
        # Apply log1p
        A_log = np.log1p(A_sym)
        
        # Normalize by total weight
        total_weight = A_log.sum()
        if total_weight > 0:
            A_norm = A_log / total_weight
        else:
            A_norm = A_log
            
        return A_norm


class GraphBiomarkerExtractor:
    """Extract comprehensive graph biomarkers from brain connectome."""
    
    def __init__(self):
        self.feature_names = []
        
    def extract_features(self, A: np.ndarray) -> np.ndarray:
        """
        Extract comprehensive graph biomarkers.
        
        Features:
        - Node strength statistics (mean, std, min, max, median, p25, p75)
        - Clustering statistics (mean, std, min, max)
        - Degree statistics (mean, std, min, max)
        - Full betweenness centrality (no sampling)
        - Global efficiency
        - Transitivity
        - Density
        - Average shortest path (largest component)
        - Modularity via Louvain
        - Small-worldness (γ, λ, σ)
        - Spectral features (5 smallest Laplacian eigenvalues, spectral radius, algebraic connectivity)
        - Edge-weight statistics (mean, std, median, p90, top-10)
        
        Args:
            A: Preprocessed 70×70 adjacency matrix
            
        Returns:
            Feature vector
        """
        features = []
        self.feature_names = []
        
        # Create NetworkX graph
        G = nx.from_numpy_array(A, create_using=nx.Graph)
        
        # === NODE STRENGTH STATISTICS ===
        node_strengths = np.array([A[i, :].sum() for i in range(A.shape[0])])
        features.extend([
            np.mean(node_strengths),
            np.std(node_strengths),
            np.min(node_strengths),
            np.max(node_strengths),
            np.median(node_strengths),
            np.percentile(node_strengths, 25),
            np.percentile(node_strengths, 75)
        ])
        self.feature_names.extend([
            'strength_mean', 'strength_std', 'strength_min', 'strength_max',
            'strength_median', 'strength_p25', 'strength_p75'
        ])
        
        # === CLUSTERING STATISTICS ===
        clustering_coeffs = np.array(list(nx.clustering(G, weight='weight').values()))
        features.extend([
            np.mean(clustering_coeffs),
            np.std(clustering_coeffs),
            np.min(clustering_coeffs),
            np.max(clustering_coeffs)
        ])
        self.feature_names.extend([
            'clustering_mean', 'clustering_std', 'clustering_min', 'clustering_max'
        ])
        
        # === DEGREE STATISTICS ===
        degrees = np.array([d for _, d in G.degree(weight='weight')])
        features.extend([
            np.mean(degrees),
            np.std(degrees),
            np.min(degrees),
            np.max(degrees)
        ])
        self.feature_names.extend([
            'degree_mean', 'degree_std', 'degree_min', 'degree_max'
        ])
        
        # === BETWEENNESS CENTRALITY (FULL, NO SAMPLING) ===
        try:
            betweenness = nx.betweenness_centrality(G, weight='weight')
            betweenness_vals = np.array(list(betweenness.values()))
            features.extend([
                np.mean(betweenness_vals),
                np.std(betweenness_vals),
                np.max(betweenness_vals)
            ])
            self.feature_names.extend(['betweenness_mean', 'betweenness_std', 'betweenness_max'])
        except:
            features.extend([0, 0, 0])
            self.feature_names.extend(['betweenness_mean', 'betweenness_std', 'betweenness_max'])
        
        # === GLOBAL EFFICIENCY ===
        try:
            global_eff = nx.global_efficiency(G)
            features.append(global_eff)
        except:
            features.append(0)
        self.feature_names.append('global_efficiency')
        
        # === TRANSITIVITY ===
        try:
            transitivity = nx.transitivity(G)
            features.append(transitivity)
        except:
            features.append(0)
        self.feature_names.append('transitivity')
        
        # === DENSITY ===
        density = nx.density(G)
        features.append(density)
        self.feature_names.append('density')
        
        # === AVERAGE SHORTEST PATH (LARGEST COMPONENT) ===
        try:
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            avg_path = nx.average_shortest_path_length(subgraph, weight='weight')
            features.append(avg_path)
        except:
            features.append(0)
        self.feature_names.append('avg_shortest_path')
        
        # === MODULARITY VIA LOUVAIN ===
        try:
            from networkx.algorithms import community
            communities = community.greedy_modularity_communities(G, weight='weight')
            modularity = community.modularity(G, communities, weight='weight')
            features.append(modularity)
        except:
            features.append(0)
        self.feature_names.append('modularity')
        
        # === SMALL-WORLDNESS (γ, λ, σ) ===
        try:
            # Real network metrics
            C_real = nx.average_clustering(G, weight='weight')
            L_real = avg_path if avg_path > 0 else nx.average_shortest_path_length(subgraph, weight='weight')
            
            # Random network approximation
            n = G.number_of_nodes()
            m = G.number_of_edges()
            p = 2 * m / (n * (n - 1))
            C_rand = p
            L_rand = np.log(n) / np.log(n * p) if n * p > 1 else 1
            
            gamma = C_real / C_rand if C_rand > 0 else 0
            lambda_val = L_real / L_rand if L_rand > 0 else 0
            sigma = gamma / lambda_val if lambda_val > 0 else 0
            
            features.extend([gamma, lambda_val, sigma])
        except:
            features.extend([0, 0, 0])
        self.feature_names.extend(['small_world_gamma', 'small_world_lambda', 'small_world_sigma'])
        
        # === SPECTRAL FEATURES ===
        try:
            # Compute normalized Laplacian
            L = nx.normalized_laplacian_matrix(G).toarray()
            eigenvalues = np.linalg.eigvalsh(L)
            eigenvalues = np.sort(eigenvalues)
            
            # Smallest 5 eigenvalues
            smallest_5 = eigenvalues[:5]
            features.extend(smallest_5.tolist())
            self.feature_names.extend([f'laplacian_eig_{i+1}' for i in range(5)])
            
            # Spectral radius (largest eigenvalue)
            spectral_radius = eigenvalues[-1]
            features.append(spectral_radius)
            self.feature_names.append('spectral_radius')
            
            # Algebraic connectivity (second smallest eigenvalue)
            algebraic_connectivity = eigenvalues[1] if len(eigenvalues) > 1 else 0
            features.append(algebraic_connectivity)
            self.feature_names.append('algebraic_connectivity')
        except:
            features.extend([0] * 7)
            self.feature_names.extend([f'laplacian_eig_{i+1}' for i in range(5)] + 
                                     ['spectral_radius', 'algebraic_connectivity'])
        
        # === EDGE-WEIGHT STATISTICS ===
        edge_weights = [w for _, _, w in G.edges(data='weight') if w is not None]
        if len(edge_weights) > 0:
            edge_weights = np.array(edge_weights)
            features.extend([
                np.mean(edge_weights),
                np.std(edge_weights),
                np.median(edge_weights),
                np.percentile(edge_weights, 90)
            ])
            
            # Top-10 edge weights
            top_10 = sorted(edge_weights, reverse=True)[:10]
            while len(top_10) < 10:
                top_10.append(0)
            features.extend(top_10)
        else:
            features.extend([0] * 14)
        
        self.feature_names.extend([
            'edge_mean', 'edge_std', 'edge_median', 'edge_p90'
        ] + [f'top_edge_{i+1}' for i in range(10)])
        
        return np.array(features)
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return self.feature_names


class MetadataFusion:
    """Fuse graph features with subject metadata."""
    
    METADATA_FIELDS = ['CCI', 'Sex', 'Age', 'Neuroticm', 'Extraversion', 
                       'Openness', 'Agreeableness', 'Conscientiousness']
    
    def __init__(self):
        pass
    
    def fuse_features(self, graph_features: np.ndarray, 
                     metadata: Dict[str, float]) -> np.ndarray:
        """
        Combine graph features with metadata into single vector.
        
        Args:
            graph_features: Graph biomarker vector
            metadata: Subject metadata dictionary
            
        Returns:
            Fused feature vector
        """
        metadata_vec = []
        for field in self.METADATA_FIELDS:
            value = metadata.get(field, np.nan)
            # Handle missing values
            if pd.isna(value):
                value = 0.0  # Impute with zero (will be standardized later)
            metadata_vec.append(value)
        
        return np.concatenate([graph_features, np.array(metadata_vec)])
    
    def get_metadata_names(self) -> List[str]:
        """Return metadata field names."""
        return self.METADATA_FIELDS


class BaselineMLPipeline:
    """Complete baseline ML pipeline for brain connectome analysis."""
    
    def __init__(self, k_folds: int = 5, random_state: int = 42):
        """
        Initialize pipeline.
        
        Args:
            k_folds: Number of CV folds
            random_state: Random seed
        """
        self.k_folds = k_folds
        self.random_state = random_state
        
        self.preprocessor = ConnectomePreprocessor()
        self.biomarker_extractor = GraphBiomarkerExtractor()
        self.metadata_fusion = MetadataFusion()
        
    def load_data(self, data_dir: str) -> Tuple[List[SubjectData], Dict[str, np.ndarray]]:
        """
        Load brain connectivity data and metadata.
        
        Args:
            data_dir: Path to data directory
            
        Returns:
            Tuple of (subject_data_list, labels_dict)
        """
        from data_loader import BrainDataLoader
        
        loader = BrainDataLoader(data_dir)
        adj_list, _, labels = loader.load_all_subjects(
            symmetrize=False,  # We'll do our own preprocessing
            add_self_loops=False
        )
        
        # Load metadata
        metadata_df = loader.metadata
        
        subjects = []
        for idx, ursi in enumerate(labels['ursi']):
            # Get metadata for this subject
            meta_row = metadata_df[metadata_df['URSI'] == ursi].iloc[0]
            metadata = {
                'CCI': meta_row.get('CCI', np.nan),
                'Sex': meta_row.get('Sex', np.nan),
                'Age': meta_row.get('Age', np.nan),
                'Neuroticm': meta_row.get('Neuroticm', np.nan),
                'Extraversion': meta_row.get('Extraversion', np.nan),
                'Openness': meta_row.get('Openness', np.nan),
                'Agreeableness': meta_row.get('Agreeableness', np.nan),
                'Conscientiousness': meta_row.get('Conscientiousness', np.nan)
            }
            
            subjects.append(SubjectData(
                adjacency=adj_list[idx],
                metadata=metadata,
                ursi=ursi
            ))
        
        return subjects, labels
    
    def extract_all_features(self, subjects: List[SubjectData]) -> np.ndarray:
        """
        Extract and fuse features for all subjects.
        
        Args:
            subjects: List of SubjectData
            
        Returns:
            Feature matrix (n_subjects × n_features)
        """
        print("Extracting features for all subjects...")
        all_features = []
        
        for subject in tqdm(subjects, desc='Feature extraction'):
            # Preprocess adjacency
            A_processed = self.preprocessor.preprocess(subject.adjacency)
            
            # Extract graph biomarkers
            graph_features = self.biomarker_extractor.extract_features(A_processed)
            
            # Fuse with metadata
            fused_features = self.metadata_fusion.fuse_features(
                graph_features, subject.metadata
            )
            
            all_features.append(fused_features)
        
        feature_matrix = np.array(all_features)
        print(f"Feature matrix shape: {feature_matrix.shape}")
        
        return feature_matrix
    
    def evaluate_models(self, 
                       X: np.ndarray,
                       labels: Dict[str, np.ndarray]) -> Dict:
        """
        Evaluate models with stratified k-fold CV.
        
        Args:
            X: Feature matrix
            labels: Dictionary of labels (sex, math, creativity)
            
        Returns:
            Results dictionary
        """
        skf = StratifiedKFold(n_splits=self.k_folds, shuffle=True, 
                             random_state=self.random_state)
        
        results = {
            'sex': {'svm': [], 'xgb': []},
            'math': {'svr': [], 'xgb': []},
            'creativity': {'svr': [], 'xgb': []}
        }
        
        y_sex = labels['sex']
        y_math = labels['math']
        y_creativity = labels['creativity']
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_sex)):
            print(f"\n{'='*60}")
            print(f"Fold {fold + 1}/{self.k_folds}")
            print(f"{'='*60}")
            
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            
            # Standardize features AFTER split
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # === SEX CLASSIFICATION ===
            print("Training sex classification models...")
            y_sex_train, y_sex_test = y_sex[train_idx], y_sex[test_idx]
            
            # Tuned RBF SVM
            svm_clf = SVC(kernel='rbf', C=10, gamma='scale', 
                         probability=True, random_state=self.random_state)
            svm_clf.fit(X_train, y_sex_train)
            y_pred = svm_clf.predict(X_test)
            y_proba = svm_clf.predict_proba(X_test)[:, 1]
            
            results['sex']['svm'].append({
                'accuracy': accuracy_score(y_sex_test, y_pred),
                'auroc': roc_auc_score(y_sex_test, y_proba)
            })
            
            # XGBoost
            if XGBOOST_AVAILABLE:
                xgb_clf = XGBClassifier(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=self.random_state,
                    eval_metric='logloss'
                )
                xgb_clf.fit(X_train, y_sex_train)
                y_pred = xgb_clf.predict(X_test)
                y_proba = xgb_clf.predict_proba(X_test)[:, 1]
                
                results['sex']['xgb'].append({
                    'accuracy': accuracy_score(y_sex_test, y_pred),
                    'auroc': roc_auc_score(y_sex_test, y_proba)
                })
            
            # === MATH REGRESSION ===
            print("Training math regression models...")
            y_math_train, y_math_test = y_math[train_idx], y_math[test_idx]
            
            # Handle NaN values
            valid_train = ~np.isnan(y_math_train)
            valid_test = ~np.isnan(y_math_test)
            
            if valid_train.sum() > 0 and valid_test.sum() > 0:
                # Tuned RBF SVR
                svr_math = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
                svr_math.fit(X_train[valid_train], y_math_train[valid_train])
                y_pred = svr_math.predict(X_test[valid_test])
                y_true = y_math_test[valid_test]
                
                results['math']['svr'].append({
                    'mae': mean_absolute_error(y_true, y_pred),
                    'r2': r2_score(y_true, y_pred),
                    'spearman': spearmanr(y_true, y_pred)[0]
                })
                
                # XGBoost
                if XGBOOST_AVAILABLE:
                    xgb_reg = XGBRegressor(
                        n_estimators=100,
                        max_depth=3,
                        learning_rate=0.1,
                        random_state=self.random_state
                    )
                    xgb_reg.fit(X_train[valid_train], y_math_train[valid_train])
                    y_pred = xgb_reg.predict(X_test[valid_test])
                    
                    results['math']['xgb'].append({
                        'mae': mean_absolute_error(y_true, y_pred),
                        'r2': r2_score(y_true, y_pred),
                        'spearman': spearmanr(y_true, y_pred)[0]
                    })
            
            # === CREATIVITY REGRESSION ===
            print("Training creativity regression models...")
            y_creativity_train = y_creativity[train_idx]
            y_creativity_test = y_creativity[test_idx]
            
            valid_train = ~np.isnan(y_creativity_train)
            valid_test = ~np.isnan(y_creativity_test)
            
            if valid_train.sum() > 0 and valid_test.sum() > 0:
                # Tuned RBF SVR
                svr_creativity = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
                svr_creativity.fit(X_train[valid_train], y_creativity_train[valid_train])
                y_pred = svr_creativity.predict(X_test[valid_test])
                y_true = y_creativity_test[valid_test]
                
                results['creativity']['svr'].append({
                    'mae': mean_absolute_error(y_true, y_pred),
                    'r2': r2_score(y_true, y_pred),
                    'spearman': spearmanr(y_true, y_pred)[0]
                })
                
                # XGBoost
                if XGBOOST_AVAILABLE:
                    xgb_reg = XGBRegressor(
                        n_estimators=100,
                        max_depth=3,
                        learning_rate=0.1,
                        random_state=self.random_state
                    )
                    xgb_reg.fit(X_train[valid_train], y_creativity_train[valid_train])
                    y_pred = xgb_reg.predict(X_test[valid_test])
                    
                    results['creativity']['xgb'].append({
                        'mae': mean_absolute_error(y_true, y_pred),
                        'r2': r2_score(y_true, y_pred),
                        'spearman': spearmanr(y_true, y_pred)[0]
                    })
        
        # Aggregate results
        aggregated = self._aggregate_results(results)
        self._print_results(aggregated)
        
        return {
            'fold_results': results,
            'aggregated': aggregated
        }
    
    def _aggregate_results(self, results: Dict) -> Dict:
        """Aggregate results across folds."""
        aggregated = {}
        
        for task in ['sex', 'math', 'creativity']:
            aggregated[task] = {}
            for model_name, fold_results in results[task].items():
                if len(fold_results) == 0:
                    continue
                    
                aggregated[task][model_name] = {}
                metric_keys = fold_results[0].keys()
                
                for metric in metric_keys:
                    values = [r[metric] for r in fold_results]
                    aggregated[task][model_name][f'{metric}_mean'] = np.mean(values)
                    aggregated[task][model_name][f'{metric}_std'] = np.std(values)
        
        return aggregated
    
    def _print_results(self, aggregated: Dict):
        """Print aggregated results."""
        print("\n" + "="*70)
        print("BASELINE ML PIPELINE RESULTS (Mean ± Std)")
        print("="*70)
        
        print("\n📊 SEX CLASSIFICATION:")
        for model_name, metrics in aggregated['sex'].items():
            if len(metrics) == 0:
                continue
            print(f"  {model_name.upper()}:")
            print(f"    Accuracy: {metrics['accuracy_mean']:.4f} ± {metrics['accuracy_std']:.4f}")
            print(f"    AUROC:    {metrics['auroc_mean']:.4f} ± {metrics['auroc_std']:.4f}")
        
        print("\n📈 MATH REGRESSION (FSIQ):")
        for model_name, metrics in aggregated['math'].items():
            if len(metrics) == 0:
                continue
            print(f"  {model_name.upper()}:")
            print(f"    MAE:      {metrics['mae_mean']:.4f} ± {metrics['mae_std']:.4f}")
            print(f"    R²:       {metrics['r2_mean']:.4f} ± {metrics['r2_std']:.4f}")
            print(f"    Spearman: {metrics['spearman_mean']:.4f} ± {metrics['spearman_std']:.4f}")
        
        print("\n🎨 CREATIVITY REGRESSION (CAQ):")
        for model_name, metrics in aggregated['creativity'].items():
            if len(metrics) == 0:
                continue
            print(f"  {model_name.upper()}:")
            print(f"    MAE:      {metrics['mae_mean']:.4f} ± {metrics['mae_std']:.4f}")
            print(f"    R²:       {metrics['r2_mean']:.4f} ± {metrics['r2_std']:.4f}")
            print(f"    Spearman: {metrics['spearman_mean']:.4f} ± {metrics['spearman_std']:.4f}")
    
    def run(self, data_dir: str) -> Dict:
        """
        Run complete baseline pipeline.
        
        Args:
            data_dir: Path to data directory
            
        Returns:
            Results dictionary
        """
        print("="*70)
        print("BASELINE ML PIPELINE FOR BRAIN CONNECTOMES")
        print("="*70)
        print("Preprocessing: Symmetrize → Zero diagonal → Log1p → Normalize")
        print("Features: Graph biomarkers + Metadata fusion")
        print("Models: Tuned RBF SVM + XGBoost")
        print(f"CV: {self.k_folds}-fold stratified")
        print("="*70)
        
        # Load data
        subjects, labels = self.load_data(data_dir)
        
        # Extract features
        X = self.extract_all_features(subjects)
        
        # Evaluate models
        results = self.evaluate_models(X, labels)
        
        return results


def main():
    """Run baseline ML pipeline."""
    pipeline = BaselineMLPipeline(k_folds=5, random_state=42)
    results = pipeline.run('../data')
    
    # Save results
    import pickle
    output_file = 'baseline_ml_results.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n✅ Results saved to {output_file}")
    

if __name__ == '__main__':
    main()
