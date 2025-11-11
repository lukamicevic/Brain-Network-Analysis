"""
Data loading and preprocessing for brain connectivity graphs.
"""
import os
import numpy as np
import pandas as pd
import scipy.io as sio
from pathlib import Path
from typing import Tuple, Dict, List
import networkx as nx


class BrainDataLoader:
    """Load and preprocess brain connectivity data."""

    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: Path to data directory containing metainfo.csv and smallgraphs/
        """
        self.data_dir = Path(data_dir)
        self.metainfo_path = self.data_dir / 'metainfo.csv'
        self.graphs_dir = self.data_dir / 'smallgraphs'

        # Load metadata
        self.metadata = pd.read_csv(self.metainfo_path)

        # Filter to only subjects with available graph files
        available_files = set([f.stem.replace('_fiber', '') for f in self.graphs_dir.glob('*.mat')])
        self.metadata = self.metadata[self.metadata['URSI'].isin(available_files)].reset_index(drop=True)
        print(f"Loaded metadata for {len(self.metadata)} subjects (with graph files)")

    def load_subject_graph(self, ursi: str) -> np.ndarray:
        """
        Load connectivity matrix for a single subject.

        Args:
            ursi: Subject ID (e.g., 'M87102217')

        Returns:
            70x70 connectivity matrix as dense numpy array
        """
        mat_file = self.graphs_dir / f'{ursi}_fiber.mat'
        data = sio.loadmat(str(mat_file))
        # Convert sparse to dense
        adj = data['fibergraph'].toarray()
        return adj

    def preprocess_adjacency(self, A: np.ndarray,
                            symmetrize: bool = True,
                            add_self_loops: bool = True,
                            threshold: float = 0.0,
                            log_scale: bool = False) -> np.ndarray:
        """
        Preprocess adjacency matrix.

        Args:
            A: Original adjacency matrix
            symmetrize: Make undirected by averaging A and A^T
            add_self_loops: Add identity matrix
            threshold: Remove edges with weight < threshold
            log_scale: Apply log(1 + x) transform to weights

        Returns:
            Preprocessed adjacency matrix
        """
        A_proc = A.copy()

        # Symmetrize (undirected graph)
        if symmetrize:
            A_proc = (A_proc + A_proc.T) / 2

        # Threshold weak connections
        if threshold > 0:
            A_proc[A_proc < threshold] = 0

        # Log-scale to tame heavy tails
        if log_scale:
            A_proc = np.log1p(A_proc)

        # Add self-loops
        if add_self_loops:
            A_proc = A_proc + np.eye(A_proc.shape[0])

        return A_proc

    def compute_node_features(self, A: np.ndarray) -> np.ndarray:
        """
        Compute node features from graph topology.

        Features per node:
        - In-strength (sum of incoming edges)
        - Out-strength (sum of outgoing edges)
        - Total strength
        - Clustering coefficient
        - Betweenness centrality (sampled for speed)
        - Eigenvector centrality
        - One-hot position encoding

        Args:
            A: Adjacency matrix (70x70)

        Returns:
            Node feature matrix (70 x num_features)
        """
        n_nodes = A.shape[0]
        features = []

        # Strength features
        in_strength = A.sum(axis=0)  # incoming
        out_strength = A.sum(axis=1)  # outgoing
        total_strength = in_strength + out_strength

        # Z-score normalize strengths
        features.append(self._zscore(in_strength).reshape(-1, 1))
        features.append(self._zscore(out_strength).reshape(-1, 1))
        features.append(self._zscore(total_strength).reshape(-1, 1))

        # Graph-based features using NetworkX
        G = nx.from_numpy_array(A, create_using=nx.DiGraph)

        # Clustering coefficient
        clustering = np.array([nx.clustering(G.to_undirected(), node)
                              for node in range(n_nodes)])
        features.append(clustering.reshape(-1, 1))

        # Betweenness centrality (sampled for speed with k nodes)
        try:
            betweenness = nx.betweenness_centrality(G, k=min(20, n_nodes))
            betweenness_arr = np.array([betweenness[i] for i in range(n_nodes)])
            features.append(betweenness_arr.reshape(-1, 1))
        except:
            features.append(np.zeros((n_nodes, 1)))

        # Eigenvector centrality
        try:
            eig_cent = nx.eigenvector_centrality(G, max_iter=1000)
            eig_arr = np.array([eig_cent[i] for i in range(n_nodes)])
            features.append(self._zscore(eig_arr).reshape(-1, 1))
        except:
            features.append(np.zeros((n_nodes, 1)))

        # One-hot position encoding (ROI identity)
        position_encoding = np.eye(n_nodes)
        features.append(position_encoding)

        # Concatenate all features
        X = np.hstack(features)

        # Replace NaN/Inf with 0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        return X.astype(np.float32)

    def _zscore(self, x: np.ndarray) -> np.ndarray:
        """Z-score normalization."""
        std = x.std()
        if std == 0:
            return np.zeros_like(x)
        return (x - x.mean()) / std

    def prepare_labels(self) -> Dict[str, np.ndarray]:
        """
        Prepare labels for all tasks.

        Returns:
            Dictionary with:
            - 'sex': Binary labels (0=F, 1=M)
            - 'math': Math capability scores (FSIQ)
            - 'creativity': Creativity scores (CAQ)
            - 'ursi': Subject IDs
        """
        labels = {}

        # Sex (binary classification) - already encoded as 0/1
        labels['sex'] = self.metadata['Sex'].values.astype(int)

        # Math capability (use FSIQ as proxy)
        labels['math'] = self.metadata['FSIQ'].values.astype(np.float32)

        # Creativity (CAQ score)
        labels['creativity'] = self.metadata['CAQ'].values.astype(np.float32)

        # Subject IDs
        labels['ursi'] = self.metadata['URSI'].values

        # Age (optional covariate)
        labels['age'] = self.metadata['Age'].values.astype(np.float32)

        return labels

    def load_all_subjects(self,
                         symmetrize: bool = True,
                         add_self_loops: bool = True,
                         threshold: float = 0.0,
                         log_scale: bool = False) -> Tuple[List[np.ndarray],
                                                            List[np.ndarray],
                                                            Dict[str, np.ndarray]]:
        """
        Load all subjects with preprocessing.

        Returns:
            Tuple of (adjacency_matrices, node_features, labels)
        """
        labels = self.prepare_labels()
        adjacency_matrices = []
        node_features = []

        print("Loading and preprocessing subjects...")
        for ursi in labels['ursi']:
            # Load raw adjacency
            A = self.load_subject_graph(ursi)

            # Preprocess
            A_proc = self.preprocess_adjacency(
                A,
                symmetrize=symmetrize,
                add_self_loops=add_self_loops,
                threshold=threshold,
                log_scale=log_scale
            )

            # Compute node features
            X = self.compute_node_features(A_proc)

            adjacency_matrices.append(A_proc)
            node_features.append(X)

        print(f"Loaded {len(adjacency_matrices)} subjects")
        print(f"Node features shape: {node_features[0].shape}")

        return adjacency_matrices, node_features, labels


if __name__ == '__main__':
    # Test data loading
    loader = BrainDataLoader('../data')
    adj_list, feat_list, labels = loader.load_all_subjects()

    print(f"\nDataset statistics:")
    print(f"Number of subjects: {len(adj_list)}")
    print(f"Graph size: {adj_list[0].shape}")
    print(f"Node features: {feat_list[0].shape}")
    print(f"\nLabel statistics:")
    print(f"Sex distribution: {np.bincount(labels['sex'])}")
    print(f"Math (FSIQ) - mean: {labels['math'].mean():.1f}, std: {labels['math'].std():.1f}")
    print(f"Creativity (CAQ) - mean: {labels['creativity'].mean():.1f}, std: {labels['creativity'].std():.1f}")
