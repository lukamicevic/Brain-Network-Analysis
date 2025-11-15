"""
PyTorch Geometric dataset for brain connectivity graphs.
"""
import torch
import numpy as np
from torch_geometric.data import Data, Dataset
from typing import List, Dict, Optional
from sklearn.preprocessing import StandardScaler


class BrainGraphDataset(Dataset):
    """PyTorch Geometric dataset for brain connectivity graphs."""

    def __init__(self,
                 adjacency_matrices: List[np.ndarray],
                 node_features: List[np.ndarray],
                 labels: Dict[str, np.ndarray],
                 indices: Optional[np.ndarray] = None,
                 train_stats: Optional[Dict] = None):
        """
        Args:
            adjacency_matrices: List of adjacency matrices (N x 70 x 70)
            node_features: List of node feature matrices (N x 70 x D)
            labels: Dictionary of labels (sex, math, creativity, etc.)
            indices: Subject indices to use (for train/val/test split)
            train_stats: Statistics from training set for normalization
        """
        super().__init__()
        self.adjacency_matrices = adjacency_matrices
        self.node_features = node_features
        self.labels = labels
        self._indices = indices if indices is not None else np.arange(len(adjacency_matrices))
        self.train_stats = train_stats

        # Normalize regression targets if train_stats provided
        if train_stats is None:
            # This is the training set - compute stats
            self.train_stats = self._compute_train_stats()

    def _compute_train_stats(self) -> Dict:
        """Compute mean/std for regression targets on this split."""
        math_vals = self.labels['math'][self._indices]
        creativity_vals = self.labels['creativity'][self._indices]

        # Remove NaN values
        math_vals = math_vals[~np.isnan(math_vals)]
        creativity_vals = creativity_vals[~np.isnan(creativity_vals)]

        stats = {
            'math_mean': math_vals.mean(),
            'math_std': math_vals.std(),
            'creativity_mean': creativity_vals.mean(),
            'creativity_std': creativity_vals.std(),
        }
        return stats

    def _normalize_target(self, value: float, mean: float, std: float) -> float:
        """Z-score normalize a target value."""
        if np.isnan(value):
            return 0.0
        if std == 0:
            return 0.0
        return (value - mean) / std

    def len(self) -> int:
        """Return number of graphs in dataset."""
        return len(self._indices)

    def indices(self):
        """Return the indices mapping for this dataset."""
        return range(len(self._indices))
    
    def get(self, idx: int) -> Data:
        """
        Get a single graph.

        Args:
            idx: Index in range [0, len(self._indices))
        
        Returns:
            PyTorch Geometric Data object with:
            - x: Node features (70 x D)
            - edge_index: Edge connectivity (2 x E)
            - edge_weight: Edge weights (E)
            - y_sex: Binary label for sex
            - y_math: Normalized math score
            - y_creativity: Normalized creativity score
            - age: Age (optional covariate)
        """
        # Map idx (0 to len-1) to actual subject index
        subj_idx = self._indices[idx]

        # Get adjacency and features
        A = self.adjacency_matrices[subj_idx]
        X = self.node_features[subj_idx]

        # Convert adjacency to edge_index and edge_weight (COO format)
        edge_index, edge_weight = self._adj_to_edge_index(A)

        # Get labels
        y_sex = self.labels['sex'][subj_idx]
        y_math_raw = self.labels['math'][subj_idx]
        y_creativity_raw = self.labels['creativity'][subj_idx]
        age = self.labels['age'][subj_idx]

        # Normalize regression targets
        y_math = self._normalize_target(
            y_math_raw,
            self.train_stats['math_mean'],
            self.train_stats['math_std']
        )
        y_creativity = self._normalize_target(
            y_creativity_raw,
            self.train_stats['creativity_mean'],
            self.train_stats['creativity_std']
        )

        # Create PyG Data object
        data = Data(
            x=torch.FloatTensor(X),
            edge_index=edge_index,
            edge_weight=edge_weight,
            y_sex=torch.LongTensor([y_sex]),
            y_math=torch.FloatTensor([y_math]),
            y_creativity=torch.FloatTensor([y_creativity]),
            age=torch.FloatTensor([age]),
            # Store raw values for evaluation
            y_math_raw=torch.FloatTensor([y_math_raw]),
            y_creativity_raw=torch.FloatTensor([y_creativity_raw]),
        )

        return data

    def _adj_to_edge_index(self, A: np.ndarray):
        """
        Convert adjacency matrix to edge_index and edge_weight.

        Args:
            A: Adjacency matrix (N x N)

        Returns:
            edge_index: (2, num_edges) tensor
            edge_weight: (num_edges,) tensor
        """
        # Find non-zero edges
        row, col = np.where(A > 0)
        edge_weight = A[row, col]

        # Create edge_index
        edge_index = torch.LongTensor(np.vstack([row, col]))
        edge_weight = torch.FloatTensor(edge_weight)

        return edge_index, edge_weight


def create_data_splits(adjacency_matrices: List[np.ndarray],
                      node_features: List[np.ndarray],
                      labels: Dict[str, np.ndarray],
                      train_idx: np.ndarray,
                      val_idx: np.ndarray,
                      test_idx: np.ndarray):
    """
    Create train/val/test datasets with proper normalization.

    Args:
        adjacency_matrices: All adjacency matrices
        node_features: All node features
        labels: All labels
        train_idx, val_idx, test_idx: Indices for each split

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Create training dataset first (computes normalization stats)
    train_dataset = BrainGraphDataset(
        adjacency_matrices,
        node_features,
        labels,
        indices=train_idx,
        train_stats=None  # Will compute from training data
    )

    # Create val/test datasets using training stats
    val_dataset = BrainGraphDataset(
        adjacency_matrices,
        node_features,
        labels,
        indices=val_idx,
        train_stats=train_dataset.train_stats
    )

    test_dataset = BrainGraphDataset(
        adjacency_matrices,
        node_features,
        labels,
        indices=test_idx,
        train_stats=train_dataset.train_stats
    )

    return train_dataset, val_dataset, test_dataset


if __name__ == '__main__':
    # Test dataset creation
    from data_loader import BrainDataLoader

    loader = BrainDataLoader('../data')
    adj_list, feat_list, labels = loader.load_all_subjects()

    # Create simple train/val/test split
    n = len(adj_list)
    indices = np.random.permutation(n)
    train_idx = indices[:int(0.7*n)]
    val_idx = indices[int(0.7*n):int(0.85*n)]
    test_idx = indices[int(0.85*n):]

    train_ds, val_ds, test_ds = create_data_splits(
        adj_list, feat_list, labels,
        train_idx, val_idx, test_idx
    )

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # Test getting a sample
    sample = train_ds[0]
    print(f"\nSample data:")
    print(f"  Node features: {sample.x.shape}")
    print(f"  Edges: {sample.edge_index.shape}")
    print(f"  Edge weights: {sample.edge_weight.shape}")
    print(f"  Sex label: {sample.y_sex.item()}")
    print(f"  Math (normalized): {sample.y_math.item():.3f}")
    print(f"  Creativity (normalized): {sample.y_creativity.item():.3f}")
