"""
GCN model for multi-task brain connectivity prediction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_add_pool
from torch_geometric.nn import BatchNorm, LayerNorm


class GCNEncoder(nn.Module):
    """GCN encoder for node embeddings."""

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = 64,
                 num_layers: int = 3,
                 dropout: float = 0.3,
                 use_batch_norm: bool = True):
        """
        Args:
            in_channels: Number of input node features
            hidden_channels: Hidden dimension size
            num_layers: Number of GCN layers
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm

        # GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))

        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # Batch normalization layers
        if use_batch_norm:
            self.batch_norms = nn.ModuleList([
                BatchNorm(hidden_channels) for _ in range(num_layers)
            ])

    def forward(self, x, edge_index, edge_weight=None):
        """
        Forward pass.

        Args:
            x: Node features (num_nodes x in_channels)
            edge_index: Edge connectivity (2 x num_edges)
            edge_weight: Edge weights (num_edges,)

        Returns:
            Node embeddings (num_nodes x hidden_channels)
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)

            if self.use_batch_norm:
                x = self.batch_norms[i](x)

            if i < self.num_layers - 1:  # No activation on last layer
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x


class AttentionPooling(nn.Module):
    """Attention-based graph pooling."""

    def __init__(self, hidden_channels: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.Tanh(),
            nn.Linear(hidden_channels // 2, 1)
        )

    def forward(self, x, batch):
        """
        Args:
            x: Node embeddings (num_nodes x hidden_channels)
            batch: Batch assignment vector

        Returns:
            Graph-level embedding (batch_size x hidden_channels)
        """
        # Compute attention scores
        att_scores = self.attention(x)  # (num_nodes, 1)
        att_weights = torch.softmax(att_scores, dim=0)

        # Weighted sum
        x_pooled = global_add_pool(x * att_weights, batch)

        return x_pooled


class MultiTaskGCN(nn.Module):
    """Multi-task GCN for sex classification and math/creativity regression."""

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = 64,
                 num_layers: int = 3,
                 dropout: float = 0.3,
                 pooling: str = 'mean',
                 head_hidden: int = 32):
        """
        Args:
            in_channels: Number of input node features
            hidden_channels: Hidden dimension for GCN layers
            num_layers: Number of GCN layers
            dropout: Dropout rate
            pooling: Pooling method ('mean', 'add', 'attention')
            head_hidden: Hidden dimension for task heads
        """
        super().__init__()

        # Shared encoder
        self.encoder = GCNEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout
        )

        # Pooling layer
        self.pooling = pooling
        if pooling == 'attention':
            self.pool_fn = AttentionPooling(hidden_channels)
        elif pooling == 'mean':
            self.pool_fn = global_mean_pool
        elif pooling == 'add':
            self.pool_fn = global_add_pool
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        # Task-specific heads
        # Sex classification head (binary)
        self.sex_head = nn.Sequential(
            nn.Linear(hidden_channels, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 2)  # Binary classification
        )

        # Math regression head
        self.math_head = nn.Sequential(
            nn.Linear(hidden_channels, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1)  # Regression
        )

        # Creativity regression head
        self.creativity_head = nn.Sequential(
            nn.Linear(hidden_channels, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1)  # Regression
        )

    def forward(self, data):
        """
        Forward pass.

        Args:
            data: PyG Data batch

        Returns:
            Dictionary with predictions for each task
        """
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch

        # Encode nodes
        x = self.encoder(x, edge_index, edge_weight)

        # Pool to graph-level
        if self.pooling == 'attention':
            g = self.pool_fn(x, batch)
        else:
            g = self.pool_fn(x, batch)

        # Task-specific predictions
        sex_logits = self.sex_head(g)
        math_pred = self.math_head(g).squeeze(-1)
        creativity_pred = self.creativity_head(g).squeeze(-1)

        return {
            'sex_logits': sex_logits,
            'math': math_pred,
            'creativity': creativity_pred,
            'graph_embedding': g  # For analysis
        }


class GATEncoder(nn.Module):
    """GAT (Graph Attention Network) encoder alternative."""

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = 64,
                 num_layers: int = 3,
                 dropout: float = 0.3,
                 heads: int = 4):
        """
        Args:
            in_channels: Number of input node features
            hidden_channels: Hidden dimension size
            num_layers: Number of GAT layers
            dropout: Dropout rate
            heads: Number of attention heads
        """
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # GAT layers
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels // heads, heads=heads, dropout=dropout))

        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels // heads, heads=heads, dropout=dropout))

        # Last layer with single head
        self.convs.append(GATConv(hidden_channels, hidden_channels, heads=1, dropout=dropout))

    def forward(self, x, edge_index, edge_weight=None):
        """Forward pass."""
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class MultiTaskGAT(nn.Module):
    """Multi-task GAT model (alternative to GCN)."""

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = 64,
                 num_layers: int = 3,
                 dropout: float = 0.3,
                 heads: int = 4,
                 pooling: str = 'mean',
                 head_hidden: int = 32):
        super().__init__()

        self.encoder = GATEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            heads=heads
        )

        self.pooling = pooling
        if pooling == 'attention':
            self.pool_fn = AttentionPooling(hidden_channels)
        elif pooling == 'mean':
            self.pool_fn = global_mean_pool
        elif pooling == 'add':
            self.pool_fn = global_add_pool

        self.sex_head = nn.Sequential(
            nn.Linear(hidden_channels, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 2)
        )

        self.math_head = nn.Sequential(
            nn.Linear(hidden_channels, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1)
        )

        self.creativity_head = nn.Sequential(
            nn.Linear(hidden_channels, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1)
        )

    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch

        x = self.encoder(x, edge_index, edge_weight)

        if self.pooling == 'attention':
            g = self.pool_fn(x, batch)
        else:
            g = self.pool_fn(x, batch)

        sex_logits = self.sex_head(g)
        math_pred = self.math_head(g).squeeze(-1)
        creativity_pred = self.creativity_head(g).squeeze(-1)

        return {
            'sex_logits': sex_logits,
            'math': math_pred,
            'creativity': creativity_pred,
            'graph_embedding': g
        }


if __name__ == '__main__':
    # Test model
    from data_loader import BrainDataLoader
    from dataset import create_data_splits
    from torch_geometric.loader import DataLoader

    loader = BrainDataLoader('../data')
    adj_list, feat_list, labels = loader.load_all_subjects()

    n = len(adj_list)
    indices = np.arange(n)
    train_idx = indices[:80]
    val_idx = indices[80:95]
    test_idx = indices[95:]

    train_ds, val_ds, test_ds = create_data_splits(
        adj_list, feat_list, labels,
        train_idx, val_idx, test_idx
    )

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)

    # Create model
    import numpy as np
    model = MultiTaskGCN(
        in_channels=feat_list[0].shape[1],
        hidden_channels=64,
        num_layers=3,
        dropout=0.3,
        pooling='mean'
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Test forward pass
    batch = next(iter(train_loader))
    outputs = model(batch)

    print(f"\nOutput shapes:")
    print(f"  Sex logits: {outputs['sex_logits'].shape}")
    print(f"  Math predictions: {outputs['math'].shape}")
    print(f"  Creativity predictions: {outputs['creativity'].shape}")
    print(f"  Graph embeddings: {outputs['graph_embedding'].shape}")
