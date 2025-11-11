"""
Training pipeline with k-fold cross-validation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, r2_score
from scipy.stats import spearmanr
from tqdm import tqdm
import copy
from typing import Dict, List, Tuple
import json


class MultiTaskLoss(nn.Module):
    """Multi-task loss with task weighting."""

    def __init__(self,
                 lambda_sex: float = 1.0,
                 lambda_math: float = 1.0,
                 lambda_creativity: float = 1.0):
        """
        Args:
            lambda_sex: Weight for sex classification loss
            lambda_math: Weight for math regression loss
            lambda_creativity: Weight for creativity regression loss
        """
        super().__init__()
        self.lambda_sex = lambda_sex
        self.lambda_math = lambda_math
        self.lambda_creativity = lambda_creativity

    def forward(self, predictions: Dict, targets: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Compute multi-task loss.

        Args:
            predictions: Model predictions dict
            targets: Ground truth dict

        Returns:
            Total loss and loss breakdown dict
        """
        # Sex classification (cross-entropy)
        loss_sex = F.cross_entropy(predictions['sex_logits'], targets['sex'])

        # Math regression (MSE)
        loss_math = F.mse_loss(predictions['math'], targets['math'])

        # Creativity regression (MSE)
        loss_creativity = F.mse_loss(predictions['creativity'], targets['creativity'])

        # Total weighted loss
        total_loss = (
            self.lambda_sex * loss_sex +
            self.lambda_math * loss_math +
            self.lambda_creativity * loss_creativity
        )

        loss_dict = {
            'total': total_loss.item(),
            'sex': loss_sex.item(),
            'math': loss_math.item(),
            'creativity': loss_creativity.item()
        }

        return total_loss, loss_dict


class Trainer:
    """Trainer for multi-task GCN."""

    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 lambda_sex: float = 1.0,
                 lambda_math: float = 1.0,
                 lambda_creativity: float = 1.0):
        """
        Args:
            model: GCN model
            device: torch device
            lambda_sex: Loss weight for sex
            lambda_math: Loss weight for math
            lambda_creativity: Loss weight for creativity
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = MultiTaskLoss(lambda_sex, lambda_math, lambda_creativity)

    def train_epoch(self,
                   train_loader: DataLoader,
                   optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        losses = {'sex': 0, 'math': 0, 'creativity': 0}
        num_batches = 0

        for batch in train_loader:
            batch = batch.to(self.device)
            optimizer.zero_grad()

            # Forward pass
            predictions = self.model(batch)

            # Prepare targets
            targets = {
                'sex': batch.y_sex.squeeze(),
                'math': batch.y_math.squeeze(),
                'creativity': batch.y_creativity.squeeze()
            }

            # Compute loss
            loss, loss_dict = self.criterion(predictions, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss_dict['total']
            losses['sex'] += loss_dict['sex']
            losses['math'] += loss_dict['math']
            losses['creativity'] += loss_dict['creativity']
            num_batches += 1

        # Average losses
        avg_losses = {
            'total': total_loss / num_batches,
            'sex': losses['sex'] / num_batches,
            'math': losses['math'] / num_batches,
            'creativity': losses['creativity'] / num_batches
        }

        return avg_losses

    @torch.no_grad()
    def evaluate(self,
                loader: DataLoader,
                return_predictions: bool = False) -> Dict:
        """
        Evaluate model on a dataset.

        Returns:
            Dictionary with metrics for each task
        """
        self.model.eval()

        all_sex_preds = []
        all_sex_probs = []
        all_sex_targets = []

        all_math_preds = []
        all_math_targets = []

        all_creativity_preds = []
        all_creativity_targets = []

        total_loss = 0
        num_batches = 0

        for batch in loader:
            batch = batch.to(self.device)

            # Forward pass
            predictions = self.model(batch)

            # Get targets
            targets = {
                'sex': batch.y_sex.squeeze(),
                'math': batch.y_math.squeeze(),
                'creativity': batch.y_creativity.squeeze()
            }

            # Compute loss
            loss, _ = self.criterion(predictions, targets)
            total_loss += loss.item()
            num_batches += 1

            # Collect predictions and targets
            # Sex
            sex_probs = F.softmax(predictions['sex_logits'], dim=1)
            sex_preds = torch.argmax(sex_probs, dim=1)
            all_sex_preds.extend(sex_preds.cpu().numpy())
            all_sex_probs.extend(sex_probs[:, 1].cpu().numpy())  # Probability of class 1
            all_sex_targets.extend(targets['sex'].cpu().numpy())

            # Math (denormalize for evaluation)
            all_math_preds.extend(predictions['math'].cpu().numpy())
            all_math_targets.extend(batch.y_math_raw.cpu().numpy())

            # Creativity (denormalize)
            all_creativity_preds.extend(predictions['creativity'].cpu().numpy())
            all_creativity_targets.extend(batch.y_creativity_raw.cpu().numpy())

        # Convert to numpy
        all_sex_preds = np.array(all_sex_preds)
        all_sex_probs = np.array(all_sex_probs)
        all_sex_targets = np.array(all_sex_targets)
        all_math_preds = np.array(all_math_preds)
        all_math_targets = np.array(all_math_targets)
        all_creativity_preds = np.array(all_creativity_preds)
        all_creativity_targets = np.array(all_creativity_targets)

        # Compute metrics
        metrics = {}

        # Sex classification
        metrics['sex_accuracy'] = accuracy_score(all_sex_targets, all_sex_preds)
        metrics['sex_auroc'] = roc_auc_score(all_sex_targets, all_sex_probs)

        # Math regression (on raw values)
        valid_math = ~np.isnan(all_math_targets)
        if valid_math.sum() > 0:
            math_preds_valid = all_math_preds[valid_math]
            math_targets_valid = all_math_targets[valid_math]
            metrics['math_mae'] = mean_absolute_error(math_targets_valid, math_preds_valid)
            metrics['math_r2'] = r2_score(math_targets_valid, math_preds_valid)
            metrics['math_spearman'], _ = spearmanr(math_targets_valid, math_preds_valid)
        else:
            metrics['math_mae'] = float('nan')
            metrics['math_r2'] = float('nan')
            metrics['math_spearman'] = float('nan')

        # Creativity regression
        valid_creativity = ~np.isnan(all_creativity_targets)
        if valid_creativity.sum() > 0:
            creativity_preds_valid = all_creativity_preds[valid_creativity]
            creativity_targets_valid = all_creativity_targets[valid_creativity]
            metrics['creativity_mae'] = mean_absolute_error(creativity_targets_valid, creativity_preds_valid)
            metrics['creativity_r2'] = r2_score(creativity_targets_valid, creativity_preds_valid)
            metrics['creativity_spearman'], _ = spearmanr(creativity_targets_valid, creativity_preds_valid)
        else:
            metrics['creativity_mae'] = float('nan')
            metrics['creativity_r2'] = float('nan')
            metrics['creativity_spearman'] = float('nan')

        metrics['loss'] = total_loss / num_batches

        if return_predictions:
            metrics['predictions'] = {
                'sex': all_sex_preds,
                'sex_probs': all_sex_probs,
                'math': all_math_preds,
                'creativity': all_creativity_preds
            }
            metrics['targets'] = {
                'sex': all_sex_targets,
                'math': all_math_targets,
                'creativity': all_creativity_targets
            }

        return metrics


def train_with_early_stopping(trainer: Trainer,
                              train_loader: DataLoader,
                              val_loader: DataLoader,
                              optimizer: torch.optim.Optimizer,
                              num_epochs: int = 500,
                              patience: int = 30,
                              verbose: bool = True) -> Tuple[nn.Module, Dict]:
    """
    Train with early stopping.

    Returns:
        Best model and training history
    """
    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0
    history = {'train': [], 'val': []}

    pbar = tqdm(range(num_epochs), desc='Training') if verbose else range(num_epochs)

    for epoch in pbar:
        # Train
        train_metrics = trainer.train_epoch(train_loader, optimizer)
        history['train'].append(train_metrics)

        # Validate
        val_metrics = trainer.evaluate(val_loader)
        history['val'].append(val_metrics)

        # Early stopping check
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_model_state = copy.deepcopy(trainer.model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Update progress bar
        if verbose:
            pbar.set_postfix({
                'train_loss': f"{train_metrics['total']:.4f}",
                'val_loss': f"{val_metrics['loss']:.4f}",
                'sex_acc': f"{val_metrics['sex_accuracy']:.3f}",
                'patience': f"{epochs_without_improvement}/{patience}"
            })

        # Early stopping
        if epochs_without_improvement >= patience:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Restore best model
    trainer.model.load_state_dict(best_model_state)

    return trainer.model, history


def run_kfold_cv(model_class,
                model_params: Dict,
                adjacency_matrices: List[np.ndarray],
                node_features: List[np.ndarray],
                labels: Dict[str, np.ndarray],
                k_folds: int = 5,
                batch_size: int = 8,
                num_epochs: int = 500,
                patience: int = 30,
                lr: float = 1e-3,
                weight_decay: float = 1e-4,
                device: torch.device = torch.device('cpu'),
                verbose: bool = True) -> Dict:
    """
    Run k-fold cross-validation.

    Returns:
        Dictionary with results for each fold and aggregated metrics
    """
    from dataset import create_data_splits

    # Stratified k-fold on sex labels
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    all_results = []
    fold_histories = []

    for fold, (train_val_idx, test_idx) in enumerate(skf.split(np.zeros(len(adjacency_matrices)), labels['sex'])):
        if verbose:
            print(f"\n{'='*50}")
            print(f"Fold {fold + 1}/{k_folds}")
            print(f"{'='*50}")

        # Further split train_val into train and val
        train_size = int(0.85 * len(train_val_idx))
        train_idx = train_val_idx[:train_size]
        val_idx = train_val_idx[train_size:]

        # Create datasets
        train_ds, val_ds, test_ds = create_data_splits(
            adjacency_matrices,
            node_features,
            labels,
            train_idx,
            val_idx,
            test_idx
        )

        # Create data loaders
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        # Create model
        model = model_class(**model_params)
        trainer = Trainer(model, device)

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Train
        best_model, history = train_with_early_stopping(
            trainer,
            train_loader,
            val_loader,
            optimizer,
            num_epochs=num_epochs,
            patience=patience,
            verbose=verbose
        )

        # Evaluate on test set
        test_metrics = trainer.evaluate(test_loader)

        if verbose:
            print(f"\nTest Results (Fold {fold + 1}):")
            print(f"  Sex - Accuracy: {test_metrics['sex_accuracy']:.4f}, AUROC: {test_metrics['sex_auroc']:.4f}")
            print(f"  Math - MAE: {test_metrics['math_mae']:.4f}, R²: {test_metrics['math_r2']:.4f}, Spearman: {test_metrics['math_spearman']:.4f}")
            print(f"  Creativity - MAE: {test_metrics['creativity_mae']:.4f}, R²: {test_metrics['creativity_r2']:.4f}, Spearman: {test_metrics['creativity_spearman']:.4f}")

        all_results.append(test_metrics)
        fold_histories.append(history)

    # Aggregate results across folds
    aggregated = {}
    metric_keys = [k for k in all_results[0].keys() if k not in ['predictions', 'targets']]

    for key in metric_keys:
        values = [r[key] for r in all_results if not np.isnan(r[key])]
        if len(values) > 0:
            aggregated[f'{key}_mean'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)

    if verbose:
        print(f"\n{'='*50}")
        print("Aggregated Results (Mean ± Std)")
        print(f"{'='*50}")
        print(f"Sex - Accuracy: {aggregated['sex_accuracy_mean']:.4f} ± {aggregated['sex_accuracy_std']:.4f}")
        print(f"      AUROC: {aggregated['sex_auroc_mean']:.4f} ± {aggregated['sex_auroc_std']:.4f}")
        print(f"Math - MAE: {aggregated['math_mae_mean']:.4f} ± {aggregated['math_mae_std']:.4f}")
        print(f"       R²: {aggregated['math_r2_mean']:.4f} ± {aggregated['math_r2_std']:.4f}")
        print(f"       Spearman: {aggregated['math_spearman_mean']:.4f} ± {aggregated['math_spearman_std']:.4f}")
        print(f"Creativity - MAE: {aggregated['creativity_mae_mean']:.4f} ± {aggregated['creativity_mae_std']:.4f}")
        print(f"             R²: {aggregated['creativity_r2_mean']:.4f} ± {aggregated['creativity_r2_std']:.4f}")
        print(f"             Spearman: {aggregated['creativity_spearman_mean']:.4f} ± {aggregated['creativity_spearman_std']:.4f}")

    return {
        'fold_results': all_results,
        'aggregated': aggregated,
        'histories': fold_histories
    }


if __name__ == '__main__':
    from data_loader import BrainDataLoader
    from model import MultiTaskGCN

    # Load data
    loader = BrainDataLoader('../data')
    adj_list, feat_list, labels = loader.load_all_subjects()

    # Model parameters
    model_params = {
        'in_channels': feat_list[0].shape[1],
        'hidden_channels': 64,
        'num_layers': 3,
        'dropout': 0.3,
        'pooling': 'mean',
        'head_hidden': 32
    }

    # Run k-fold CV
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    results = run_kfold_cv(
        model_class=MultiTaskGCN,
        model_params=model_params,
        adjacency_matrices=adj_list,
        node_features=feat_list,
        labels=labels,
        k_folds=5,
        batch_size=8,
        num_epochs=500,
        patience=30,
        lr=1e-3,
        weight_decay=1e-4,
        device=device,
        verbose=True
    )

    # Save results
    import pickle
    with open('results_kfold.pkl', 'wb') as f:
        pickle.dump(results, f)

    print("\nResults saved to results_kfold.pkl")
