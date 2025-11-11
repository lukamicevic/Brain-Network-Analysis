# Brain Connectivity GCN

Multi-task Graph Convolutional Network for brain connectivity analysis.

## Tasks
1. **Sex Classification**: Binary classification (Male vs Female)
2. **Math Capability**: Regression on FSIQ scores
3. **Creativity**: Regression on CAQ scores

## Project Structure

```
GCN/
├── data_loader.py      # Data loading and preprocessing
├── dataset.py          # PyTorch Geometric dataset
├── model.py           # GCN/GAT architectures
├── train.py           # Training pipeline with k-fold CV
├── baselines.py       # Baseline models (RF, SVM, etc.)
├── main.py            # Main training script
└── README.md          # This file
```

## Setup

1. **Activate virtual environment**:
```bash
source venv/bin/activate
```

2. **Install dependencies** (already done):
- PyTorch
- PyTorch Geometric
- scikit-learn
- NetworkX
- scipy, numpy, pandas

## Usage

### Quick Start

Train GCN with default settings:
```bash
python main.py --mode train
```

Run baseline models only:
```bash
python main.py --mode baseline
```

Run both GCN and baselines:
```bash
python main.py --mode both
```

### Advanced Usage

**Use GAT instead of GCN**:
```bash
python main.py --model gat --hidden_channels 64 --num_layers 3
```

**Adjust hyperparameters**:
```bash
python main.py \
    --hidden_channels 128 \
    --num_layers 4 \
    --dropout 0.5 \
    --pooling attention \
    --lr 5e-4 \
    --batch_size 16
```

**Adjust loss weights** (emphasize math task):
```bash
python main.py \
    --lambda_sex 1.0 \
    --lambda_math 2.0 \
    --lambda_creativity 1.0
```

**Custom experiment**:
```bash
python main.py \
    --exp_name my_experiment \
    --model gcn \
    --hidden_channels 64 \
    --num_layers 3 \
    --dropout 0.3 \
    --k_folds 5 \
    --num_epochs 500 \
    --patience 30
```

## Model Architecture

### GCN Encoder
- Multiple GCN layers with batch normalization
- ReLU activation and dropout
- Edge weights incorporated in message passing

### Graph Pooling
- **Mean pooling**: Average node embeddings
- **Add pooling**: Sum node embeddings
- **Attention pooling**: Learned attention weights

### Task Heads
- **Sex head**: 2-layer MLP → 2 classes (binary classification)
- **Math head**: 2-layer MLP → scalar (regression)
- **Creativity head**: 2-layer MLP → scalar (regression)

### Multi-Task Loss
```
L = λ_sex * CE + λ_math * MSE + λ_creativity * MSE
```

## Data Preprocessing

1. **Symmetrize**: `A = (A + A^T) / 2` (undirected graph)
2. **Add self-loops**: `A = A + I`
3. **Optional**: Log-scale edge weights, threshold weak edges

## Node Features

Per-node features (76-dimensional):
- **Strength features** (3): in/out/total strength (z-scored)
- **Clustering coefficient** (1)
- **Betweenness centrality** (1)
- **Eigenvector centrality** (1)
- **Position encoding** (70): one-hot ROI identity

## Baseline Models

Hand-crafted graph features (~40D):
- Node strength statistics
- Global efficiency
- Clustering coefficient
- Betweenness centrality
- Assortativity
- Top-k edge weights
- Edge weight statistics

Baseline classifiers/regressors:
- Logistic Regression / Ridge
- Random Forest
- SVM / SVR

## Evaluation Metrics

**Sex Classification**:
- Accuracy
- AUROC

**Math/Creativity Regression**:
- MAE (Mean Absolute Error)
- R² (Coefficient of Determination)
- Spearman correlation

All metrics reported as **mean ± std** across k folds.

## Output

Results saved to `results/<exp_name>/`:
- `args.json`: Experiment configuration
- `gcn_results.pkl`: Full k-fold CV results
- `gcn_aggregated.json`: Aggregated metrics
- `baseline_results.pkl`: Baseline results (if run)

## Training Protocol

- **5-fold stratified cross-validation** (stratified on sex)
- **85/15 train/val split** within each fold
- **Early stopping** (patience=30 epochs)
- **Adam optimizer** with weight decay
- **Target normalization** (z-score on training set)

## Tips

1. **Small dataset (N=114)**: Use small models, dropout, weight decay
2. **Multi-task learning**: Adjust loss weights if one task dominates
3. **Feature engineering**: Node features are crucial (topology + position)
4. **Pooling**: Try attention pooling for better graph representations
5. **Baselines**: Always compare against hand-crafted features

## Example Output

```
Fold 1/5
Training: 100%|████████████| 500/500 [02:30<00:00]
Early stopping at epoch 234

Test Results (Fold 1):
  Sex - Accuracy: 0.8261, AUROC: 0.8947
  Math - MAE: 8.32, R²: 0.24, Spearman: 0.51
  Creativity - MAE: 12.15, R²: 0.18, Spearman: 0.43

...

Aggregated Results (Mean ± Std):
Sex - Accuracy: 0.8158 ± 0.0234
      AUROC: 0.8821 ± 0.0189
Math - MAE: 8.67 ± 0.52
       R²: 0.22 ± 0.05
       Spearman: 0.48 ± 0.07
```

## Citation

Dataset from:
```
Brain Networks Dataset
https://www.andrew.cmu.edu/user/lakoglu/courses/95828/S17/projectsources/brainnetworks.rar
```
