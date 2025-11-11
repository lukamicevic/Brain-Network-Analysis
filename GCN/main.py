#!/usr/bin/env python3
"""
Main script for training GCN on brain connectivity data.

Usage:
    python main.py --mode train              # Train GCN with k-fold CV
    python main.py --mode baseline           # Run baseline models
    python main.py --mode both               # Run both
    python main.py --model gat               # Use GAT instead of GCN
"""
import argparse
import torch
import pickle
import json
from pathlib import Path

from data_loader import BrainDataLoader
from model import MultiTaskGCN, MultiTaskGAT
from train import run_kfold_cv
from baselines import BaselineModels


def parse_args():
    parser = argparse.ArgumentParser(description='Train GCN on brain connectivity data')

    # Mode
    parser.add_argument('--mode', type=str, default='both',
                       choices=['train', 'baseline', 'both'],
                       help='Training mode')

    # Data
    parser.add_argument('--data_dir', type=str, default='../data',
                       help='Path to data directory')
    parser.add_argument('--symmetrize', action='store_true', default=True,
                       help='Symmetrize adjacency matrices')
    parser.add_argument('--add_self_loops', action='store_true', default=True,
                       help='Add self-loops to graphs')
    parser.add_argument('--log_scale', action='store_true', default=False,
                       help='Apply log scaling to edge weights')
    parser.add_argument('--threshold', type=float, default=0.0,
                       help='Threshold for edge weights')

    # Model
    parser.add_argument('--model', type=str, default='gcn',
                       choices=['gcn', 'gat'],
                       help='Model architecture')
    parser.add_argument('--hidden_channels', type=int, default=64,
                       help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='Number of GCN/GAT layers')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--pooling', type=str, default='mean',
                       choices=['mean', 'add', 'attention'],
                       help='Graph pooling method')
    parser.add_argument('--head_hidden', type=int, default=32,
                       help='Hidden dimension for task heads')

    # Training
    parser.add_argument('--k_folds', type=int, default=5,
                       help='Number of folds for cross-validation')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=500,
                       help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=30,
                       help='Early stopping patience')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')

    # Loss weights
    parser.add_argument('--lambda_sex', type=float, default=1.0,
                       help='Loss weight for sex classification')
    parser.add_argument('--lambda_math', type=float, default=1.0,
                       help='Loss weight for math regression')
    parser.add_argument('--lambda_creativity', type=float, default=1.0,
                       help='Loss weight for creativity regression')

    # Output
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for results')
    parser.add_argument('--exp_name', type=str, default=None,
                       help='Experiment name (default: auto-generated)')

    # Device
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cuda/cpu/auto)')

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Generate experiment name
    if args.exp_name is None:
        args.exp_name = f"{args.model}_h{args.hidden_channels}_l{args.num_layers}_d{args.dropout}"

    exp_dir = output_dir / args.exp_name
    exp_dir.mkdir(exist_ok=True, parents=True)

    # Save args
    with open(exp_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("="*70)
    print("Brain Connectivity Multi-Task Learning")
    print("="*70)
    print(f"Experiment: {args.exp_name}")
    print(f"Output directory: {exp_dir}")
    print()

    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}\n")

    # Load data
    print("Loading data...")
    loader = BrainDataLoader(args.data_dir)
    adj_list, feat_list, labels = loader.load_all_subjects(
        symmetrize=args.symmetrize,
        add_self_loops=args.add_self_loops,
        threshold=args.threshold,
        log_scale=args.log_scale
    )
    print(f"Loaded {len(adj_list)} subjects")
    print(f"Node features shape: {feat_list[0].shape}\n")

    # Run baselines
    if args.mode in ['baseline', 'both']:
        print("\n" + "="*70)
        print("RUNNING BASELINE MODELS")
        print("="*70)

        baseline = BaselineModels()
        baseline_results = baseline.evaluate_baseline(
            adj_list,
            labels,
            k_folds=args.k_folds
        )

        # Save baseline results
        with open(exp_dir / 'baseline_results.pkl', 'wb') as f:
            pickle.dump(baseline_results, f)
        print(f"\nBaseline results saved to {exp_dir / 'baseline_results.pkl'}")

    # Run GCN/GAT
    if args.mode in ['train', 'both']:
        print("\n" + "="*70)
        print(f"TRAINING {args.model.upper()} MODEL")
        print("="*70)

        # Select model class
        if args.model == 'gcn':
            model_class = MultiTaskGCN
        elif args.model == 'gat':
            model_class = MultiTaskGAT
        else:
            raise ValueError(f"Unknown model: {args.model}")

        # Model parameters
        model_params = {
            'in_channels': feat_list[0].shape[1],
            'hidden_channels': args.hidden_channels,
            'num_layers': args.num_layers,
            'dropout': args.dropout,
            'pooling': args.pooling,
            'head_hidden': args.head_hidden
        }

        print(f"Model: {args.model.upper()}")
        print(f"Parameters: {model_params}\n")

        # Run k-fold CV
        gcn_results = run_kfold_cv(
            model_class=model_class,
            model_params=model_params,
            adjacency_matrices=adj_list,
            node_features=feat_list,
            labels=labels,
            k_folds=args.k_folds,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            patience=args.patience,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
            verbose=True
        )

        # Save GCN results
        with open(exp_dir / f'{args.model}_results.pkl', 'wb') as f:
            pickle.dump(gcn_results, f)
        print(f"\n{args.model.upper()} results saved to {exp_dir / f'{args.model}_results.pkl'}")

        # Save aggregated results as JSON for easy viewing
        with open(exp_dir / f'{args.model}_aggregated.json', 'w') as f:
            json.dump(gcn_results['aggregated'], f, indent=2)

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print(f"Results saved to: {exp_dir}")


if __name__ == '__main__':
    main()
