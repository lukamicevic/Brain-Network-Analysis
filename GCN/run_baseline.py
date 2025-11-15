#!/usr/bin/env python3
"""
Simple runner script for the baseline ML pipeline.

Usage:
    python run_baseline.py
"""
from baseline_pipeline import BaselineMLPipeline
import pickle
from pathlib import Path


def main():
    """Run the complete baseline ML pipeline."""
    
    # Initialize pipeline
    pipeline = BaselineMLPipeline(
        k_folds=5,
        random_state=42
    )
    
    # Run pipeline
    results = pipeline.run('../data')
    
    # Save results
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / 'baseline_ml_results.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n{'='*70}")
    print(f"✅ Results saved to {output_file}")
    print(f"{'='*70}")
    
    # Print feature info
    print(f"\nFeature extraction details:")
    print(f"  Graph biomarkers: {len(pipeline.biomarker_extractor.feature_names)} features")
    print(f"  Metadata fields: {len(pipeline.metadata_fusion.METADATA_FIELDS)} features")
    print(f"  Total features: {len(pipeline.biomarker_extractor.feature_names) + len(pipeline.metadata_fusion.METADATA_FIELDS)}")


if __name__ == '__main__':
    main()
