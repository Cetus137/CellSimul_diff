#!/usr/bin/env python3
"""
Analyze training data to determine realistic cell center statistics.
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_training_data(patches_dir="/users/kir-fritzsche/aif490/devel/tissue_analysis/CellSimul_diff/data/processed/train"):
    """Analyze cell density and spacing from training data."""
    patches_dir = Path(patches_dir)
    centre_files = list(patches_dir.glob("*_centres.npy"))
    
    print(f"Analyzing {len(centre_files)} training patches...")
    
    cell_counts = []
    min_distances = []
    
    for cf in centre_files[:500]:  # Sample 500 patches
        centres = np.load(cf)
        cell_counts.append(len(centres))
        
        if len(centres) > 1:
            # Compute minimum distance between centers
            from scipy.spatial.distance import pdist
            dists = pdist(centres)
            min_distances.append(dists.min())
    
    cell_counts = np.array(cell_counts)
    min_distances = np.array(min_distances)
    
    print(f"\nCell Count Statistics:")
    print(f"  Mean: {cell_counts.mean():.1f}")
    print(f"  Median: {np.median(cell_counts):.1f}")
    print(f"  Min: {cell_counts.min()}")
    print(f"  Max: {cell_counts.max()}")
    print(f"  Std: {cell_counts.std():.1f}")
    print(f"  25th percentile: {np.percentile(cell_counts, 25):.1f}")
    print(f"  75th percentile: {np.percentile(cell_counts, 75):.1f}")
    
    print(f"\nMinimum Distance Statistics:")
    print(f"  Mean: {min_distances.mean():.1f} pixels")
    print(f"  Median: {np.median(min_distances):.1f} pixels")
    print(f"  Min: {min_distances.min():.1f} pixels")
    print(f"  Max: {min_distances.max():.1f} pixels")
    
    return {
        'cell_counts': cell_counts,
        'min_distances': min_distances,
        'mean_cells': cell_counts.mean(),
        'median_cells': np.median(cell_counts),
        'mean_min_dist': min_distances.mean()
    }

if __name__ == "__main__":
    stats = analyze_training_data()
