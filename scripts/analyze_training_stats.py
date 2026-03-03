#!/usr/bin/env python3
"""
Analyze 3D training data to derive realistic cell-centre generation statistics.

Outputs:
  - N distribution (mean, std, percentiles)
  - Per-cell nearest-neighbour distance distribution
  - 16-bin marginal KDE grids for z, y, x (normalised to sum=1)

With --save_stats OUTPUT.yaml: writes a ready-to-paste YAML block for
configs/frame1_3d.yaml under the `centre_generation_3d` key.

Usage:
  python scripts/analyze_training_stats.py \
      --patches_dir data_live_node1_3d/train data_live_node2_3d/train \
      --save_stats centre_stats_3d.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial import cKDTree

# ── Helpers ───────────────────────────────────────────────────────────────────

def _marginal_histogram(coords_normalised: np.ndarray, n_bins: int = 16) -> np.ndarray:
    """
    Build a normalised 1-D histogram from (0,1)-normalised coordinates.

    Args:
        coords_normalised: 1-D array in [0, 1]
        n_bins: Number of bins (default 16)

    Returns:
        weights: 1-D float array of length n_bins summing to 1.0.
                 Any zero-count bin is replaced by a small positive floor
                 so generation never has dead zones.
    """
    counts, _ = np.histogram(coords_normalised, bins=n_bins, range=(0.0, 1.0))
    counts = counts.astype(np.float64)
    floor = max(counts.max() * 0.01, 1.0)      # 1% of peak or at least 1 sample
    counts = np.maximum(counts, floor)
    return (counts / counts.sum()).astype(np.float32)


def _collect_statistics(patches_dirs, max_patches=None):
    """Load all centre files from one or more train directories and collect stats."""
    all_files = []
    for d in patches_dirs:
        d = Path(d)
        found = sorted(d.glob("*_centres.npy"))
        print(f"  {d}: {len(found)} centre files")
        all_files.extend(found)

    if max_patches is not None:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(all_files), size=min(max_patches, len(all_files)), replace=False)
        all_files = [all_files[i] for i in sorted(idx)]

    print(f"\nProcessing {len(all_files)} patches total...")

    cell_counts = []
    per_cell_nn_dists = []        # one entry per cell
    all_z_norm = []               # pooled normalised coordinates
    all_y_norm = []
    all_x_norm = []

    n_skipped = 0
    for cf in all_files:
        c = np.load(cf).astype(np.float32)   # (N, 3) in (z, y, x) order
        if len(c) < 2:
            n_skipped += 1
            continue

        N = len(c)
        cell_counts.append(N)

        # Per-cell nearest-neighbour distances (k=2: exclude self)
        tree = cKDTree(c)
        dists, _ = tree.query(c, k=2)
        per_cell_nn_dists.extend(dists[:, 1].tolist())   # column 1 = nearest other cell

        # Normalise coordinates to [0, 1] (assume 128-voxel patches)
        # Uses the actual min/max of centres to be resolution-agnostic,
        # clamped within [0, 1] to handle border-edge cases.
        patch_size = 128.0   # expected from config; could be auto-detected
        all_z_norm.extend(np.clip(c[:, 0] / patch_size, 0.0, 1.0).tolist())
        all_y_norm.extend(np.clip(c[:, 1] / patch_size, 0.0, 1.0).tolist())
        all_x_norm.extend(np.clip(c[:, 2] / patch_size, 0.0, 1.0).tolist())

    if n_skipped:
        print(f"  Skipped {n_skipped} patches with < 2 cells.")

    cell_counts   = np.array(cell_counts, dtype=np.float32)
    nn_dists      = np.array(per_cell_nn_dists, dtype=np.float32)
    all_z_norm    = np.array(all_z_norm, dtype=np.float32)
    all_y_norm    = np.array(all_y_norm, dtype=np.float32)
    all_x_norm    = np.array(all_x_norm, dtype=np.float32)

    return cell_counts, nn_dists, all_z_norm, all_y_norm, all_x_norm


def _print_dist(label: str, arr: np.ndarray, unit: str = "") -> None:
    percs = np.percentile(arr, [5, 10, 25, 50, 75, 90, 95])
    suf = f" {unit}" if unit else ""
    print(f"\n{label}:")
    print(f"  N = {len(arr)}")
    print(f"  mean  = {arr.mean():.2f}{suf}   std = {arr.std():.2f}{suf}")
    print(f"  min   = {arr.min():.2f}{suf}   max = {arr.max():.2f}{suf}")
    print(f"  p5={percs[0]:.2f}  p10={percs[1]:.2f}  p25={percs[2]:.2f}  "
          f"p50={percs[3]:.2f}  p75={percs[4]:.2f}  p90={percs[5]:.2f}  p95={percs[6]:.2f}{suf}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract realistic cell-centre statistics from 3D training patches.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--patches_dir', nargs='+', required=True,
        help='One or more train split directories containing *_centres.npy files '
             '(e.g. data_live_node1_3d/train data_live_node2_3d/train)'
    )
    parser.add_argument(
        '--save_stats', type=str, default=None,
        help='If provided, write a YAML centre_generation_3d block to this file'
    )
    parser.add_argument(
        '--max_patches', type=int, default=None,
        help='Cap the number of patches processed (useful for quick checks)'
    )
    parser.add_argument(
        '--n_bins', type=int, default=16,
        help='Number of histogram bins for marginal KDE grids (default: 16)'
    )
    args = parser.parse_args()

    print("=" * 65)
    print("3D Training Centre Statistics")
    print("=" * 65)
    print("Scanning directories:")
    cell_counts, nn_dists, z_norm, y_norm, x_norm = _collect_statistics(
        args.patches_dir, max_patches=args.max_patches
    )

    if len(cell_counts) == 0:
        print("ERROR: No valid patches found.")
        sys.exit(1)

    # ── Print distributions ───────────────────────────────────────────────────
    _print_dist("Cell count N per patch", cell_counts, "cells")
    _print_dist("Per-cell nearest-neighbour distance", nn_dists, "voxels")

    # ── Marginal density histograms ───────────────────────────────────────────
    grid_z = _marginal_histogram(z_norm, args.n_bins)
    grid_y = _marginal_histogram(y_norm, args.n_bins)
    grid_x = _marginal_histogram(x_norm, args.n_bins)

    print(f"\nMarginal KDE grids ({args.n_bins} bins, normalised to sum=1):")
    np.set_printoptions(precision=4, suppress=True)
    print(f"  z: {grid_z}")
    print(f"  y: {grid_y}")
    print(f"  x: {grid_x}")

    # Check how non-uniform the grids are (peak/trough ratio)
    for name, g in [("z", grid_z), ("y", grid_y), ("x", grid_x)]:
        ratio = g.max() / g.min()
        print(f"  {name} peak/trough ratio: {ratio:.2f}  "
              f"({'significant spatial variation' if ratio > 2 else 'roughly uniform'})")

    # ── Recommended config values ─────────────────────────────────────────────
    n_mean   = float(cell_counts.mean())
    n_std    = float(cell_counts.std())
    n_min    = int(np.percentile(cell_counts, 5))
    n_max    = int(np.percentile(cell_counts, 95))
    # Use 5th percentile of per-cell NN distances as the safe exclusion radius
    min_dist = float(np.percentile(nn_dists, 5))

    print("\n" + "=" * 65)
    print("Recommended centre_generation_3d config values:")
    print("=" * 65)
    print(f"  n_mean:       {n_mean:.1f}")
    print(f"  n_std:        {n_std:.1f}")
    print(f"  n_min:        {n_min}")
    print(f"  n_max:        {n_max}")
    print(f"  min_distance: {min_dist:.2f}  (5th percentile of per-cell NN dists)")

    # ── Optionally save YAML ──────────────────────────────────────────────────
    if args.save_stats:
        stats_block = {
            'centre_generation_3d': {
                'border_margin':  10,
                'max_attempts':   50,
                'n_mean':         round(n_mean, 2),
                'n_std':          round(n_std, 2),
                'n_min':          n_min,
                'n_max':          n_max,
                'min_distance':   round(min_dist, 2),
                'density_grid_z': [round(float(v), 6) for v in grid_z],
                'density_grid_y': [round(float(v), 6) for v in grid_y],
                'density_grid_x': [round(float(v), 6) for v in grid_x],
            }
        }
        out_path = Path(args.save_stats)
        with open(out_path, 'w') as f:
            yaml.dump(stats_block, f, default_flow_style=False, sort_keys=False)
        print(f"\nSaved stats to: {out_path}")
        print("Paste the 'centre_generation_3d' block into configs/frame1_3d.yaml")

    print("=" * 65)


if __name__ == "__main__":
    main()
