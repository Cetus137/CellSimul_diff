"""
Generate 3D cell centres with various spatial distribution strategies.
"""

import numpy as np
from typing import List, Optional, Tuple
from scipy.spatial import cKDTree


def generate_random_centres_simple3d(
    volume_shape: Tuple[int, int, int],
    num_cells: int,
    border_margin: int = 10,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate random 3D cell centres (uniform distribution).

    Note: no minimum-distance constraint — cells may overlap.
    Prefer generate_realistic_centres3d for production use.

    Args:
        volume_shape: (D, H, W) volume shape
        num_cells: Exact number of cells to generate
        border_margin: Pixels to exclude from edges
        seed: Random seed

    Returns:
        centres: (N, 3) array in (z, y, x) order
    """
    if seed is not None:
        np.random.seed(seed)

    d, h, w = volume_shape
    z = np.random.uniform(border_margin, d - border_margin, num_cells)
    y = np.random.uniform(border_margin, h - border_margin, num_cells)
    x = np.random.uniform(border_margin, w - border_margin, num_cells)

    return np.column_stack([z, y, x]).astype(np.float32)


def generate_random_centres_poisson3d(
    volume_shape: Tuple[int, int, int],
    density: float = 0.00001,
    min_distance: float = 8.0,
    border_margin: int = 10,
    max_attempts: int = 30,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate 3D cell centres using Poisson disk sampling.

    Args:
        volume_shape: (D, H, W) volume shape
        density: Expected cells per voxel (default ~20 cells in 128^3)
        min_distance: Minimum spacing between centres
        border_margin: Pixels to exclude from edges
        max_attempts: Attempts per point before giving up
        seed: Random seed

    Returns:
        centres: (N, 3) array in (z, y, x) order
    """
    if seed is not None:
        np.random.seed(seed)

    d, h, w = volume_shape
    valid_volume = (d - 2 * border_margin) * (h - 2 * border_margin) * (w - 2 * border_margin)
    expected_cells = int(valid_volume * density)

    # Initialise with first point
    z0 = np.random.uniform(border_margin, d - border_margin)
    y0 = np.random.uniform(border_margin, h - border_margin)
    x0 = np.random.uniform(border_margin, w - border_margin)

    centres = [[z0, y0, x0]]
    active_list = [0]

    while active_list and len(centres) < expected_cells * 2:
        idx = np.random.choice(active_list)
        z_ref, y_ref, x_ref = centres[idx]

        # Build tree once per active-list iteration, not per candidate (bug fix)
        tree = cKDTree(centres) if len(centres) > 1 else None

        found = False
        for _ in range(max_attempts):
            # Candidate in spherical annulus [r, 2r]
            angle_theta = np.random.uniform(0, 2 * np.pi)
            angle_phi = np.random.uniform(0, np.pi)
            r = np.random.uniform(min_distance, 2 * min_distance)

            z_new = z_ref + r * np.cos(angle_theta) * np.sin(angle_phi)
            y_new = y_ref + r * np.sin(angle_theta) * np.sin(angle_phi)
            x_new = x_ref + r * np.cos(angle_phi)

            if not (border_margin <= z_new < d - border_margin and
                    border_margin <= y_new < h - border_margin and
                    border_margin <= x_new < w - border_margin):
                continue

            if tree is not None:
                dist, _ = tree.query([z_new, y_new, x_new])
                if dist < min_distance:
                    continue

            centres.append([z_new, y_new, x_new])
            active_list.append(len(centres) - 1)
            found = True
            break

        if not found:
            active_list.remove(idx)

    return np.array(centres, dtype=np.float32)


def generate_centres_from_training_distribution3d(
    volume_shape: Tuple[int, int, int],
    mean_cells: float = 20.0,
    std_cells: float = 8.0,
    mean_min_dist: float = 8.0,
    border_margin: int = 10,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate 3D centres matching training data statistics (parametric, no KDE).

    Prefer generate_realistic_centres3d when measured statistics are available.

    Args:
        volume_shape: (D, H, W) volume shape
        mean_cells: Mean cell count from training data
        std_cells: Std deviation of cell count
        mean_min_dist: Average minimum distance between cells
        border_margin: Pixels to exclude from edges
        seed: Random seed

    Returns:
        centres: (N, 3) array in (z, y, x) order
    """
    if seed is not None:
        np.random.seed(seed)

    num_cells = max(1, int(np.random.normal(mean_cells, std_cells)))

    d, h, w = volume_shape
    valid_volume = (d - 2 * border_margin) * (h - 2 * border_margin) * (w - 2 * border_margin)
    density = num_cells / valid_volume

    return generate_random_centres_poisson3d(
        volume_shape=volume_shape,
        density=density,
        min_distance=mean_min_dist,
        border_margin=border_margin,
        seed=None,
    )


def _sample_from_marginal(
    weights: List[float],
    n_samples: int,
    coord_min: float,
    coord_max: float,
) -> np.ndarray:
    """
    Draw n_samples from a discrete marginal distribution via inverse-CDF.

    Args:
        weights: Probability mass per bin (any positive values; normalised internally).
        n_samples: Number of samples to draw.
        coord_min: Lower bound of the sampling range (voxels).
        coord_max: Upper bound of the sampling range (voxels).

    Returns:
        samples: (n_samples,) float32 array in [coord_min, coord_max].
    """
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    n_bins = len(weights)

    # CDF over bin edges [0, 1]
    cdf = np.concatenate([[0.0], np.cumsum(weights)])
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    # Uniform samples → normalised coordinate [0, 1] via linear interpolation
    u = np.random.uniform(0.0, 1.0, n_samples)
    normalised = np.interp(u, cdf, bin_edges)

    return (coord_min + normalised * (coord_max - coord_min)).astype(np.float32)


def _sample_from_joint3d(
    grid_3d: np.ndarray,
    n_samples: int,
    z_min: float, z_max: float,
    y_min: float, y_max: float,
    x_min: float, x_max: float,
) -> np.ndarray:
    """
    Draw n_samples positions from a joint 3D density histogram.

    Args:
        grid_3d:  (B, B, B) normalised probability array (sums ~1).
        n_samples: Number of (z, y, x) triples to draw.
        z_min / z_max: Output range for z coordinates (voxels).
        y_min / y_max: Output range for y coordinates (voxels).
        x_min / x_max: Output range for x coordinates (voxels).

    Returns:
        samples: (n_samples, 3) float32 array, columns = [z, y, x].
    """
    B = grid_3d.shape[0]
    flat = grid_3d.ravel().astype(np.float64)
    flat /= flat.sum()                             # exact normalisation

    # Sample flat bin indices according to the joint probability
    bin_indices = np.random.choice(flat.size, size=n_samples, p=flat)

    # Decode flat index → (iz, iy, ix)
    iz = bin_indices // (B * B)
    iy = (bin_indices // B) % B
    ix = bin_indices % B

    # Jitter uniformly within each bin (bin width = 1/B in normalised space)
    bw = 1.0 / B
    z_norm = iz * bw + np.random.uniform(0.0, bw, size=n_samples)
    y_norm = iy * bw + np.random.uniform(0.0, bw, size=n_samples)
    x_norm = ix * bw + np.random.uniform(0.0, bw, size=n_samples)

    z = (z_min + z_norm * (z_max - z_min)).astype(np.float32)
    y = (y_min + y_norm * (y_max - y_min)).astype(np.float32)
    x = (x_min + x_norm * (x_max - x_min)).astype(np.float32)

    return np.stack([z, y, x], axis=1)


def generate_realistic_centres3d(
    volume_shape: Tuple[int, int, int],
    n_mean: float,
    n_std: float,
    n_min: int,
    n_max: int,
    min_distance: float,
    density_grid_z: List[float],
    density_grid_y: List[float],
    density_grid_x: List[float],
    density_grid_3d: Optional[List[float]] = None,
    n_bins_joint: int = 16,
    border_margin: int = 10,
    max_attempts: int = 50,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate 3D cell centres from data-derived statistics using marginal KDE.

    Cell count N is sampled from a truncated Normal distribution.  Candidate
    positions are drawn via inverse-CDF sampling from per-axis marginal density
    histograms measured on real training patches, then accepted/rejected based
    on a minimum-distance constraint (Matern-hardcore-style thinning).

    The density_grid_* arrays are typically produced by
    ``scripts/analyze_training_stats.py --save_stats`` and stored in the config.

    Args:
        volume_shape:    (D, H, W) target volume shape.
        n_mean:          Mean cell count measured from training patches.
        n_std:           Std deviation of cell count.
        n_min:           Minimum allowed cell count (hard clip).
        n_max:           Maximum allowed cell count (hard clip).
        min_distance:    Minimum voxel distance between any two centres.
                         Use the 5th percentile of per-cell NN distances from
                         real patches (output of analyze_training_stats.py).
        density_grid_z:  Marginal histogram over z (any length >= 2, sums to 1).
        density_grid_y:  Marginal histogram over y.
        density_grid_x:  Marginal histogram over x.
        density_grid_3d: Optional joint 3D histogram (n_bins_joint^3 values,
                         row-major z/y/x order, as produced by
                         analyze_training_stats.py --n_bins_3d N).  When
                         provided this replaces the three independent marginals,
                         preserving spatial correlations between axes.
        n_bins_joint:    Cube root of len(density_grid_3d) (default 16).
        border_margin:   Voxels excluded from each edge.
        max_attempts:    Max fresh candidates tried before declaring the volume
                         full and returning whatever has been accepted so far.
        seed:            Random seed.

    Returns:
        centres: (N_actual, 3) float32 array in (z, y, x) order.
                 N_actual may be less than the sampled N if the volume is too
                 dense to accommodate all cells without violating min_distance.
    """
    if seed is not None:
        np.random.seed(seed)

    d, h, w = volume_shape
    z_min, z_max = float(border_margin), float(d - border_margin)
    y_min, y_max = float(border_margin), float(h - border_margin)
    x_min, x_max = float(border_margin), float(w - border_margin)

    # Pre-build joint grid if supplied
    _joint_grid: Optional[np.ndarray] = None
    if density_grid_3d is not None:
        _joint_grid = np.asarray(density_grid_3d, dtype=np.float32).reshape(
            n_bins_joint, n_bins_joint, n_bins_joint
        )

    # --- Sample target N from truncated Normal ---
    n_target = int(np.round(np.random.normal(n_mean, n_std)))
    n_target = int(np.clip(n_target, n_min, n_max))

    accepted: List[List[float]] = []
    tree: Optional[cKDTree] = None
    consecutive_failures = 0

    while len(accepted) < n_target:
        # Draw a batch of candidates from joint 3D grid (preferred) or
        # independent marginals (fallback when joint grid not available)
        batch = max(n_target * 4, 64)
        if _joint_grid is not None:
            candidates = _sample_from_joint3d(
                _joint_grid, batch,
                z_min, z_max, y_min, y_max, x_min, x_max,
            )
        else:
            cz = _sample_from_marginal(density_grid_z, batch, z_min, z_max)
            cy = _sample_from_marginal(density_grid_y, batch, y_min, y_max)
            cx = _sample_from_marginal(density_grid_x, batch, x_min, x_max)
            candidates = np.stack([cz, cy, cx], axis=1)  # (batch, 3)

        batch_accepted = 0
        for pt in candidates:
            if len(accepted) >= n_target:
                break

            if tree is not None:
                dist, _ = tree.query(pt)
                if dist < min_distance:
                    continue

            accepted.append(pt.tolist())
            # Rebuild tree only after acceptance (O(N log N) total, not per-candidate)
            tree = cKDTree(accepted)
            batch_accepted += 1

        if batch_accepted == 0:
            consecutive_failures += 1
            if consecutive_failures >= max_attempts:
                # Volume too dense to fit more cells — return what we have
                break
        else:
            consecutive_failures = 0

    if len(accepted) == 0:
        # Fallback: return a single centre at the volume midpoint
        return np.array([[d / 2.0, h / 2.0, w / 2.0]], dtype=np.float32)

    return np.array(accepted, dtype=np.float32)
