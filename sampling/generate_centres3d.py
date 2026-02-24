"""
Generate 3D cell centres with various spatial distribution strategies.
"""

import numpy as np
from typing import Tuple, Optional
from scipy.spatial import cKDTree


def generate_random_centres_simple3d(
    volume_shape: Tuple[int, int, int],
    num_cells: int,
    border_margin: int = 10,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate random 3D cell centres (uniform distribution).
    
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
    valid_volume = (d - 2*border_margin) * (h - 2*border_margin) * (w - 2*border_margin)
    expected_cells = int(valid_volume * density)
    
    # Initialize with first point
    z0 = np.random.uniform(border_margin, d - border_margin)
    y0 = np.random.uniform(border_margin, h - border_margin)
    x0 = np.random.uniform(border_margin, w - border_margin)
    
    centres = [[z0, y0, x0]]
    active_list = [0]
    
    while active_list and len(centres) < expected_cells * 2:  # Allow overshoot
        idx = np.random.choice(active_list)
        z_ref, y_ref, x_ref = centres[idx]
        
        found = False
        for _ in range(max_attempts):
            # Generate candidate in annulus [r, 2r]
            angle_theta = np.random.uniform(0, 2*np.pi)
            angle_phi = np.random.uniform(0, np.pi)
            r = np.random.uniform(min_distance, 2*min_distance)
            
            z_new = z_ref + r * np.cos(angle_theta) * np.sin(angle_phi)
            y_new = y_ref + r * np.sin(angle_theta) * np.sin(angle_phi)
            x_new = x_ref + r * np.cos(angle_phi)
            
            # Check bounds
            if not (border_margin <= z_new < d - border_margin and
                    border_margin <= y_new < h - border_margin and
                    border_margin <= x_new < w - border_margin):
                continue
            
            # Check distance to existing centres
            if len(centres) > 1:
                tree = cKDTree(centres)
                dist, _ = tree.query([z_new, y_new, x_new])
                if dist < min_distance:
                    continue
            
            centres.append([z_new, y_new, x_new])
            active_list.append(len(centres) - 1)
            found = True
            break
        
        if not found:
            active_list.remove(idx)
    
    centres = np.array(centres, dtype=np.float32)
    return centres


def generate_centres_from_training_distribution3d(
    volume_shape: Tuple[int, int, int],
    mean_cells: float = 20.0,
    std_cells: float = 8.0,
    mean_min_dist: float = 8.0,
    border_margin: int = 10,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate 3D centres matching training data statistics.
    
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
    
    # Sample number of cells from Gaussian
    num_cells = int(np.random.normal(mean_cells, std_cells))
    num_cells = max(1, num_cells)  # At least 1 cell
    
    # Use Poisson sampling with computed density
    d, h, w = volume_shape
    valid_volume = (d - 2*border_margin) * (h - 2*border_margin) * (w - 2*border_margin)
    density = num_cells / valid_volume
    
    return generate_random_centres_poisson3d(
        volume_shape=volume_shape,
        density=density,
        min_distance=mean_min_dist,
        border_margin=border_margin,
        seed=None  # Already seeded above
    )
