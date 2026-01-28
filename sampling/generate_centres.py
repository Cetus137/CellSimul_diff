"""
Generate realistic cell centres for inference.

Creates synthetic cell centre distributions that match training data statistics.
"""

import numpy as np
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_random_centres_poisson(
    image_shape: Tuple[int, int] = (256, 256),
    density: float = 0.0004,  # cells per pixel
    min_distance: float = 10.0,  # minimum pixels between centres
    border_margin: int = 20,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate random cell centres using Poisson disk sampling.
    
    Creates spatially distributed centres with minimum distance constraints,
    mimicking realistic cell distributions.
    
    Args:
        image_shape: (H, W) output image size
        density: Expected cells per pixel (0.0004 ≈ 26 cells in 256x256)
        min_distance: Minimum distance between centres in pixels
        border_margin: Minimum distance from image borders
        seed: Random seed for reproducibility
    
    Returns:
        centres: Array of shape (N, 2) with (y, x) coordinates
    
    Note:
        Default density of 0.0004 gives ~26 cells per 256x256 image,
        which is typical for sparse cell cultures. Adjust based on your data.
    """
    if seed is not None:
        np.random.seed(seed)
    
    h, w = image_shape
    expected_cells = int(h * w * density)
    
    centres = []
    max_attempts = expected_cells * 50  # Safety limit
    attempts = 0
    
    while len(centres) < expected_cells and attempts < max_attempts:
        attempts += 1
        
        # Generate random candidate
        y = np.random.uniform(border_margin, h - border_margin)
        x = np.random.uniform(border_margin, w - border_margin)
        candidate = np.array([y, x])
        
        # Check minimum distance to existing centres
        if len(centres) == 0:
            centres.append(candidate)
            continue
        
        centres_array = np.array(centres)
        distances = np.linalg.norm(centres_array - candidate, axis=1)
        
        if np.all(distances >= min_distance):
            centres.append(candidate)
    
    if len(centres) == 0:
        logger.warning("No centres generated - constraints too strict!")
        return np.empty((0, 2), dtype=np.float32)
    
    logger.info(f"Generated {len(centres)} centres (expected {expected_cells})")
    return np.array(centres, dtype=np.float32)


def generate_random_centres_simple(
    image_shape: Tuple[int, int] = (256, 256),
    num_cells: int = 20,
    border_margin: int = 20,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate random cell centres with uniform distribution.
    
    Simple approach for quick testing - no minimum distance constraints.
    
    Args:
        image_shape: (H, W) output image size
        num_cells: Number of cells to generate
        border_margin: Minimum distance from image borders
        seed: Random seed for reproducibility
    
    Returns:
        centres: Array of shape (N, 2) with (y, x) coordinates
    """
    if seed is not None:
        np.random.seed(seed)
    
    h, w = image_shape
    
    y_coords = np.random.uniform(border_margin, h - border_margin, num_cells)
    x_coords = np.random.uniform(border_margin, w - border_margin, num_cells)
    
    centres = np.stack([y_coords, x_coords], axis=1).astype(np.float32)
    
    logger.info(f"Generated {num_cells} centres (uniform random)")
    return centres


def generate_centres_from_training_distribution(
    image_shape: Tuple[int, int] = (256, 256),
    mean_cells: float = 20.0,
    std_cells: float = 8.0,
    mean_min_dist: float = 15.0,
    border_margin: int = 20,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate centres matching training data statistics.
    
    Samples number of cells from normal distribution and places them with
    realistic spacing constraints.
    
    Args:
        image_shape: (H, W) output image size
        mean_cells: Mean number of cells from training data
        std_cells: Standard deviation of cell count
        mean_min_dist: Mean minimum distance between cells from training
        border_margin: Minimum distance from borders
        seed: Random seed
    
    Returns:
        centres: Array of shape (N, 2) with (y, x) coordinates
    
    Note:
        Use scripts/analyze_training_stats.py to determine realistic
        values for mean_cells, std_cells, and mean_min_dist from your data.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Sample number of cells from normal distribution
    num_cells = max(3, int(np.random.normal(mean_cells, std_cells)))
    
    # Use Poisson disk sampling with training-derived minimum distance
    min_distance = max(5.0, mean_min_dist * 0.7)  # Slightly relaxed
    
    centres = generate_random_centres_poisson(
        image_shape=image_shape,
        density=num_cells / (image_shape[0] * image_shape[1]),
        min_distance=min_distance,
        border_margin=border_margin,
        seed=None  # Already seeded above
    )
    
    return centres


def load_centres_from_training(
    patches_dir: str,
    patch_index: int = 0
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Load actual centres from a training patch for testing.
    
    Useful for verifying that inference produces similar results to training data.
    
    Args:
        patches_dir: Path to processed patches directory
        patch_index: Index of patch to load (0-based)
    
    Returns:
        centres: Cell centres from training data
        image_shape: Shape of the patch
    """
    from pathlib import Path
    
    patches_path = Path(patches_dir)
    centre_files = sorted(patches_path.glob("*_centres.npy"))
    
    if patch_index >= len(centre_files):
        raise ValueError(f"Patch index {patch_index} out of range (max: {len(centre_files)-1})")
    
    centres = np.load(centre_files[patch_index])
    
    # Load corresponding image to get shape
    image_file = centre_files[patch_index].parent / centre_files[patch_index].name.replace("_centres", "_image")
    image = np.load(image_file)
    
    if image.ndim == 3:
        image_shape = image.shape[1:]  # (C, H, W) -> (H, W)
    else:
        image_shape = image.shape  # (H, W)
    
    logger.info(f"Loaded {len(centres)} centres from patch {patch_index}")
    return centres, image_shape


# Example usage and statistics
if __name__ == "__main__":
    print("Testing cell centre generation...")
    
    # Method 1: Simple uniform random
    centres1 = generate_random_centres_simple(num_cells=20, seed=42)
    print(f"\nUniform random: {len(centres1)} centres")
    print(f"  Range Y: [{centres1[:, 0].min():.1f}, {centres1[:, 0].max():.1f}]")
    print(f"  Range X: [{centres1[:, 1].min():.1f}, {centres1[:, 1].max():.1f}]")
    
    # Method 2: Poisson disk sampling
    centres2 = generate_random_centres_poisson(density=0.0003, min_distance=12, seed=42)
    print(f"\nPoisson disk: {len(centres2)} centres")
    
    # Method 3: From training distribution (example parameters)
    centres3 = generate_centres_from_training_distribution(
        mean_cells=20, std_cells=8, mean_min_dist=15, seed=42
    )
    print(f"\nTraining distribution: {len(centres3)} centres")
