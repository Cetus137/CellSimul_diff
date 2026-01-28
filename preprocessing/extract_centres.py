"""
Extract cell centres from instance segmentation masks.

This module computes centroids from instance masks, filtering out
invalid or edge-case objects based on size constraints.
"""

import numpy as np
from scipy import ndimage
from typing import List, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_centres_from_mask(
    mask: np.ndarray,
    min_area: int = 100,
    max_area: int = 10000,
) -> np.ndarray:
    """
    Extract cell centres from an instance segmentation mask.
    
    Args:
        mask: Instance mask where each cell has a unique integer ID (H, W)
        min_area: Minimum cell area in pixels (discard smaller)
        max_area: Maximum cell area in pixels (discard larger)
    
    Returns:
        centres: Array of shape (N, 2) with (y, x) coordinates of cell centres
    
    Note:
        Background is assumed to be 0. Cells with IDs > 0 are processed.
    """
    centres = []
    unique_ids = np.unique(mask)
    
    # Remove background
    unique_ids = unique_ids[unique_ids > 0]
    
    for cell_id in unique_ids:
        # Create binary mask for this cell
        cell_mask = (mask == cell_id)
        
        # Compute area
        area = np.sum(cell_mask)
        
        # Filter by size
        if area < min_area or area > max_area:
            continue
        
        # Compute centroid using center of mass
        # Returns (y, x) coordinates
        centroid = ndimage.center_of_mass(cell_mask)
        
        # Skip if centroid calculation failed
        if np.isnan(centroid[0]) or np.isnan(centroid[1]):
            logger.warning(f"Invalid centroid for cell {cell_id}, skipping")
            continue
        
        centres.append(centroid)
    
    # Ensure proper shape even when empty (0, 2) instead of (0,)
    if len(centres) == 0:
        return np.empty((0, 2), dtype=np.float32)
    
    return np.array(centres, dtype=np.float32)


def filter_border_cells(
    centres: np.ndarray,
    image_shape: Tuple[int, int],
    border_margin: int = 10
) -> np.ndarray:
    """
    Remove cells whose centres are too close to image borders.
    
    This is important for patch extraction to avoid incomplete cells.
    
    Args:
        centres: Array of shape (N, 2) with (y, x) coordinates
        image_shape: (H, W) of the image
        border_margin: Minimum distance from border in pixels
    
    Returns:
        filtered_centres: Array of valid centres
    """
    # Handle empty input
    if len(centres) == 0:
        return centres
    
    h, w = image_shape
    
    # Boolean mask for valid centres
    valid = (
        (centres[:, 0] >= border_margin) &
        (centres[:, 0] < h - border_margin) &
        (centres[:, 1] >= border_margin) &
        (centres[:, 1] < w - border_margin)
    )
    
    return centres[valid]


def save_centres(centres: np.ndarray, output_path: str) -> None:
    """Save centres to a .npy file."""
    np.save(output_path, centres)
    logger.info(f"Saved {len(centres)} centres to {output_path}")


def load_centres(path: str) -> np.ndarray:
    """Load centres from a .npy file."""
    centres = np.load(path)
    logger.info(f"Loaded {len(centres)} centres from {path}")
    return centres


if __name__ == "__main__":
    # Example usage
    import tifffile
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python extract_centres.py <mask_path> <output_path>")
        sys.exit(1)
    
    mask_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # Load mask
    mask = tifffile.imread(mask_path)
    
    if mask is None:
        logger.error(f"Failed to load mask from {mask_path}")
        sys.exit(1)
    
    # Extract centres
    centres = extract_centres_from_mask(mask)
    
    # Filter border cells
    centres = filter_border_cells(centres, mask.shape[:2])
    
    # Save
    save_centres(centres, output_path)
    
    logger.info(f"Extracted {len(centres)} valid centres")
