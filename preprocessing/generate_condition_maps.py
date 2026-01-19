"""
Generate conditioning maps from cell centres.

This module creates multi-channel conditioning tensors from point locations:
1. Centre heatmap: Gaussian blobs at each centre
2. Distance-to-nearest-centre map: Euclidean distance field
3. Boundary likelihood map: Derived from Voronoi ridges via distance field
"""

import numpy as np
from scipy import ndimage
from scipy.spatial import distance_matrix, Voronoi
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_centre_heatmap(
    centres: np.ndarray,
    image_shape: Tuple[int, int],
    sigma: float = 3.0
) -> np.ndarray:
    """
    Generate a Gaussian heatmap from cell centres.
    
    Args:
        centres: Array of shape (N, 2) with (y, x) coordinates
        image_shape: (H, W) of output heatmap
        sigma: Standard deviation of Gaussian blobs
    
    Returns:
        heatmap: Float array of shape (H, W) with Gaussian blobs at centres
    
    Note:
        Multiple overlapping Gaussians are summed (not max-pooled).
        Values are NOT normalized to [0, 1] - peak value is 1.0 per Gaussian.
    """
    h, w = image_shape
    heatmap = np.zeros((h, w), dtype=np.float32)
    
    if len(centres) == 0:
        return heatmap
    
    # Create coordinate grids
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    
    # Add Gaussian for each centre
    for cy, cx in centres:
        # Compute squared distance from this centre
        dist_sq = (y_coords - cy) ** 2 + (x_coords - cx) ** 2
        
        # Gaussian formula: exp(-dist^2 / (2 * sigma^2))
        gaussian = np.exp(-dist_sq / (2 * sigma ** 2))
        
        # Accumulate (sum overlapping Gaussians)
        heatmap += gaussian
    
    return heatmap


def generate_distance_map(
    centres: np.ndarray,
    image_shape: Tuple[int, int],
    normalize: bool = True
) -> np.ndarray:
    """
    Generate Euclidean distance-to-nearest-centre map.
    
    Args:
        centres: Array of shape (N, 2) with (y, x) coordinates
        image_shape: (H, W) of output map
        normalize: If True, normalize distances to [0, 1] using max distance
    
    Returns:
        distance_map: Float array of shape (H, W)
    
    Note:
        This is a continuous field representing distance to the nearest cell centre.
        Used as a geometric prior for the diffusion model.
    """
    h, w = image_shape
    
    # Handle empty centres case
    if len(centres) == 0:
        # Return maximum distance everywhere
        return np.ones((h, w), dtype=np.float32) * np.sqrt(h**2 + w**2)
    
    # Create coordinate grids
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    coords = np.stack([y_coords.ravel(), x_coords.ravel()], axis=1)  # (H*W, 2)
    
    # Compute pairwise distances: (H*W, N)
    distances = distance_matrix(coords, centres)
    
    # Take minimum distance to any centre
    min_distances = np.min(distances, axis=1)
    
    # Reshape back to image
    distance_map = min_distances.reshape(h, w).astype(np.float32)
    
    # Normalize to [0, 1]
    if normalize and distance_map.max() > 0:
        distance_map = distance_map / distance_map.max()
    
    return distance_map


def generate_boundary_map(
    centres: np.ndarray,
    image_shape: Tuple[int, int],
    sigma: float = 2.0,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Generate boundary likelihood map from cell centres.
    
    This uses a soft-assignment approach based on distance ratios:
    - Points equidistant from two centres are likely boundaries
    - Computed via entropy of distance-based assignment probabilities
    
    Args:
        centres: Array of shape (N, 2) with (y, x) coordinates
        image_shape: (H, W) of output map
        sigma: Temperature parameter for soft assignment
        threshold: Not used currently, kept for API compatibility
    
    Returns:
        boundary_map: Float array of shape (H, W) in [0, 1]
    
    Note:
        Alternative implementation could use Voronoi ridges directly,
        but this soft approach is more robust for sparse centres.
    """
    h, w = image_shape
    
    # Handle edge cases
    if len(centres) == 0:
        return np.zeros((h, w), dtype=np.float32)
    
    if len(centres) == 1:
        # Single cell: no boundaries
        return np.zeros((h, w), dtype=np.float32)
    
    # Create coordinate grids
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    coords = np.stack([y_coords.ravel(), x_coords.ravel()], axis=1)  # (H*W, 2)
    
    # Compute pairwise distances: (H*W, N)
    distances = distance_matrix(coords, centres)
    
    # Soft assignment via softmax over negative distances
    # Points close to a centre get high probability for that centre
    # Points equidistant get similar probabilities -> high entropy
    probs = np.exp(-distances / sigma)  # (H*W, N)
    probs = probs / (probs.sum(axis=1, keepdims=True) + 1e-10)  # Normalize
    
    # Compute entropy: -sum(p * log(p))
    # High entropy = uncertain assignment = likely boundary
    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    
    # Normalize entropy to [0, 1]
    # Maximum entropy for N centres is log(N)
    max_entropy = np.log(len(centres))
    boundary_map = (entropy / max_entropy).reshape(h, w).astype(np.float32)
    
    return boundary_map


def generate_conditioning_maps(
    centres: np.ndarray,
    image_shape: Tuple[int, int],
    heatmap_sigma: float = 3.0,
    boundary_sigma: float = 2.0
) -> np.ndarray:
    """
    Generate all conditioning maps as a stacked tensor.
    
    Args:
        centres: Array of shape (N, 2) with (y, x) coordinates
        image_shape: (H, W) of output maps
        heatmap_sigma: Sigma for centre heatmap Gaussians
        boundary_sigma: Temperature for boundary soft-assignment
    
    Returns:
        conditioning: Float array of shape (3, H, W) with:
            [0]: Centre heatmap
            [1]: Distance-to-nearest-centre map (normalized)
            [2]: Boundary likelihood map
    
    Note:
        This is the critical tensor that conditions the diffusion model.
        All channels are in [0, 1] range (approximately for heatmap).
    """
    h, w = image_shape
    
    # Generate individual maps
    heatmap = generate_centre_heatmap(centres, image_shape, sigma=heatmap_sigma)
    distance_map = generate_distance_map(centres, image_shape, normalize=True)
    boundary_map = generate_boundary_map(centres, image_shape, sigma=boundary_sigma)
    
    # Stack along channel dimension: (3, H, W)
    conditioning = np.stack([heatmap, distance_map, boundary_map], axis=0)
    
    return conditioning.astype(np.float32)


if __name__ == "__main__":
    # Example usage and visualization
    import matplotlib.pyplot as plt
    
    # Create synthetic centres
    np.random.seed(42)
    centres = np.random.rand(20, 2) * 200  # 20 cells in 200x200 image
    image_shape = (200, 200)
    
    # Generate conditioning maps
    conditioning = generate_conditioning_maps(centres, image_shape)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(conditioning[0], cmap='hot')
    axes[0].set_title('Centre Heatmap')
    axes[0].axis('off')
    
    axes[1].imshow(conditioning[1], cmap='viridis')
    axes[1].set_title('Distance Map')
    axes[1].axis('off')
    
    axes[2].imshow(conditioning[2], cmap='plasma')
    axes[2].set_title('Boundary Map')
    axes[2].axis('off')
    
    # Overlay centres
    for ax in axes:
        ax.scatter(centres[:, 1], centres[:, 0], c='cyan', s=20, marker='x')
    
    plt.tight_layout()
    plt.savefig('conditioning_maps_example.png', dpi=150, bbox_inches='tight')
    logger.info("Saved example to conditioning_maps_example.png")
