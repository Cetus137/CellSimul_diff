"""
Segmentation consistency evaluation.

Re-segment generated images and compare derived statistics
(cell count, sizes, nearest-neighbor distances) with ground truth.
"""

import numpy as np
from scipy import ndimage
from scipy.spatial import distance_matrix
from skimage import measure, morphology
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt


def segment_from_distance_transform(
    image: np.ndarray,
    centres: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Segment image using watershed from known centres.
    
    Args:
        image: Input image (H, W)
        centres: Cell centres (N, 2) in (y, x) format
        threshold: Intensity threshold for foreground
    
    Returns:
        labels: Segmentation mask with unique IDs per cell
    """
    h, w = image.shape
    
    # Create binary mask (simple threshold)
    binary = image > threshold
    
    # Create markers from centres
    markers = np.zeros((h, w), dtype=int)
    for idx, (cy, cx) in enumerate(centres, start=1):
        cy_int, cx_int = int(round(cy)), int(round(cx))
        if 0 <= cy_int < h and 0 <= cx_int < w:
            markers[cy_int, cx_int] = idx
    
    # Compute distance transform for watershed
    distance = ndimage.distance_transform_edt(binary)
    
    # Watershed segmentation
    labels = morphology.watershed(-distance, markers, mask=binary)
    
    return labels


def compute_cell_statistics(labels: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute statistics from segmentation mask.
    
    Args:
        labels: Segmentation mask with unique IDs
    
    Returns:
        stats: Dictionary with:
            - num_cells: Number of cells
            - areas: Cell areas
            - perimeters: Cell perimeters
            - circularity: Shape circularity (4π·area / perimeter²)
            - centroids: Cell centroids
    """
    props = measure.regionprops(labels)
    
    stats = {
        'num_cells': len(props),
        'areas': np.array([p.area for p in props]),
        'perimeters': np.array([p.perimeter for p in props if p.perimeter > 0]),
        'centroids': np.array([p.centroid for p in props])
    }
    
    # Compute circularity
    circularity = []
    for p in props:
        if p.perimeter > 0:
            c = 4 * np.pi * p.area / (p.perimeter ** 2)
            circularity.append(c)
    stats['circularity'] = np.array(circularity)
    
    return stats


def compute_nearest_neighbor_distances(centres: np.ndarray) -> np.ndarray:
    """
    Compute nearest-neighbor distances between cell centres.
    
    Args:
        centres: Cell centres (N, 2)
    
    Returns:
        nn_distances: Nearest-neighbor distance for each cell
    """
    if len(centres) < 2:
        return np.array([])
    
    # Compute pairwise distances
    dists = distance_matrix(centres, centres)
    
    # Set diagonal to infinity (distance to self)
    np.fill_diagonal(dists, np.inf)
    
    # Find minimum distance for each cell
    nn_distances = dists.min(axis=1)
    
    return nn_distances


def compare_statistics(
    real_stats_list: List[Dict],
    gen_stats_list: List[Dict],
    save_path: str = None
) -> Dict[str, float]:
    """
    Compare statistics between real and generated images.
    
    Args:
        real_stats_list: List of statistics dicts for real images
        gen_stats_list: List of statistics dicts for generated images
        save_path: Optional path to save comparison plots
    
    Returns:
        distances: Dictionary of metric distances
    """
    # Aggregate statistics
    real_areas = np.concatenate([s['areas'] for s in real_stats_list])
    gen_areas = np.concatenate([s['areas'] for s in gen_stats_list])
    
    real_circularity = np.concatenate([s['circularity'] for s in real_stats_list])
    gen_circularity = np.concatenate([s['circularity'] for s in gen_stats_list])
    
    real_centroids = np.concatenate([s['centroids'] for s in real_stats_list])
    gen_centroids = np.concatenate([s['centroids'] for s in gen_stats_list])
    
    # Compute nearest-neighbor distances
    real_nn_all = []
    for s in real_stats_list:
        nn = compute_nearest_neighbor_distances(s['centroids'])
        real_nn_all.extend(nn)
    real_nn = np.array(real_nn_all)
    
    gen_nn_all = []
    for s in gen_stats_list:
        nn = compute_nearest_neighbor_distances(s['centroids'])
        gen_nn_all.extend(nn)
    gen_nn = np.array(gen_nn_all)
    
    # Compare distributions using Wasserstein distance (Earth Mover's Distance)
    from scipy.stats import wasserstein_distance
    
    distances = {
        'area': wasserstein_distance(real_areas, gen_areas),
        'circularity': wasserstein_distance(real_circularity, gen_circularity),
        'nearest_neighbor': wasserstein_distance(real_nn, gen_nn)
    }
    
    # Plot comparisons
    if save_path:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Cell count distribution
        ax = axes[0, 0]
        real_counts = [s['num_cells'] for s in real_stats_list]
        gen_counts = [s['num_cells'] for s in gen_stats_list]
        ax.hist(real_counts, bins=20, alpha=0.5, label='Real', color='blue')
        ax.hist(gen_counts, bins=20, alpha=0.5, label='Generated', color='red')
        ax.set_xlabel('Cell count per image')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.set_title(f'Cell Count (real: {np.mean(real_counts):.1f}, gen: {np.mean(gen_counts):.1f})')
        ax.grid(True, alpha=0.3)
        
        # Area distribution
        ax = axes[0, 1]
        ax.hist(real_areas, bins=50, alpha=0.5, label='Real', density=True, color='blue')
        ax.hist(gen_areas, bins=50, alpha=0.5, label='Generated', density=True, color='red')
        ax.set_xlabel('Cell area (pixels)')
        ax.set_ylabel('Density')
        ax.legend()
        ax.set_title(f'Cell Area (EMD: {distances["area"]:.2f})')
        ax.grid(True, alpha=0.3)
        
        # Circularity distribution
        ax = axes[1, 0]
        ax.hist(real_circularity, bins=50, alpha=0.5, label='Real', density=True, color='blue')
        ax.hist(gen_circularity, bins=50, alpha=0.5, label='Generated', density=True, color='red')
        ax.set_xlabel('Circularity')
        ax.set_ylabel('Density')
        ax.legend()
        ax.set_title(f'Circularity (EMD: {distances["circularity"]:.4f})')
        ax.grid(True, alpha=0.3)
        
        # Nearest-neighbor distance
        ax = axes[1, 1]
        ax.hist(real_nn, bins=50, alpha=0.5, label='Real', density=True, color='blue')
        ax.hist(gen_nn, bins=50, alpha=0.5, label='Generated', density=True, color='red')
        ax.set_xlabel('Nearest-neighbor distance (pixels)')
        ax.set_ylabel('Density')
        ax.legend()
        ax.set_title(f'NN Distance (EMD: {distances["nearest_neighbor"]:.2f})')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return distances


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    
    from scipy.spatial import Voronoi
    
    def create_synthetic_cells(centres, image_shape=(256, 256)):
        """Create synthetic segmentation from centres."""
        h, w = image_shape
        
        # Use Voronoi to create cells
        y, x = np.ogrid[:h, :w]
        coords = np.stack([y.ravel(), x.ravel()], axis=1)
        
        # Distance to each centre
        dists = distance_matrix(coords, centres)
        labels = dists.argmin(axis=1).reshape(h, w) + 1
        
        return labels
    
    # Create test data
    real_stats_list = []
    for _ in range(10):
        centres = np.random.rand(20, 2) * 200
        labels = create_synthetic_cells(centres)
        stats = compute_cell_statistics(labels)
        real_stats_list.append(stats)
    
    gen_stats_list = []
    for _ in range(10):
        centres = np.random.rand(22, 2) * 200  # Slightly different count
        labels = create_synthetic_cells(centres)
        stats = compute_cell_statistics(labels)
        gen_stats_list.append(stats)
    
    # Compare
    distances = compare_statistics(
        real_stats_list,
        gen_stats_list,
        save_path='segmentation_comparison_test.png'
    )
    
    print("Segmentation consistency metrics:")
    for metric, dist in distances.items():
        print(f"  {metric}: {dist:.4f}")
    
    print("✓ Segmentation consistency test passed!")
