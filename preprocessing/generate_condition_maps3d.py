"""
Generate 3D conditioning maps from cell centre positions.

Two channels only (boundary channel omitted for 3D):
  0 — Centre heatmap:    Gaussian blobs at each centre
  1 — Distance map:      Euclidean distance to nearest centre (percentile-normalised)

All output maps are float32 in [0, 1].
Centres are expected in (z, y, x) order.
"""

import numpy as np
from scipy.spatial import cKDTree
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_centre_heatmap3d(
    centres: np.ndarray,
    volume_shape: Tuple[int, int, int],
    sigma: float = 3.0
) -> np.ndarray:
    """
    Generate a 3D Gaussian heatmap from cell centres.

    Args:
        centres: (N, 3) float array in (z, y, x) order.
        volume_shape: (D, H, W) of the output volume.
        sigma: Standard deviation of the Gaussian blobs (isotropic, in voxels).

    Returns:
        heatmap: float32 array of shape (D, H, W) in [0, 1].
    """
    D, H, W = volume_shape
    heatmap = np.zeros((D, H, W), dtype=np.float32)

    if len(centres) == 0:
        return heatmap

    # Coordinate grids: shape (D, H, W) each
    z_grid, y_grid, x_grid = np.meshgrid(
        np.arange(D), np.arange(H), np.arange(W), indexing='ij'
    )

    two_sigma_sq = 2.0 * sigma ** 2

    for cz, cy, cx in centres:
        dist_sq = (z_grid - cz) ** 2 + (y_grid - cy) ** 2 + (x_grid - cx) ** 2
        heatmap += np.exp(-dist_sq / two_sigma_sq)

    return np.clip(heatmap, 0.0, 1.0)


def generate_distance_map3d(
    centres: np.ndarray,
    volume_shape: Tuple[int, int, int],
    normalize: bool = True,
    percentile: float = 95.0
) -> np.ndarray:
    """
    Generate a 3D Euclidean distance-to-nearest-centre map.

    Uses a cKDTree for efficiency (avoids the O(N·D·H·W) distance matrix).

    Args:
        centres: (N, 3) float array in (z, y, x) order.
        volume_shape: (D, H, W) of the output volume.
        normalize: If True, percentile-normalise to [0, 1].
        percentile: Percentile used for normalisation (default 95).

    Returns:
        distance_map: float32 array of shape (D, H, W) in [0, 1] when
            normalize=True, otherwise raw Euclidean distances in voxels.
    """
    D, H, W = volume_shape

    if len(centres) == 0:
        return np.ones((D, H, W), dtype=np.float32)

    # Build flat coordinate grid (D*H*W, 3)
    z_grid, y_grid, x_grid = np.meshgrid(
        np.arange(D), np.arange(H), np.arange(W), indexing='ij'
    )
    coords = np.stack([z_grid.ravel(), y_grid.ravel(), x_grid.ravel()], axis=1)

    # Nearest-neighbour query
    tree = cKDTree(centres)
    min_dists, _ = tree.query(coords, k=1)  # (D*H*W,)

    distance_map = min_dists.reshape(D, H, W).astype(np.float32)

    if normalize:
        d_scale = np.percentile(distance_map, percentile)
        if d_scale < 1e-6:
            d_scale = distance_map.max()
        if d_scale < 1e-6:
            return np.zeros_like(distance_map)
        distance_map = np.clip(distance_map / d_scale, 0.0, 1.0)

    return distance_map


def generate_conditioning_maps3d(
    centres: np.ndarray,
    volume_shape: Tuple[int, int, int],
    heatmap_sigma: float = 3.0,
    distance_percentile: float = 95.0,
    active_channels: Optional[dict] = None
) -> np.ndarray:
    """
    Generate stacked 3D conditioning maps from cell centres.

    Channel order (when both active):
        [0] — Centre heatmap      (generated when active_channels['heatmap'] is True)
        [1] — Distance map        (generated when active_channels['distance'] is True)

    Boundary channel is intentionally omitted for the 3D model (entropy boundary
    requires O(N·D·H·W) memory which is prohibitive at 128³).

    Args:
        centres: (N, 3) float array in (z, y, x) order.
        volume_shape: (D, H, W) of each output channel.
        heatmap_sigma: Gaussian sigma for heatmap blobs (voxels).
        distance_percentile: Percentile for distance map normalisation.
        active_channels: Dict with boolean flags, e.g.
            {'heatmap': True, 'distance': True}.
            None → both channels active.

    Returns:
        conditioning: float32 array of shape (C, D, H, W) where
            C = number of active channels (1 or 2).
    """
    if active_channels is None:
        active_channels = {'heatmap': True, 'distance': True}

    maps = []

    if active_channels.get('heatmap', True):
        maps.append(
            generate_centre_heatmap3d(centres, volume_shape, sigma=heatmap_sigma)
        )

    if active_channels.get('distance', True):
        maps.append(
            generate_distance_map3d(
                centres, volume_shape,
                normalize=True,
                percentile=distance_percentile
            )
        )

    if len(maps) == 0:
        raise ValueError("active_channels: at least one channel must be enabled")

    return np.stack(maps, axis=0).astype(np.float32)  # (C, D, H, W)


if __name__ == "__main__":
    import time

    np.random.seed(42)
    centres = np.random.rand(30, 3) * 128  # 30 cells in a 128³ patch
    shape = (128, 128, 128)

    t0 = time.time()
    cond = generate_conditioning_maps3d(centres, shape)
    print(f"Generated conditioning in {time.time() - t0:.2f}s")
    print(f"Shape: {cond.shape}  dtype: {cond.dtype}")
    print(f"Heatmap range: [{cond[0].min():.3f}, {cond[0].max():.3f}]")
    print(f"Distance range: [{cond[1].min():.3f}, {cond[1].max():.3f}]")
