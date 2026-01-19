"""
Extract overlapping patches from microscopy images with their corresponding centres.

Patches are extracted with configurable overlap and filtered by cell count.
This creates training samples of (image_patch, centres_list) pairs.
"""

import numpy as np
import tifffile
from pathlib import Path
from typing import List, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_patches_with_centres(
    image: np.ndarray,
    centres: np.ndarray,
    patch_size: int = 256,
    stride: int = 128,
    min_cells: int = 3
) -> List[Dict]:
    """
    Extract overlapping patches from an image along with centres in each patch.
    
    Args:
        image: Input image of shape (H, W) or (H, W, C)
        centres: Cell centres of shape (N, 2) with (y, x) coordinates
        patch_size: Size of square patches to extract
        stride: Step size for sliding window (stride < patch_size gives overlap)
        min_cells: Minimum number of cell centres required in a patch
    
    Returns:
        patches: List of dictionaries, each containing:
            - 'image': Image patch of shape (patch_size, patch_size)
            - 'centres': Centres in patch-relative coordinates (M, 2)
            - 'global_position': (top, left) position of patch in original image
            - 'num_cells': Number of cells in this patch
    
    Note:
        Patches at image borders are discarded if they would be incomplete.
        Centre coordinates are converted to patch-relative (0 to patch_size).
    """
    h, w = image.shape[:2]
    patches = []
    
    # Compute patch grid positions
    top_positions = range(0, h - patch_size + 1, stride)
    left_positions = range(0, w - patch_size + 1, stride)
    
    for top in top_positions:
        for left in left_positions:
            # Extract patch boundaries
            bottom = top + patch_size
            right = left + patch_size
            
            # Extract image patch
            if len(image.shape) == 2:
                image_patch = image[top:bottom, left:right]
            else:
                image_patch = image[top:bottom, left:right, :]
            
            # Find centres within this patch
            in_patch = (
                (centres[:, 0] >= top) &
                (centres[:, 0] < bottom) &
                (centres[:, 1] >= left) &
                (centres[:, 1] < right)
            )
            
            patch_centres = centres[in_patch].copy()
            
            # Filter by minimum cell count
            if len(patch_centres) < min_cells:
                continue
            
            # Convert centres to patch-relative coordinates
            patch_centres[:, 0] -= top
            patch_centres[:, 1] -= left
            
            # Store patch info
            patches.append({
                'image': image_patch,
                'centres': patch_centres,
                'global_position': (top, left),
                'num_cells': len(patch_centres)
            })
    
    logger.info(f"Extracted {len(patches)} patches from {h}x{w} image")
    return patches


def save_patch_dataset(
    patches: List[Dict],
    output_dir: Path,
    prefix: str = "patch"
) -> None:
    """
    Save patches and their centres to disk.
    
    Args:
        patches: List of patch dictionaries from extract_patches_with_centres
        output_dir: Directory to save patches
        prefix: Filename prefix for patches
    
    Saves:
        - {prefix}_{i}_image.npy: Image patch
        - {prefix}_{i}_centres.npy: Centres in patch coordinates
        - {prefix}_{i}_meta.npz: Metadata (global_position, num_cells)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, patch in enumerate(patches):
        base_name = f"{prefix}_{i:05d}"
        
        # Save image patch
        np.save(output_dir / f"{base_name}_image.npy", patch['image'])
        
        # Save centres
        np.save(output_dir / f"{base_name}_centres.npy", patch['centres'])
        
        # Save metadata
        np.savez(
            output_dir / f"{base_name}_meta.npz",
            global_position=patch['global_position'],
            num_cells=patch['num_cells']
        )
    
    logger.info(f"Saved {len(patches)} patches to {output_dir}")


def load_patch_dataset(
    patches_dir: Path,
    indices: List[int] = None
) -> List[Dict]:
    """
    Load patches from disk.
    
    Args:
        patches_dir: Directory containing saved patches
        indices: Optional list of patch indices to load (loads all if None)
    
    Returns:
        patches: List of patch dictionaries
    """
    patches_dir = Path(patches_dir)
    
    # Find all patches if indices not specified
    if indices is None:
        image_files = sorted(patches_dir.glob("*_image.npy"))
        indices = [
            int(f.stem.split('_')[-2])
            for f in image_files
        ]
    
    patches = []
    for i in indices:
        base_name = f"patch_{i:05d}"
        
        # Load components
        image = np.load(patches_dir / f"{base_name}_image.npy")
        centres = np.load(patches_dir / f"{base_name}_centres.npy")
        meta = np.load(patches_dir / f"{base_name}_meta.npz")
        
        patches.append({
            'image': image,
            'centres': centres,
            'global_position': tuple(meta['global_position']),
            'num_cells': int(meta['num_cells'])
        })
    
    logger.info(f"Loaded {len(patches)} patches from {patches_dir}")
    return patches


def visualize_patches(
    patches: List[Dict],
    num_samples: int = 9,
    save_path: str = None
) -> None:
    """
    Visualize a grid of patches with their centres overlaid.
    
    Args:
        patches: List of patch dictionaries
        num_samples: Number of patches to show
        save_path: Optional path to save figure
    """
    import matplotlib.pyplot as plt
    
    num_samples = min(num_samples, len(patches))
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(num_samples):
        patch = patches[i]
        image = patch['image']
        centres = patch['centres']
        
        # Display image
        axes[i].imshow(image, cmap='gray')
        
        # Overlay centres
        axes[i].scatter(
            centres[:, 1], centres[:, 0],
            c='cyan', s=30, marker='x', linewidths=2
        )
        
        axes[i].set_title(f"Patch {i} ({patch['num_cells']} cells)")
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved visualization to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python extract_patches.py <image_path> <centres_path> <output_dir>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    centres_path = sys.argv[2]
    output_dir = sys.argv[3]
    
    # Load data
    image = tifffile.imread(image_path)
    centres = np.load(centres_path)
    
    if image is None:
        logger.error(f"Failed to load image from {image_path}")
        sys.exit(1)
    
    # Extract patches
    patches = extract_patches_with_centres(
        image,
        centres,
        patch_size=256,
        stride=128,
        min_cells=3
    )
    
    # Save patches
    save_patch_dataset(patches, Path(output_dir), prefix="patch")
    
    # Visualize a few
    visualize_patches(patches, num_samples=9, save_path=Path(output_dir) / "preview.png")
