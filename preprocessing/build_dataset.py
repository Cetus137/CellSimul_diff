"""
Build complete dataset from raw microscopy images and masks.

This orchestrates the full preprocessing pipeline:
1. Extract centres from all masks
2. Extract patches from all images
3. Create train/val/test splits
4. Save dataset index for efficient loading
"""

import numpy as np
import tifffile
from pathlib import Path
from typing import List, Dict, Tuple
import yaml
import json
from tqdm import tqdm
import logging

from .extract_centres import extract_centres_from_mask, filter_border_cells
from .extract_patches import extract_patches_with_centres, save_patch_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_image_mask_pair(
    image_path: Path,
    mask_path: Path,
    config: Dict
) -> List[Dict]:
    """
    Process a single image-mask pair into patches.
    
    Args:
        image_path: Path to microscopy image
        mask_path: Path to instance segmentation mask
        config: Configuration dictionary
    
    Returns:
        patches: List of patch dictionaries
    """
    # Load image and mask
    image = tifffile.imread(str(image_path))
    mask = tifffile.imread(str(mask_path))
    
    if image is None or mask is None:
        logger.warning(f"Failed to load {image_path} or {mask_path}, skipping")
        return []
    
    # Extract centres from mask
    centres = extract_centres_from_mask(
        mask,
        min_area=config['preprocessing']['min_cell_area'],
        max_area=config['preprocessing']['max_cell_area']
    )
    
    # Filter border cells
    centres = filter_border_cells(centres, mask.shape[:2], border_margin=10)
    
    if len(centres) == 0:
        logger.warning(f"No valid centres found in {mask_path}, skipping")
        return []
    
    # Extract patches
    patches = extract_patches_with_centres(
        image,
        centres,
        patch_size=config['preprocessing']['patch_size'],
        stride=config['preprocessing']['patch_stride'],
        min_cells=config['preprocessing']['min_cells_per_patch']
    )
    
    # Add source filename to each patch
    for patch in patches:
        patch['source_image'] = image_path.stem
    
    return patches


def build_dataset(config_path: str = "configs/data.yaml") -> None:
    """
    Build complete dataset from raw data.
    
    Args:
        config_path: Path to data configuration YAML
    
    Pipeline:
        1. Load configuration
        2. Find all image-mask pairs
        3. Extract patches from each pair
        4. Create train/val/test splits
        5. Save patches and create dataset index
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup paths
    images_dir = Path(config['raw_data']['images_dir'])
    masks_dir = Path(config['raw_data']['masks_dir'])
    output_dir = Path(config['processed_data']['patches_dir'])
    metadata_dir = Path(config['processed_data']['metadata_dir'])
    splits_dir = Path(config['splits']['splits_dir'])
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_suffix = config['raw_data']['image_suffix']
    image_files = sorted(images_dir.glob(f"*{image_suffix}"))
    
    if len(image_files) == 0:
        logger.error(f"No images found in {images_dir} with suffix {image_suffix}")
        return
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Process all image-mask pairs
    all_patches = []
    for image_path in tqdm(image_files, desc="Processing images"):
        # Construct corresponding mask path
        # Assumes mask filename is {mask_prefix}{image_name}
        mask_prefix = config['raw_data'].get('mask_prefix', 'masks_')
        mask_name = mask_prefix + image_path.name
        mask_path = masks_dir / mask_name
        
        if not mask_path.exists():
            logger.warning(f"Mask not found for {image_path.name}, skipping")
            continue
        
        # Process this pair
        patches = process_image_mask_pair(image_path, mask_path, config)
        all_patches.extend(patches)
    
    logger.info(f"Extracted {len(all_patches)} total patches")
    
    if len(all_patches) == 0:
        logger.error("No patches extracted! Check your data and configuration.")
        return
    
    # Create train/val/test splits
    splits = create_splits(
        len(all_patches),
        train_frac=config['splits']['train_fraction'],
        val_frac=config['splits']['val_fraction'],
        test_frac=config['splits']['test_fraction'],
        random_seed=config['splits']['random_seed']
    )
    
    # Save patches by split
    for split_name, indices in splits.items():
        split_patches = [all_patches[i] for i in indices]
        split_output_dir = output_dir / split_name
        
        save_patch_dataset(split_patches, split_output_dir, prefix="patch")
        
        logger.info(f"{split_name}: {len(indices)} patches")
    
    # Save dataset index (mapping of patch indices to files)
    dataset_index = {
        'total_patches': len(all_patches),
        'splits': {k: len(v) for k, v in splits.items()},
        'config': config
    }
    
    with open(metadata_dir / 'dataset_index.json', 'w') as f:
        json.dump(dataset_index, f, indent=2)
    
    # Save split indices
    for split_name, indices in splits.items():
        np.save(splits_dir / f"{split_name}_indices.npy", np.array(indices))
    
    logger.info(f"Dataset built successfully!")
    logger.info(f"  Train: {len(splits['train'])} patches")
    logger.info(f"  Val: {len(splits['val'])} patches")
    logger.info(f"  Test: {len(splits['test'])} patches")


def create_splits(
    num_samples: int,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    random_seed: int = 42
) -> Dict[str, np.ndarray]:
    """
    Create random train/val/test splits.
    
    Args:
        num_samples: Total number of samples
        train_frac: Fraction for training
        val_frac: Fraction for validation
        test_frac: Fraction for testing
        random_seed: Random seed for reproducibility
    
    Returns:
        splits: Dictionary with 'train', 'val', 'test' keys mapping to index arrays
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1"
    
    # Shuffle indices
    np.random.seed(random_seed)
    indices = np.random.permutation(num_samples)
    
    # Compute split points
    train_size = int(num_samples * train_frac)
    val_size = int(num_samples * val_frac)
    
    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    return {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices
    }


if __name__ == "__main__":
    import sys
    
    config_path = "configs/data.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    build_dataset(config_path)
