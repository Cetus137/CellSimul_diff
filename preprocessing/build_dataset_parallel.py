"""
Parallelized dataset building for large-scale preprocessing.

Uses multiprocessing to speed up patch extraction from thousands of images.
Optimized for high-core-count systems like Google Colab.
"""

import numpy as np
import tifffile
from pathlib import Path
from typing import List, Dict, Tuple
import yaml
import json
from tqdm import tqdm
import logging
from multiprocessing import Pool, cpu_count
from functools import partial

from .extract_centres import extract_centres_from_mask, filter_border_cells
from .extract_patches import extract_patches_with_centres, save_patch_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_single_image(
    args: Tuple[Path, Path, Dict]
) -> Tuple[List[Dict], str]:
    """
    Process a single image-mask pair (worker function for multiprocessing).
    
    Args:
        args: Tuple of (image_path, mask_path, config)
    
    Returns:
        Tuple of (patches, source_name)
    """
    image_path, mask_path, config = args
    
    try:
        # Load image and mask
        image = tifffile.imread(str(image_path))
        mask = tifffile.imread(str(mask_path))
        
        if image is None or mask is None:
            logger.warning(f"Failed to load {image_path.name}, skipping")
            return [], image_path.stem
        
        # Extract centres from mask
        centres = extract_centres_from_mask(
            mask,
            min_area=config['preprocessing']['min_cell_area'],
            max_area=config['preprocessing']['max_cell_area']
        )
        
        # Filter border cells
        centres = filter_border_cells(centres, mask.shape[:2], border_margin=10)
        
        if len(centres) == 0:
            return [], image_path.stem
        
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
        
        return patches, image_path.stem
        
    except Exception as e:
        logger.error(f"Error processing {image_path.name}: {e}")
        return [], image_path.stem


def build_dataset_parallel(
    config_path: str = "configs/data.yaml",
    num_workers: int = None,
    chunksize: int = 1
) -> None:
    """
    Build complete dataset from raw data using parallel processing.
    
    Args:
        config_path: Path to data configuration YAML
        num_workers: Number of parallel workers (None = auto-detect CPU count)
        chunksize: Number of images per worker batch (1 = process one at a time)
    
    Pipeline:
        1. Load configuration
        2. Find all image-mask pairs
        3. Extract patches in parallel using multiprocessing
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
    
    # Prepare image-mask pairs
    mask_prefix = config['raw_data'].get('mask_prefix', 'masks_')
    pairs = []
    
    for image_path in image_files:
        mask_name = mask_prefix + image_path.name
        mask_path = masks_dir / mask_name
        
        if not mask_path.exists():
            logger.warning(f"Mask not found for {image_path.name}, skipping")
            continue
        
        pairs.append((image_path, mask_path, config))
    
    logger.info(f"Found {len(pairs)} valid image-mask pairs")
    
    # Determine number of workers
    if num_workers is None:
        num_workers = cpu_count()
    
    logger.info(f"Using {num_workers} parallel workers")
    logger.info(f"Chunksize: {chunksize}")
    
    # Process all pairs in parallel
    all_patches = []
    successful = 0
    failed = 0
    
    with Pool(processes=num_workers) as pool:
        # Use imap_unordered for better progress tracking
        results = pool.imap_unordered(
            process_single_image,
            pairs,
            chunksize=chunksize
        )
        
        # Collect results with progress bar
        for patches, source_name in tqdm(results, total=len(pairs), desc="Processing images"):
            if len(patches) > 0:
                all_patches.extend(patches)
                successful += 1
            else:
                failed += 1
    
    logger.info(f"Processing complete:")
    logger.info(f"  Successful: {successful}")
    logger.info(f"  Failed/Empty: {failed}")
    logger.info(f"  Total patches: {len(all_patches)}")
    
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
    logger.info("Saving patches to disk...")
    for split_name, indices in splits.items():
        split_patches = [all_patches[i] for i in indices]
        split_output_dir = output_dir / split_name
        
        save_patch_dataset(split_patches, split_output_dir, prefix="patch")
        
        logger.info(f"{split_name}: {len(indices)} patches")
    
    # Save dataset index
    dataset_index = {
        'total_patches': len(all_patches),
        'splits': {k: len(v) for k, v in splits.items()},
        'config': config,
        'num_workers_used': num_workers,
        'num_source_images': successful
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
    logger.info(f"  Speedup: ~{num_workers}x (with {num_workers} cores)")


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
    num_workers = None  # Auto-detect
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    if len(sys.argv) > 2:
        num_workers = int(sys.argv[2])
    
    build_dataset_parallel(config_path, num_workers=num_workers)
