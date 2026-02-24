"""
Create a split assignment file that maps cube file indices to train/val/test splits.

This file is used by array jobs to know which split folder to write patches to.

Usage:
    python preprocessing/create_split_assignment.py configs/frame1_3d.yaml
"""

import numpy as np
import yaml
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_split_assignment(config_path: str = "configs/frame1_3d.yaml") -> None:
    """
    Create split assignment file from cube file list.
    
    Assigns each cube file index to train/val/test based on configured fractions.
    Saves to <patches_dir>/split_assignment.txt
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    cubes_dir = Path(config['raw_data']['cubes_dir'])
    output_dir = Path(config['processed_data']['patches_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cube_suffix = config['raw_data'].get('cube_suffix', '.tif')
    mask_suffix = config['raw_data'].get('mask_suffix', '_masks.tif')

    # Discover cube files (exclude mask files)
    cube_files = sorted(
        f for f in cubes_dir.glob(f"*{cube_suffix}")
        if not f.name.endswith(mask_suffix)
    )
    
    if not cube_files:
        logger.error(f"No cube files found in {cubes_dir}")
        return
    
    n_files = len(cube_files)
    logger.info(f"Found {n_files} cube files")
    
    # Create splits
    train_frac = config['splits']['train_fraction']
    val_frac = config['splits']['val_fraction']
    test_frac = config['splits']['test_fraction']
    random_seed = config['splits']['random_seed']
    
    np.random.seed(random_seed)
    indices = np.random.permutation(n_files)
    
    n_train = int(n_files * train_frac)
    n_val = int(n_files * val_frac)
    
    # Assign splits
    split_assignment = [''] * n_files
    for idx in indices[:n_train]:
        split_assignment[idx] = 'train'
    for idx in indices[n_train:n_train + n_val]:
        split_assignment[idx] = 'val'
    for idx in indices[n_train + n_val:]:
        split_assignment[idx] = 'test'
    
    # Save to file
    assignment_file = output_dir / "split_assignment.txt"
    with open(assignment_file, 'w') as f:
        for i, (cube_file, split) in enumerate(zip(cube_files, split_assignment)):
            f.write(f"{i}\t{split}\t{cube_file.name}\n")
    
    logger.info("=" * 60)
    logger.info("SPLIT ASSIGNMENT CREATED")
    logger.info("=" * 60)
    logger.info(f"  Total files:  {n_files}")
    logger.info(f"  Train files:  {n_train} ({train_frac*100:.0f}%)")
    logger.info(f"  Val files:    {n_val} ({val_frac*100:.0f}%)")
    logger.info(f"  Test files:   {n_files - n_train - n_val} ({test_frac*100:.0f}%)")
    logger.info(f"  Saved to:     {assignment_file}")
    logger.info("=" * 60)


if __name__ == "__main__":
    import sys
    cfg = sys.argv[1] if len(sys.argv) > 1 else "configs/frame1_3d.yaml"
    create_split_assignment(cfg)
