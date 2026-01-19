"""
Main script for data preprocessing.

Orchestrates the entire preprocessing pipeline from raw images to training-ready patches.
"""

import argparse
import yaml
from pathlib import Path
import logging

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.build_dataset import build_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess microscopy images for diffusion model training"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/data.yaml',
        help='Path to data configuration file'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force reprocessing even if processed data exists'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check if raw data exists
    images_dir = Path(config['raw_data']['images_dir'])
    masks_dir = Path(config['raw_data']['masks_dir'])
    
    if not images_dir.exists():
        logger.error(f"Images directory not found: {images_dir}")
        logger.info("Please place your microscopy images in data/raw/images/")
        return
    
    if not masks_dir.exists():
        logger.error(f"Masks directory not found: {masks_dir}")
        logger.info("Please place your instance segmentation masks in data/raw/masks/")
        return
    
    # Check if processed data already exists
    patches_dir = Path(config['processed_data']['patches_dir'])
    if patches_dir.exists() and not args.force:
        logger.warning(f"Processed data already exists at {patches_dir}")
        logger.warning("Use --force to reprocess")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            logger.info("Preprocessing cancelled")
            return
    
    # Run preprocessing pipeline
    logger.info("="*60)
    logger.info("Starting preprocessing pipeline")
    logger.info("="*60)
    logger.info(f"Images directory: {images_dir}")
    logger.info(f"Masks directory: {masks_dir}")
    logger.info(f"Output directory: {patches_dir}")
    logger.info(f"Patch size: {config['preprocessing']['patch_size']}")
    logger.info(f"Patch stride: {config['preprocessing']['patch_stride']}")
    logger.info(f"Min cells per patch: {config['preprocessing']['min_cells_per_patch']}")
    logger.info("="*60)
    
    try:
        build_dataset(str(config_path))
        logger.info("="*60)
        logger.info("Preprocessing completed successfully!")
        logger.info("="*60)
        logger.info("Next steps:")
        logger.info("  1. Review the generated patches")
        logger.info("  2. Adjust configuration if needed")
        logger.info("  3. Run: python scripts/train.py")
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise


if __name__ == "__main__":
    main()
