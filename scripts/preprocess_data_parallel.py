"""
Parallelized preprocessing script for large-scale datasets.

Use this for 1000+ images to dramatically speed up preprocessing.
Optimized for multi-core systems like Google Colab.
"""

import argparse
import yaml
from pathlib import Path
import logging
from multiprocessing import cpu_count

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.build_dataset_parallel import build_dataset_parallel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Parallel preprocessing for large microscopy datasets"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/data.yaml',
        help='Path to data configuration file'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help=f'Number of parallel workers (default: auto-detect, max: {cpu_count()})'
    )
    parser.add_argument(
        '--chunksize',
        type=int,
        default=1,
        help='Images per worker batch (default: 1)'
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
        
        # In non-interactive environments (like Colab), skip the prompt
        if sys.stdin.isatty():
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                logger.info("Preprocessing cancelled")
                return
        else:
            logger.info("Non-interactive mode detected, stopping. Use --force to override.")
            return
    
    # Determine workers
    num_workers = args.workers if args.workers else cpu_count()
    logger.info(f"System has {cpu_count()} CPU cores")
    logger.info(f"Using {num_workers} parallel workers")
    
    # Run parallel preprocessing pipeline
    logger.info("="*60)
    logger.info("Starting PARALLEL preprocessing pipeline")
    logger.info("="*60)
    logger.info(f"Images directory: {images_dir}")
    logger.info(f"Masks directory: {masks_dir}")
    logger.info(f"Output directory: {patches_dir}")
    logger.info(f"Patch size: {config['preprocessing']['patch_size']}")
    logger.info(f"Patch stride: {config['preprocessing']['patch_stride']}")
    logger.info(f"Min cells per patch: {config['preprocessing']['min_cells_per_patch']}")
    logger.info(f"Parallel workers: {num_workers}")
    logger.info(f"Chunksize: {args.chunksize}")
    logger.info(f"Expected speedup: ~{num_workers}x")
    logger.info("="*60)
    
    try:
        build_dataset_parallel(
            str(config_path),
            num_workers=num_workers,
            chunksize=args.chunksize
        )
        logger.info("="*60)
        logger.info("Parallel preprocessing completed successfully!")
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
