"""
Main sampling script.

Generate synthetic microscopy images from cell centres using a trained model.
"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
import logging

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from sampling.sample_from_centres import (
    load_model,
    sample_from_centres,
    visualize_sample,
    create_grid
)
from preprocessing.generate_condition_maps import generate_conditioning_maps

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate samples from trained diffusion model"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--model_config',
        type=str,
        default='configs/model.yaml',
        help='Path to model configuration'
    )
    parser.add_argument(
        '--data_config',
        type=str,
        default='configs/data.yaml',
        help='Path to data configuration'
    )
    parser.add_argument(
        '--centres',
        type=str,
        default=None,
        help='Path to .npy file with centres (optional, generates random if not provided)'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=16,
        help='Number of samples to generate'
    )
    parser.add_argument(
        '--num_cells',
        type=int,
        default=20,
        help='Number of cells per sample (if generating random centres)'
    )
    parser.add_argument(
        '--image_size',
        type=int,
        default=256,
        help='Image size (H=W)'
    )
    parser.add_argument(
        '--use_cfg',
        action='store_true',
        help='Use classifier-free guidance'
    )
    parser.add_argument(
        '--guidance_scale',
        type=float,
        default=3.0,
        help='CFG guidance scale'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='samples',
        help='Output directory for samples'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--save_grid',
        action='store_true',
        help='Save samples as a grid'
    )
    
    args = parser.parse_args()
    
    # Set device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = 'cpu'
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("Sampling from diffusion model")
    logger.info("="*60)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Device: {device}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Number of samples: {args.num_samples}")
    logger.info(f"Image size: {args.image_size}x{args.image_size}")
    logger.info(f"Classifier-free guidance: {args.use_cfg}")
    if args.use_cfg:
        logger.info(f"Guidance scale: {args.guidance_scale}")
    logger.info("="*60)
    
    # Load model
    logger.info("Loading model...")
    model = load_model(
        checkpoint_path=args.checkpoint,
        config_path=args.model_config,
        device=device
    )
    model.eval()
    
    # Load or generate centres
    if args.centres:
        logger.info(f"Loading centres from {args.centres}")
        centres_array = np.load(args.centres)
        
        # If single set of centres, replicate for num_samples
        if centres_array.ndim == 2:
            centres_list = [centres_array for _ in range(args.num_samples)]
        else:
            centres_list = [centres_array[i] for i in range(len(centres_array))]
    else:
        logger.info(f"Generating random centres ({args.num_cells} cells per image)")
        centres_list = []
        for i in range(args.num_samples):
            centres = np.random.rand(args.num_cells, 2) * args.image_size
            centres_list.append(centres)
    
    # Load data config for conditioning parameters
    with open(args.data_config, 'r') as f:
        data_config = yaml.safe_load(f)
    
    heatmap_sigma = data_config['preprocessing']['centre_heatmap_sigma']
    
    # Generate samples
    logger.info("Generating samples...")
    images = []
    conditioning_maps = []
    
    for i, centres in enumerate(centres_list):
        logger.info(f"Sampling {i+1}/{len(centres_list)}...")
        
        # Generate conditioning
        conditioning = generate_conditioning_maps(
            centres,
            (args.image_size, args.image_size),
            heatmap_sigma=heatmap_sigma
        )
        conditioning_maps.append(conditioning)
        
        # Sample image
        image = sample_from_centres(
            model,
            centres,
            image_shape=(args.image_size, args.image_size),
            heatmap_sigma=heatmap_sigma,
            use_cfg=args.use_cfg,
            guidance_scale=args.guidance_scale,
            device=device
        )
        images.append(image)
        
        # Save individual sample with conditioning
        visualize_sample(
            image,
            centres,
            conditioning=conditioning,
            save_path=output_dir / f'sample_{i:03d}.png'
        )
    
    logger.info(f"Saved {len(images)} individual samples")
    
    # Save grid if requested
    if args.save_grid:
        logger.info("Creating sample grid...")
        grid = create_grid(images)
        
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 12))
        plt.imshow(grid, cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_dir / 'grid.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info("Saved sample grid")
    
    # Save metadata
    metadata = {
        'checkpoint': str(args.checkpoint),
        'num_samples': args.num_samples,
        'image_size': args.image_size,
        'use_cfg': args.use_cfg,
        'guidance_scale': args.guidance_scale if args.use_cfg else None,
        'seed': args.seed
    }
    
    import json
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("="*60)
    logger.info("Sampling completed!")
    logger.info("="*60)
    logger.info(f"Samples saved to: {output_dir}")


if __name__ == "__main__":
    main()
