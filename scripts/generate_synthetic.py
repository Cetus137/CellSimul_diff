#!/usr/bin/env python3
"""
Generate synthetic cell images from trained diffusion model.

Simple command-line interface for inference.
"""

import argparse
import sys
from pathlib import Path

from sampling.inference_pipeline import CellSynthesizer
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic cell microscopy images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 100 images with Poisson distribution
  python scripts/generate_synthetic.py --checkpoint checkpoints/best.pt --num_samples 100
  
  # Generate with specific cell density
  python scripts/generate_synthetic.py --checkpoint checkpoints/best.pt --density 0.0005 --num_samples 50
  
  # Generate with fixed number of cells per image
  python scripts/generate_synthetic.py --checkpoint checkpoints/best.pt --method simple --num_cells 25
  
  # Use training distribution statistics
  python scripts/generate_synthetic.py --checkpoint checkpoints/best.pt --method training_dist --mean_cells 22 --std_cells 7
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    
    # Generation parameters
    parser.add_argument(
        '--num_samples',
        type=int,
        default=10,
        help='Number of images to generate (default: 10)'
    )
    
    parser.add_argument(
        '--method',
        type=str,
        choices=['simple', 'poisson', 'training_dist', 'from_file'],
        default='poisson',
        help='Cell centre generation method (default: poisson)'
    )
    
    parser.add_argument(
        '--centres_file',
        type=str,
        default=None,
        help='Path to .npy file containing centres (required when method=from_file)'
    )
    
    # Method-specific parameters
    parser.add_argument(
        '--num_cells',
        type=int,
        default=20,
        help='Number of cells per image (for method=simple, default: 20)'
    )
    
    parser.add_argument(
        '--density',
        type=float,
        default=0.0003,
        help='Cell density per pixel (for method=poisson, default: 0.0003 ≈ 20 cells/256²)'
    )
    
    parser.add_argument(
        '--min_distance',
        type=float,
        default=12.0,
        help='Minimum distance between cell centres in pixels (default: 12)'
    )
    
    parser.add_argument(
        '--mean_cells',
        type=float,
        default=20.0,
        help='Mean cell count (for method=training_dist, default: 20)'
    )
    
    parser.add_argument(
        '--std_cells',
        type=float,
        default=8.0,
        help='Std dev of cell count (for method=training_dist, default: 8)'
    )
    
    # Sampling parameters
    parser.add_argument(
        '--use_cfg',
        action='store_true',
        default=True,
        help='Use classifier-free guidance (default: True)'
    )
    
    parser.add_argument(
        '--no_cfg',
        action='store_true',
        help='Disable classifier-free guidance'
    )
    
    parser.add_argument(
        '--guidance_scale',
        type=float,
        default=3.0,
        help='CFG guidance scale (default: 3.0)'
    )
    
    # Output parameters
    parser.add_argument(
        '--output_dir',
        type=str,
        default='generated_samples',
        help='Output directory (default: generated_samples)'
    )
    
    parser.add_argument(
        '--prefix',
        type=str,
        default='synthetic',
        help='Filename prefix (default: synthetic)'
    )
    
    parser.add_argument(
        '--no_visualization',
        action='store_true',
        help='Skip saving PNG visualizations with centre overlays'
    )
    
    # Config files
    parser.add_argument(
        '--config',
        type=str,
        default='configs/frame1.yaml',
        help='Path to unified config (default: configs/frame1.yaml)'
    )
    
    # Other
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='Device to use (overrides config)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed (overrides config random_seed)'
    )
    
    args = parser.parse_args()

    # Load config to resolve defaults
    import yaml
    with open(args.config, 'r') as f:
        _cfg = yaml.safe_load(f)
    device = args.device or _cfg.get('device', 'cuda')
    seed   = args.seed   if args.seed is not None else _cfg.get('random_seed', 42)
    
    # Handle CFG flag
    use_cfg = args.use_cfg and not args.no_cfg
    
    # Print configuration
    logger.info("="*70)
    logger.info("Synthetic Cell Image Generation")
    logger.info("="*70)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Method: {args.method}")
    logger.info(f"Number of samples: {args.num_samples}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Classifier-free guidance: {use_cfg}")
    if use_cfg:
        logger.info(f"  Guidance scale: {args.guidance_scale}")
    logger.info("="*70)
    
    # Method-specific info
    if args.method == 'simple':
        logger.info(f"Parameters: {args.num_cells} cells per image")
    elif args.method == 'poisson':
        logger.info(f"Parameters: density={args.density}, min_distance={args.min_distance}")
    elif args.method == 'training_dist':
        logger.info(f"Parameters: mean={args.mean_cells}, std={args.std_cells}, min_distance={args.min_distance}")
    elif args.method == 'from_file':
        logger.info(f"Parameters: centres_file={args.centres_file}")
    logger.info("="*70)
    
    try:
        # Initialize synthesizer
        synthesizer = CellSynthesizer(
            checkpoint_path=args.checkpoint,
            config=args.config,
            device=device
        )
        
        # Prepare centre generation parameters
        centre_kwargs = {}
        if args.method == 'simple':
            centre_kwargs['num_cells'] = args.num_cells
        elif args.method == 'poisson':
            centre_kwargs['density'] = args.density
            centre_kwargs['min_distance'] = args.min_distance
        elif args.method == 'training_dist':
            centre_kwargs['mean_cells'] = args.mean_cells
            centre_kwargs['std_cells'] = args.std_cells
            centre_kwargs['min_distance'] = args.min_distance
        elif args.method == 'from_file':
            centre_kwargs['centres_file'] = args.centres_file
        
        # Generate and save each sample immediately
        logger.info(f"Generating {args.num_samples} samples (saving each on the fly)...")
        all_cell_counts = []
        for i in range(args.num_samples):
            sample_seed = None if seed is None else seed + i
            centres = synthesizer.generate_centres(
                method=args.method,
                seed=sample_seed,
                **centre_kwargs
            )
            result = synthesizer.generate_image(
                centres=centres,
                use_cfg=use_cfg,
                guidance_scale=args.guidance_scale
            )
            synthesizer.save_samples(
                [result['image']],
                [centres],
                output_dir=args.output_dir,
                prefix=args.prefix,
                save_centres=True,
                save_visualization=not args.no_visualization,
                start_idx=i,
            )
            all_cell_counts.append(len(centres))
            logger.info(f"  Saved sample {i + 1}/{args.num_samples}  "
                        f"({len(centres)} cells)")

        logger.info("="*70)
        logger.info("✓ Generation complete!")
        logger.info("="*70)
        logger.info(f"Generated {args.num_samples} images")
        logger.info(f"Cell counts: min={min(all_cell_counts)}, "
                   f"max={max(all_cell_counts)}, "
                   f"mean={sum(all_cell_counts)/len(all_cell_counts):.1f}")
        logger.info(f"Output: {args.output_dir}/")
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
