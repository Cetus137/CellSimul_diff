#!/usr/bin/env python3
"""
Generate synthetic 3D cell volumes from trained diffusion model.

Command-line interface for 3D inference.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import tifffile
import logging
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

from sampling.generate_centres3d import (
    generate_random_centres_simple3d,
    generate_random_centres_poisson3d,
    generate_centres_from_training_distribution3d
)
from sampling.sample_from_centres3d import load_model3d, sample_from_centres3d
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def save_3d_visualization(
    volume: np.ndarray,
    centres: np.ndarray,
    save_path: Path,
    heatmap: np.ndarray = None,
    slice_fractions: tuple = (0.5, 0.5, 0.5)
) -> None:
    """
    Create and save orthogonal slice visualization with centres overlaid.
    
    Args:
        volume: 3D volume (D, H, W)
        centres: Cell centres (N, 3) in (z, y, x) order
        save_path: Path to save PNG
        heatmap: Optional conditioning heatmap (D, H, W) to show side-by-side
        slice_fractions: Which slices to show (z_frac, y_frac, x_frac)
    """
    D, H, W = volume.shape
    z_frac, y_frac, x_frac = slice_fractions
    
    # Select slice indices
    z_slice = int(D * z_frac)
    y_slice = int(H * y_frac)
    x_slice = int(W * x_frac)
    
    # Extract slices from volume
    xy_slice = volume[z_slice, :, :]  # Z fixed, show Y-X (top-down view)
    xz_slice = volume[:, y_slice, :]  # Y fixed, show Z-X (front view)
    yz_slice = volume[:, :, x_slice]  # X fixed, show Z-Y (side view)
    
    # Determine number of rows (1 if no heatmap, 2 if heatmap provided)
    n_rows = 2 if heatmap is not None else 1
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
    
    # Make axes indexable even if only 1 row
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    z_tolerance = 5
    
    # Row 0: Generated Volume
    # XY slice (top-down)
    axes[0, 0].imshow(xy_slice, cmap='gray', origin='lower')
    mask = np.abs(centres[:, 0] - z_slice) < z_tolerance
    if mask.any():
        axes[0, 0].scatter(centres[mask, 2], centres[mask, 1], 
                           c='red', marker='+', s=100, linewidths=2, alpha=0.8)
    axes[0, 0].set_title(f'Generated - XY Slice (z={z_slice}/{D})')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    
    # XZ slice (front)
    axes[0, 1].imshow(xz_slice.T, cmap='gray', origin='lower', aspect='auto')
    mask = np.abs(centres[:, 1] - y_slice) < z_tolerance
    if mask.any():
        axes[0, 1].scatter(centres[mask, 2], centres[mask, 0], 
                           c='red', marker='+', s=100, linewidths=2, alpha=0.8)
    axes[0, 1].set_title(f'Generated - XZ Slice (y={y_slice}/{H})')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Z')
    
    # YZ slice (side)
    axes[0, 2].imshow(yz_slice.T, cmap='gray', origin='lower', aspect='auto')
    mask = np.abs(centres[:, 2] - x_slice) < z_tolerance
    if mask.any():
        axes[0, 2].scatter(centres[mask, 1], centres[mask, 0], 
                           c='red', marker='+', s=100, linewidths=2, alpha=0.8)
    axes[0, 2].set_title(f'Generated - YZ Slice (x={x_slice}/{W})')
    axes[0, 2].set_xlabel('Y')
    axes[0, 2].set_ylabel('Z')
    
    # Row 1: Heatmap Conditioning (if provided)
    if heatmap is not None:
        # Extract slices from heatmap
        hm_xy_slice = heatmap[z_slice, :, :]
        hm_xz_slice = heatmap[:, y_slice, :]
        hm_yz_slice = heatmap[:, :, x_slice]
        
        # XY slice
        axes[1, 0].imshow(hm_xy_slice, cmap='hot', origin='lower')
        mask = np.abs(centres[:, 0] - z_slice) < z_tolerance
        if mask.any():
            axes[1, 0].scatter(centres[mask, 2], centres[mask, 1], 
                               c='cyan', marker='+', s=100, linewidths=2, alpha=0.8)
        axes[1, 0].set_title(f'Heatmap - XY Slice (z={z_slice}/{D})')
        axes[1, 0].set_xlabel('X')
        axes[1, 0].set_ylabel('Y')
        
        # XZ slice
        axes[1, 1].imshow(hm_xz_slice.T, cmap='hot', origin='lower', aspect='auto')
        mask = np.abs(centres[:, 1] - y_slice) < z_tolerance
        if mask.any():
            axes[1, 1].scatter(centres[mask, 2], centres[mask, 0], 
                               c='cyan', marker='+', s=100, linewidths=2, alpha=0.8)
        axes[1, 1].set_title(f'Heatmap - XZ Slice (y={y_slice}/{H})')
        axes[1, 1].set_xlabel('X')
        axes[1, 1].set_ylabel('Z')
        
        # YZ slice
        axes[1, 2].imshow(hm_yz_slice.T, cmap='hot', origin='lower', aspect='auto')
        mask = np.abs(centres[:, 2] - x_slice) < z_tolerance
        if mask.any():
            axes[1, 2].scatter(centres[mask, 1], centres[mask, 0], 
                               c='cyan', marker='+', s=100, linewidths=2, alpha=0.8)
        axes[1, 2].set_title(f'Heatmap - YZ Slice (x={x_slice}/{W})')
        axes[1, 2].set_xlabel('Y')
        axes[1, 2].set_ylabel('Z')
    
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic 3D cell volumes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 10 volumes with Poisson distribution
  python scripts/generate_synthetic3d.py --checkpoint checkpoints/frame1_3d/best.pt --num_samples 10
  
  # Generate with specific cell density
  python scripts/generate_synthetic3d.py --checkpoint checkpoints/frame1_3d/best.pt --density 0.00001 --num_samples 5
  
  # Use from file
  python scripts/generate_synthetic3d.py --checkpoint checkpoints/frame1_3d/best.pt --method from_file --centres_file data/test/patch_00000_centres.npy
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to trained 3D model checkpoint'
    )
    
    # Generation parameters
    parser.add_argument(
        '--num_samples',
        type=int,
        default=10,
        help='Number of volumes to generate (default: 10)'
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
        help='Number of cells per volume (for method=simple, default: 20)'
    )
    
    parser.add_argument(
        '--density',
        type=float,
        default=0.00001,
        help='Cell density per voxel (for method=poisson, default: 0.00001 ≈ 20 cells/128³)'
    )
    
    parser.add_argument(
        '--min_distance',
        type=float,
        default=8.0,
        help='Minimum distance between cell centres in voxels (default: 8)'
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
        default='synthetic_cells_3d',
        help='Output directory (default: synthetic_cells_3d)'
    )
    
    parser.add_argument(
        '--prefix',
        type=str,
        default='synthetic_3d',
        help='Filename prefix (default: synthetic_3d)'
    )
    
    # Config files
    parser.add_argument(
        '--config',
        type=str,
        default='configs/frame1_3d.yaml',
        help='Path to unified config (default: configs/frame1_3d.yaml)'
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
    
    parser.add_argument(
        '--no_visualization',
        action='store_true',
        help='Disable visualization PNG generation (default: False)'
    )
    
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    device = args.device or cfg.get('device', 'cuda')
    seed = args.seed if args.seed is not None else cfg.get('random_seed', 42)
    volume_size = cfg['preprocessing']['patch_size']
    heatmap_sigma = cfg['preprocessing']['centre_heatmap_sigma']
    
    # Resolve active conditioning channels
    conditioning_cfg = cfg.get('unet', {}).get('conditioning')
    active_channels = {k: bool(v) for k, v in conditioning_cfg.items()} if conditioning_cfg else None
    
    # Handle CFG flag
    use_cfg = args.use_cfg and not args.no_cfg
    
    # Print configuration
    logger.info("="*70)
    logger.info("Synthetic 3D Cell Volume Generation")
    logger.info("="*70)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Method: {args.method}")
    logger.info(f"Number of samples: {args.num_samples}")
    logger.info(f"Volume size: {volume_size}³")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Classifier-free guidance: {use_cfg}")
    if use_cfg:
        logger.info(f"  Guidance scale: {args.guidance_scale}")
    logger.info("="*70)
    
    # Method-specific info
    if args.method == 'simple':
        logger.info(f"Parameters: {args.num_cells} cells per volume")
    elif args.method == 'poisson':
        logger.info(f"Parameters: density={args.density}, min_distance={args.min_distance}")
    elif args.method == 'training_dist':
        logger.info(f"Parameters: mean={args.mean_cells}, std={args.std_cells}, min_distance={args.min_distance}")
    elif args.method == 'from_file':
        logger.info(f"Parameters: centres_file={args.centres_file}")
    logger.info("="*70)
    
    try:
        # Load model
        logger.info("Loading 3D model...")
        model = load_model3d(args.checkpoint, args.config, device)
        logger.info("Model loaded successfully")
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate samples
        logger.info(f"Generating {args.num_samples} samples...")
        all_cell_counts = []
        
        volume_shape = (volume_size, volume_size, volume_size)
        
        for i in range(args.num_samples):
            sample_seed = None if seed is None else seed + i
            
            # Generate centres
            if args.method == 'simple':
                centres = generate_random_centres_simple3d(
                    volume_shape=volume_shape,
                    num_cells=args.num_cells,
                    border_margin=10,
                    seed=sample_seed
                )
            elif args.method == 'poisson':
                centres = generate_random_centres_poisson3d(
                    volume_shape=volume_shape,
                    density=args.density,
                    min_distance=args.min_distance,
                    border_margin=10,
                    seed=sample_seed
                )
            elif args.method == 'training_dist':
                centres = generate_centres_from_training_distribution3d(
                    volume_shape=volume_shape,
                    mean_cells=args.mean_cells,
                    std_cells=args.std_cells,
                    mean_min_dist=args.min_distance,
                    border_margin=10,
                    seed=sample_seed
                )
            elif args.method == 'from_file':
                if args.centres_file is None:
                    raise ValueError("--centres_file required for method=from_file")
                centres = np.load(args.centres_file)
            else:
                raise ValueError(f"Unknown method: {args.method}")
            
            # Generate volume
            volume, metadata = sample_from_centres3d(
                model=model,
                centres=centres,
                volume_size=volume_size,
                heatmap_sigma=heatmap_sigma,
                active_channels=active_channels,
                device=device,
                use_cfg=use_cfg,
                guidance_scale=args.guidance_scale
            )
            
            # Save volume as TIF
            volume_file = output_dir / f"{args.prefix}_{i:04d}.tif"
            tifffile.imwrite(str(volume_file), volume.astype(np.float32))
            
            # Save heatmap conditioning (channel 0 of condition_maps)
            heatmap = metadata['condition_maps'][0]  # (D, H, W)
            heatmap_file = output_dir / f"{args.prefix}_{i:04d}_heatmap.tif"
            tifffile.imwrite(str(heatmap_file), heatmap.astype(np.float32))
            
            # Save centres
            centres_file = output_dir / f"{args.prefix}_{i:04d}_centres.npy"
            np.save(str(centres_file), centres)
            
            # Save visualization
            if not args.no_visualization:
                viz_file = output_dir / f"{args.prefix}_{i:04d}_viz.png"
                save_3d_visualization(volume, centres, viz_file, heatmap=heatmap)
            
            all_cell_counts.append(len(centres))
            logger.info(f"  Saved sample {i + 1}/{args.num_samples}  "
                       f"({len(centres)} cells)")
        
        logger.info("="*70)
        logger.info("✓ Generation complete!")
        logger.info("="*70)
        logger.info(f"Generated {args.num_samples} volumes")
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
