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

from sampling.generate_centres3d import (
    generate_random_centres_simple3d,
    generate_random_centres_poisson3d,
    generate_centres_from_training_distribution3d,
    generate_realistic_centres3d,
)
from sampling.sample_from_centres3d import load_model3d, sample_from_centres3d, sample_batch_from_centres3d
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
        choices=['simple', 'poisson', 'training_dist', 'from_file', 'realistic'],
        default='realistic',
        help='Cell centre generation method (default: realistic)'
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

    parser.add_argument(
        '--ddim_steps',
        type=int,
        default=200,
        help=(
            'DDIM denoising steps (default: 200, ~5x faster than DDPM-1000). '
            'Set to 100 for ~10x or 50 for ~20x speedup. '
            'Use 0 to fall back to the original full DDPM-1000.'
        )
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
        '--batch_size',
        type=int,
        default=1,
        help=(
            'Number of volumes to denoise in a single GPU pass (default: 1). '
            'Set to 2 for ~2x throughput at ~2x peak VRAM cost. '
            'Recommended: 2 on A100/RTX8000, 1 on V100 if VRAM is limited.'
        )
    )
    
    parser.add_argument(
        '--no_visualization',
        action='store_true',
        help='Disable visualization PNG generation (default: False)'
    )

    parser.add_argument(
        '--no_heatmap',
        action='store_true',
        help='Disable saving of _heatmap.tif conditioning files (default: False)'
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

    # Load data-derived centre generation statistics (populated by analyze_training_stats.py)
    cg_cfg = cfg.get('centre_generation_3d', {})
    realistic_params = {
        'n_mean':          cg_cfg.get('n_mean',        20.0),
        'n_std':           cg_cfg.get('n_std',          8.0),
        'n_min':           cg_cfg.get('n_min',          5),
        'n_max':           cg_cfg.get('n_max',          60),
        'min_distance':    cg_cfg.get('min_distance',   8.0),
        'density_grid_z':  cg_cfg.get('density_grid_z', [1.0] * 16),
        'density_grid_y':  cg_cfg.get('density_grid_y', [1.0] * 16),
        'density_grid_x':  cg_cfg.get('density_grid_x', [1.0] * 16),
        'density_grid_3d': cg_cfg.get('density_grid_3d', None),
        'n_bins_joint':    cg_cfg.get('n_bins_joint',   16),
        'border_margin':   cg_cfg.get('border_margin',  10),
        'max_attempts':    cg_cfg.get('max_attempts',   50),
    }
    
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
        if args.num_samples > 1:
            cf = Path(args.centres_file)
            if cf.is_dir():
                logger.info(f"  from_file: cycling through .npy files in {cf}")
            else:
                logger.warning(
                    f"from_file: single file with num_samples={args.num_samples}. "
                    "Each sample will receive Gaussian-jittered centres (sigma=0 voxel). "
                    "Pass a directory of .npy files to use distinct centre sets."
                )
    elif args.method == 'realistic':
        logger.info(
            f"Parameters: n_mean={realistic_params['n_mean']:.1f}, "
            f"n_std={realistic_params['n_std']:.1f}, "
            f"min_distance={realistic_params['min_distance']:.1f}"
        )
        if not cfg.get('centre_generation_3d'):
            logger.warning(
                "centre_generation_3d not found in config — using fallback defaults. "
                "Run scripts/analyze_training_stats.py --save_stats to populate real values."
            )
    logger.info("="*70)
    
    try:
        # Load model
        logger.info("Loading 3D model...")
        model = load_model3d(args.checkpoint, args.config, device)
        logger.info("Model loaded successfully")

        ddim_steps_eff = args.ddim_steps if args.ddim_steps and args.ddim_steps > 0 else None
        if ddim_steps_eff:
            logger.info(f"Sampler: DDIM ({ddim_steps_eff} steps)  [~{1000 // ddim_steps_eff}x faster than DDPM-1000]")
        else:
            logger.info("Sampler: DDPM (1000 steps)")

        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate samples
        batch_size = args.batch_size
        logger.info(f"Generating {args.num_samples} samples (batch_size={batch_size})...")
        all_cell_counts = []

        volume_shape = (volume_size, volume_size, volume_size)

        def _generate_centres(bi: int) -> np.ndarray:
            """Generate centres for sample index bi."""
            sample_seed = None if seed is None else seed + bi
            if args.method == 'simple':
                return generate_random_centres_simple3d(
                    volume_shape=volume_shape,
                    num_cells=args.num_cells,
                    border_margin=10,
                    seed=sample_seed,
                )
            elif args.method == 'poisson':
                return generate_random_centres_poisson3d(
                    volume_shape=volume_shape,
                    density=args.density,
                    min_distance=args.min_distance,
                    border_margin=10,
                    seed=sample_seed,
                )
            elif args.method == 'training_dist':
                return generate_centres_from_training_distribution3d(
                    volume_shape=volume_shape,
                    mean_cells=args.mean_cells,
                    std_cells=args.std_cells,
                    mean_min_dist=args.min_distance,
                    border_margin=10,
                    seed=sample_seed,
                )
            elif args.method == 'from_file':
                if args.centres_file is None:
                    raise ValueError("--centres_file required for method=from_file")
                cf_path = Path(args.centres_file)
                if cf_path.is_dir():
                    npy_files = sorted(cf_path.glob("*_centres.npy"))
                    if not npy_files:
                        raise ValueError(f"No *_centres.npy files found in {cf_path}")
                    return np.load(npy_files[bi % len(npy_files)])
                else:
                    c = np.load(cf_path).astype(np.float32)
                    if args.num_samples > 1 and bi > 0:
                        jitter = np.random.normal(0.0, 0.0, c.shape).astype(np.float32)
                        c = np.clip(c + jitter, 10, volume_size - 10)
                    return c
            elif args.method == 'realistic':
                return generate_realistic_centres3d(
                    volume_shape=volume_shape,
                    seed=sample_seed,
                    **realistic_params,
                )
            else:
                raise ValueError(f"Unknown method: {args.method}")

        i = 0
        while i < args.num_samples:
            batch_indices = list(range(i, min(i + batch_size, args.num_samples)))
            batch_centres = [_generate_centres(bi) for bi in batch_indices]

            # Single denoising pass for the whole batch
            volumes, metadatas = sample_batch_from_centres3d(
                model=model,
                centres_list=batch_centres,
                volume_size=volume_size,
                heatmap_sigma=heatmap_sigma,
                active_channels=active_channels,
                device=device,
                use_cfg=use_cfg,
                guidance_scale=args.guidance_scale,
                ddim_steps=ddim_steps_eff,
            )

            # Build ordered channel names from active_channels so filenames match content
            _ch_order = ['heatmap', 'distance']
            _active_ch_names = [k for k in _ch_order if (active_channels or {}).get(k, True)]

            # Save each volume in the batch
            for j, (bi, volume, metadata) in enumerate(
                zip(batch_indices, volumes, metadatas)
            ):
                centres = batch_centres[j]
                cond_maps = metadata['condition_maps']  # (n_geom, D, H, W)

                volume_file  = output_dir / f"{args.prefix}_{bi:04d}.tif"
                centres_file = output_dir / f"{args.prefix}_{bi:04d}_centres.npy"

                tifffile.imwrite(str(volume_file), volume.astype(np.float32))
                np.save(str(centres_file), centres)

                if not args.no_heatmap:
                    # Save every active conditioning channel with its proper name
                    for ch_idx, ch_name in enumerate(_active_ch_names):
                        if ch_idx < cond_maps.shape[0]:
                            ch_file = output_dir / f"{args.prefix}_{bi:04d}_cond_{ch_name}.tif"
                            tifffile.imwrite(str(ch_file), cond_maps[ch_idx].astype(np.float32))

                if not args.no_visualization:
                    viz_file = output_dir / f"{args.prefix}_{bi:04d}_viz.png"
                    # Pass first channel to visualization (used for slice overlay row)
                    save_3d_visualization(volume, centres, viz_file, heatmap=cond_maps[0])

                all_cell_counts.append(len(centres))
                logger.info(
                    f"  Saved sample {bi + 1}/{args.num_samples}  ({len(centres)} cells)"
                )

            i += len(batch_indices)
        
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
