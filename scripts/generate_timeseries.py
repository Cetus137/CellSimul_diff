"""
Generate synthetic time-series of cell microscopy images with temporal dynamics.

This script generates a sequence of images where cell centres undergo small 
displacements between consecutive timepoints, simulating cell migration or drift.

Author: CellSimul Diffusion Project
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import tifffile
from skimage import exposure
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from sampling.inference_pipeline import CellSynthesizer
from sampling.generate_centres import (
    generate_random_centres_simple,
    generate_random_centres_poisson,
    generate_centres_from_training_distribution
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def displace_centres(
    centres: np.ndarray,
    displacement_std: float = 2.0,
    displacement_mean: float = 0.0,
    directional_bias: Optional[Tuple[float, float]] = None,
    boundary_margin: int = 20,
    image_shape: Tuple[int, int] = (256, 256),
    min_distance: float = 12.0,
    max_iterations: int = 50,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Apply small random displacements to cell centres with collision avoidance.
    
    Args:
        centres: Array of shape (N, 2) with (y, x) coordinates
        displacement_std: Standard deviation of displacement in pixels
        displacement_mean: Mean displacement magnitude
        directional_bias: Optional (dy, dx) bias for directional motion
        boundary_margin: Minimum distance from image boundary
        image_shape: (height, width) of image
        min_distance: Minimum allowed distance between centres
        max_iterations: Maximum attempts to resolve collisions
        seed: Random seed
    
    Returns:
        displaced_centres: New centres array with same shape
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    # Random displacement
    displacements = rng.normal(
        displacement_mean, 
        displacement_std, 
        size=centres.shape
    )
    
    # Add directional bias if specified
    if directional_bias is not None:
        displacements[:, 0] += directional_bias[0]  # y direction
        displacements[:, 1] += directional_bias[1]  # x direction
    
    # Apply displacement
    new_centres = centres + displacements
    
    # Resolve collisions - ensure minimum distance between centres
    for iteration in range(max_iterations):
        collisions_resolved = True
        
        for i in range(len(new_centres)):
            for j in range(i + 1, len(new_centres)):
                # Calculate distance
                diff = new_centres[i] - new_centres[j]
                dist = np.sqrt(np.sum(diff ** 2))
                
                if dist < min_distance and dist > 0:
                    # Push centres apart along the line connecting them
                    collisions_resolved = False
                    push_direction = diff / dist
                    push_amount = (min_distance - dist) / 2.0
                    
                    new_centres[i] += push_direction * push_amount
                    new_centres[j] -= push_direction * push_amount
        
        if collisions_resolved:
            break
    
    return new_centres


def generate_timeseries(
    synthesizer: CellSynthesizer,
    initial_centres: np.ndarray,
    num_timepoints: int,
    displacement_std: float = 2.0,
    displacement_mean: float = 0.0,
    directional_bias: Optional[Tuple[float, float]] = None,
    min_distance: float = 12.0,
    noise_correlation: float = 0.0,
    temporal_smoothness: float = 0.0,
    match_histograms: bool = False,
    use_cfg: bool = True,
    guidance_scale: float = 3.0,
    output_dir: str = './timeseries',
    prefix: str = 'timeseries',
    save_centres: bool = True,
    save_visualization: bool = True,
    save_stack: bool = True,
    seed: Optional[int] = None
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Generate a time-series of images with temporal cell dynamics.
    
    Args:
        synthesizer: CellSynthesizer instance
        initial_centres: Starting centres of shape (N, 2)
        num_timepoints: Number of timepoints to generate
        displacement_std: Std deviation of displacement per timestep
        displacement_mean: Mean displacement magnitude
        directional_bias: Optional (dy, dx) directional bias
        min_distance: Minimum allowed distance between centres
        noise_correlation: Spatial correlation between consecutive frames (0-1)
        temporal_smoothness: AR(1) smoothing coefficient for noise evolution (0-1)
            0.0 = independent noise per frame
            0.7 = moderate temporal smoothing
            0.9 = very smooth evolution
        match_histograms: Whether to apply histogram matching post-processing
            Uses first frame as reference to ensure consistent intensity distributions
        use_cfg: Whether to use classifier-free guidance
        guidance_scale: CFG scale if use_cfg=True
        output_dir: Directory to save results
        prefix: Prefix for output filenames
        save_centres: Whether to save centres at each timepoint
        save_visualization: Whether to save visualization overlays
        save_stack: Whether to save complete (T, Y, X) TIFF stack
        seed: Random seed for displacement
    
    Returns:
        images_list: List of generated images
        centres_list: List of centres at each timepoint
    """
    os.makedirs(output_dir, exist_ok=True)
    
    images_list = []
    centres_list = []
    current_centres = initial_centres.copy()
    
    logger.info(f"Generating time-series with {num_timepoints} timepoints")
    logger.info(f"Initial centres: {len(initial_centres)} cells")
    logger.info(f"Displacement: mean={displacement_mean}, std={displacement_std}")
    logger.info(f"Minimum distance: {min_distance} pixels")
    logger.info(f"Noise correlation (rho): {noise_correlation}")
    if directional_bias is not None:
        logger.info(f"Directional bias: dy={directional_bias[0]}, dx={directional_bias[1]}")
    
    # Create shared initial noise (x_T) for correlation
    import torch
    device = synthesizer.device
    H, W = synthesizer.image_size, synthesizer.image_size
    x_T = torch.randn((1, 1, H, W), device=device)
    
    # Base seeds for noise generation
    if seed is None:
        seed = 42
    seed_shared = seed
    
    for t in tqdm(range(num_timepoints), desc="Generating timepoints"):
        # Generate image using correlated sampling
        seed_unique = seed_shared + 10000 + t  # Different unique seed per frame
        
        result = synthesizer.generate_image_correlated(
            centres=current_centres,
            x_T=x_T,
            seed_shared=seed_shared,
            seed_unique=seed_unique,
            rho=noise_correlation,
            guidance_scale=guidance_scale if use_cfg else 0.0,
            return_conditioning=False
        )
        
        image = result['image']
        images_list.append(image)
        centres_list.append(current_centres.copy())
        
        # Save image
        image_filename = os.path.join(output_dir, f"{prefix}_t{t:04d}.tiff")
        from PIL import Image
        # Convert from [-1, 1] to [0, 255]
        image_uint8 = ((image + 1) / 2 * 255).astype(np.uint8)
        Image.fromarray(image_uint8).save(image_filename)
        
        # Save centres
        if save_centres:
            centres_filename = os.path.join(output_dir, f"{prefix}_t{t:04d}_centres.npy")
            np.save(centres_filename, current_centres)
        
        # Save visualization
        if save_visualization:
            import matplotlib.pyplot as plt
            viz_filename = os.path.join(output_dir, f"{prefix}_t{t:04d}_overlay.png")
            
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(image, cmap='gray', vmin=-1, vmax=1)
            ax.scatter(current_centres[:, 1], current_centres[:, 0], 
                      c='red', marker='x', s=50, linewidths=2)
            ax.set_title(f'Timepoint {t}/{num_timepoints-1}')
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(viz_filename, dpi=150, bbox_inches='tight')
            plt.close()
        
        # Displace centres for next timepoint (skip on last iteration)
        if t < num_timepoints - 1:
            timepoint_seed = seed + t if seed is not None else None
            current_centres = displace_centres(
                current_centres,
                displacement_std=displacement_std,
                displacement_mean=displacement_mean,
                directional_bias=directional_bias,
                min_distance=min_distance,
                image_shape=(synthesizer.image_size, synthesizer.image_size),
                seed=timepoint_seed
            )
    
    # Save complete time-series as single (T, Y, X) TIFF stack
    if save_stack:
        stack = np.stack(images_list, axis=0)  # Shape: (T, Y, X)
        
        # Apply histogram matching if requested
        if match_histograms and num_timepoints > 1:
            logger.info("Applying histogram matching to ensure consistent intensity distributions...")
            reference_frame = stack[0]  # Use first frame as reference
            matched_stack = np.zeros_like(stack)
            matched_stack[0] = reference_frame
            
            for t in range(1, num_timepoints):
                matched_stack[t] = exposure.match_histograms(
                    stack[t], 
                    reference_frame,
                    channel_axis=None
                )
            
            stack = matched_stack
            logger.info("Histogram matching complete")
        
        stack_filename = os.path.join(output_dir, f"{prefix}_stack.tiff")
        
        # Save as 32-bit float to preserve original data range [-1, 1]
        tifffile.imwrite(
            stack_filename,
            stack.astype(np.float32),
            metadata={'axes': 'TYX'}
        )
        logger.info(f"Saved time-series stack: {stack_filename} with shape {stack.shape}")
    
    logger.info(f"Time-series generation complete!")
    logger.info(f"Saved {num_timepoints} images to {output_dir}")
    
    return images_list, centres_list


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic time-series of cell microscopy images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate time-series from file centres
  python scripts/generate_timeseries.py --checkpoint checkpoints/best.pt \\
    --centres_file data/processed/test/patch_00000_centres.npy \\
    --num_timepoints 50 --displacement_std 2.0
  
  # Generate with directional bias (drift)
  python scripts/generate_timeseries.py --checkpoint checkpoints/best.pt \\
    --method poisson --num_timepoints 100 \\
    --displacement_std 1.5 --directional_bias 0.5 0.3
  
  # Generate with random walk (no bias)
  python scripts/generate_timeseries.py --checkpoint checkpoints/best.pt \\
    --method training_dist --num_timepoints 30 --displacement_std 3.0
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    
    # Time-series parameters
    parser.add_argument(
        '--num_timepoints',
        type=int,
        default=2,
        help='Number of timepoints to generate (default: 2)'
    )
    
    parser.add_argument(
        '--displacement_std',
        type=float,
        default=2.0,
        help='Standard deviation of displacement per timestep in pixels (default: 2.0)'
    )
    
    parser.add_argument(
        '--displacement_mean',
        type=float,
        default=0.0,
        help='Mean displacement magnitude (default: 0.0)'
    )
    
    parser.add_argument(
        '--directional_bias',
        type=float,
        nargs=2,
        default=None,
        metavar=('DY', 'DX'),
        help='Directional bias as (dy, dx) in pixels per timestep (default: None)'
    )
    
    parser.add_argument(
        '--min_distance',
        type=float,
        default=12.0,
        help='Minimum allowed distance between centres in pixels (default: 12.0)'
    )
    
    parser.add_argument(
        '--noise_correlation',
        type=float,
        default=0.0,
        help='Spatial correlation between consecutive frames, 0-1 (default: 0.0, independent)'
    )
    
    parser.add_argument(
        '--temporal_smoothness',
        type=float,
        default=0.0,
        help='AR(1) coefficient for smooth noise evolution, 0-1 (default: 0.0, no smoothing). '
             'Higher values create smoother temporal transitions. Recommended: 0.7 for moderate, 0.9 for very smooth.'
    )
    
    parser.add_argument(
        '--match_histograms',
        action='store_true',
        help='Apply histogram matching post-processing to match all frames to first frame. '
             'Ensures consistent intensity distributions across time (default: False)'
    )
    
    # Initial centres configuration
    parser.add_argument(
        '--method',
        type=str,
        default='poisson',
        choices=['simple', 'poisson', 'training_dist', 'from_file'],
        help='Initial centre generation method (default: poisson)'
    )
    
    parser.add_argument(
        '--centres_file',
        type=str,
        default=None,
        help='Path to .npy file with initial centres (required when method=from_file)'
    )
    
    parser.add_argument(
        '--num_cells',
        type=int,
        default=20,
        help='Number of cells (for method=simple, default: 20)'
    )
    
    parser.add_argument(
        '--density',
        type=float,
        default=0.0003,
        help='Cell density in cells/pixel (for method=poisson, default: 0.0003)'
    )
    
    parser.add_argument(
        '--initial_min_distance',
        type=float,
        default=12.0,
        help='Minimum distance between cells for initial generation (default: 12.0)'
    )
    
    parser.add_argument(
        '--mean_cells',
        type=float,
        default=20.0,
        help='Mean number of cells (for method=training_dist, default: 20.0)'
    )
    
    parser.add_argument(
        '--std_cells',
        type=float,
        default=8.0,
        help='Std of number of cells (for method=training_dist, default: 8.0)'
    )
    
    # Model parameters
    parser.add_argument(
        '--guidance_scale',
        type=float,
        default=3.0,
        help='Classifier-free guidance scale (default: 3.0)'
    )
    
    parser.add_argument(
        '--no_cfg',
        action='store_true',
        help='Disable classifier-free guidance'
    )
    
    # Output configuration
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./timeseries_output',
        help='Output directory (default: ./timeseries_output)'
    )
    
    parser.add_argument(
        '--prefix',
        type=str,
        default='timeseries',
        help='Prefix for output filenames (default: timeseries)'
    )
    
    parser.add_argument(
        '--no_visualization',
        action='store_true',
        help='Disable visualization overlays (saves time)'
    )
    
    parser.add_argument(
        '--no_save_centres',
        action='store_true',
        help='Do not save centres at each timepoint'
    )
    
    parser.add_argument(
        '--no_stack',
        action='store_true',
        help='Do not save complete (T, Y, X) TIFF stack'
    )
    
    # Configuration files
    parser.add_argument(
        '--model_config',
        type=str,
        default='configs/model.yaml',
        help='Path to model config (default: configs/model.yaml)'
    )
    
    parser.add_argument(
        '--data_config',
        type=str,
        default='configs/data.yaml',
        help='Path to data config (default: configs/data.yaml)'
    )
    
    # System
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (default: cuda if available)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (default: None)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.method == 'from_file' and args.centres_file is None:
        parser.error("--centres_file is required when method=from_file")
    
    # Set random seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    # Log configuration
    use_cfg = not args.no_cfg
    logger.info("="*70)
    logger.info("SYNTHETIC TIME-SERIES GENERATION")
    logger.info("="*70)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Number of timepoints: {args.num_timepoints}")
    logger.info(f"Displacement std: {args.displacement_std} pixels")
    logger.info(f"Displacement mean: {args.displacement_mean}")
    if args.directional_bias is not None:
        logger.info(f"Directional bias: dy={args.directional_bias[0]}, dx={args.directional_bias[1]}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Classifier-free guidance: {use_cfg}")
    if use_cfg:
        logger.info(f"  Guidance scale: {args.guidance_scale}")
    logger.info("="*70)
    
    # Initial centres method
    logger.info(f"Initial centres method: {args.method}")
    if args.method == 'from_file':
        logger.info(f"  Centres file: {args.centres_file}")
    elif args.method == 'simple':
        logger.info(f"  Number of cells: {args.num_cells}")
    elif args.method == 'poisson':
        logger.info(f"  Density: {args.density}, min_distance: {args.min_distance}")
    elif args.method == 'training_dist':
        logger.info(f"  Mean cells: {args.mean_cells}, std: {args.std_cells}")
    logger.info("="*70)
    
    try:
        # Initialize synthesizer
        synthesizer = CellSynthesizer(
            checkpoint_path=args.checkpoint,
            model_config=args.model_config,
            data_config=args.data_config,
            device=args.device
        )
        
        # Generate or load initial centres
        if args.method == 'from_file':
            initial_centres = np.load(args.centres_file)
            logger.info(f"Loaded {len(initial_centres)} initial centres from {args.centres_file}")
        else:
            image_shape = (synthesizer.image_size, synthesizer.image_size)
            
            if args.method == 'simple':
                initial_centres = generate_random_centres_simple(
                    image_shape=image_shape,
                    num_cells=args.num_cells,
                    border_margin=20,
                    seed=args.seed
                )
            elif args.method == 'poisson':
                initial_centres = generate_random_centres_poisson(
                    image_shape=image_shape,
                    density=args.density,
                    min_distance=args.initial_min_distance,
                    border_margin=20,
                    seed=args.seed
                )
            elif args.method == 'training_dist':
                initial_centres = generate_centres_from_training_distribution(
                    image_shape=image_shape,
                    mean_cells=args.mean_cells,
                    std_cells=args.std_cells,
                    mean_min_dist=args.initial_min_distance,
                    border_margin=20,
                    seed=args.seed
                )
            
            logger.info(f"Generated {len(initial_centres)} initial centres using '{args.method}' method")
        
        # Generate time-series
        images, centres = generate_timeseries(
            synthesizer=synthesizer,
            initial_centres=initial_centres,
            num_timepoints=args.num_timepoints,
            displacement_std=args.displacement_std,
            displacement_mean=args.displacement_mean,
            directional_bias=tuple(args.directional_bias) if args.directional_bias else None,
            min_distance=args.min_distance,
            noise_correlation=args.noise_correlation,
            temporal_smoothness=args.temporal_smoothness,
            match_histograms=args.match_histograms,
            use_cfg=use_cfg,
            guidance_scale=args.guidance_scale,
            output_dir=args.output_dir,
            prefix=args.prefix,
            save_centres=not args.no_save_centres,
            save_visualization=not args.no_visualization,
            save_stack=not args.no_stack,
            seed=args.seed
        )
        
        logger.info("="*70)
        logger.info("SUCCESS!")
        logger.info(f"Generated {len(images)} images")
        logger.info(f"Output saved to: {args.output_dir}")
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
