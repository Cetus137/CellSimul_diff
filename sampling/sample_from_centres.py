"""
Sample from trained diffusion models.

Generate synthetic microscopy images conditioned on cell centres.
"""

import torch
import numpy as np
import yaml
from pathlib import Path
from typing import Optional, List
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

from models.unet import ConditionalUNet
from models.diffusion import DDPM
from preprocessing.generate_condition_maps import generate_conditioning_maps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(
    checkpoint_path: str,
    config_path: str = "configs/frame1.yaml",
    device: str = 'cuda'
) -> DDPM:
    """
    Load trained diffusion model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to model configuration
        device: Device to load model on
    
    Returns:
        model: Loaded DDPM model
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    conditioning_cfg = config['unet'].get('conditioning')
    if conditioning_cfg is not None:
        condition_channels = sum(1 for v in conditioning_cfg.values() if v)
    else:
        condition_channels = config['unet']['condition_channels']

    unet = ConditionalUNet(
        in_channels=config['unet']['in_channels'],
        out_channels=config['unet']['out_channels'],
        condition_channels=condition_channels,
        base_channels=config['unet']['base_channels'],
        channel_multipliers=config['unet']['channel_multipliers'],
        num_res_blocks=config['unet']['num_res_blocks'],
        attention_resolutions=config['unet']['attention_resolutions'],
        num_heads=config['unet']['num_heads'],
        time_emb_dim=config['unet']['time_emb_dim'],
        num_groups=config['unet']['norm_groups'],
        dropout=config['unet']['dropout']
    )
    
    model = DDPM(
        model=unet,
        timesteps=config['diffusion']['timesteps'],
        beta_schedule=config['diffusion']['beta_schedule'],
        beta_start=config['diffusion'].get('beta_start', 0.0001),
        beta_end=config['diffusion'].get('beta_end', 0.02),
        prediction_type=config['diffusion']['prediction_type'],
        loss_type=config['diffusion']['loss_type']
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle both raw model state dicts and full checkpoints
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")
    elif 'ema_shadow' in checkpoint:
        # Load EMA weights
        for name, param in model.named_parameters():
            if name in checkpoint['ema_shadow']:
                param.data = checkpoint['ema_shadow'][name]
        logger.info("Loaded EMA weights from checkpoint")
    else:
        model.load_state_dict(checkpoint)
        logger.info("Loaded model weights")
    
    model.eval()
    return model


def sample_from_centres(
    model: DDPM,
    centres: np.ndarray,
    image_shape: tuple = (256, 256),
    heatmap_sigma: float = 3.0,
    boundary_sigma: float = 2.0,
    use_cfg: bool = True,
    guidance_scale: float = 3.0,
    device: str = 'cuda',
    active_channels: Optional[dict] = None
) -> np.ndarray:
    """
    Generate image from cell centres.
    
    Args:
        model: Trained DDPM model
        centres: Cell centres of shape (N, 2) with (y, x) coordinates
        image_shape: Output image shape (H, W)
        heatmap_sigma: Sigma for centre heatmap
        boundary_sigma: Temperature for boundary map
        use_cfg: Use classifier-free guidance
        guidance_scale: CFG guidance scale
        device: Device to sample on
        active_channels: Dict of booleans for which conditioning channels to
            use, e.g. {'heatmap': True, 'distance': True, 'boundary': False}.
            None (default) = all three active.
    
    Returns:
        image: Generated image of shape (H, W)
    """
    # Generate conditioning maps
    # CRITICAL: Use same parameters as training!
    conditioning = generate_conditioning_maps(
        centres,
        image_shape,
        heatmap_sigma=heatmap_sigma,
        boundary_sigma=boundary_sigma,
        boundary_method='entropy',  # Must match training
        distance_percentile=95.0,   # CRITICAL: Must match training
        active_channels=active_channels
    )
    
    # Convert to tensor and add batch dimension
    conditioning = torch.from_numpy(conditioning).unsqueeze(0).float().to(device)  # (1, 3, H, W)
    
    # DIAGNOSTIC: Print conditioning statistics
    print(f"\n{'='*60}")
    print(f"INFERENCE CONDITIONING DIAGNOSTICS")
    print(f"{'='*60}")
    print(f"Conditioning shape: {conditioning.shape}")
    for c in range(conditioning.shape[1]):
        c_data = conditioning[:, c]
        print(f"Channel {c}: min={c_data.min():.4f}, max={c_data.max():.4f}, "
              f"mean={c_data.mean():.4f}, std={c_data.std():.4f}")
    print(f"{'='*60}\n")
    
    # Sample
    with torch.no_grad():
        if use_cfg:
            samples = model.sample_with_cfg(
                conditioning,
                guidance_scale=guidance_scale,
                clip_denoised=True
            )
        else:
            samples = model.sample(
                conditioning,
                clip_denoised=True
            )
    
    # Convert to numpy and denormalize
    image = samples[0, 0].cpu().numpy()  # (H, W)
    
    # DIAGNOSTIC: Print output statistics BEFORE denormalization
    print(f"\n{'='*60}")
    print(f"INFERENCE OUTPUT DIAGNOSTICS")
    print(f"{'='*60}")
    print(f"Raw output (before denorm): min={image.min():.4f}, max={image.max():.4f}, "
          f"mean={image.mean():.4f}, std={image.std():.4f}")
    
    # Denormalize from [-1, 1] to [0, 1]
    image = (image + 1.0) / 2.0
    image = np.clip(image, 0, 1)
    
    print(f"After denorm to [0,1]: min={image.min():.4f}, max={image.max():.4f}, "
          f"mean={image.mean():.4f}, std={image.std():.4f}")
    print(f"{'='*60}\n")
    
    return image


def batch_sample_from_centres_list(
    model: DDPM,
    centres_list: List[np.ndarray],
    image_shape: tuple = (256, 256),
    use_cfg: bool = True,
    guidance_scale: float = 3.0,
    device: str = 'cuda'
) -> List[np.ndarray]:
    """
    Generate multiple images from a list of centre configurations.
    
    Args:
        model: Trained DDPM model
        centres_list: List of centre arrays
        image_shape: Output image shape
        use_cfg: Use CFG
        guidance_scale: CFG scale
        device: Device
    
    Returns:
        images: List of generated images
    """
    images = []
    
    for centres in tqdm(centres_list, desc="Sampling"):
        image = sample_from_centres(
            model,
            centres,
            image_shape=image_shape,
            use_cfg=use_cfg,
            guidance_scale=guidance_scale,
            device=device
        )
        images.append(image)
    
    return images


def visualize_sample(
    image: np.ndarray,
    centres: np.ndarray,
    conditioning: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
):
    """
    Visualize generated sample with centres and conditioning.
    
    Args:
        image: Generated image (H, W)
        centres: Cell centres (N, 2)
        conditioning: Optional conditioning maps (3, H, W)
        save_path: Optional path to save figure
    """
    if conditioning is not None:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        axes[0].imshow(conditioning[0], cmap='hot')
        axes[0].set_title('Centre Heatmap')
        axes[0].axis('off')
        
        axes[1].imshow(conditioning[1], cmap='viridis')
        axes[1].set_title('Distance Map')
        axes[1].axis('off')
        
        axes[2].imshow(conditioning[2], cmap='plasma')
        axes[2].set_title('Boundary Map')
        axes[2].axis('off')
        
        axes[3].imshow(image, cmap='gray')
        axes[3].scatter(centres[:, 1], centres[:, 0], c='cyan', s=20, marker='x', alpha=0.7)
        axes[3].set_title('Generated Image')
        axes[3].axis('off')
    else:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        
        ax.imshow(image, cmap='gray')
        ax.scatter(centres[:, 1], centres[:, 0], c='cyan', s=30, marker='x', linewidths=2)
        ax.set_title(f'Generated Image ({len(centres)} cells)')
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved visualization to {save_path}")
    else:
        plt.show()


def create_grid(images: List[np.ndarray], grid_size: Optional[tuple] = None) -> np.ndarray:
    """
    Create a grid of images.
    
    Args:
        images: List of images (H, W)
        grid_size: Optional (rows, cols), inferred if None
    
    Returns:
        grid: Grid image
    """
    n = len(images)
    
    if grid_size is None:
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
    else:
        rows, cols = grid_size
    
    h, w = images[0].shape
    grid = np.zeros((rows * h, cols * w), dtype=np.float32)
    
    for idx, img in enumerate(images):
        if idx >= rows * cols:
            break
        
        i = idx // cols
        j = idx % cols
        
        grid[i*h:(i+1)*h, j*w:(j+1)*w] = img
    
    return grid


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Sample from trained diffusion model")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/frame1.yaml', help='Model config')
    parser.add_argument('--num_samples', type=int, default=4, help='Number of samples to generate')
    parser.add_argument('--num_cells', type=int, default=20, help='Number of cells per sample')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--use_cfg', action='store_true', help='Use classifier-free guidance')
    parser.add_argument('--guidance_scale', type=float, default=3.0, help='CFG guidance scale')
    parser.add_argument('--output_dir', type=str, default='samples', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = load_model(args.checkpoint, args.config, args.device)
    
    # Generate random centres
    np.random.seed(42)
    centres_list = []
    for i in range(args.num_samples):
        centres = np.random.rand(args.num_cells, 2) * args.image_size
        centres_list.append(centres)
    
    # Sample images
    logger.info(f"Generating {args.num_samples} samples with {args.num_cells} cells each")
    images = batch_sample_from_centres_list(
        model,
        centres_list,
        image_shape=(args.image_size, args.image_size),
        use_cfg=args.use_cfg,
        guidance_scale=args.guidance_scale,
        device=args.device
    )
    
    # Save individual samples
    for i, (image, centres) in enumerate(zip(images, centres_list)):
        visualize_sample(
            image,
            centres,
            save_path=output_dir / f'sample_{i:03d}.png'
        )
    
    # Create grid
    grid = create_grid(images)
    plt.figure(figsize=(12, 12))
    plt.imshow(grid, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / 'grid.png', dpi=150, bbox_inches='tight')
    
    logger.info(f"Saved {args.num_samples} samples to {output_dir}")
