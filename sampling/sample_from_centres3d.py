"""
3D sampling utilities for generating synthetic volumes from trained diffusion model.
"""

import torch
import numpy as np
import yaml
from pathlib import Path
from typing import Optional, Tuple
import logging
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.diffusion3d import DDPM3D
from models.unet3d import ConditionalUNet3D
from preprocessing.generate_condition_maps3d import generate_conditioning_maps3d

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model3d(checkpoint_path: str, config_path: str, device: str = "cuda") -> DDPM3D:
    """
    Load trained 3D diffusion model from checkpoint.
    
    Args:
        checkpoint_path: Path to .pt checkpoint
        config_path: Path to unified config file
        device: Device to load model on
    
    Returns:
        model: Loaded DDPM3D model in eval mode
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Determine input channels
    conditioning_cfg = config.get('unet', {}).get('conditioning')
    if conditioning_cfg is not None:
        condition_channels = sum([v for v in conditioning_cfg.values()])
    else:
        condition_channels = 3  # default: heatmap + distance + boundary
    
    # Create U-Net
    unet = ConditionalUNet3D(
        in_channels=config['unet']['in_channels'],
        out_channels=config['unet']['out_channels'],
        condition_channels=condition_channels,
        base_channels=config['unet']['base_channels'],
        channel_multipliers=config['unet']['channel_multipliers'],
        num_res_blocks=config['unet']['num_res_blocks'],
        num_groups=config['unet']['norm_groups'],
        time_emb_dim=config['unet']['time_emb_dim'],
        dropout=config['unet'].get('dropout', 0.0)
    )
    
    # Create diffusion model
    model = DDPM3D(
        model=unet,
        timesteps=config['diffusion']['timesteps'],
        beta_start=config['diffusion']['beta_start'],
        beta_end=config['diffusion']['beta_end'],
        beta_schedule=config['diffusion'].get('beta_schedule', 'linear')
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


@torch.no_grad()
def sample_from_centres3d(
    model: DDPM3D,
    centres: np.ndarray,
    volume_size: int = 128,
    heatmap_sigma: float = 3.0,
    active_channels: Optional[dict] = None,
    device: str = "cuda",
    use_cfg: bool = False,
    guidance_scale: float = 3.0,
    ddim_steps: Optional[int] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Generate a 3D volume from cell centres using trained diffusion model.

    Args:
        model: Trained DDPM3D model
        centres: (N, 3) array of cell centres in (z, y, x) order
        volume_size: Cubic volume side length
        heatmap_sigma: Gaussian sigma for centre heatmap
        active_channels: Dict specifying which conditioning channels are active
        device: Computation device
        use_cfg: Whether to use classifier-free guidance
        guidance_scale: CFG scale (if use_cfg=True)
        ddim_steps: If set, use DDIM with this many steps instead of full DDPM.
                    Recommended: 200 (~5x speedup), 100 (~10x), 50 (~20x).
                    None = original DDPM-1000.

    Returns:
        volume: Generated volume (D, H, W)
        metadata: Dict with centres and conditioning
    """
    model.eval()

    # Generate conditioning maps
    volume_shape = (volume_size, volume_size, volume_size)
    condition_maps = generate_conditioning_maps3d(
        centres=centres,
        volume_shape=volume_shape,
        heatmap_sigma=heatmap_sigma,
        active_channels=active_channels
    )

    # Prepare conditioning tensor
    condition = torch.from_numpy(condition_maps).unsqueeze(0).to(device)  # (1, C, D, H, W)

    # Route to DDIM or DDPM depending on ddim_steps.
    # NOTE: FP16 autocast is intentionally NOT used — the model was trained in FP32
    # and small FP16 precision errors accumulate across denoising steps, producing
    # visibly noisier outputs. DDIM alone already gives a large speedup.
    if ddim_steps is not None:
        if use_cfg:
            sample = model.sample_ddim_with_cfg(
                conditioning=condition,
                guidance_scale=guidance_scale,
                ddim_steps=ddim_steps,
            )
        else:
            sample = model.sample_ddim(
                conditioning=condition,
                ddim_steps=ddim_steps,
            )
    else:
        # Original DDPM (1000 steps)
        if use_cfg:
            sample = model.sample_with_cfg(
                conditioning=condition,
                guidance_scale=guidance_scale
            )
        else:
            sample = model.sample(
                conditioning=condition
            )

    # Convert to numpy
    volume = sample[0, 0].cpu().numpy()  # (D, H, W)

    metadata = {
        'centres': centres,
        'condition_maps': condition_maps,
        'num_cells': len(centres),
        'use_cfg': use_cfg,
        'guidance_scale': guidance_scale if use_cfg else None,
        'ddim_steps': ddim_steps,
    }

    return volume, metadata


@torch.no_grad()
def sample_batch_from_centres3d(
    model: DDPM3D,
    centres_list: list,
    volume_size: int = 128,
    heatmap_sigma: float = 3.0,
    active_channels: Optional[dict] = None,
    device: str = "cuda",
    use_cfg: bool = False,
    guidance_scale: float = 3.0,
    ddim_steps: Optional[int] = None,
) -> Tuple[list, list]:
    """
    Generate a batch of 3D volumes in a single DDPM/DDIM pass.

    Stacks conditioning tensors from all volumes into (B, C, D, H, W) and
    runs a single denoising loop, giving ~B× throughput over calling
    sample_from_centres3d B times.

    Args:
        model: Trained DDPM3D model
        centres_list: List of (N_i, 3) centre arrays, one per volume
        volume_size: Cubic volume side length
        heatmap_sigma: Gaussian sigma for centre heatmap
        active_channels: Dict specifying which conditioning channels are active
        device: Computation device
        use_cfg: Whether to use classifier-free guidance
        guidance_scale: CFG scale (if use_cfg=True)
        ddim_steps: If set, use DDIM with this many steps (None = full DDPM-1000)

    Returns:
        volumes: List of (D, H, W) numpy arrays, one per input
        metadatas: List of metadata dicts, one per input
    """
    model.eval()
    B = len(centres_list)
    volume_shape = (volume_size, volume_size, volume_size)

    # Build conditioning maps for each volume and stack into (B, C, D, H, W)
    condition_maps_list = []
    for centres in centres_list:
        cmap = generate_conditioning_maps3d(
            centres=centres,
            volume_shape=volume_shape,
            heatmap_sigma=heatmap_sigma,
            active_channels=active_channels,
        )
        condition_maps_list.append(cmap)

    condition = torch.from_numpy(np.stack(condition_maps_list, axis=0)).to(device)

    # NOTE: FP16 autocast is intentionally NOT used — the model was trained in FP32
    # and small FP16 precision errors accumulate across denoising steps, producing
    # visibly noisier outputs.
    if ddim_steps is not None:
        if use_cfg:
            samples = model.sample_ddim_with_cfg(condition, guidance_scale, ddim_steps)
        else:
            samples = model.sample_ddim(condition, ddim_steps)
    else:
        if use_cfg:
            samples = model.sample_with_cfg(condition, guidance_scale)
        else:
            samples = model.sample(condition)

    # Split batch back into individual volumes
    volumes = [samples[b, 0].cpu().numpy() for b in range(B)]

    metadatas = [
        {
            'centres': centres_list[b],
            'condition_maps': condition_maps_list[b],
            'num_cells': len(centres_list[b]),
            'use_cfg': use_cfg,
            'guidance_scale': guidance_scale if use_cfg else None,
            'ddim_steps': ddim_steps,
        }
        for b in range(B)
    ]

    return volumes, metadatas
