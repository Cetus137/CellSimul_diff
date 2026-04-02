"""
Training script for the 3D centre-conditioned diffusion model.

Mirrors scripts/train.py exactly, but imports from the *3d counterparts:
    models.unet3d           → ConditionalUNet3D
    models.diffusion3d      → DDPM3D
    datasets.centre_condition_dataset3d → get_dataloader3d

The 2D train.py and all 2D modules are NOT touched.
"""

import argparse
import yaml
import torch
import torch.optim as optim
from pathlib import Path
import logging
import numpy as np

from models.unet3d import ConditionalUNet3D
from models.diffusion3d import DDPM3D
from datasets.centre_condition_dataset3d import get_dataloader3d
from training.trainer3d import Trainer3D

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Config helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_conditioning_config(config: dict):
    """Return (active_channels dict, condition_channels int) from config."""
    conditioning = config['unet'].get('conditioning')
    if conditioning is not None:
        active = {k: bool(v) for k, v in conditioning.items()}
        n_channels = sum(1 for v in active.values() if v)
    else:
        n_channels = config['unet']['condition_channels']
        active = None
    return active, n_channels


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

def create_model(config: dict) -> DDPM3D:
    active_channels, condition_channels = get_conditioning_config(config)

    unet = ConditionalUNet3D(
        in_channels=config['unet']['in_channels'],
        out_channels=config['unet']['out_channels'],
        condition_channels=condition_channels,
        base_channels=config['unet']['base_channels'],
        channel_multipliers=config['unet']['channel_multipliers'],
        num_res_blocks=config['unet']['num_res_blocks'],
        time_emb_dim=config['unet']['time_emb_dim'],
        num_groups=config['unet']['norm_groups'],
        dropout=config['unet']['dropout']
    )

    ddpm = DDPM3D(
        model=unet,
        timesteps=config['diffusion']['timesteps'],
        beta_schedule=config['diffusion']['beta_schedule'],
        beta_start=config['diffusion'].get('beta_start', 0.0001),
        beta_end=config['diffusion'].get('beta_end', 0.02),
        prediction_type=config['diffusion']['prediction_type'],
        loss_type=config['diffusion']['loss_type']
    )

    return ddpm


# ──────────────────────────────────────────────────────────────────────────────
# Optimizer / scheduler (identical logic to train.py)
# ──────────────────────────────────────────────────────────────────────────────

def create_optimizer(model: DDPM3D, config: dict) -> torch.optim.Optimizer:
    opt_cfg = config['optimizer']
    if opt_cfg['type'] == 'AdamW':
        return optim.AdamW(
            model.parameters(),
            lr=opt_cfg['learning_rate'],
            weight_decay=opt_cfg['weight_decay'],
            betas=opt_cfg['betas'],
            eps=opt_cfg['eps']
        )
    elif opt_cfg['type'] == 'Adam':
        return optim.Adam(
            model.parameters(),
            lr=opt_cfg['learning_rate'],
            weight_decay=opt_cfg.get('weight_decay', 0),
            betas=opt_cfg.get('betas', (0.9, 0.999)),
            eps=opt_cfg.get('eps', 1e-8)
        )
    else:
        raise ValueError(f"Unknown optimizer type: {opt_cfg['type']}")


def create_scheduler(optimizer, config: dict):
    sched_cfg = config['scheduler']
    if sched_cfg['type'] == 'cosine':
        from torch.optim.lr_scheduler import LambdaLR
        warmup = sched_cfg['warmup_steps']
        total  = config['training']['num_epochs'] * 1000

        def warmup_cosine(step):
            if step < warmup:
                return step / warmup
            progress = (step - warmup) / max(1, total - warmup)
            return 0.5 * (1 + np.cos(np.pi * progress))

        return LambdaLR(optimizer, warmup_cosine)
    elif sched_cfg['type'] == 'constant':
        return None
    else:
        raise ValueError(f"Unknown scheduler type: {sched_cfg['type']}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train 3D centre-conditioned diffusion model")
    parser.add_argument('--config', type=str, default='configs/frame1_3d.yaml')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)

    device = args.device or config['device']
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = 'cpu'

    seed = config['random_seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    logger.info("=" * 60)
    logger.info("Starting 3D training")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Config: {args.config}")

    # ── Conditioning config ────────────────────────────────────────────────────
    active_channels, condition_channels = get_conditioning_config(config)
    logger.info(f"Conditioning: {condition_channels} channels  {active_channels or ''}")

    # ── Debug / overfit mode ───────────────────────────────────────────────────
    debug_overfit = config['training'].get('debug_overfit', {}).get('enabled', False)
    if debug_overfit:
        logger.warning("DEBUG OVERFIT MODE — small subset only")
        num_patches   = config['training']['debug_overfit'].get('num_patches', 8)
        augment_train = not config['training']['debug_overfit'].get('disable_augmentation', True)
        p_uncond      = 0.0
    else:
        num_patches   = None
        augment_train = config['augmentation'].get('random_flip_horizontal', False)
        small_data    = config['training'].get('small_data_mode', {})
        p_uncond      = small_data.get('p_uncond', 0.0) if small_data.get('enabled') \
                        else config['diffusion']['cfg'].get('dropout_prob', 0.0)

    # ── Data loaders ───────────────────────────────────────────────────────────
    train_loader = get_dataloader3d(
        patches_dir=config['processed_data']['patches_dir'],  # str or list
        split='train',
        batch_size=config['dataloader']['batch_size'],
        num_workers=config['dataloader']['num_workers'],
        heatmap_sigma=config['preprocessing']['centre_heatmap_sigma'],
        augment=augment_train,
        p_uncond=p_uncond,
        shuffle=True,
        active_channels=active_channels
    )

    if debug_overfit and num_patches is not None:
        from torch.utils.data import Subset
        indices = list(range(min(num_patches, len(train_loader.dataset))))
        subset_ds = Subset(train_loader.dataset, indices)
        train_loader = torch.utils.data.DataLoader(
            subset_ds,
            batch_size=config['dataloader']['batch_size'],
            shuffle=True,
            num_workers=config['dataloader']['num_workers'],
            pin_memory=True,
            drop_last=True
        )
        logger.info(f"Debug mode: {len(indices)} patches")

    val_loader = get_dataloader3d(
        patches_dir=config['processed_data']['patches_dir'],  # str or list
        split='val',
        batch_size=config['dataloader']['batch_size'],
        num_workers=config['dataloader']['num_workers'],
        heatmap_sigma=config['preprocessing']['centre_heatmap_sigma'],
        augment=False,
        p_uncond=0.0,
        shuffle=False,
        active_channels=active_channels
    )

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val   batches: {len(val_loader)}")

    # ── Model ──────────────────────────────────────────────────────────────────
    model = create_model(config)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    # ── Optimizer / scheduler ──────────────────────────────────────────────────
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)

    # ── Low-noise bias ─────────────────────────────────────────────────────────
    small_data_cfg   = config['training'].get('small_data_mode', {})
    low_noise_bias   = small_data_cfg.get('low_noise_bias', False)
    low_noise_frac   = small_data_cfg.get('low_noise_fraction', 0.3)
    low_noise_weight = small_data_cfg.get('low_noise_weight', 3.0)

    # ── Geometry channel names for visualizer ──────────────────────────────────
    _KNOWN_GEOM_NAMES = {'heatmap': 'Heatmap', 'distance': 'Distance', 'boundary': 'Boundary'}
    if active_channels is not None:
        geom_channel_names = [_KNOWN_GEOM_NAMES[k] for k, v in active_channels.items()
                              if v and k in _KNOWN_GEOM_NAMES]
    else:
        geom_channel_names = None

    # ── Trainer3D (swaps in 3-D orthogonal-slice visualizer) ──────────────────
    trainer = Trainer3D(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=config['training'].get('checkpoint_dir', 'checkpoints/frame1_3d'),
        use_ema=config['training']['ema']['enabled'],
        ema_decay=config['training']['ema']['decay'],
        ema_update_every=config['training']['ema']['update_every'],
        mixed_precision=config['training']['mixed_precision'],
        gradient_clip=config['training']['gradient_clip'],
        log_every=config['training']['log_every'],
        save_every=config['training']['save_every'],
        validate_every=config['training']['validate_every'],
        visualize=config['training'].get('visualize', True),
        viz_dir=config['training'].get('viz_dir', 'visualizations/frame1_3d'),
        low_noise_bias=low_noise_bias,
        low_noise_fraction=low_noise_frac,
        low_noise_weight=low_noise_weight,
        n_geom_channels=condition_channels,
        geom_channel_names=geom_channel_names,
    )

    if args.resume:
        logger.info(f"Resuming from: {args.resume}")
        trainer.load_checkpoint(args.resume)

    logger.info("=" * 60)
    logger.info("Starting training loop")
    logger.info("=" * 60)

    try:
        trainer.train(num_epochs=config['training']['num_epochs'])
    except KeyboardInterrupt:
        logger.info("Training interrupted")
        trainer.save_checkpoint(name='interrupted.pt')
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    logger.info("Done. Checkpoints saved to: "
                f"{config['training'].get('checkpoint_dir', 'checkpoints/frame1_3d')}")


if __name__ == "__main__":
    main()
