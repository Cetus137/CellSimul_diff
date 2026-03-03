"""
Train the 3D frame-2 conditional diffusion model (temporally-conditioned 3D DDPM).

Combines:
  - 3D models (ConditionalUNet3D, DDPM3D)
  - Temporal pair dataset (TemporalPairDataset3D)
  - 3D trainer (Trainer3D)

Usage:
    python -m scripts.train_frame2_3d --config configs/frame2_3d.yaml

    # Debug overfit (8 pairs):
    python -m scripts.train_frame2_3d --config configs/frame2_3d.yaml --debug_overfit

    # Resume:
    python -m scripts.train_frame2_3d --config configs/frame2_3d.yaml --resume checkpoints/frame2_3d/latest.pt
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import yaml

sys.path.append(str(Path(__file__).parent.parent))

from datasets.temporal_pair_dataset3d import get_temporal_dataloader_3d
from models.diffusion3d import DDPM3D
from models.unet3d import ConditionalUNet3D
from training.trainer3d import Trainer3D

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Config helpers
# ------------------------------------------------------------------

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


_GEOM_ORDER = ['heatmap', 'distance']  # 3D only has 2 geometry channels


def get_conditioning_config(cfg):
    """
    Return (active_channels dict, geom_names list, n_geom int, total condition_channels int).

    For 3D frame2:
      - Geometry channels: heatmap (0), distance (1)
      - If prev_frame=True: +1 channel for V_t
      - Total: n_geom + (1 if prev_frame else 0)
    """
    u = cfg.get('unet', {})
    conditioning = u.get('conditioning')
    if conditioning is None:
        # Legacy integer — 2 channels for 3D geometry (heatmap + distance)
        condition_channels = u.get('condition_channels', 2)
        prev_frame = bool(u.get('prev_frame', False))
        n_geom = condition_channels - (1 if prev_frame else 0)
        active_channels = {k: True for k in _GEOM_ORDER[:n_geom]}
    else:
        active_channels = {k: bool(v) for k, v in conditioning.items() if k in _GEOM_ORDER}
        n_geom = sum(1 for v in active_channels.values() if v)
        prev_frame = bool(u.get('prev_frame', False))
        condition_channels = n_geom + (1 if prev_frame else 0)

    geom_names = [k.capitalize() for k in _GEOM_ORDER if active_channels.get(k, False)]
    return active_channels, geom_names, n_geom, condition_channels


def create_model(cfg):
    _active, _names, _n_geom, condition_channels = get_conditioning_config(cfg)
    u = cfg["unet"]
    unet = ConditionalUNet3D(
        in_channels=u["in_channels"],
        out_channels=u["out_channels"],
        condition_channels=condition_channels,
        base_channels=u["base_channels"],
        channel_multipliers=u["channel_multipliers"],
        num_res_blocks=u["num_res_blocks"],
        time_emb_dim=u["time_emb_dim"],
        num_groups=u["norm_groups"],
        dropout=u["dropout"],
    )
    d = cfg["diffusion"]
    ddpm = DDPM3D(
        model=unet,
        timesteps=d["timesteps"],
        beta_schedule=d["beta_schedule"],
        beta_start=d.get("beta_start", 0.0001),
        beta_end=d.get("beta_end", 0.02),
        prediction_type=d["prediction_type"],
        loss_type=d["loss_type"],
    )
    return ddpm


def create_optimizer(model, cfg):
    o = cfg["optimizer"]
    if o["type"] == "AdamW":
        return optim.AdamW(
            model.parameters(),
            lr=o["learning_rate"],
            weight_decay=o["weight_decay"],
            betas=o["betas"],
            eps=o["eps"],
        )
    raise ValueError(f"Unknown optimizer: {o['type']}")


def create_scheduler(optimizer, cfg):
    s = cfg["scheduler"]
    if s["type"] == "cosine":
        from torch.optim.lr_scheduler import LambdaLR
        warmup = s["warmup_steps"]
        total = cfg["training"]["num_epochs"] * 1000

        def fn(step):
            if step < warmup:
                return step / max(warmup, 1)
            progress = (step - warmup) / max(total - warmup, 1)
            return 0.5 * (1 + np.cos(np.pi * progress))

        return LambdaLR(optimizer, fn)
    if s["type"] == "constant":
        return None
    raise ValueError(f"Unknown scheduler: {s['type']}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train the 3D frame-2 conditional diffusion model."
    )
    parser.add_argument(
        "--config", default="configs/frame2_3d.yaml",
        help="Path to training config (data + model + training settings)"
    )
    parser.add_argument(
        "--resume", default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--device", default=None,
        help="Device: cuda / cpu (auto-detected when omitted)"
    )
    parser.add_argument(
        "--debug_overfit", action="store_true",
        help="Train on tiny subset to verify conditioning is learned"
    )
    args = parser.parse_args()

    # ---- Config -------------------------------------------------------
    cfg = load_config(args.config)

    # ---- Device -------------------------------------------------------
    if args.device:
        device = args.device
    else:
        device = cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # ---- Seeds --------------------------------------------------------
    seed = cfg.get("random_seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # ---- Conditioning config ------------------------------------------
    active_channels, geom_names, n_geom, condition_channels = get_conditioning_config(cfg)
    logger.info("Geometry conditioning channels (%d): %s", n_geom, geom_names)
    logger.info("Total condition_channels (geom + V_t): %d", condition_channels)

    # ---- Data loaders -------------------------------------------------
    # pairs_dir may be a single string or a list of strings (multi-dataset)
    _pairs_dirs_raw = cfg["temporal_dataset"]["pairs_dir"]
    if isinstance(_pairs_dirs_raw, str):
        _pairs_dirs_raw = [_pairs_dirs_raw]
    train_dir = [str(Path(d) / "train") for d in _pairs_dirs_raw]
    val_dir   = [str(Path(d) / "val")   for d in _pairs_dirs_raw]
    logger.info("Training from %d pairs director%s", len(_pairs_dirs_raw),
                "y" if len(_pairs_dirs_raw) == 1 else "ies")

    batch_size = cfg.get("dataloader", {}).get("batch_size", 2)  # 3D: default 2
    num_workers = cfg.get("dataloader", {}).get("num_workers", 4)
    overfit_n = cfg.get("training", {}).get("debug_overfit", {}).get("num_patches", 8)

    heatmap_sigma = cfg.get("preprocessing", {}).get("centre_heatmap_sigma", 3.0)
    distance_percentile = cfg.get("preprocessing", {}).get("distance_percentile", 95.0)

    if args.debug_overfit:
        logger.warning("=" * 60)
        logger.warning("DEBUG OVERFIT MODE — training on %d pairs", overfit_n)
        logger.warning("=" * 60)
        train_loader = get_temporal_dataloader_3d(
            split_dir=train_dir,
            batch_size=min(batch_size, overfit_n),
            num_workers=0,
            augment=False,
            heatmap_sigma=heatmap_sigma,
            distance_percentile=distance_percentile,
            overfit_n=overfit_n,
            shuffle=True,
            active_channels=active_channels,
            p_uncond=0.0,   # no dropout in debug mode
        )
        val_loader = get_temporal_dataloader_3d(
            split_dir=val_dir,
            batch_size=min(batch_size, overfit_n),
            num_workers=0,
            augment=False,
            heatmap_sigma=heatmap_sigma,
            distance_percentile=distance_percentile,
            overfit_n=overfit_n,
            shuffle=False,
            active_channels=active_channels,
            p_uncond=0.0,
        )
    else:
        # Read CFG dropout probability from config.
        # Priority: training.small_data_mode.p_uncond (if enabled)
        # Fallback: diffusion.cfg.dropout_prob (if CFG enabled)
        # Default: 0.0 (no dropout, disables CFG training)
        sm_cfg = cfg.get("training", {}).get("small_data_mode", {})
        diff_cfg = cfg.get("diffusion", {}).get("cfg", {})
        if sm_cfg.get("enabled", False):
            p_uncond = sm_cfg.get("p_uncond", 0.0)
        elif diff_cfg.get("enabled", False):
            p_uncond = diff_cfg.get("dropout_prob", 0.0)
        else:
            p_uncond = 0.0
        logger.info("CFG conditioning dropout probability: %.3f", p_uncond)

        augment_train = cfg.get("augmentation", {}).get("enabled", True)
        train_loader = get_temporal_dataloader_3d(
            split_dir=train_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            augment=augment_train,
            heatmap_sigma=heatmap_sigma,
            distance_percentile=distance_percentile,
            shuffle=True,
            active_channels=active_channels,
            p_uncond=p_uncond,
        )
        val_loader = get_temporal_dataloader_3d(
            split_dir=val_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            augment=False,
            heatmap_sigma=heatmap_sigma,
            distance_percentile=distance_percentile,
            shuffle=False,
            active_channels=active_channels,
            p_uncond=0.0,   # never drop conditioning at validation
        )

    logger.info("Train batches: %d   Val batches: %d",
                len(train_loader), len(val_loader))

    # ---- Model --------------------------------------------------------
    logger.info("Creating 3D frame-2 model (condition_channels=%d)...", condition_channels)
    model = create_model(cfg)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info("Parameters: %s", f"{num_params:,}")

    # ---- Optimizer / scheduler ----------------------------------------
    optimizer = create_optimizer(model, cfg)
    scheduler = create_scheduler(optimizer, cfg)

    # ---- Trainer ------------------------------------------------------
    tr = cfg.get("training", {})
    sm = tr.get("small_data_mode", {})
    ema = tr.get("ema", {})

    trainer = Trainer3D(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=tr.get("checkpoint_dir", "checkpoints/frame2_3d"),
        use_ema=ema.get("enabled", True),
        ema_decay=ema.get("decay", 0.9999),
        ema_update_every=ema.get("update_every", 10),
        mixed_precision=tr.get("mixed_precision", False),
        gradient_clip=tr.get("gradient_clip", 1.0),
        log_every=tr.get("log_every", 100),
        save_every=tr.get("save_every", 5000),
        validate_every=tr.get("validate_every", 2000),
        visualize=tr.get("visualize", True),
        viz_dir=tr.get("viz_dir", "visualizations/frame2_3d"),
        low_noise_bias=sm.get("low_noise_bias", False) and not args.debug_overfit,
        low_noise_fraction=sm.get("low_noise_fraction", 0.3),
        low_noise_weight=sm.get("low_noise_weight", 3.0),
        n_geom_channels=n_geom,
        geom_channel_names=geom_names,
    )

    if args.resume:
        logger.info("Resuming from: %s", args.resume)
        trainer.load_checkpoint(args.resume)

    # ---- Train --------------------------------------------------------
    num_epochs = tr.get("num_epochs", 500)
    logger.info("Starting 3D frame-2 training for %d epochs", num_epochs)
    try:
        trainer.train(num_epochs=num_epochs)
    except KeyboardInterrupt:
        logger.info("Interrupted — saving checkpoint...")
        trainer.save_checkpoint(name="interrupted.pt")
    except Exception as exc:
        logger.error("Training failed: %s", exc)
        raise

    logger.info("Training complete. Checkpoints: %s", tr.get("checkpoint_dir", "checkpoints/frame2_3d"))


if __name__ == "__main__":
    main()
