"""
Two-frame temporally linked sampling pipeline.

Implements the generative process:

    Frame 1:  I_t       ~ p( I    | C_t )
    Frame 2:  I_{t+1}   ~ p( I    | C_{t+1}, I_t )

where C_t and C_{t+1} are 3-channel conditioning maps derived from cell-centre
positions at times t and t+1 respectively.

Usage examples
--------------
# Provide centres files for both frames:
python -m sampling.sample_two_frame \\
    --frame1_ckpt checkpoints/first_attempt/best.pt \\
    --frame2_ckpt checkpoints/frame2/best.pt \\
    --centres_t  synthetic_cells/synthetic_0000_centres.npy \\
    --centres_t1 synthetic_cells/synthetic_0001_centres.npy \\
    --out_dir out/two_frame

# Use EMA weights from the checkpoints:
python -m sampling.sample_two_frame ... --use_ema

# Enable classifier-free guidance:
python -m sampling.sample_two_frame ... --cfg --guidance_scale 3.0
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.append(str(Path(__file__).parent.parent))

from models.diffusion import DDPM
from models.unet import ConditionalUNet
from preprocessing.generate_condition_maps import generate_conditioning_maps
from utils.normalization import to_zero_one

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------

def _build_model(cfg: dict, device: str) -> DDPM:
    """Instantiate a DDPM from config dict (no weights)."""
    u = cfg["unet"]
    unet = ConditionalUNet(
        in_channels=u["in_channels"],
        out_channels=u["out_channels"],
        condition_channels=u["condition_channels"],
        base_channels=u["base_channels"],
        channel_multipliers=u["channel_multipliers"],
        num_res_blocks=u["num_res_blocks"],
        attention_resolutions=u["attention_resolutions"],
        num_heads=u["num_heads"],
        time_emb_dim=u["time_emb_dim"],
        num_groups=u["norm_groups"],
        dropout=u["dropout"],
    )
    d = cfg["diffusion"]
    return DDPM(
        model=unet,
        timesteps=d["timesteps"],
        beta_schedule=d["beta_schedule"],
        beta_start=d.get("beta_start", 0.0001),
        beta_end=d.get("beta_end", 0.02),
        prediction_type=d["prediction_type"],
        loss_type=d["loss_type"],
    ).to(device)


def load_model(checkpoint_path: str, config_path: str, device: str,
               use_ema: bool = False) -> DDPM:
    """
    Load a trained DDPM from *checkpoint_path* using architecture from
    *config_path*.

    If *use_ema* is True and the checkpoint contains an EMA shadow, those
    weights are loaded instead of the raw model weights.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model = _build_model(cfg, device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if use_ema and "ema_shadow" in ckpt:
        # Apply EMA weights parameter-by-parameter
        shadow: dict = ckpt["ema_shadow"]
        state = model.state_dict()
        for name in state:
            if name in shadow:
                state[name] = shadow[name]
        model.load_state_dict(state)
        logger.info("Loaded EMA weights from %s", checkpoint_path)
    elif "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(
            "Loaded model weights from %s  (step=%s)",
            checkpoint_path,
            ckpt.get("step", "?"),
        )
    else:
        # Assume bare state-dict
        model.load_state_dict(ckpt)
        logger.info("Loaded bare state-dict from %s", checkpoint_path)

    model.eval()
    return model


# ------------------------------------------------------------------
# Conditioning helpers
# ------------------------------------------------------------------

def centres_to_cond(
    centres: np.ndarray,
    image_shape: tuple,
    heatmap_sigma: float = 3.0,
    boundary_sigma: float = 2.0,
    device: str = "cpu",
    batch_size: int = 1,
) -> torch.Tensor:
    """
    Build a (B, 3, H, W) float32 conditioning tensor from *centres*.

    centres: (N, 2) float array with (y, x) coordinates.
    """
    maps = generate_conditioning_maps(
        centres, image_shape,
        heatmap_sigma=heatmap_sigma,
        boundary_sigma=boundary_sigma,
    )  # (3, H, W) float32
    cond = torch.from_numpy(maps).unsqueeze(0)       # (1, 3, H, W)
    cond = cond.expand(batch_size, -1, -1, -1)       # (B, 3, H, W)
    return cond.to(device)


# ------------------------------------------------------------------
# Two-frame sampling
# ------------------------------------------------------------------

@torch.no_grad()
def sample_two_frames(
    model_frame1: DDPM,
    model_frame2: DDPM,
    centres_t: np.ndarray,
    centres_t1: np.ndarray,
    image_shape: tuple = (256, 256),
    heatmap_sigma: float = 3.0,
    boundary_sigma: float = 2.0,
    use_cfg: bool = False,
    guidance_scale: float = 3.0,
    device: str = "cpu",
    batch_size: int = 1,
):
    """
    Run the full two-frame pipeline.

    Parameters
    ----------
    model_frame1 : trained DDPM with condition_channels=3
    model_frame2 : trained DDPM with condition_channels=4
    centres_t    : (N0, 2) cell centres for frame t
    centres_t1   : (N1, 2) cell centres for frame t+1
    image_shape  : (H, W) of output images
    use_cfg      : use classifier-free guidance during sampling
    guidance_scale : CFG scale (only used when use_cfg=True)
    device       : torch device string

    Returns
    -------
    img_t  : (B, 1, H, W) float32 in [-1, 1]  — frame t
    img_t1 : (B, 1, H, W) float32 in [-1, 1]  — frame t+1
    """
    H, W = image_shape

    # ---- Frame 1: I_t ~ p(I | C_t) -----------------------------------
    cond_t = centres_to_cond(
        centres_t, image_shape, heatmap_sigma, boundary_sigma,
        device=device, batch_size=batch_size,
    )  # (B, 3, H, W)

    logger.info("Sampling frame 1  (condition_channels=3) ...")
    if use_cfg:
        img_t = model_frame1.sample_with_cfg(
            conditioning=cond_t,
            guidance_scale=guidance_scale,
        )
    else:
        img_t = model_frame1.sample(conditioning=cond_t)
    # img_t: (B, 1, H, W) in [-1, 1]

    # ---- Frame 2: I_{t+1} ~ p(I | C_{t+1}, I_t) ----------------------
    cond_t1_maps = centres_to_cond(
        centres_t1, image_shape, heatmap_sigma, boundary_sigma,
        device=device, batch_size=batch_size,
    )  # (B, 3, H, W)

    # Build 4-channel conditioning: [C_{t+1} maps (3ch) | I_t (1ch)]
    frame2_cond = torch.cat([cond_t1_maps, img_t], dim=1)  # (B, 4, H, W)

    logger.info("Sampling frame 2  (condition_channels=4) ...")
    if use_cfg:
        img_t1 = model_frame2.sample_with_cfg(
            conditioning=frame2_cond,
            guidance_scale=guidance_scale,
        )
    else:
        img_t1 = model_frame2.sample(conditioning=frame2_cond)
    # img_t1: (B, 1, H, W) in [-1, 1]

    return img_t, img_t1


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate a temporally consistent image pair."
    )
    parser.add_argument(
        "--frame1_ckpt", required=True,
        help="Path to frame-1 checkpoint (condition_channels=3)"
    )
    parser.add_argument(
        "--frame2_ckpt", required=True,
        help="Path to frame-2 checkpoint (condition_channels=4)"
    )
    parser.add_argument(
        "--frame1_config", default="configs/frame1.yaml",
        help="Model config for frame 1"
    )
    parser.add_argument(
        "--frame2_config", default="configs/frame2.yaml",
        help="Model config for frame 2"
    )
    parser.add_argument(
        "--centres_t", required=True,
        help="Path to .npy file with cell centres at time t  (N, 2)"
    )
    parser.add_argument(
        "--centres_t1", required=True,
        help="Path to .npy file with cell centres at time t+1  (N, 2)"
    )
    parser.add_argument(
        "--image_size", type=int, default=256,
        help="Spatial size H=W of the generated images"
    )
    parser.add_argument(
        "--out_dir", default="two_frame_output",
        help="Directory to write output images and arrays"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Number of samples to generate in parallel"
    )
    parser.add_argument(
        "--use_ema", action="store_true",
        help="Load EMA weights from checkpoints"
    )
    parser.add_argument(
        "--cfg", action="store_true",
        help="Use classifier-free guidance during sampling"
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=3.0,
        help="CFG guidance scale"
    )
    parser.add_argument(
        "--heatmap_sigma", type=float, default=3.0
    )
    parser.add_argument(
        "--boundary_sigma", type=float, default=2.0
    )
    parser.add_argument(
        "--save_tiff", action="store_true",
        help="Also save each pair as a (2, H, W) float32 TIFF  "
             "(frame t in slice 0, frame t+1 in slice 1)"
    )
    parser.add_argument(
        "--device", default=None,
        help="Device: cuda / cpu (auto-detected when omitted)"
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load centres -------------------------------------------------
    centres_t  = np.load(args.centres_t).astype(np.float32)
    centres_t1 = np.load(args.centres_t1).astype(np.float32)
    logger.info("centres_t: %d cells   centres_t1: %d cells",
                len(centres_t), len(centres_t1))

    # ---- Load models --------------------------------------------------
    logger.info("Loading frame-1 model from %s", args.frame1_ckpt)
    model1 = load_model(
        args.frame1_ckpt, args.frame1_config,
        device=device, use_ema=args.use_ema,
    )

    logger.info("Loading frame-2 model from %s", args.frame2_ckpt)
    model2 = load_model(
        args.frame2_ckpt, args.frame2_config,
        device=device, use_ema=args.use_ema,
    )

    # ---- Sample -------------------------------------------------------
    image_shape = (args.image_size, args.image_size)
    img_t, img_t1 = sample_two_frames(
        model_frame1=model1,
        model_frame2=model2,
        centres_t=centres_t,
        centres_t1=centres_t1,
        image_shape=image_shape,
        heatmap_sigma=args.heatmap_sigma,
        boundary_sigma=args.boundary_sigma,
        use_cfg=args.cfg,
        guidance_scale=args.guidance_scale,
        device=device,
        batch_size=args.batch_size,
    )

    # ---- Save ---------------------------------------------------------
    img_t_np  = img_t.cpu().numpy()   # (B, 1, H, W)
    img_t1_np = img_t1.cpu().numpy()  # (B, 1, H, W)

    for b in range(img_t_np.shape[0]):
        np.save(out_dir / f"frame_t_sample{b:03d}.npy",  img_t_np[b, 0])
        np.save(out_dir / f"frame_t1_sample{b:03d}.npy", img_t1_np[b, 0])

    logger.info("Saved %d frame pairs to %s", img_t_np.shape[0], out_dir)

    # ---- Optional: save concatenated (2, H, W) TIFF ------------------
    if args.save_tiff:
        import tifffile
        for b in range(img_t_np.shape[0]):
            # Rescale from [-1, 1] to [0, 1] float32
            frame_t  = (img_t_np[b, 0]  + 1.0) / 2.0
            frame_t1 = (img_t1_np[b, 0] + 1.0) / 2.0
            # Stack along axis 0: shape (2, H, W), float32 in [0, 1]
            pair = np.stack([frame_t, frame_t1], axis=0).astype(np.float32)
            tiff_path = out_dir / f"pair_sample{b:03d}.tif"
            tifffile.imwrite(str(tiff_path), pair)
        logger.info("Saved %d TIFF pairs (2, H, W) [0,1] to %s", img_t_np.shape[0], out_dir)

    # ---- Optional: save quick PNG previews ----------------------------
    try:
        import matplotlib.pyplot as plt
        for b in range(img_t_np.shape[0]):
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            f_t  = to_zero_one(img_t_np[b, 0])
            f_t1 = to_zero_one(img_t1_np[b, 0])
            diff = np.abs(f_t1 - f_t)

            axes[0].imshow(f_t,  cmap="gray", vmin=0, vmax=1)
            axes[0].set_title("Frame t  (I_t)")
            axes[1].imshow(f_t1, cmap="gray", vmin=0, vmax=1)
            axes[1].set_title("Frame t+1  (I_{t+1})")
            axes[2].imshow(diff, cmap="hot",  vmin=0, vmax=0.5)
            axes[2].set_title("|I_{t+1} − I_t|")

            for ax in axes:
                ax.axis("off")

            fig.tight_layout()
            fig.savefig(out_dir / f"pair_preview_sample{b:03d}.png", dpi=120)
            plt.close(fig)

        logger.info("Saved PNG previews to %s", out_dir)
    except Exception as exc:
        logger.warning("Could not save PNG previews: %s", exc)


if __name__ == "__main__":
    main()
