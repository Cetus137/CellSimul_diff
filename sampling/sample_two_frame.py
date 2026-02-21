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


_GEOM_ORDER = ["heatmap", "distance", "boundary"]


def _resolve_conditioning(cfg: dict) -> tuple[dict, int, bool, int]:
    """Resolve conditioning settings from config.

    Returns:
        active_geom: dict of active geometry channels in canonical order
        n_geom: number of geometry conditioning channels
        prev_frame: whether previous frame is appended as an extra channel
        condition_channels: total conditioning channels expected by the UNet
    """
    u = cfg.get("unet", {})
    conditioning = u.get("conditioning")

    if conditioning is None:
        # Legacy: explicit integer means first N geometry channels are active.
        n_geom = int(u.get("condition_channels", 3))
        if n_geom < 1 or n_geom > 3:
            raise ValueError(
                f"Legacy unet.condition_channels must be 1..3, got {n_geom}"
            )
        active_geom = {k: (i < n_geom) for i, k in enumerate(_GEOM_ORDER)}
    else:
        active_geom = {k: bool(conditioning.get(k, False)) for k in _GEOM_ORDER}
        n_geom = sum(1 for v in active_geom.values() if v)
        if n_geom < 1:
            raise ValueError("unet.conditioning must enable at least one channel")

    prev_frame = bool(u.get("prev_frame", False))
    condition_channels = n_geom + (1 if prev_frame else 0)
    return active_geom, n_geom, prev_frame, condition_channels


# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------

def _build_model(cfg: dict, device: str) -> DDPM:
    """Instantiate a DDPM from config dict (no weights)."""
    u = cfg["unet"]
    _active_geom, _n_geom, _prev_frame, condition_channels = _resolve_conditioning(cfg)
    unet = ConditionalUNet(
        in_channels=u["in_channels"],
        out_channels=u["out_channels"],
        condition_channels=condition_channels,
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

    def _infer_ckpt_condition_channels() -> int | None:
        """Infer total condition_channels from checkpoint, if possible."""
        state_dict = ckpt.get("model_state_dict") if isinstance(ckpt, dict) else None
        if state_dict is None and isinstance(ckpt, dict):
            # bare state-dict checkpoint
            state_dict = ckpt
        if not isinstance(state_dict, dict):
            return None

        w = None
        if "model.init_conv.weight" in state_dict:
            w = state_dict["model.init_conv.weight"]
        elif "init_conv.weight" in state_dict:
            w = state_dict["init_conv.weight"]
        if w is None or not hasattr(w, "shape"):
            return None
        in_channels = int(cfg.get("unet", {}).get("in_channels", 1))
        # w shape: (base_channels, in_channels + condition_channels, kH, kW)
        return int(w.shape[1]) - in_channels

    def _rethrow_with_context(exc: Exception) -> None:
        inferred = _infer_ckpt_condition_channels()
        if inferred is None:
            raise
        _active_geom, _n_geom, _prev_frame, cfg_cond = _resolve_conditioning(cfg)
        raise RuntimeError(
            "Checkpoint/config architecture mismatch while loading weights. "
            f"Config expects condition_channels={cfg_cond}, but checkpoint implies "
            f"condition_channels={inferred}. Align `unet.conditioning`/`unet.prev_frame` "
            "in the config with how the checkpoint was trained."
        ) from exc

    if use_ema and "ema_shadow" in ckpt:
        # Apply EMA weights parameter-by-parameter
        shadow: dict = ckpt["ema_shadow"]
        state = model.state_dict()
        for name in state:
            if name in shadow:
                state[name] = shadow[name]
        try:
            model.load_state_dict(state)
        except Exception as exc:
            _rethrow_with_context(exc)
        logger.info("Loaded EMA weights from %s", checkpoint_path)
    elif "model_state_dict" in ckpt:
        try:
            model.load_state_dict(ckpt["model_state_dict"])
        except Exception as exc:
            _rethrow_with_context(exc)
        logger.info(
            "Loaded model weights from %s  (step=%s)",
            checkpoint_path,
            ckpt.get("step", "?"),
        )
    else:
        # Assume bare state-dict
        try:
            model.load_state_dict(ckpt)
        except Exception as exc:
            _rethrow_with_context(exc)
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
    active_channels: dict | None = None,
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
        active_channels=active_channels,
    )  # (3, H, W) float32
    cond = torch.from_numpy(maps).unsqueeze(0)       # (1, C, H, W)
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
    active_geom_frame1: dict | None = None,
    active_geom_frame2: dict | None = None,
    prev_frame2: bool = True,
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
        active_channels=active_geom_frame1,
        device=device, batch_size=batch_size,
    )  # (B, C1, H, W)

    expected_c1 = int(getattr(model_frame1.model, "condition_channels", cond_t.shape[1]))
    if cond_t.shape[1] != expected_c1:
        raise ValueError(
            f"Frame-1 conditioning channel mismatch: model expects {expected_c1} "
            f"but generated {cond_t.shape[1]}. Check frame1 config conditioning flags." 
        )

    logger.info("Sampling frame 1  (condition_channels=%d) ...", cond_t.shape[1])
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
        active_channels=active_geom_frame2,
        device=device, batch_size=batch_size,
    )  # (B, Cg, H, W)

    # Build conditioning for frame 2.
    if prev_frame2:
        frame2_cond = torch.cat([cond_t1_maps, img_t], dim=1)  # (B, Cg+1, H, W)
    else:
        frame2_cond = cond_t1_maps

    expected_c2 = int(getattr(model_frame2.model, "condition_channels", frame2_cond.shape[1]))
    if frame2_cond.shape[1] != expected_c2:
        raise ValueError(
            f"Frame-2 conditioning channel mismatch: model expects {expected_c2} "
            f"but generated {frame2_cond.shape[1]}. Check frame2 config conditioning/prev_frame." 
        )

    logger.info("Sampling frame 2  (condition_channels=%d) ...", frame2_cond.shape[1])
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
        help="Path to frame-1 checkpoint"
    )
    parser.add_argument(
        "--frame2_ckpt", required=True,
        help="Path to frame-2 checkpoint"
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
    with open(args.frame1_config) as f:
        cfg1 = yaml.safe_load(f)
    with open(args.frame2_config) as f:
        cfg2 = yaml.safe_load(f)

    active_geom1, _n_geom1, prev1, cond_ch1 = _resolve_conditioning(cfg1)
    active_geom2, _n_geom2, prev2, cond_ch2 = _resolve_conditioning(cfg2)
    if prev1:
        logger.warning("frame1 config has unet.prev_frame=true; ignoring for frame-1 sampling")

    logger.info("Loading frame-1 model from %s", args.frame1_ckpt)
    model1 = load_model(
        args.frame1_ckpt, args.frame1_config,
        device=device, use_ema=args.use_ema,
    )

    if int(getattr(model1.model, "condition_channels", cond_ch1)) != cond_ch1:
        logger.warning(
            "frame1 config expects condition_channels=%d but model reports %d; "
            "if sampling fails, align config with checkpoint",
            cond_ch1,
            int(getattr(model1.model, "condition_channels", -1)),
        )

    logger.info("Loading frame-2 model from %s", args.frame2_ckpt)
    model2 = load_model(
        args.frame2_ckpt, args.frame2_config,
        device=device, use_ema=args.use_ema,
    )

    if int(getattr(model2.model, "condition_channels", cond_ch2)) != cond_ch2:
        logger.warning(
            "frame2 config expects condition_channels=%d but model reports %d; "
            "if sampling fails, align config with checkpoint",
            cond_ch2,
            int(getattr(model2.model, "condition_channels", -1)),
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
        active_geom_frame1=active_geom1,
        active_geom_frame2=active_geom2,
        prev_frame2=prev2,
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
        try:
            from skimage.exposure import match_histograms as _match_histograms
            _have_skimage = True
        except ImportError:
            _have_skimage = False
            logger.warning("skimage not available; skipping histogram matching in previews")

        for b in range(img_t_np.shape[0]):
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            f_t  = to_zero_one(img_t_np[b, 0])
            f_t1 = to_zero_one(img_t1_np[b, 0])

            # Histogram-match f_t1 to f_t so both panels share the same
            # intensity distribution, making the diff reveal structure only.
            if _have_skimage:
                f_t1_disp = _match_histograms(f_t1, f_t).astype(np.float32)
                t1_title  = "Frame t+1  (I_{t+1}, hist-matched)"
            else:
                f_t1_disp = f_t1
                t1_title  = "Frame t+1  (I_{t+1})"

            diff = np.abs(f_t1_disp - f_t)

            axes[0].imshow(f_t,       cmap="gray", vmin=0, vmax=1)
            axes[0].set_title("Frame t  (I_t)")
            axes[1].imshow(f_t1_disp, cmap="gray", vmin=0, vmax=1)
            axes[1].set_title(t1_title)
            axes[2].imshow(diff,      cmap="hot",  vmin=0, vmax=0.5)
            axes[2].set_title("|I_{t+1} − I_t|")

            # Overlay cell centres as red crosses (y,x → scatter takes x,y)
            if len(centres_t) > 0:
                axes[0].scatter(
                    centres_t[:, 1], centres_t[:, 0],
                    c="red", marker="x", s=40, linewidths=1.5,
                    label=f"{len(centres_t)} cells",
                )
                axes[0].legend(loc="upper right", fontsize=7, framealpha=0.7)
            if len(centres_t1) > 0:
                axes[1].scatter(
                    centres_t1[:, 1], centres_t1[:, 0],
                    c="red", marker="x", s=40, linewidths=1.5,
                    label=f"{len(centres_t1)} cells",
                )
                axes[1].legend(loc="upper right", fontsize=7, framealpha=0.7)

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
