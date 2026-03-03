"""
Two-frame temporally linked 3D sampling pipeline.

Implements the generative process:

    Frame 1:  V_t       ~ p( V    | C_t )
    Frame 2:  V_{t+1}   ~ p( V    | C_{t+1}, V_t )

where C_t and C_{t+1} are 2-channel conditioning maps (heatmap + distance)
derived from 3D cell-centre positions at times t and t+1 respectively.

Usage examples
--------------
# Realistic mode — centres derived from training-data statistics (recommended):
python -m sampling.sample_two_frame3d \\
    --frame1_ckpt checkpoints/frame1_3d/best.pt \\
    --frame2_ckpt checkpoints/frame2_3d/best.pt \\
    --method realistic \\
    --out_dir out/two_frame_3d

# Provide explicit centres files for both frames:
python -m sampling.sample_two_frame3d \\
    --frame1_ckpt checkpoints/frame1_3d/best.pt \\
    --frame2_ckpt checkpoints/frame2_3d/best.pt \\
    --method from_file \\
    --centres_t  synthetic_cells/synthetic_0000_centres.npy \\
    --centres_t1 synthetic_cells/synthetic_0001_centres.npy \\
    --out_dir out/two_frame_3d

# Use EMA weights from the checkpoints:
python -m sampling.sample_two_frame3d ... --use_ema

# Enable classifier-free guidance:
python -m sampling.sample_two_frame3d ... --cfg --guidance_scale 3.0
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
import tifffile
from skimage.exposure import match_histograms

sys.path.append(str(Path(__file__).parent.parent))

from models.diffusion3d import DDPM3D
from models.unet3d import ConditionalUNet3D
from preprocessing.generate_condition_maps3d import generate_conditioning_maps3d
from sampling.generate_centres3d import generate_realistic_centres3d
from utils.normalization import to_zero_one

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Histogram matching helper
# ------------------------------------------------------------------

def _histogram_match_volume(
    source: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """
    Match the intensity histogram of `source` to `reference`.

    Both arrays are float32 in [-1, 1] (or any consistent range).
    Returns source rescaled to match reference's cumulative distribution,
    clipped to the original data range.

    Parameters
    ----------
    source    : (D, H, W) float32 — volume whose histogram will be adjusted
    reference : (D, H, W) float32 — volume whose histogram is the target

    Returns
    -------
    matched : (D, H, W) float32
    """
    vmin = min(source.min(), reference.min())
    vmax = max(source.max(), reference.max())
    matched = match_histograms(source.astype(np.float64),
                               reference.astype(np.float64))
    return np.clip(matched, vmin, vmax).astype(np.float32)


_GEOM_ORDER = ["heatmap", "distance"]  # 3D only has 2 geometry channels


def _resolve_conditioning(cfg: dict) -> tuple[dict, int, bool, int]:
    """Resolve conditioning settings from config.

    Returns:
        active_geom: dict of active geometry channels
        n_geom: number of geometry conditioning channels
        prev_frame: whether previous frame is appended as an extra channel
        condition_channels: total conditioning channels expected by the UNet
    """
    u = cfg.get("unet", {})
    conditioning = u.get("conditioning")

    if conditioning is None:
        # Legacy: explicit integer
        condition_channels = int(u.get("condition_channels", 2))
        prev_frame = bool(u.get("prev_frame", False))
        n_geom = condition_channels - (1 if prev_frame else 0)
        if n_geom < 1 or n_geom > 2:
            raise ValueError(
                f"Legacy unet.condition_channels for 3D must imply 1 or 2 geom channels, got {n_geom}"
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

def _build_model(cfg: dict, device: str) -> DDPM3D:
    """Instantiate a DDPM3D from config dict (no weights)."""
    u = cfg["unet"]
    _active_geom, _n_geom, _prev_frame, condition_channels = _resolve_conditioning(cfg)
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
    return DDPM3D(
        model=unet,
        timesteps=d["timesteps"],
        beta_schedule=d["beta_schedule"],
        beta_start=d.get("beta_start", 0.0001),
        beta_end=d.get("beta_end", 0.02),
        prediction_type=d["prediction_type"],
        loss_type=d["loss_type"],
    ).to(device)


def load_model(checkpoint_path: str, config_path: str, device: str,
               use_ema: bool = False) -> DDPM3D:
    """
    Load a trained DDPM3D from *checkpoint_path* using architecture from
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
        # w shape: (base_channels, in_channels + condition_channels, kD, kH, kW)
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

def centres_to_cond_3d(
    centres: np.ndarray,
    volume_shape: tuple,
    heatmap_sigma: float = 3.0,
    distance_percentile: float = 95.0,
    active_channels: dict | None = None,
    device: str = "cpu",
    batch_size: int = 1,
) -> torch.Tensor:
    """
    Build a (B, n_geom, D, H, W) float32 conditioning tensor from *centres*.

    centres: (N, 3) float array with (z, y, x) coordinates.
    """
    maps = generate_conditioning_maps3d(
        centres, volume_shape,
        heatmap_sigma=heatmap_sigma,
        distance_percentile=distance_percentile,
        active_channels=active_channels,
    )  # (n_geom, D, H, W) float32
    cond = torch.from_numpy(maps).unsqueeze(0)       # (1, n_geom, D, H, W)
    cond = cond.expand(batch_size, -1, -1, -1, -1)   # (B, n_geom, D, H, W)
    return cond.to(device)


# ------------------------------------------------------------------
# Two-frame sampling
# ------------------------------------------------------------------

@torch.no_grad()
def sample_two_frames_3d(
    model_frame1: DDPM3D,
    model_frame2: DDPM3D,
    centres_t: np.ndarray,
    centres_t1: np.ndarray,
    volume_shape: tuple = (128, 128, 128),
    heatmap_sigma: float = 3.0,
    distance_percentile: float = 95.0,
    use_cfg: bool = False,
    guidance_scale: float = 3.0,
    device: str = "cpu",
    batch_size: int = 1,
    active_geom_frame1: dict | None = None,
    active_geom_frame2: dict | None = None,
    prev_frame2: bool = True,
    match_histograms: bool = False,
):
    """
    Run the full 3D two-frame pipeline.

    Parameters
    ----------
    model_frame1     : trained DDPM3D (frame1_3d model)
    model_frame2     : trained DDPM3D (frame2_3d model, with prev_frame conditioning)
    centres_t        : (N0, 3) cell centres for frame t in (z, y, x)
    centres_t1       : (N1, 3) cell centres for frame t+1 in (z, y, x)
    volume_shape     : (D, H, W) of output volumes
    use_cfg          : use classifier-free guidance during sampling
    guidance_scale   : CFG scale (only used when use_cfg=True)
    device           : torch device string
    match_histograms : if True, match the histogram of V_{t+1} to V_t
                       so adjacent frames share the same intensity distribution.
                       Applied per-sample in the batch after generation.

    Returns
    -------
    vol_t  : (B, 1, D, H, W) float32 in [-1, 1]  — frame t
    vol_t1 : (B, 1, D, H, W) float32 in [-1, 1]  — frame t+1
    """
    D, H, W = volume_shape

    # ---- Frame 1: V_t ~ p(V | C_t) -----------------------------------
    cond_t = centres_to_cond_3d(
        centres_t, volume_shape, heatmap_sigma, distance_percentile,
        active_channels=active_geom_frame1,
        device=device, batch_size=batch_size,
    )  # (B, n_geom, D, H, W)

    expected_c1 = int(getattr(model_frame1.model, "condition_channels", cond_t.shape[1]))
    if cond_t.shape[1] != expected_c1:
        raise ValueError(
            f"Frame-1 conditioning channel mismatch: model expects {expected_c1} "
            f"but generated {cond_t.shape[1]}. Check frame1_3d config conditioning flags." 
        )

    logger.info("Sampling 3D frame 1  (condition_channels=%d) ...", cond_t.shape[1])
    if use_cfg:
        vol_t = model_frame1.sample_with_cfg(
            conditioning=cond_t,
            guidance_scale=guidance_scale,
        )
    else:
        vol_t = model_frame1.sample(conditioning=cond_t)
    # vol_t: (B, 1, D, H, W) in [-1, 1]

    # ---- Frame 2: V_{t+1} ~ p(V | C_{t+1}, V_t) ----------------------
    cond_t1_maps = centres_to_cond_3d(
        centres_t1, volume_shape, heatmap_sigma, distance_percentile,
        active_channels=active_geom_frame2,
        device=device, batch_size=batch_size,
    )  # (B, n_geom, D, H, W)

    # Build conditioning for frame 2.
    # V_t is normalised to [0, 1] to match training convention in
    # TemporalPairDataset3D where vol_t0 is rescaled via clip(v*0.5+0.5, 0, 1).
    if prev_frame2:
        vol_t_01 = (vol_t * 0.5 + 0.5).clamp(0.0, 1.0)  # [-1,1] → [0,1]
        frame2_cond = torch.cat([cond_t1_maps, vol_t_01], dim=1)  # (B, n_geom+1, D, H, W)
    else:
        frame2_cond = cond_t1_maps

    expected_c2 = int(getattr(model_frame2.model, "condition_channels", frame2_cond.shape[1]))
    if frame2_cond.shape[1] != expected_c2:
        raise ValueError(
            f"Frame-2 conditioning channel mismatch: model expects {expected_c2} "
            f"but generated {frame2_cond.shape[1]}. Check frame2_3d config conditioning/prev_frame." 
        )

    logger.info("Sampling 3D frame 2  (condition_channels=%d) ...", frame2_cond.shape[1])
    if use_cfg:
        vol_t1 = model_frame2.sample_with_cfg(
            conditioning=frame2_cond,
            guidance_scale=guidance_scale,
        )
    else:
        vol_t1 = model_frame2.sample(conditioning=frame2_cond)
    # vol_t1: (B, 1, D, H, W) in [-1, 1]

    # ---- Optional: histogram match V_{t+1} → V_t ----------------------
    if match_histograms:
        logger.info("Applying histogram matching: V_{t+1} → V_t histogram ...")
        vol_t_np  = vol_t.cpu().numpy()   # (B, 1, D, H, W)
        vol_t1_np = vol_t1.cpu().numpy()
        for b in range(vol_t_np.shape[0]):
            vol_t1_np[b, 0] = _histogram_match_volume(
                source=vol_t1_np[b, 0],
                reference=vol_t_np[b, 0],
            )
        vol_t1 = torch.from_numpy(vol_t1_np).to(vol_t.device)

    return vol_t, vol_t1


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate a temporally consistent 3D volume pair."
    )
    parser.add_argument(
        "--frame1_ckpt", required=True,
        help="Path to frame-1 3D checkpoint"
    )
    parser.add_argument(
        "--frame2_ckpt", required=True,
        help="Path to frame-2 3D checkpoint"
    )
    parser.add_argument(
        "--frame1_config", default="configs/frame1_3d.yaml",
        help="Model config for frame 1"
    )
    parser.add_argument(
        "--frame2_config", default="configs/frame2_3d.yaml",
        help="Model config for frame 2"
    )
    parser.add_argument(
        "--method", default="from_file", choices=["from_file", "realistic"],
        help=(
            "Centre generation strategy. "
            "'from_file' (default): load from --centres_t / --centres_t1 .npy files. "
            "'realistic': generate t centres from training-data statistics (centre_generation_3d "
            "block in frame1_config), then derive t+1 centres by applying a small Gaussian "
            "displacement (--displacement_sigma) to each t centre."
        ),
    )
    parser.add_argument(
        "--centres_t", default=None,
        help="Path to .npy file with cell centres at time t  (N, 3) in (z,y,x). "
             "Required when --method from_file."
    )
    parser.add_argument(
        "--centres_t1", default=None,
        help="Path to .npy file with cell centres at time t+1  (N, 3) in (z,y,x). "
             "Required when --method from_file."
    )
    parser.add_argument(
        "--displacement_sigma", type=float, default=3.0,
        help=(
            "Std-dev (voxels) of the Gaussian displacement applied to each t centre to "
            "produce t+1 centres in 'realistic' mode. Models typical inter-frame cell motion. "
            "Default: 3.0 voxels."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for realistic centre generation (default: None = non-deterministic)."
    )
    parser.add_argument(
        "--volume_size", type=int, default=128,
        help="Spatial size D=H=W of the generated volumes"
    )
    parser.add_argument(
        "--out_dir", default="two_frame_3d_output",
        help="Directory to write output volumes"
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
        "--match_histograms", action="store_true",
        help="Match V_{t+1} intensity histogram to V_t after generation"
    )
    parser.add_argument(
        "--heatmap_sigma", type=float, default=3.0
    )
    parser.add_argument(
        "--distance_percentile", type=float, default=95.0
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

    # ---- Load configs early (needed for realistic centre generation) --
    with open(args.frame1_config) as f:
        cfg1 = yaml.safe_load(f)
    with open(args.frame2_config) as f:
        cfg2 = yaml.safe_load(f)

    # ---- Load / generate centres --------------------------------------
    volume_shape = (args.volume_size, args.volume_size, args.volume_size)

    if args.method == "from_file":
        if args.centres_t is None or args.centres_t1 is None:
            parser.error("--centres_t and --centres_t1 are required when --method from_file")
        centres_t  = np.load(args.centres_t).astype(np.float32)
        centres_t1 = np.load(args.centres_t1).astype(np.float32)
        logger.info("Loaded centres_t: %d cells   centres_t1: %d cells",
                    len(centres_t), len(centres_t1))

    elif args.method == "realistic":
        # Build realistic_params from the frame1 config
        cg_cfg = cfg1.get("centre_generation_3d", {})
        if not cg_cfg:
            logger.warning(
                "centre_generation_3d block not found in frame1 config — "
                "using fallback defaults. Run scripts/analyze_training_stats.py "
                "--save_stats to populate real values."
            )
        realistic_params = {
            "n_mean":         cg_cfg.get("n_mean",        20.0),
            "n_std":          cg_cfg.get("n_std",          8.0),
            "n_min":          cg_cfg.get("n_min",          5),
            "n_max":          cg_cfg.get("n_max",          60),
            "min_distance":   cg_cfg.get("min_distance",   8.0),
            "density_grid_z": cg_cfg.get("density_grid_z", [1.0] * 16),
            "density_grid_y": cg_cfg.get("density_grid_y", [1.0] * 16),
            "density_grid_x": cg_cfg.get("density_grid_x", [1.0] * 16),
            "border_margin":  cg_cfg.get("border_margin",  10),
            "max_attempts":   cg_cfg.get("max_attempts",   50),
        }

        rng = np.random.default_rng(args.seed)
        seed_t = int(rng.integers(0, 2**31)) if args.seed is not None else None

        logger.info(
            "Generating t centres (realistic): n_mean=%.1f, n_std=%.1f, "
            "min_distance=%.1f, displacement_sigma=%.1f",
            realistic_params["n_mean"], realistic_params["n_std"],
            realistic_params["min_distance"], args.displacement_sigma,
        )

        centres_t = generate_realistic_centres3d(
            volume_shape=volume_shape,
            seed=seed_t,
            **realistic_params,
        )

        # Derive t+1 centres by applying a Gaussian displacement to each t centre.
        # This models the small inter-frame motion of cells without generating an
        # entirely independent layout (which would be biologically unrealistic).
        border_margin = realistic_params["border_margin"]
        d, h, w = volume_shape
        displacement = rng.normal(
            0.0, args.displacement_sigma, centres_t.shape
        ).astype(np.float32)
        centres_t1 = centres_t + displacement
        # Clamp each axis to stay within the valid interior
        centres_t1[:, 0] = np.clip(centres_t1[:, 0], border_margin, d - border_margin)
        centres_t1[:, 1] = np.clip(centres_t1[:, 1], border_margin, h - border_margin)
        centres_t1[:, 2] = np.clip(centres_t1[:, 2], border_margin, w - border_margin)

        logger.info(
            "Generated centres_t: %d cells   centres_t1: %d cells (displaced, σ=%.1f vox)",
            len(centres_t), len(centres_t1), args.displacement_sigma,
        )

    else:
        raise ValueError(f"Unknown --method: {args.method}")

    # ---- Load models --------------------------------------------------
    active_geom1, _n_geom1, prev1, cond_ch1 = _resolve_conditioning(cfg1)
    active_geom2, _n_geom2, prev2, cond_ch2 = _resolve_conditioning(cfg2)
    if prev1:
        logger.warning("frame1_3d config has unet.prev_frame=true; ignoring for frame-1 sampling")

    logger.info("Loading frame-1 3D model from %s", args.frame1_ckpt)
    model1 = load_model(
        args.frame1_ckpt, args.frame1_config,
        device=device, use_ema=args.use_ema,
    )

    if int(getattr(model1.model, "condition_channels", cond_ch1)) != cond_ch1:
        logger.warning(
            "frame1_3d config expects condition_channels=%d but model reports %d; "
            "if sampling fails, align config with checkpoint",
            cond_ch1,
            int(getattr(model1.model, "condition_channels", -1)),
        )

    logger.info("Loading frame-2 3D model from %s", args.frame2_ckpt)
    model2 = load_model(
        args.frame2_ckpt, args.frame2_config,
        device=device, use_ema=args.use_ema,
    )

    if int(getattr(model2.model, "condition_channels", cond_ch2)) != cond_ch2:
        logger.warning(
            "frame2_3d config expects condition_channels=%d but model reports %d; "
            "if sampling fails, align config with checkpoint",
            cond_ch2,
            int(getattr(model2.model, "condition_channels", -1)),
        )

    # ---- Sample -------------------------------------------------------
    vol_t, vol_t1 = sample_two_frames_3d(
        model_frame1=model1,
        model_frame2=model2,
        centres_t=centres_t,
        centres_t1=centres_t1,
        volume_shape=volume_shape,
        heatmap_sigma=args.heatmap_sigma,
        distance_percentile=args.distance_percentile,
        use_cfg=args.cfg,
        guidance_scale=args.guidance_scale,
        device=device,
        batch_size=args.batch_size,
        active_geom_frame1=active_geom1,
        active_geom_frame2=active_geom2,
        prev_frame2=prev2,
        match_histograms=args.match_histograms,
    )

    # ---- Save ---------------------------------------------------------
    vol_t_np  = vol_t.cpu().numpy()   # (B, 1, D, H, W)
    vol_t1_np = vol_t1.cpu().numpy()  # (B, 1, D, H, W)

    for b in range(vol_t_np.shape[0]):
        # Save as .npy (raw [-1, 1] data)
        np.save(out_dir / f"frame_t_sample{b:03d}.npy",  vol_t_np[b, 0])
        np.save(out_dir / f"frame_t1_sample{b:03d}.npy", vol_t1_np[b, 0])
        
        # Save as TIFF (rescaled to [0, 1])
        vol_t_01  = to_zero_one(vol_t_np[b, 0]).astype(np.float32)
        vol_t1_01 = to_zero_one(vol_t1_np[b, 0]).astype(np.float32)
        
        tifffile.imwrite(
            str(out_dir / f"frame_t_sample{b:03d}.tif"),
            vol_t_01
        )
        tifffile.imwrite(
            str(out_dir / f"frame_t1_sample{b:03d}.tif"),
            vol_t1_01
        )
        
        # Save paired volume: stack as (2, D, H, W)
        pair = np.stack([vol_t_01, vol_t1_01], axis=0).astype(np.float32)
        tifffile.imwrite(
            str(out_dir / f"pair_sample{b:03d}.tif"),
            pair
        )

    logger.info("Saved %d 3D frame pairs to %s", vol_t_np.shape[0], out_dir)
    logger.info("  - Individual volumes: frame_t_sample***.tif and frame_t1_sample***.tif")
    logger.info("  - Paired volumes (2,D,H,W): pair_sample***.tif")


if __name__ == "__main__":
    main()
