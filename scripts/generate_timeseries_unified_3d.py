"""
Autoregressive 3D timeseries generation using the unified model.

Generates a sequence of N volumetric frames autoregressively:

    V_0 ~ p(V | heatmap_0,  zeros     )   [frame 0 — no previous frame]
    V_1 ~ p(V | heatmap_1,  V_0 in 01 )
    V_2 ~ p(V | heatmap_2,  V_1 in 01 )
    ...

A single unified checkpoint (configs/unified_3d.yaml) is used throughout.
Each generated V_t is reused immediately as the prev-frame conditioning for V_{t+1}.

Two centre-generation modes
---------------------------
  --method from_files  (default)
      Load pre-generated centre files from --centres_dir.
      Files must be named: synthetic_3d_NNNN_centres.npy

  --method realistic
      Generate frame-0 centres from data statistics stored in the config's
      centre_generation_3d block (populate with analyze_training_stats.py
      --save_stats).  Each subsequent frame's centres are derived by applying
      an isotropic Gaussian displacement (--displacement_sigma voxels) to the
      previous frame's centres, modelling realistic inter-frame cell motion.

Usage:
    # Load pre-computed centre files:
    python -m scripts.generate_timeseries_unified_3d \\
        --checkpoint checkpoints/unified_3d/best.pt \\
        --config     configs/unified_3d.yaml \\
        --centres_dir synthetic_cells_3d/ \\
        --start_t 0 --num_frames 10 \\
        --out_dir timeseries_output/unified_3d/

    # Generate centres on-the-fly:
    python -m scripts.generate_timeseries_unified_3d \\
        --checkpoint checkpoints/unified_3d/best.pt \\
        --config     configs/unified_3d.yaml \\
        --method realistic \\
        --displacement_sigma 3.0 \\
        --num_frames 10 \\
        --out_dir timeseries_output/unified_3d/

Output per frame:
    t{TTTT}_vol.tif             — generated volume [0,1] float32
    t{TTTT}_centres.npy         — cell centres used for heatmap conditioning
    timeseries_t0000-tNNNN.tif  — full T×D×H×W TIFF stack
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import tifffile
import yaml
from skimage.exposure import match_histograms as skimage_match_histograms

sys.path.append(str(Path(__file__).parent.parent))

from sampling.generate_centres3d import generate_realistic_centres3d
from sampling.sample_two_frame3d import (
    load_model,
    centres_to_cond_3d,
    _resolve_conditioning,
)
from utils.normalization import to_zero_one

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Centre helpers
# ──────────────────────────────────────────────────────────────────────────────


def displace_centres(
    centres: np.ndarray,
    sigma: float,
    volume_shape: tuple,
    border_margin: int,
    rng: np.random.Generator,
    min_distance: float = 0.0,
    max_iterations: int = 50,
) -> np.ndarray:
    """
    Derive next-frame centres by applying isotropic Gaussian displacement.

    Each cell moves independently by a 3D Gaussian offset with std=sigma
    voxels, then is clamped to stay within the valid interior.  If
    ``min_distance > 0`` an iterative pairwise push-apart pass is run after
    the clamp to enforce the minimum inter-cell spacing (same strategy as the
    2-D script).

    Args:
        centres:       (N, 3) float32 current-frame centres in (z, y, x) order.
        sigma:         Displacement std-dev in voxels.  Typical value: 2–5.
        volume_shape:  (D, H, W) volume dimensions.
        border_margin: Minimum distance from each volume edge (voxels).
        rng:           numpy Generator instance for reproducibility.
        min_distance:  Minimum pairwise inter-cell distance (voxels).  Set to
                       the config ``min_distance`` value (≈25.6 vox) to match
                       the hard-core Matérn constraint used at frame 0.
                       0 disables the check entirely.
        max_iterations: Maximum number of repulsion iterations before giving up.

    Returns:
        (N, 3) float32 displaced centres, clamped within the valid interior.
    """
    d, h, w = volume_shape
    delta = rng.normal(0.0, sigma, centres.shape).astype(np.float32)
    displaced = (centres + delta).astype(np.float32)

    # ── border clamp ──────────────────────────────────────────────────────────
    displaced[:, 0] = np.clip(displaced[:, 0], border_margin, d - border_margin)
    displaced[:, 1] = np.clip(displaced[:, 1], border_margin, h - border_margin)
    displaced[:, 2] = np.clip(displaced[:, 2], border_margin, w - border_margin)

    # ── pairwise push-apart to restore min_distance ───────────────────────────
    if min_distance > 0.0 and len(displaced) > 1:
        for _ in range(max_iterations):
            moved = False
            for i in range(len(displaced)):
                for j in range(i + 1, len(displaced)):
                    diff = displaced[i] - displaced[j]
                    dist = float(np.sqrt(np.dot(diff, diff)))
                    if 1e-6 < dist < min_distance:
                        push = diff / dist * (min_distance - dist) * 0.5
                        displaced[i] += push
                        displaced[j] -= push
                        moved = True
            if not moved:
                break
        # re-clamp after repulsion (cells may have been pushed towards edges)
        displaced[:, 0] = np.clip(displaced[:, 0], border_margin, d - border_margin)
        displaced[:, 1] = np.clip(displaced[:, 1], border_margin, h - border_margin)
        displaced[:, 2] = np.clip(displaced[:, 2], border_margin, w - border_margin)

    return displaced


# ──────────────────────────────────────────────────────────────────────────────
# Centre file discovery
# ──────────────────────────────────────────────────────────────────────────────

def find_centres_files(
    centres_dir: Path,
    start_t: int,
    num_frames: int,
) -> List[Path]:
    """
    Find centre files for timepoints [start_t, start_t + num_frames - 1].

    Supports naming patterns (tried in order):
        synthetic_3d_TTTT_centres.npy   ← preferred (matches synthetic_cells_3d/)
        synthetic_TTTT_centres.npy
        timeseries_tTTTT_centres.npy
        centres_tTTTT.npy
        centres_TTTT.npy
    """
    files = []
    for t in range(start_t, start_t + num_frames):
        candidates = [
            centres_dir / f"synthetic_3d_{t:04d}_centres.npy",
            centres_dir / f"synthetic_{t:04d}_centres.npy",
            centres_dir / f"timeseries_t{t:04d}_centres.npy",
            centres_dir / f"centres_t{t:04d}.npy",
            centres_dir / f"centres_{t:04d}.npy",
        ]
        found = next((c for c in candidates if c.exists()), None)
        if found is None:
            raise FileNotFoundError(
                f"Could not find centres file for timepoint {t} in {centres_dir}.\n"
                f"Tried: {[c.name for c in candidates]}"
            )
        files.append(found)
    return files


# ──────────────────────────────────────────────────────────────────────────────
# Single-frame sampler
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def sample_frame(
    model,
    centres: np.ndarray,
    volume_shape: tuple,
    prev_vol: Optional[torch.Tensor],   # None → frame 0; (1,1,D,H,W) in [-1,1] otherwise
    active_geom: dict,
    heatmap_sigma: float,
    distance_percentile: float,
    use_cfg: bool,
    guidance_scale: float,
    device: str,
    prev_vol_active_frac: float = 1.0,
) -> torch.Tensor:
    """
    Sample one frame.

    Conditioning:
        Channel 0: heatmap of `centres` in [0,1]
        Channel 1: prev_vol rescaled to [0,1]  (zeros when prev_vol is None)

    Args:
        prev_vol_active_frac: Fraction of denoising steps (counting from t=0)
            during which ch1 (prev_vol) is active. E.g. 0.3 keeps ch1 zeroed
            for the first 70% of steps (global structure driven by heatmap) and
            restores it for the final 30% (texture refinement). 1.0 = no effect.
            Automatically a no-op when prev_vol is None (frame 0).

    Returns: (1, 1, D, H, W) float32 tensor in [-1, 1].
    """
    # Build geometry conditioning: (1, n_geom, D, H, W)
    geom = centres_to_cond_3d(
        centres=centres,
        volume_shape=volume_shape,
        heatmap_sigma=heatmap_sigma,
        distance_percentile=distance_percentile,
        active_channels=active_geom,
        device=device,
        batch_size=1,
    )

    # Build prev-frame channel: zeros for frame 0, real volume for subsequent frames
    if prev_vol is None:
        prev_ch = torch.zeros(1, 1, *volume_shape, device=device)
    else:
        prev_ch = (prev_vol * 0.5 + 0.5).clamp(0.0, 1.0)  # [-1,1] → [0,1]

    # Concatenate: (1, n_geom+1, D, H, W)
    conditioning = torch.cat([geom, prev_ch], dim=1)

    # s=1 is mathematically identical to a single conditioned pass — skip the
    # unconditional forward entirely to avoid wasting 2× compute.
    if use_cfg and guidance_scale != 1.0:
        vol = model.sample_with_cfg(
            conditioning=conditioning,
            guidance_scale=guidance_scale,
            prev_vol_active_frac=prev_vol_active_frac,
        )
    else:
        vol = model.sample(
            conditioning=conditioning,
            prev_vol_active_frac=prev_vol_active_frac,
        )

    return vol  # (1, 1, D, H, W) in [-1, 1]


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Autoregressive 3D timeseries generation — unified model"
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to unified_3d checkpoint (.pt)"
    )
    parser.add_argument(
        "--config", default="configs/unified_3d.yaml",
        help="Path to unified_3d config YAML"
    )
    parser.add_argument(
        "--method", default="from_files",
        choices=["from_files", "realistic"],
        help=(
            "Centre generation strategy. "
            "'from_files': load synthetic_3d_NNNN_centres.npy from --centres_dir. "
            "'realistic': generate frame-0 centres from config statistics, then evolve "
            "each subsequent frame with Gaussian displacement (--displacement_sigma)."
        )
    )
    parser.add_argument(
        "--centres_dir", default=None,
        help="Directory with synthetic_3d_NNNN_centres.npy files (required for --method from_files)"
    )
    parser.add_argument(
        "--displacement_sigma", type=float, default=3.0,
        help="Std-dev (voxels) of inter-frame Gaussian displacement (--method realistic). "
             "Typical range: 2–6 voxels. Default: 3.0"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for centre generation (--method realistic only)"
    )
    parser.add_argument(
        "--start_t", type=int, default=0,
        help="Index of the first timepoint (default: 0)"
    )
    parser.add_argument(
        "--num_frames", type=int, default=10,
        help="Number of frames to generate (default: 10)"
    )
    parser.add_argument(
        "--volume_size", type=int, default=128,
        help="Cubic volume side length D=H=W (default: 128)"
    )
    parser.add_argument(
        "--out_dir", default="timeseries_output/unified_3d",
        help="Output directory for generated volumes"
    )
    parser.add_argument(
        "--use_ema", action="store_true",
        help="Load EMA weights from checkpoint (recommended)"
    )
    parser.add_argument(
        "--cfg", action="store_true",
        help="Use classifier-free guidance during sampling"
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=3.0,
        help="CFG guidance scale for frames 1+ (only used with --cfg, default: 3.0)"
    )
    parser.add_argument(
        "--guidance_scale_t0", type=float, default=None,
        help=(
            "CFG guidance scale for frame 0 only (no prev-frame channel). "
            "If None, uses --guidance_scale for all frames. "
            "Frame 0 benefits from a higher scale (e.g. 3.0–5.0) because "
            "CFG only amplifies the heatmap channel, not prev_vol."
        )
    )
    parser.add_argument(
        "--heatmap_sigma", type=float, default=3.0,
        help="Gaussian sigma for centre heatmap conditioning (voxels)"
    )
    parser.add_argument(
        "--save_heatmaps", action="store_true",
        help="Also save the heatmap conditioning volume for each frame as t{T}_heatmap.tif"
    )
    parser.add_argument(
        "--match_histograms", action="store_true",
        help=(
            "Match each generated frame's histogram to the previous frame before "
            "passing it as prev-frame conditioning. Prevents autoregressive intensity "
            "drift. Applied in [-1,1] space; skipped for frame 0 (no reference). "
            "Default: off."
        )
    )
    parser.add_argument(
        "--prev_vol_active_frac", type=float, default=1.0,
        help=(
            "Fraction of denoising steps (counting from t=0, i.e. the low-noise end) "
            "during which the prev-frame channel (ch1) is active. "
            "0.3 = ch1 zeroed for the first 70%% of steps (heatmap drives placement), "
            "then restored for the final 30%% (texture refinement). "
            "1.0 = always active (default, backward-compatible). "
            "0.0 = always zeroed (pure heatmap-only, no temporal link). "
            "No effect on frame 0 (prev_vol is already zeros)."
        )
    )
    parser.add_argument(
        "--distance_percentile", type=float, default=95.0,
        help="Percentile for distance map normalisation (unused when distance=false)"
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

    # ── Load config & model ────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    active_geom, n_geom, prev_frame, cond_ch = _resolve_conditioning(cfg)
    if not prev_frame:
        logger.warning(
            "Config has prev_frame=false — this is not a unified/autoregressive config. "
            "Frame conditioning will be geometry-only throughout."
        )

    logger.info("Conditioning: n_geom=%d  prev_frame=%s  total_cond_ch=%d",
                n_geom, prev_frame, cond_ch)

    logger.info("Loading model from %s ...", args.checkpoint)
    model = load_model(
        args.checkpoint, args.config,
        device=device, use_ema=args.use_ema,
    )
    model.eval()

    # ── Build centres sequence ─────────────────────────────────────────────────
    volume_shape = (args.volume_size, args.volume_size, args.volume_size)

    if args.method == "from_files":
        if args.centres_dir is None:
            parser.error("--centres_dir is required when --method from_files")
        centres_dir = Path(args.centres_dir)
        centres_files = find_centres_files(centres_dir, args.start_t, args.num_frames)
        logger.info("Found %d centre files starting at t=%d", len(centres_files), args.start_t)
        centres_sequence = [np.load(p).astype(np.float32) for p in centres_files]

    else:  # realistic
        cg = cfg.get("centre_generation_3d")
        if not cg:
            parser.error(
                "centre_generation_3d block not found in config. "
                "Run scripts/analyze_training_stats.py --save_stats to populate it, "
                "or use --method from_files instead."
            )
        rng = np.random.default_rng(args.seed)
        border_margin = int(cg.get("border_margin", 10))

        # Frame 0: draw from training-data statistics
        c0 = generate_realistic_centres3d(
            volume_shape=volume_shape,
            n_mean=float(cg["n_mean"]),
            n_std=float(cg["n_std"]),
            n_min=int(cg["n_min"]),
            n_max=int(cg["n_max"]),
            min_distance=float(cg["min_distance"]),
            density_grid_z=cg["density_grid_z"],
            density_grid_y=cg["density_grid_y"],
            density_grid_x=cg["density_grid_x"],
            density_grid_3d=cg.get("density_grid_3d"),
            n_bins_joint=int(cg.get("n_bins_joint", 16)),
            border_margin=border_margin,
            max_attempts=int(cg.get("max_attempts", 50)),
            seed=int(rng.integers(0, 2**31)),
        )
        logger.info(
            "Frame-0 centres (realistic): %d cells  displacement_sigma=%.1f vox",
            len(c0), args.displacement_sigma,
        )

        # Subsequent frames: Gaussian displacement of previous centres
        _disp_min_dist = float(cg.get("min_distance", 0.0))
        _disp_max_iter = int(cg.get("max_attempts", 50))
        centres_sequence = [c0]
        for _ in range(args.num_frames - 1):
            centres_sequence.append(
                displace_centres(
                    centres_sequence[-1],
                    sigma=args.displacement_sigma,
                    volume_shape=volume_shape,
                    border_margin=border_margin,
                    rng=rng,
                    min_distance=_disp_min_dist,
                    max_iterations=_disp_max_iter,
                )
            )

    # ── Autoregressive generation loop ────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Generating %d frames autoregressively  [method=%s]",
                args.num_frames, args.method)
    logger.info("=" * 60)

    prev_vol: Optional[torch.Tensor] = None   # None → frame 0 uses zero prev-channel
    all_volumes: List[np.ndarray] = []
    all_heatmaps: List[np.ndarray] = []

    for frame_idx, centres in enumerate(centres_sequence):
        t = args.start_t + frame_idx

        mode = "frame-0 (zero prev)" if prev_vol is None else f"conditioned on t={t-1}"
        logger.info("Frame %d/%d  t=%04d  n_cells=%d  [%s]",
                    frame_idx + 1, args.num_frames, t, len(centres), mode)

        frame_guidance = (
            args.guidance_scale_t0
            if (args.guidance_scale_t0 is not None and prev_vol is None)
            else args.guidance_scale
        )
        if frame_idx == 0 and args.guidance_scale_t0 is not None:
            logger.info("  → frame-0 CFG scale: %.2f", frame_guidance)

        vol = sample_frame(
            model=model,
            centres=centres,
            volume_shape=volume_shape,
            prev_vol=prev_vol,
            active_geom=active_geom,
            heatmap_sigma=args.heatmap_sigma,
            distance_percentile=args.distance_percentile,
            use_cfg=args.cfg,
            guidance_scale=frame_guidance,
            device=device,
            prev_vol_active_frac=args.prev_vol_active_frac,
        )
        # vol: (1, 1, D, H, W) in [-1, 1]

        # Optional histogram matching: match this frame's global statistics to
        # the previous frame so that autoregressive intensity drift is suppressed.
        # Applied in [-1,1] before both saving and prev_vol assignment so the
        # conditioning tensor and the saved TIFF are always consistent.
        if args.match_histograms and prev_vol is not None:
            ref_np  = prev_vol[0, 0].cpu().numpy()           # (D,H,W) in [-1,1]
            src_np  = vol[0, 0].cpu().numpy()                 # (D,H,W) in [-1,1]
            matched = skimage_match_histograms(
                src_np.astype(np.float64), ref_np.astype(np.float64)
            )
            matched = np.clip(matched, -1.0, 1.0).astype(np.float32)
            vol = torch.from_numpy(matched).unsqueeze(0).unsqueeze(0).to(vol.device)
            logger.info("  → histogram matched to t=%04d", t - 1)

        # Save frame as TIFF in [0, 1]
        vol_np = vol[0, 0].cpu().numpy()   # (D, H, W)
        vol_01 = to_zero_one(vol_np).astype(np.float32)
        tifffile.imwrite(str(out_dir / f"t{t:04d}_vol.tif"), vol_01)
        np.save(str(out_dir / f"t{t:04d}_centres.npy"), centres)

        if args.save_heatmaps:
            heatmap = centres_to_cond_3d(
                centres=centres,
                volume_shape=volume_shape,
                heatmap_sigma=args.heatmap_sigma,
                distance_percentile=args.distance_percentile,
                active_channels=active_geom,
                device="cpu",
                batch_size=1,
            )[0, 0].numpy()  # (D, H, W) — channel 0 is always the heatmap
            tifffile.imwrite(str(out_dir / f"t{t:04d}_heatmap.tif"), heatmap.astype(np.float32))
            all_heatmaps.append(heatmap.astype(np.float32))
            logger.info("  → saved t%04d_heatmap.tif", t)

        all_volumes.append(vol_01)
        logger.info("  → saved t%04d_vol.tif", t)

        # Pass this frame as the prev-frame conditioning for the next step
        prev_vol = vol  # (1, 1, D, H, W) in [-1, 1], stays on device

    # ── Save full timeseries stack ─────────────────────────────────────────────
    t_end = args.start_t + args.num_frames - 1
    stack = np.stack(all_volumes, axis=0)   # (T, D, H, W)
    stack_path = out_dir / f"timeseries_t{args.start_t:04d}-t{t_end:04d}.tif"
    tifffile.imwrite(str(stack_path), stack)

    if args.save_heatmaps and all_heatmaps:
        heatmap_stack = np.stack(all_heatmaps, axis=0)   # (T, D, H, W)
        heatmap_stack_path = out_dir / f"timeseries_heatmap_t{args.start_t:04d}-t{t_end:04d}.tif"
        tifffile.imwrite(str(heatmap_stack_path), heatmap_stack)
        logger.info("Heatmap stack:    %s  shape=%s", heatmap_stack_path.name, heatmap_stack.shape)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Done. %d frames written to %s", args.num_frames, out_dir)
    logger.info("Timeseries stack: %s  shape=%s", stack_path.name, stack.shape)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
