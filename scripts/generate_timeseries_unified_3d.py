"""
Autoregressive 3D timeseries generation using the unified model.

Generates a sequence of N volumetric frames autoregressively:

    V_0 ~ p(V | heatmap_0,  zeros     )   [frame 0 — no previous frame]
    V_1 ~ p(V | heatmap_1,  V_0 in 01 )
    V_2 ~ p(V | heatmap_2,  V_1 in 01 )
    ...

A single unified checkpoint (configs/unified_3d.yaml) is used throughout.
Each generated V_t is reused immediately as the prev-frame conditioning for V_{t+1}.

Three centre-generation modes
------------------------------
  --method from_files  (default)
      Load pre-generated centre files from --centres_dir.
      Files must be named: synthetic_3d_NNNN_centres.npy

  --method realistic
      Generate frame-0 centres from data statistics stored in the config's
      centre_generation_3d block (populate with analyze_training_stats.py
      --save_stats).  Each subsequent frame's centres are derived by applying
      an isotropic Gaussian displacement (--displacement_sigma voxels) to the
      previous frame's centres, modelling realistic inter-frame cell motion.

  --method real_centres
      Sample frame-0 centres from a real training patch (guaranteed in-distribution).
      Load *_centres.npy files from --patches_dir (e.g. data_live_node2_3d/train),
      pick one at random per sample, then evolve each subsequent frame using a
      log-normal displacement whose parameters are fitted to the training data:
        magnitude ~ LogNormal(μ=ln(median), σ=shape)
        direction ~ isotropic uniform on unit sphere
      Defaults: --lognormal_median 4.99  --lognormal_sigma 0.751.
      No cell birth or death; all cells persist across all frames.

Usage:
    # Load pre-computed centre files:
    python -m scripts.generate_timeseries_unified_3d \\
        --checkpoint checkpoints/unified_3d/best.pt \\
        --config     configs/unified_3d.yaml \\
        --centres_dir synthetic_cells_3d/ \\
        --start_t 0 --num_frames 10 \\
        --out_dir timeseries_output/unified_3d/

    # Generate centres on-the-fly from config stats:
    python -m scripts.generate_timeseries_unified_3d \\
        --checkpoint checkpoints/unified_3d/best.pt \\
        --config     configs/unified_3d.yaml \\
        --method realistic \\
        --displacement_sigma 3.0 \\
        --num_frames 10 \\
        --out_dir timeseries_output/unified_3d/

    # Start from real training centres (recommended):
    python -m scripts.generate_timeseries_unified_3d \\
        --checkpoint checkpoints/unified_3d/best.pt \\
        --config     configs/unified_3d.yaml \\
        --method real_centres \\
        --patches_dir data_live_node2_3d/train \\
        --num_frames 10 \\
        --out_dir timeseries_output/unified_3d/

Output per frame:
    t{TTTT}_vol.tif             — generated volume [0,1] float32
    t{TTTT}_centres.npy         — cell centres used for heatmap conditioning
    timeseries_t0000-tNNNN.tif  — full T×D×H×W TIFF stack
"""

import argparse
import csv
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import tifffile
import yaml
from skimage.exposure import match_histograms as skimage_match_histograms

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


def displace_centres_lognormal(
    centres: np.ndarray,
    mu: float,
    sigma: float,
    volume_shape: tuple,
    border_margin: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Derive next-frame centres using log-normal displacement magnitudes.

    Each cell moves by a displacement whose magnitude is drawn from
    LogNormal(mu, sigma) and whose direction is drawn uniformly from the
    unit sphere (isotropic).  Cells are clamped to stay within the valid
    interior after displacement.

    Parameters
    ----------
    centres : (N, 3) float32  current-frame centres in (z, y, x) order.
    mu      : log-normal location parameter  = ln(median displacement).
    sigma   : log-normal scale  (shape) parameter.
    volume_shape : (D, H, W) volume dimensions.
    border_margin : minimum distance from each volume edge (voxels).
    rng     : numpy Generator instance for reproducibility.

    Returns
    -------
    (N, 3) float32 displaced centres, clamped within the valid interior.
    """
    N = len(centres)
    if N == 0:
        return centres.copy()

    # Displacement magnitudes from log-normal distribution
    magnitudes = np.exp(rng.normal(mu, sigma, N)).astype(np.float32)  # (N,)

    # Isotropic random directions on the unit sphere
    directions = rng.standard_normal((N, 3)).astype(np.float32)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    directions /= norms  # unit vectors

    displaced = (centres + magnitudes[:, None] * directions).astype(np.float32)

    d, h, w = volume_shape
    displaced[:, 0] = np.clip(displaced[:, 0], border_margin, d - border_margin)
    displaced[:, 1] = np.clip(displaced[:, 1], border_margin, h - border_margin)
    displaced[:, 2] = np.clip(displaced[:, 2], border_margin, w - border_margin)

    return displaced


def filter_centres_to_window(
    sim_centres: np.ndarray,
    padding: int,
    volume_size: int,
) -> tuple:
    """
    Split simulation-space centres into those inside the visible model window
    and a full (N, 3) array with NaN for absent cells.

    The model window is the inner ``volume_size³`` region of the simulation
    volume: coords in ``[padding, padding + volume_size)`` on every axis.

    Args:
        sim_centres: ``(N, 3)`` float32 centres in simulation space,
                     e.g. ``[0, sim_size)`` with ``sim_size = volume_size + 2*padding``.
        padding:     Number of voxels of border padding on each side.
        volume_size: Side length of the visible model volume (e.g. 128).

    Returns:
        inner_centres: ``(M, 3)`` float32 coords in model space ``[0, volume_size)``
                       for cells currently inside the window.  May be empty.
        saved_centres: ``(N, 3)`` float32 in model space.  Cells outside the
                       window have ``NaN`` coords.  Row index = persistent
                       instance ID is always preserved.
    """
    lo = float(padding)
    hi = float(padding + volume_size)
    in_window = (
        (sim_centres[:, 0] >= lo) & (sim_centres[:, 0] < hi) &
        (sim_centres[:, 1] >= lo) & (sim_centres[:, 1] < hi) &
        (sim_centres[:, 2] >= lo) & (sim_centres[:, 2] < hi)
    )
    model_coords = sim_centres - float(padding)          # shift to model space
    saved = np.where(in_window[:, np.newaxis], model_coords, np.nan).astype(np.float32)
    inner = model_coords[in_window].astype(np.float32)
    return inner, saved


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

def build_frame_conditioning(
    centres: np.ndarray,
    volume_shape: tuple,
    prev_vol: Optional[torch.Tensor],
    active_geom: dict,
    heatmap_sigma: float,
    distance_percentile: float,
    device: str,
) -> torch.Tensor:
    """
    Build a ``(1, C, D, H, W)`` conditioning tensor for a single sample.

    Channel 0: Gaussian heatmap of ``centres`` in [0, 1].
    Channel 1: ``prev_vol`` rescaled to [0, 1]; zeros when ``prev_vol is None``.
    """
    geom = centres_to_cond_3d(
        centres=centres,
        volume_shape=volume_shape,
        heatmap_sigma=heatmap_sigma,
        distance_percentile=distance_percentile,
        active_channels=active_geom,
        device=device,
        batch_size=1,
    )
    prev_ch = (
        torch.zeros(1, 1, *volume_shape, device=device)
        if prev_vol is None
        else (prev_vol * 0.5 + 0.5).clamp(0.0, 1.0)   # [-1,1] → [0,1]
    )
    return torch.cat([geom, prev_ch], dim=1)  # (1, C, D, H, W)


@torch.no_grad()
def run_model_batch(
    model,
    conditioning: torch.Tensor,   # (B, C, D, H, W)
    use_cfg: bool,
    guidance_scale: float,
    prev_vol_active_frac: float = 1.0,
) -> torch.Tensor:
    """
    Run the diffusion model for a batch of conditionings.

    ``guidance_scale == 1.0`` is mathematically identical to a single conditioned
    pass, so the unconditioned forward is skipped in that case.

    Returns: ``(B, 1, D, H, W)`` float32 tensor in [-1, 1].
    """
    if use_cfg and guidance_scale != 1.0:
        return model.sample_with_cfg(
            conditioning=conditioning,
            guidance_scale=guidance_scale,
            prev_vol_active_frac=prev_vol_active_frac,
        )
    return model.sample(
        conditioning=conditioning,
        prev_vol_active_frac=prev_vol_active_frac,
    )


@torch.no_grad()
def sample_frame(
    model,
    centres: np.ndarray,
    volume_shape: tuple,
    prev_vol: Optional[torch.Tensor],
    active_geom: dict,
    heatmap_sigma: float,
    distance_percentile: float,
    use_cfg: bool,
    guidance_scale: float,
    device: str,
    prev_vol_active_frac: float = 1.0,
) -> torch.Tensor:
    """
    Sample one frame (single-sample convenience wrapper).

    Returns: ``(1, 1, D, H, W)`` float32 tensor in [-1, 1].
    """
    conditioning = build_frame_conditioning(
        centres=centres,
        volume_shape=volume_shape,
        prev_vol=prev_vol,
        active_geom=active_geom,
        heatmap_sigma=heatmap_sigma,
        distance_percentile=distance_percentile,
        device=device,
    )
    return run_model_batch(model, conditioning, use_cfg, guidance_scale, prev_vol_active_frac)


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
        choices=["from_files", "realistic", "real_centres"],
        help=(
            "Centre generation strategy. "
            "'from_files': load synthetic_3d_NNNN_centres.npy from --centres_dir. "
            "'realistic': generate frame-0 centres from config statistics, then evolve "
            "each subsequent frame with Gaussian displacement (--displacement_sigma). "
            "'real_centres': load a random real training patch from --patches_dir for frame 0, "
            "then evolve with log-normal displacement (--lognormal_median, --lognormal_sigma)."
        )
    )
    parser.add_argument(
        "--centres_dir", default=None,
        help="Directory with synthetic_3d_NNNN_centres.npy files (required for --method from_files)"
    )
    parser.add_argument(
        "--patches_dir", default=None,
        help="Directory containing *_centres.npy training patches "
             "(required for --method real_centres, e.g. data_live_node2_3d/train)"
    )
    parser.add_argument(
        "--displacement_sigma", type=float, default=3.0,
        help="Std-dev (voxels) of inter-frame Gaussian displacement (--method realistic). "
             "Typical range: 2–6 voxels. Default: 3.0"
    )
    parser.add_argument(
        "--lognormal_median", type=float, default=4.99,
        help="Median displacement magnitude in voxels for --method real_centres. "
             "Sets the log-normal location: mu=ln(median). "
             "Fitted to node2 training data. Default: 4.99"
    )
    parser.add_argument(
        "--lognormal_sigma", type=float, default=0.751,
        help="Log-normal shape (scale) parameter for --method real_centres. "
             "Fitted to node2 training data. Default: 0.751"
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
    parser.add_argument(
        "--num_samples", type=int, default=1,
        help="Number of independent timeseries to generate. Each sample draws its own "
             "centre sequence (--method realistic) or shares the file-based centres but "
             "gets a different diffusion noise realisation (--method from_files). "
             "When > 1, outputs are written to sample_NNNN/ subdirectories. (default: 1)"
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=1,
        help="Number of samples processed simultaneously as a GPU batch. Higher values "
             "reduce wall time but require proportionally more GPU memory. "
             "Must divide --num_samples cleanly or the last batch will be smaller. "
             "(default: 1)"
    )
    parser.add_argument(
        "--simulation_padding", type=int, default=16,
        help="(--method realistic only) Voxel padding added on every side of the "
             "visible volume for simulation purposes. E.g. 16 → centres are simulated "
             "in a (volume_size + 2*16)³ space; only those inside the inner volume_size³ "
             "window are passed to the heatmap. Cells that drift outside that window "
             "appear as NaN rows in saved centre files, preserving row-index = instance-ID "
             "across frames.  0 disables the feature (backward-compatible). (default: 16)"
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

    # ── Volume & simulation dimensions ────────────────────────────────────────
    volume_shape = (args.volume_size, args.volume_size, args.volume_size)
    sim_padding  = args.simulation_padding
    sim_size     = args.volume_size + 2 * sim_padding
    sim_shape    = (sim_size, sim_size, sim_size)
    # Sequences are stored in simulation space only for realistic + padding > 0.
    # real_centres patches are already in model space — no sim-space needed.
    centres_are_in_sim_space = (args.method == "realistic" and sim_padding > 0)
    if centres_are_in_sim_space:
        logger.info(
            "Simulation space: %d³  (visible %d³ + %d px padding each side)",
            sim_size, args.volume_size, sim_padding,
        )

    # ── Build ALL centre sequences (one per sample) ───────────────────────────
    all_centre_sequences: List[List[np.ndarray]] = []

    if args.method == "from_files":
        if args.centres_dir is None:
            parser.error("--centres_dir is required when --method from_files")
        centres_dir = Path(args.centres_dir)
        centres_files = find_centres_files(centres_dir, args.start_t, args.num_frames)
        logger.info("Found %d centre files starting at t=%d", len(centres_files), args.start_t)
        _base_seq = [np.load(p).astype(np.float32) for p in centres_files]
        # All samples share the same spatial centres; diversity comes from diffusion noise.
        all_centre_sequences = [_base_seq] * args.num_samples

    elif args.method == "realistic":
        cg = cfg.get("centre_generation_3d")
        if not cg:
            parser.error(
                "centre_generation_3d block not found in config. "
                "Run scripts/analyze_training_stats.py --save_stats to populate it, "
                "or use --method from_files instead."
            )
        rng = np.random.default_rng(args.seed)
        border_margin = int(cg.get("border_margin", 10))
        _disp_min_dist = float(cg.get("min_distance", 0.0))
        _disp_max_iter = int(cg.get("max_attempts", 50))

        # When simulation_padding > 0 generate in the larger sim_shape so
        # cells can drift in/out of the visible volume_shape window.  Scale
        # cell counts proportionally so density inside the window is preserved.
        _gen_shape  = sim_shape if centres_are_in_sim_space else volume_shape
        _vol_ratio  = float(sim_size ** 3) / float(args.volume_size ** 3) \
                      if centres_are_in_sim_space else 1.0

        for s in range(args.num_samples):
            c0 = generate_realistic_centres3d(
                volume_shape=_gen_shape,
                n_mean=float(cg["n_mean"]) * _vol_ratio,
                n_std=float(cg["n_std"])   * _vol_ratio,
                n_min=max(1, int(cg["n_min"] * _vol_ratio)),
                n_max=int(cg["n_max"] * _vol_ratio),
                min_distance=float(cg["min_distance"]),
                density_grid_z=cg["density_grid_z"],
                density_grid_y=cg["density_grid_y"],
                density_grid_x=cg["density_grid_x"],
                density_grid_3d=cg.get("density_grid_3d"),
                n_bins_joint=int(cg.get("n_bins_joint", 16)),
                border_margin=border_margin,
                max_attempts=int(cg.get("max_attempts", 50)),
                seed=int(rng.integers(0, 2**31)),  # independent seed per sample
            )
            seq: List[np.ndarray] = [c0]
            for _ in range(args.num_frames - 1):
                seq.append(
                    displace_centres(
                        seq[-1],
                        sigma=args.displacement_sigma,
                        volume_shape=_gen_shape,
                        border_margin=border_margin,
                        rng=rng,
                        min_distance=_disp_min_dist,
                        max_iterations=_disp_max_iter,
                    )
                )
            all_centre_sequences.append(seq)
            if centres_are_in_sim_space:
                _, c0_saved = filter_centres_to_window(c0, sim_padding, args.volume_size)
                n_visible = int(np.sum(~np.isnan(c0_saved[:, 0])))
                logger.info(
                    "Sample %04d: %d cells total in sim  (%d visible at t=0)  sigma=%.1f vox",
                    s, len(c0), n_visible, args.displacement_sigma,
                )
            else:
                logger.info(
                    "Sample %04d centre sequence: %d cells  displacement_sigma=%.1f vox",
                    s, len(c0), args.displacement_sigma,
                )

    else:  # real_centres
        if args.patches_dir is None:
            parser.error("--patches_dir is required for --method real_centres")
        patches_dir = Path(args.patches_dir)
        centre_files = sorted(patches_dir.glob("*_centres.npy"))
        if not centre_files:
            parser.error(f"No *_centres.npy files found in {patches_dir}")
        logger.info(
            "Found %d real centre files in %s", len(centre_files), patches_dir
        )

        rng = np.random.default_rng(args.seed)
        border_margin = int(cfg.get("centre_generation_3d", {}).get("border_margin", 10))
        lognormal_mu = float(np.log(args.lognormal_median))
        logger.info(
            "Log-normal displacement: median=%.2f vox  sigma=%.3f  (mu=%.4f)",
            args.lognormal_median, args.lognormal_sigma, lognormal_mu,
        )

        for s in range(args.num_samples):
            # Pick a random real centre file for frame 0
            c0_file = centre_files[rng.integers(0, len(centre_files))]
            c0 = np.load(c0_file).astype(np.float32)
            logger.info(
                "Sample %04d: %d real centres from %s", s, len(c0), c0_file.name
            )

            seq: List[np.ndarray] = [c0]
            for _ in range(args.num_frames - 1):
                seq.append(
                    displace_centres_lognormal(
                        seq[-1],
                        mu=lognormal_mu,
                        sigma=args.lognormal_sigma,
                        volume_shape=volume_shape,
                        border_margin=border_margin,
                        rng=rng,
                    )
                )
            all_centre_sequences.append(seq)

    # ── Helper: per-sample output directory ───────────────────────────────────
    def sample_out_dir(s: int) -> Path:
        """Return (and create) the output directory for sample s."""
        d = out_dir if args.num_samples == 1 else out_dir / f"sample_{s:04d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ── Autoregressive generation (batched across samples) ────────────────────
    n_batches = (args.num_samples + args.sample_batch_size - 1) // args.sample_batch_size
    t_end = args.start_t + args.num_frames - 1
    logger.info("=" * 60)
    logger.info(
        "Generating %d sample(s) × %d frames  "
        "[sample_batch_size=%d  n_batches=%d  method=%s]",
        args.num_samples, args.num_frames,
        args.sample_batch_size, n_batches, args.method,
    )
    logger.info("=" * 60)

    for batch_start in range(0, args.num_samples, args.sample_batch_size):
        batch_idxs = list(range(
            batch_start,
            min(batch_start + args.sample_batch_size, args.num_samples)
        ))
        B = len(batch_idxs)
        batch_num = batch_start // args.sample_batch_size + 1
        logger.info("── Batch %d/%d  samples=%s", batch_num, n_batches, batch_idxs)

        # Per-sample state for this batch
        batch_prev_vols: List[Optional[torch.Tensor]] = [None] * B
        batch_all_volumes: List[List[np.ndarray]] = [[] for _ in range(B)]
        batch_all_heatmaps: List[List[np.ndarray]] = [[] for _ in range(B)]
        # List of (timepoint, saved_centres (N,3)) tuples for the tracks CSV
        batch_all_centres: List[List[tuple]] = [[] for _ in range(B)]

        for frame_idx in range(args.num_frames):
            t = args.start_t + frame_idx
            is_frame0 = (frame_idx == 0)

            frame_guidance = (
                args.guidance_scale_t0
                if (args.guidance_scale_t0 is not None and is_frame0)
                else args.guidance_scale
            )
            if is_frame0 and args.guidance_scale_t0 is not None:
                logger.info("  Frame 0 CFG scale: %.2f", frame_guidance)

            # Filter sim-space centres to visible window (or pass through)
            frame_inner: List[np.ndarray] = []
            frame_saved: List[np.ndarray] = []
            for b, si in enumerate(batch_idxs):
                raw = all_centre_sequences[si][frame_idx]
                if centres_are_in_sim_space:
                    inner, saved = filter_centres_to_window(raw, sim_padding, args.volume_size)
                else:
                    inner, saved = raw, raw
                frame_inner.append(inner)
                frame_saved.append(saved)

            # Build stacked conditioning  (B, C, D, H, W)
            cond_list = [
                build_frame_conditioning(
                    centres=frame_inner[b],          # model-space, visible cells only
                    volume_shape=volume_shape,
                    prev_vol=batch_prev_vols[b],
                    active_geom=active_geom,
                    heatmap_sigma=args.heatmap_sigma,
                    distance_percentile=args.distance_percentile,
                    device=device,
                )
                for b in range(B)
            ]
            cond_batch = torch.cat(cond_list, dim=0)  # (B, C, D, H, W)

            # Single forward pass for all B samples
            vols = run_model_batch(
                model,
                cond_batch,
                use_cfg=args.cfg,
                guidance_scale=frame_guidance,
                prev_vol_active_frac=args.prev_vol_active_frac,
            )  # (B, 1, D, H, W) in [-1, 1]

            # ── Per-sample post-processing, histogram matching, save ──────────
            for b, si in enumerate(batch_idxs):
                vol = vols[b:b+1]  # (1, 1, D, H, W) in [-1, 1]
                sdir = sample_out_dir(si)

                # Optional histogram matching to suppress intensity drift
                if args.match_histograms and batch_prev_vols[b] is not None:
                    ref_np = batch_prev_vols[b][0, 0].cpu().numpy()
                    src_np = vol[0, 0].cpu().numpy()
                    matched = skimage_match_histograms(
                        src_np.astype(np.float64), ref_np.astype(np.float64)
                    )
                    matched = np.clip(matched, -1.0, 1.0).astype(np.float32)
                    vol = torch.from_numpy(matched).unsqueeze(0).unsqueeze(0).to(vol.device)
                    logger.info("  → [s%04d] histogram matched to t=%04d", si, t - 1)

                # frame_saved: (N_total, 3), NaN for cells outside visible window.
                # frame_inner: (M, 3), model-space coords of visible cells only.
                saved_centres = frame_saved[b]
                inner_centres = frame_inner[b]
                n_active = int(np.sum(~np.isnan(saved_centres[:, 0]))) \
                           if saved_centres.dtype == np.float32 and np.any(np.isnan(saved_centres)) \
                           else len(saved_centres)

                # Save frame as TIFF in [0, 1]
                vol_np = vol[0, 0].cpu().numpy()
                vol_01 = to_zero_one(vol_np).astype(np.float32)
                tifffile.imwrite(str(sdir / f"t{t:04d}_vol.tif"), vol_01)
                # Centres file: (N_total, 3) with NaN for absent cells.
                # Row index = persistent instance ID across all frames.
                np.save(str(sdir / f"t{t:04d}_centres.npy"), saved_centres)
                logger.info(
                    "  → [s%04d] t%04d_vol.tif  n_active=%d/%d",
                    si, t, n_active, len(saved_centres)
                )

                if args.save_heatmaps:
                    # Use inner_centres (model-space, visible only) for the heatmap
                    heatmap = centres_to_cond_3d(
                        centres=inner_centres,
                        volume_shape=volume_shape,
                        heatmap_sigma=args.heatmap_sigma,
                        distance_percentile=args.distance_percentile,
                        active_channels=active_geom,
                        device="cpu",
                        batch_size=1,
                    )[0, 0].numpy()  # (D, H, W) — channel 0 is always the heatmap
                    tifffile.imwrite(
                        str(sdir / f"t{t:04d}_heatmap.tif"),
                        heatmap.astype(np.float32),
                    )
                    batch_all_heatmaps[b].append(heatmap.astype(np.float32))

                batch_all_volumes[b].append(vol_01)
                batch_all_centres[b].append((t, saved_centres))
                # Keep this frame's volume as prev conditioning for next frame
                batch_prev_vols[b] = vol

        # ── Save per-sample timeseries stacks ─────────────────────────────────
        for b, si in enumerate(batch_idxs):
            sdir = sample_out_dir(si)
            stack = np.stack(batch_all_volumes[b], axis=0)  # (T, D, H, W)
            stack_path = sdir / f"timeseries_t{args.start_t:04d}-t{t_end:04d}.tif"
            tifffile.imwrite(str(stack_path), stack)
            logger.info(
                "[s%04d] timeseries stack: %s  shape=%s", si, stack_path.name, stack.shape
            )

            if args.save_heatmaps and batch_all_heatmaps[b]:
                hm_stack = np.stack(batch_all_heatmaps[b], axis=0)  # (T, D, H, W)
                hm_path = sdir / f"timeseries_heatmap_t{args.start_t:04d}-t{t_end:04d}.tif"
                tifffile.imwrite(str(hm_path), hm_stack)
                logger.info("[s%04d] heatmap stack:    %s  shape=%s", si, hm_path.name, hm_stack.shape)

            # ── Tracks CSV: timepoint, track_id, x, y, z ──────────────────────
            # Centres are stored internally as (z, y, x); CSV exports as x, y, z.
            # NaN rows = cell absent from visible volume at that timepoint.
            csv_path = sdir / f"timeseries_t{args.start_t:04d}-t{t_end:04d}_tracks.csv"
            with open(csv_path, "w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["timepoint", "track_id", "x", "y", "z"])
                for tp, centres_arr in batch_all_centres[b]:
                    for track_id, (cz, cy, cx) in enumerate(centres_arr):
                        writer.writerow([
                            tp,
                            track_id,
                            "" if np.isnan(cx) else f"{cx:.4f}",
                            "" if np.isnan(cy) else f"{cy:.4f}",
                            "" if np.isnan(cz) else f"{cz:.4f}",
                        ])
            logger.info("[s%04d] tracks CSV:       %s", si, csv_path.name)

    logger.info("")
    logger.info("=" * 60)
    logger.info(
        "Done. %d sample(s) × %d frames written to %s",
        args.num_samples, args.num_frames, out_dir,
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
