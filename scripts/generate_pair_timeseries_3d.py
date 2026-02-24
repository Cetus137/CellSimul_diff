"""
Generate a 3D time-series of volumetric frame pairs using the frame2_3d model.

This script automates:
  1. Loading centres for multiple timepoints
  2. Repeatedly calling the two-frame 3D pipeline to generate consecutive pairs
  3. Saving all volumes in a structured output directory

Usage:
    python -m scripts.generate_pair_timeseries_3d \\
        --frame1_ckpt checkpoints/frame1_3d/best.pt \\
        --frame2_ckpt checkpoints/frame2_3d/best.pt \\
        --centres_dir synthetic_cells/ \\
        --start_t 0 \\
        --num_pairs 10 \\
        --out_dir timeseries_pairs_3d/
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import tifffile
import yaml

sys.path.append(str(Path(__file__).parent.parent))

from sampling.sample_two_frame3d import (
    load_model,
    sample_two_frames_3d,
    _resolve_conditioning,
)
from utils.normalization import to_zero_one

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)


def find_centres_files(
    centres_dir: Path,
    start_t: int,
    num_pairs: int,
) -> List[Path]:
    """
    Find centre files for the required timepoints.
    
    Looks for files matching patterns:
      - synthetic_TTTT_centres.npy
      - timeseries_tTTTT_centres.npy
      - centres_tTTTT.npy
    
    Returns:
        List of paths to centre files for timepoints [start_t, start_t + num_pairs]
    """
    timepoints = list(range(start_t, start_t + num_pairs + 1))
    centres_files = []
    
    for t in timepoints:
        # Try multiple naming patterns
        candidates = [
            centres_dir / f"synthetic_{t:04d}_centres.npy",
            centres_dir / f"timeseries_t{t:04d}_centres.npy",
            centres_dir / f"centres_t{t:04d}.npy",
            centres_dir / f"centres_{t:04d}.npy",
        ]
        
        found = None
        for candidate in candidates:
            if candidate.exists():
                found = candidate
                break
        
        if found is None:
            raise FileNotFoundError(
                f"Could not find centres file for timepoint {t} in {centres_dir}. "
                f"Tried patterns: synthetic_####_centres.npy, timeseries_t####_centres.npy, "
                f"centres_t####.npy, centres_####.npy"
            )
        
        centres_files.append(found)
    
    return centres_files


def main():
    parser = argparse.ArgumentParser(
        description="Generate 3D pair time-series using frame2_3d model"
    )
    parser.add_argument(
        "--frame1_ckpt", required=True,
        help="Path to frame1_3d checkpoint"
    )
    parser.add_argument(
        "--frame2_ckpt", required=True,
        help="Path to frame2_3d checkpoint"
    )
    parser.add_argument(
        "--frame1_config", default="configs/frame1_3d.yaml",
        help="Config for frame1_3d model"
    )
    parser.add_argument(
        "--frame2_config", default="configs/frame2_3d.yaml",
        help="Config for frame2_3d model"
    )
    parser.add_argument(
        "--centres_dir", required=True,
        help="Directory containing centres files (synthetic_TTTT_centres.npy)"
    )
    parser.add_argument(
        "--start_t", type=int, default=0,
        help="Starting timepoint index"
    )
    parser.add_argument(
        "--num_pairs", type=int, default=10,
        help="Number of consecutive pairs to generate"
    )
    parser.add_argument(
        "--volume_size", type=int, default=128,
        help="Size of generated volumes (D=H=W)"
    )
    parser.add_argument(
        "--out_dir", default="timeseries_pairs_3d",
        help="Output directory for generated volumes"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Batch size (number of samples per pair)"
    )
    parser.add_argument(
        "--use_ema", action="store_true",
        help="Load EMA weights from checkpoints"
    )
    parser.add_argument(
        "--cfg", action="store_true",
        help="Use classifier-free guidance"
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=3.0,
        help="CFG guidance scale"
    )
    parser.add_argument(
        "--heatmap_sigma", type=float, default=3.0
    )
    parser.add_argument(
        "--distance_percentile", type=float, default=95.0
    )
    parser.add_argument(
        "--device", default=None,
        help="Device: cuda / cpu"
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load configs -------------------------------------------------
    with open(args.frame1_config) as f:
        cfg1 = yaml.safe_load(f)
    with open(args.frame2_config) as f:
        cfg2 = yaml.safe_load(f)

    active_geom1, _n_geom1, prev1, cond_ch1 = _resolve_conditioning(cfg1)
    active_geom2, _n_geom2, prev2, cond_ch2 = _resolve_conditioning(cfg2)

    if prev1:
        logger.warning("frame1_3d config has prev_frame=true; ignoring for frame1")

    # ---- Load models --------------------------------------------------
    logger.info("Loading frame1_3d model from %s", args.frame1_ckpt)
    model1 = load_model(
        args.frame1_ckpt, args.frame1_config,
        device=device, use_ema=args.use_ema,
    )

    logger.info("Loading frame2_3d model from %s", args.frame2_ckpt)
    model2 = load_model(
        args.frame2_ckpt, args.frame2_config,
        device=device, use_ema=args.use_ema,
    )

    # ---- Find centres files -------------------------------------------
    centres_dir = Path(args.centres_dir)
    centres_files = find_centres_files(centres_dir, args.start_t, args.num_pairs)
    logger.info("Found %d centres files for %d pairs", len(centres_files), args.num_pairs)

    # ---- Generate pairs -----------------------------------------------
    volume_shape = (args.volume_size, args.volume_size, args.volume_size)

    logger.info("=" * 60)
    logger.info("Generating %d consecutive 3D frame pairs", args.num_pairs)
    logger.info("=" * 60)

    all_volumes = []  # For optional full timeseries stack

    for pair_idx in range(args.num_pairs):
        t0 = args.start_t + pair_idx
        t1 = t0 + 1

        centres_t  = np.load(centres_files[pair_idx]).astype(np.float32)
        centres_t1 = np.load(centres_files[pair_idx + 1]).astype(np.float32)

        logger.info("")
        logger.info("Pair %d/%d:  t=%d → t=%d  (%d cells → %d cells)",
                    pair_idx + 1, args.num_pairs, t0, t1,
                    len(centres_t), len(centres_t1))

        # Generate pair
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
        )

        # Save pair
        vol_t_np  = vol_t.cpu().numpy()   # (B, 1, D, H, W)
        vol_t1_np = vol_t1.cpu().numpy()

        for b in range(vol_t_np.shape[0]):
            # Save individual volumes
            vol_t_01  = to_zero_one(vol_t_np[b, 0]).astype(np.float32)
            vol_t1_01 = to_zero_one(vol_t1_np[b, 0]).astype(np.float32)

            tifffile.imwrite(
                str(out_dir / f"t{t0:04d}_sample{b:03d}.tif"),
                vol_t_01
            )
            tifffile.imwrite(
                str(out_dir / f"t{t1:04d}_sample{b:03d}.tif"),
                vol_t1_01
            )

            # Save centres for reference
            np.save(out_dir / f"t{t0:04d}_centres.npy", centres_t)
            np.save(out_dir / f"t{t1:04d}_centres.npy", centres_t1)

            # Accumulate for timeseries stack (first batch only)
            if b == 0:
                if pair_idx == 0:
                    all_volumes.append(vol_t_01)
                all_volumes.append(vol_t1_01)

        logger.info("  ✓ Saved pair to %s", out_dir)

    # ---- Save full timeseries as single TIFF stack --------------------
    if len(all_volumes) > 0:
        stack = np.stack(all_volumes, axis=0)  # (T, D, H, W)
        stack_path = out_dir / f"timeseries_t{args.start_t:04d}-t{args.start_t + args.num_pairs:04d}.tif"
        tifffile.imwrite(str(stack_path), stack)
        logger.info("")
        logger.info("=" * 60)
        logger.info("Saved full timeseries stack: %s", stack_path)
        logger.info("  Shape: %s  (T=%d, D=%d, H=%d, W=%d)",
                    stack.shape, stack.shape[0], stack.shape[1], stack.shape[2], stack.shape[3])
        logger.info("=" * 60)

    logger.info("")
    logger.info("Complete! Generated %d pairs → %d volumes in %s",
                args.num_pairs, len(all_volumes), out_dir)


if __name__ == "__main__":
    main()
