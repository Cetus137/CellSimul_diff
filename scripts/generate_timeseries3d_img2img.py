#!/usr/bin/env python3
"""
SDEdit img2img temporal generation from real paired data.

For each pair discovered in PAIRS_DIR:
  - Loads V_t0 (real frame 0) and C_t1 (real centres at frame 1)
  - Corrupts V_t0 with noise at t_start via q(x_{t_start} | x_0)
  - Denoises conditioned on the heatmap of C_t1 to produce predicted V_t1
  - Saves predicted V_t1 alongside the real V_t1 (if --save_real) for comparison

Expected pair file naming convention (as in data_live_node1_3d/pairs/):
  pair3d_p{ID}_{SS}_vol_t0.npy
  pair3d_p{ID}_{SS}_vol_t1.npy
  pair3d_p{ID}_{SS}_centres_t0.npy   (optional — not used)
  pair3d_p{ID}_{SS}_centres_t1.npy   (required)

Usage:
  python scripts/generate_timeseries3d_img2img.py \\
      --checkpoint checkpoints/frame1_3d_combined_noD_noZ_raw/best.pt \\
      --config configs/frame1_3d.yaml \\
      --pairs_dir data_live_node1_3d/pairs \\
      --output_dir synthetic_cells_3d/img2img/t_start_250 \\
      --t_start 250 \\
      --use_ddim \\
      --max_pairs 50
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import tifffile
import yaml

from sampling.sample_from_centres3d import load_model3d, sample_batch_img2img3d

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ── Volume normalisation ───────────────────────────────────────────────────────

def load_and_normalise(path: Path) -> np.ndarray:
    """
    Load a .npy volume and normalise to [-1, 1] to match model training range.

    Handles:
      Already in [-1, 1]  -> pass through (clipped)
      uint8               -> / 255 -> [0,1] -> [-1,1]
      uint16 / large float -> percentile clip to [0,1] -> [-1,1]
    """
    vol = np.load(path).astype(np.float32)

    # Pass through if already training-normalised
    if vol.min() >= -1.05 and vol.max() <= 1.05:
        return np.clip(vol, -1.0, 1.0)

    # Map to [0, 1]
    p_low  = np.percentile(vol, 1.0)
    p_high = np.percentile(vol, 99.0)
    if p_high > p_low:
        vol = np.clip((vol - p_low) / (p_high - p_low), 0.0, 1.0)
    else:
        vol = np.zeros_like(vol)

    # [0, 1] -> [-1, 1]
    return (vol * 2.0 - 1.0).astype(np.float32)


# ── Pair discovery ─────────────────────────────────────────────────────────────

def find_pairs(pairs_dir: Path) -> list:
    """
    Recursively scan pairs_dir for matched (vol_t0, vol_t1, centres_t1) triples.

    Returns a list of dicts with keys: stem, vol_t0, vol_t1, centres_t1.
    Pairs where vol_t1 or centres_t1 are missing are silently skipped.
    """
    vol_t0_files = sorted(pairs_dir.rglob('*_vol_t0.npy'))
    pairs = []

    for vol_t0 in vol_t0_files:
        stem = vol_t0.name.replace('_vol_t0.npy', '')
        d    = vol_t0.parent

        vol_t1     = d / f'{stem}_vol_t1.npy'
        centres_t1 = d / f'{stem}_centres_t1.npy'

        if not vol_t1.exists():
            logger.debug(f'Missing vol_t1 for {stem}, skipping')
            continue
        if not centres_t1.exists():
            logger.debug(f'Missing centres_t1 for {stem}, skipping')
            continue

        pairs.append({
            'stem':       stem,
            'vol_t0':     vol_t0,
            'vol_t1':     vol_t1,
            'centres_t1': centres_t1,
        })

    return pairs


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='SDEdit img2img temporal generation from real paired data'
    )
    parser.add_argument('--checkpoint', required=True,
                        help='Path to trained model checkpoint (.pt)')
    parser.add_argument('--config',     required=True,
                        help='Path to model config YAML')
    parser.add_argument('--pairs_dir',  required=True,
                        help='Directory containing pair3d_*_vol_t0/t1.npy files')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for generated volumes')
    parser.add_argument('--t_start', type=int, default=250,
                        help='Forward-noise timestep (default: 250). '
                             'Lower = more faithful to V_t0; higher = more change.')
    parser.add_argument('--device',     default='cuda',
                        help='Device: cuda or cpu')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Pairs processed per GPU pass (reduce to 1 if OOM)')
    parser.add_argument('--max_pairs',  type=int, default=None,
                        help='Cap number of pairs processed (useful for quick tests)')
    parser.add_argument('--use_ddim',   action='store_true',
                        help='Use DDIM instead of full DDPM (much faster)')
    parser.add_argument('--ddim_steps', type=int, default=50,
                        help='Reference DDIM step count (only used with --use_ddim)')
    parser.add_argument('--ddim_eta',   type=float, default=0.0,
                        help='DDIM eta stochasticity (0=deterministic)')
    parser.add_argument('--heatmap_sigma', type=float, default=3.0,
                        help='Gaussian sigma for centre heatmap conditioning')
    parser.add_argument('--save_src',     action='store_true',
                        help='Save source V_t0 (first frame) to output dir')
    parser.add_argument('--save_heatmap', action='store_true',
                        help='Save C_t1 conditioning heatmap to output dir')
    parser.add_argument('--save_real',    action='store_true',
                        help='Save real V_t1 (second frame) to output dir for comparison')
    parser.add_argument('--no_visualization', action='store_true',
                        help='Skip saving orthogonal-slice PNG visualisations')
    args = parser.parse_args()

    # ── Setup ──────────────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info('=' * 56)
    logger.info('SDEdit Img2Img 3D Generation')
    logger.info('=' * 56)
    logger.info(f'  t_start     : {args.t_start}')
    logger.info(f'  use_ddim    : {args.use_ddim}'
                + (f'  (steps={args.ddim_steps})' if args.use_ddim else ''))
    logger.info(f'  batch_size  : {args.batch_size}')
    logger.info(f'  pairs_dir   : {args.pairs_dir}')
    logger.info(f'  output_dir  : {output_dir}')
    logger.info('=' * 56)

    # ── Discover pairs ─────────────────────────────────────────────────────────
    all_pairs = find_pairs(Path(args.pairs_dir))
    if not all_pairs:
        logger.error(f'No valid pairs found in {args.pairs_dir}')
        sys.exit(1)

    if args.max_pairs is not None:
        all_pairs = all_pairs[:args.max_pairs]

    logger.info(f'Pairs to process: {len(all_pairs)}')

    # ── Load model ─────────────────────────────────────────────────────────────
    logger.info('Loading model...')
    model = load_model3d(args.checkpoint, args.config, args.device)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    active_channels = cfg.get('unet', {}).get('conditioning', None)

    # ── Batch-process pairs ────────────────────────────────────────────────────
    n_done = 0
    i = 0
    while i < len(all_pairs):
        batch = all_pairs[i: i + args.batch_size]

        source_vols = [load_and_normalise(p['vol_t0'])                   for p in batch]
        centres_t1  = [np.load(p['centres_t1']).astype(np.float32)       for p in batch]

        volumes, metadatas = sample_batch_img2img3d(
            model=model,
            source_volumes=source_vols,
            centres_list=centres_t1,
            t_start=args.t_start,
            heatmap_sigma=args.heatmap_sigma,
            active_channels=active_channels,
            device=args.device,
            use_ddim=args.use_ddim,
            ddim_steps=args.ddim_steps,
            ddim_eta=args.ddim_eta,
        )

        for pair_info, vol_src, vol_pred, centres, meta in zip(
                batch, source_vols, volumes, centres_t1, metadatas):
            stem = pair_info['stem']

            # Source V_t0 (first frame — input to the noise/denoise process)
            if args.save_src:
                tifffile.imwrite(
                    str(output_dir / f'{stem}_src_t0.tif'),
                    vol_src.astype(np.float32)
                )

            # Predicted V_t1 (second / target frame) — always saved
            tifffile.imwrite(
                str(output_dir / f'{stem}_pred_t1.tif'),
                vol_pred.astype(np.float32)
            )

            # Conditioning heatmap (channel 0 of C_t1 condition maps)
            if args.save_heatmap:
                heatmap = meta['condition_maps'][0]  # (D, H, W)
                tifffile.imwrite(
                    str(output_dir / f'{stem}_heatmap_t1.tif'),
                    heatmap.astype(np.float32)
                )

            # Conditioning centres used
            np.save(str(output_dir / f'{stem}_centres_t1.npy'), centres)

            # Real V_t1 for side-by-side comparison
            if args.save_real:
                real_vol = load_and_normalise(pair_info['vol_t1'])
                tifffile.imwrite(
                    str(output_dir / f'{stem}_real_t1.tif'),
                    real_vol.astype(np.float32)
                )

            n_done += 1
            logger.info(f'[{n_done}/{len(all_pairs)}]  {stem}_pred_t1.tif')

        i += len(batch)

    logger.info('')
    logger.info(f'Done. {n_done} predicted volumes written to {output_dir}')


if __name__ == '__main__':
    main()
