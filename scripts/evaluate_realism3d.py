"""
Realism evaluation: compare real vs synthetic 3D cell volumes.

Runs two independent evaluation tracks:

  Track 1 — Pixel-level metrics (no GPU required):
      SNR, 3D radial power spectrum, voxel intensity histograms (Wasserstein + KL).

  Track 2 — Embedding-based metrics (GPU recommended):
      Fréchet Distance and linear probe balanced accuracy in the frozen
      frame1_3d UNet encoder's latent space.  t-SNE scatter saved as PNG.

Results are written to:
    <out_dir>/evaluation_summary.yaml
    <out_dir>/power_spectrum_3d.png
    <out_dir>/intensity_histogram_3d.png
    <out_dir>/tsne_embeddings.png          (embedding track only)

Usage
-----
# Full evaluation (pixel + embedding):
python -m scripts.evaluate_realism3d \\
    --real_dirs  data_live_node1_3d/test data_live_node2_3d/test \\
    --syn_dir    synthetic_cells_3d/ \\
    --ckpt       checkpoints/frame1_3d/best.pt \\
    --config     configs/frame1_3d.yaml \\
    --out_dir    evaluation/results/

# Pixel metrics only (no encoder checkpoint needed):
python -m scripts.evaluate_realism3d \\
    --real_dirs  data_live_node1_3d/test \\
    --syn_dir    synthetic_cells_3d/ \\
    --no_encoder \\
    --out_dir    evaluation/results/
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tifffile
import torch

from utils.normalization import normalize_raw_image
from evaluation.metrics3d import compare_all_metrics
from evaluation.embedding3d import (
    extract_embeddings, frechet_distance, linear_probe, plot_umap,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ─── Data loaders ─────────────────────────────────────────────────────────────

def load_real_pairs(
    real_dirs: List[str],
    max_samples: int = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Load real (image, centres) pairs from one or more test directories.

    Expects files matching: *_image.npy + *_centres.npy

    Returns list of (image_np (D,H,W) float32, centres_np (N,3) float32).
    """
    tuples = []
    for d in real_dirs:
        image_files = sorted(Path(d).glob("*_image.npy"))
        for img_f in image_files:
            centres_f = Path(str(img_f).replace("_image.npy", "_centres.npy"))
            if not centres_f.exists():
                logger.warning("Missing centres for %s — skipping", img_f.name)
                continue
            image_np   = np.load(img_f).astype(np.float32)
            centres_np = np.load(centres_f).astype(np.float32)
            tuples.append((image_np, centres_np))
            if max_samples and len(tuples) >= max_samples:
                break
        if max_samples and len(tuples) >= max_samples:
            break
    logger.info("Loaded %d real (image, centres) pairs", len(tuples))
    return tuples


def load_synthetic_pairs(
    syn_dir: str,
    max_samples: int = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Load synthetic (image, centres) pairs from a generation output directory.

    Supports:
        synthetic_3d_NNNN.tif  + synthetic_3d_NNNN_centres.npy
        synthetic_3d_NNNN.npy  + synthetic_3d_NNNN_centres.npy

    The image is returned as (D, H, W) float32 in whatever range it was saved.
    (normalisation to [-1,1] is applied inside build_joint_input; normalise_raw_image
    handles the conversion for pixel metrics later.)

    Returns list of (image_np (D,H,W) float32, centres_np (N,3) float32).
    """
    tuples = []
    syn_path = Path(syn_dir)

    # Try .tif first, fall back to .npy
    image_files = sorted(syn_path.glob("synthetic_3d_????.tif"))
    if not image_files:
        image_files = sorted(syn_path.glob("synthetic_3d_????.npy"))

    for img_f in image_files:
        stem = img_f.stem                                      # e.g. synthetic_3d_0000
        centres_f = syn_path / f"{stem}_centres.npy"
        if not centres_f.exists():
            logger.warning("Missing centres for %s — skipping", img_f.name)
            continue

        if img_f.suffix == ".tif":
            image_np = tifffile.imread(str(img_f)).astype(np.float32)
        else:
            image_np = np.load(img_f).astype(np.float32)

        # Ensure (D, H, W) — drop channel dim if present
        if image_np.ndim == 4 and image_np.shape[0] == 1:
            image_np = image_np[0]
        elif image_np.ndim == 4:
            image_np = image_np[0]   # take first channel if multi-channel

        centres_np = np.load(centres_f).astype(np.float32)
        tuples.append((image_np, centres_np))

        if max_samples and len(tuples) >= max_samples:
            break

    logger.info("Loaded %d synthetic (image, centres) pairs from %s", len(tuples), syn_dir)
    return tuples


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate realism of synthetic 3D cell volumes."
    )
    parser.add_argument(
        "--real_dirs", nargs="+", required=True,
        help="One or more directories containing *_image.npy + *_centres.npy real test data",
    )
    parser.add_argument(
        "--syn_dir", required=True,
        help="Directory containing synthetic_3d_NNNN.tif + synthetic_3d_NNNN_centres.npy",
    )
    parser.add_argument(
        "--ckpt", default="checkpoints/frame1_3d/best.pt",
        help="Path to frame1_3d DDPM checkpoint for the frozen encoder",
    )
    parser.add_argument(
        "--config", default="configs/frame1_3d.yaml",
        help="Config YAML corresponding to --ckpt",
    )
    parser.add_argument(
        "--out_dir", default="evaluation/results",
        help="Output directory for plots and summary YAML",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Volumes per encoder forward pass (reduce if GPU OOM)",
    )
    parser.add_argument(
        "--max_real", type=int, default=None,
        help="Limit number of real volumes loaded (default: all)",
    )
    parser.add_argument(
        "--max_syn", type=int, default=None,
        help="Limit number of synthetic volumes loaded (default: all)",
    )
    parser.add_argument(
        "--no_encoder", action="store_true",
        help="Skip embedding track — run pixel metrics only (no GPU / checkpoint needed)",
    )
    parser.add_argument(
        "--no_pixel", action="store_true",
        help="Skip pixel-level metrics (Track 1) — run encoder track only",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device for encoder inference: cuda / cpu (auto-detected when omitted)",
    )
    parser.add_argument(
        "--heatmap_sigma", type=float, default=3.0,
        help="Heatmap sigma passed to generate_conditioning_maps3d",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Load data ──────────────────────────────────────────────────────────────
    real_tuples = load_real_pairs(args.real_dirs, max_samples=args.max_real)
    syn_tuples  = load_synthetic_pairs(args.syn_dir, max_samples=args.max_syn)

    if not real_tuples:
        logger.error("No real pairs found. Check --real_dirs.")
        sys.exit(1)
    if not syn_tuples:
        logger.error("No synthetic pairs found. Check --syn_dir.")
        sys.exit(1)

    # ── Track 1: Pixel-level metrics ──────────────────────────────────────────
    if not args.no_pixel:
        logger.info("=" * 60)
        logger.info("Track 1: Pixel-level metrics")
        logger.info("=" * 60)

        real_vols = [normalize_raw_image(t[0]) for t in real_tuples]   # → [0, 1]
        syn_vols  = [normalize_raw_image(t[0]) for t in syn_tuples]     # → [0, 1]

        pixel_summary = compare_all_metrics(
            real_vols=real_vols,
            syn_vols=syn_vols,
            out_dir=str(out_dir),
        )
        summary = dict(pixel_summary)
    else:
        logger.info("Pixel track skipped (--no_pixel).")
        summary = {}

    # ── Track 2: Embedding metrics ────────────────────────────────────────────
    if not args.no_encoder:
        logger.info("=" * 60)
        logger.info("Track 2: Embedding metrics (frozen UNet encoder)")
        logger.info("=" * 60)
        logger.info("Loading encoder from %s ...", args.ckpt)

        from models.encoder3d import CellEncoder3D
        encoder = CellEncoder3D.from_ddpm_checkpoint(
            ckpt_path=args.ckpt,
            config_path=args.config,
            device=device,
        )

        # Reconstruct active_channels and prev_frame flag from config
        with open(args.config) as _f:
            config = yaml.safe_load(_f)
        _cond_cfg = config.get('unet', {}).get('conditioning', {})
        _active_channels = {k: bool(v) for k, v in _cond_cfg.items()} if _cond_cfg else None
        _prev_frame = bool(config.get('unet', {}).get('prev_frame', False))

        logger.info("Extracting embeddings — real set (%d volumes) ...", len(real_tuples))
        Z_real = extract_embeddings(
            encoder=encoder,
            data_tuples=real_tuples,
            device=device,
            batch_size=args.batch_size,
            heatmap_sigma=args.heatmap_sigma,
            active_channels=_active_channels,
            prev_frame=_prev_frame,
        )

        logger.info("Extracting embeddings — synthetic set (%d volumes) ...", len(syn_tuples))
        Z_syn = extract_embeddings(
            encoder=encoder,
            data_tuples=syn_tuples,
            device=device,
            batch_size=args.batch_size,
            heatmap_sigma=args.heatmap_sigma,
            active_channels=_active_channels,
            prev_frame=_prev_frame,
        )

        logger.info("Computing Fréchet Distance ...")
        fd = frechet_distance(Z_real, Z_syn)
        logger.info("  Fréchet Distance = %.4f", fd)

        logger.info("Training linear probe ...")
        probe_acc = linear_probe(Z_real, Z_syn)
        logger.info("  Linear probe balanced accuracy = %.4f", probe_acc)

        logger.info("Generating UMAP plot ...")
        umap_path = str(out_dir / "umap_embeddings.png")
        plot_umap(
            Z_real=Z_real,
            Z_syn=Z_syn,
            save_path=umap_path,
        )

        # Save raw embeddings for downstream analysis
        np.save(str(out_dir / "embeddings_real.npy"), Z_real)
        np.save(str(out_dir / "embeddings_syn.npy"),  Z_syn)
        logger.info("Saved embeddings → embeddings_real.npy / embeddings_syn.npy")

        summary["frechet_distance"] = round(float(fd), 4)
        summary["linear_probe_balanced_accuracy"] = round(float(probe_acc), 4)
        summary["embedding_dim"] = int(Z_real.shape[1])
        summary["encoder_ckpt"] = str(args.ckpt)

    else:
        logger.info("Embedding track skipped (--no_encoder).")

    # ── Save summary YAML ─────────────────────────────────────────────────────
    summary_path = out_dir / "evaluation_summary.yaml"

    # Strip any numpy types before YAML serialisation
    def _to_python(v):
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    summary_clean = {k: _to_python(v) for k, v in summary.items()}

    with open(summary_path, "w") as f:
        yaml.dump(summary_clean, f, default_flow_style=False, sort_keys=False)

    logger.info("=" * 60)
    logger.info("Evaluation complete.  Summary:")
    for k, v in summary_clean.items():
        if isinstance(v, float):
            logger.info("  %-42s = %.4f", k, v)
        else:
            logger.info("  %-42s = %s", k, v)
    logger.info("Results saved to: %s", out_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
