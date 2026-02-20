"""
Build temporal pair dataset from data_multiz raw/segmented tiles.

For each spatial position (C, z, y, x) that has masks at consecutive
timepoints T and T+1, this script:
  1. Loads both raw images and their instance-segmentation masks
  2. Extracts cell centres from each mask
  3. Normalises both images to [-1, 1]
  4. Saves per-pair arrays to data_multiz/processed/pairs/{train,val,test}/

Split is 80 / 10 / 10 on unique spatial positions to prevent leakage of
the same spatial region across splits.

Usage:
    python -m preprocessing.build_temporal_dataset            # default paths
    python -m preprocessing.build_temporal_dataset --help
"""

import argparse
import re
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
import tifffile

from preprocessing.extract_centres import extract_centres_from_mask
from utils.normalization import normalize_raw_image, to_minus_one_one

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Filename parsing
# ------------------------------------------------------------------

_RAW_RE = re.compile(
    r"tile_T(\d+)_C(\d+)_z(\d+)_y(\d+)_x(\d+)\.tif$"
)
_MASK_RE = re.compile(
    r"tile_T(\d+)_C(\d+)_z(\d+)_y(\d+)_x(\d+)_masks\.tif$"
)


def _parse_raw(name: str):
    """Return (t, C, z, y, x) ints or None."""
    m = _RAW_RE.match(name)
    if m:
        return tuple(int(v) for v in m.groups())
    return None


def _parse_mask(name: str):
    """Return (t, C, z, y, x) ints or None."""
    m = _MASK_RE.match(name)
    if m:
        return tuple(int(v) for v in m.groups())
    return None


# ------------------------------------------------------------------
# Pair discovery
# ------------------------------------------------------------------

def discover_pairs(raw_dir: Path, seg_dir: Path):
    """
    Return a list of tuples:
        (t0, t1, pos_key, raw0_path, raw1_path, mask0_path, mask1_path)
    where pos_key = (C, z, y, x) and t1 == t0 + 1.
    """
    # Build lookup: (t, C, z, y, x) -> raw_path
    raw_lookup: dict = {}
    for p in raw_dir.glob("tile_T*_C*_z*_y*_x*.tif"):
        parsed = _parse_raw(p.name)
        if parsed is not None:
            raw_lookup[parsed] = p

    # Build lookup: (t, C, z, y, x) -> mask_path
    mask_lookup: dict = {}
    for p in seg_dir.glob("tile_T*_C*_z*_y*_x*_masks.tif"):
        parsed = _parse_mask(p.name)
        if parsed is not None:
            mask_lookup[parsed] = p

    # Group mask tiles by spatial position -> sorted list of T values
    pos_to_times: dict = defaultdict(list)
    for (t, C, z, y, x) in mask_lookup:
        pos_to_times[(C, z, y, x)].append(t)

    for key in pos_to_times:
        pos_to_times[key].sort()

    # Find consecutive (t, t+1) pairs where both raw + mask exist
    pairs = []
    for (C, z, y, x), times in pos_to_times.items():
        t_set = set(times)
        for t0 in times:
            t1 = t0 + 1
            if t1 not in t_set:
                continue
            key0 = (t0, C, z, y, x)
            key1 = (t1, C, z, y, x)
            if key0 not in raw_lookup or key1 not in raw_lookup:
                continue
            if key0 not in mask_lookup or key1 not in mask_lookup:
                continue
            pairs.append((
                t0, t1,
                (C, z, y, x),
                raw_lookup[key0], raw_lookup[key1],
                mask_lookup[key0], mask_lookup[key1],
            ))

    return pairs


# ------------------------------------------------------------------
# Data splitting (on unique spatial positions)
# ------------------------------------------------------------------

def split_by_position(pairs, train_frac=0.8, val_frac=0.1, seed=42):
    """
    Split *pairs* into train / val / test by unique spatial position
    (C, z, y, x) to prevent leakage of the same location between splits.
    """
    unique_pos = sorted(set(p[2] for p in pairs))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_pos)

    n = len(unique_pos)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_pos = set(unique_pos[:n_train])
    val_pos = set(unique_pos[n_train: n_train + n_val])
    test_pos = set(unique_pos[n_train + n_val:])

    train = [p for p in pairs if p[2] in train_pos]
    val   = [p for p in pairs if p[2] in val_pos]
    test  = [p for p in pairs if p[2] in test_pos]

    return train, val, test


# ------------------------------------------------------------------
# Processing a single pair
# ------------------------------------------------------------------

def process_pair(raw0_path: Path, raw1_path: Path,
                 mask0_path: Path, mask1_path: Path,
                 min_cells: int = 3):
    """
    Load, validate, and preprocess one temporal pair.

    Returns a dict with keys:
        img_t0, img_t1       : float32 [-1,1], shape (H, W)
        centres_t0, centres_t1: float32, shape (N, 2)  [(y,x) pixels]
    or None if the pair should be skipped (too few cells).
    """
    img0 = np.squeeze(tifffile.imread(raw0_path).astype(np.float32))
    img1 = np.squeeze(tifffile.imread(raw1_path).astype(np.float32))
    msk0 = np.squeeze(tifffile.imread(mask0_path))
    msk1 = np.squeeze(tifffile.imread(mask1_path))

    centres0 = extract_centres_from_mask(msk0)
    centres1 = extract_centres_from_mask(msk1)

    if len(centres0) < min_cells or len(centres1) < min_cells:
        return None

    img0 = to_minus_one_one(normalize_raw_image(img0))
    img1 = to_minus_one_one(normalize_raw_image(img1))

    return {
        "img_t0":      img0.astype(np.float32),
        "img_t1":      img1.astype(np.float32),
        "centres_t0":  centres0.astype(np.float32),
        "centres_t1":  centres1.astype(np.float32),
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def build_split(pairs, out_dir: Path, split_name: str, min_cells: int = 3):
    """Process a list of pairs and save to out_dir/split_name/."""
    split_dir = out_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    skipped = 0
    total = len(pairs)
    log_every = max(1, total // 20)   # log at every ~5 %

    logger.info(f"  [{split_name}] processing {total} pairs ...")
    for i, (t0, t1, pos, raw0, raw1, msk0, msk1) in enumerate(pairs):
        result = process_pair(raw0, raw1, msk0, msk1, min_cells=min_cells)
        if result is None:
            skipped += 1
        else:
            prefix = split_dir / f"pair_{saved:05d}"
            np.save(f"{prefix}_img_t0.npy",      result["img_t0"])
            np.save(f"{prefix}_img_t1.npy",      result["img_t1"])
            np.save(f"{prefix}_centres_t0.npy",  result["centres_t0"])
            np.save(f"{prefix}_centres_t1.npy",  result["centres_t1"])
            saved += 1

        if (i + 1) % log_every == 0 or (i + 1) == total:
            pct = 100.0 * (i + 1) / total
            logger.info(
                f"  [{split_name}]  {i+1}/{total}  ({pct:.0f}%)  "
                f"saved={saved}  skipped={skipped}"
            )

    logger.info(
        f"  [{split_name}] done — saved={saved}  skipped={skipped} "
        f"(of {total} candidates)"
    )
    return saved


def main():
    parser = argparse.ArgumentParser(description="Build temporal pair dataset.")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config (e.g. configs/frame2.yaml); individual args override config values",
    )
    parser.add_argument(
        "--raw_dir",
        default=None,
        help="Directory with raw tile .tif files",
    )
    parser.add_argument(
        "--seg_dir",
        default=None,
        help="Directory with mask tile _masks.tif files",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output base directory",
    )
    parser.add_argument("--train_frac", type=float, default=None)
    parser.add_argument("--val_frac",   type=float, default=None)
    parser.add_argument("--seed",       type=int,   default=None)
    parser.add_argument("--min_cells",  type=int,   default=None)
    args = parser.parse_args()

    # Load config and pull temporal_dataset section
    cfg_td: dict = {}
    seed_default = 42
    if args.config:
        with open(args.config) as fh:
            cfg = yaml.safe_load(fh)
        cfg_td = cfg.get('temporal_dataset', {})
        seed_default = cfg.get('random_seed', 42)

    # Individual CLI args take priority; config values are the fallback
    raw_dir   = Path(args.raw_dir  or cfg_td.get('raw_dir',     'data_multiz/raw'))
    seg_dir   = Path(args.seg_dir  or cfg_td.get('seg_dir',     'data_multiz/segmented'))
    out_dir   = Path(args.out_dir  or cfg_td.get('out_dir',     'data_multiz/processed/pairs'))
    train_frac = args.train_frac   if args.train_frac  is not None else cfg_td.get('train_frac',  0.8)
    val_frac   = args.val_frac     if args.val_frac    is not None else cfg_td.get('val_frac',    0.1)
    seed       = args.seed         if args.seed        is not None else cfg_td.get('seed',        seed_default)
    min_cells  = args.min_cells    if args.min_cells   is not None else cfg_td.get('min_cells',   3)

    logger.info(f"Scanning raw:  {raw_dir}")
    logger.info(f"Scanning seg:  {seg_dir}")

    pairs = discover_pairs(raw_dir, seg_dir)
    logger.info(f"Found {len(pairs)} consecutive-T tile pairs")

    if len(pairs) == 0:
        logger.error("No pairs found — check paths and filename patterns.")
        return

    train, val, test = split_by_position(
        pairs,
        train_frac=train_frac,
        val_frac=val_frac,
        seed=seed,
    )
    logger.info(
        f"Split (by spatial position): "
        f"train={len(train)}  val={len(val)}  test={len(test)}"
    )

    build_split(train, out_dir, "train", min_cells=min_cells)
    build_split(val,   out_dir, "val",   min_cells=min_cells)
    build_split(test,  out_dir, "test",  min_cells=min_cells)

    logger.info("Done.")


if __name__ == "__main__":
    main()
