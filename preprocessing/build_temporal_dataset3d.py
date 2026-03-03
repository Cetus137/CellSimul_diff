"""
Build 3D temporal pair dataset from data_live_node1_3d temporal tiles.

For each spatial position (tile + z/y/x coords) that has 256³ volumes at 
consecutive timepoints T and T+1, this script:
  1. Loads both raw 256³ volumes and their instance-segmentation masks
  2. Crops to 128³ patches (following frame1_3d logic: z_start, 2×2 Y/X tiling)
  3. Extracts cell centres from each mask
  4. Normalises both volumes to [-1, 1]
  5. Saves per-pair arrays to data_live_node1_3d/pairs/{train,val,test}/

Split is done on unique spatial positions (tile + coords) to prevent leakage
of the same spatial region across splits.

Usage:
    python -m preprocessing.build_temporal_dataset3d                    # default paths
    python -m preprocessing.build_temporal_dataset3d --config configs/frame2_3d.yaml
"""

import argparse
import re
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
import tifffile

from utils.normalization import normalize_raw_image, to_minus_one_one
from scipy import ndimage as _ndimage

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Filename parsing
# ------------------------------------------------------------------

# Example: timepoint_0048_b2-2a_2c_pos6-01_crop_C1_t0-65_z72-328_y598-1622_x568-1592_tile_0012_z0-256_y448-704_x448-704.tif
_RAW_RE = re.compile(
    r"timepoint_(\d+)_(?:.*_)?tile_(\d+)_z(\d+-\d+)_y(\d+-\d+)_x(\d+-\d+)\.tif$"
)
_MASK_RE = re.compile(
    r"timepoint_(\d+)_(?:.*_)?tile_(\d+)_z(\d+-\d+)_y(\d+-\d+)_x(\d+-\d+)_masks\.tif$"
)


def _parse_raw(name: str):
    """Return (timepoint, tile, z_range, y_range, x_range) or None."""
    m = _RAW_RE.match(name)
    if m:
        tp, tile, z_range, y_range, x_range = m.groups()
        return (int(tp), int(tile), z_range, y_range, x_range)
    return None


def _parse_mask(name: str):
    """Return (timepoint, tile, z_range, y_range, x_range) or None."""
    m = _MASK_RE.match(name)
    if m:
        tp, tile, z_range, y_range, x_range = m.groups()
        return (int(tp), int(tile), z_range, y_range, x_range)
    return None


# ------------------------------------------------------------------
# Centre extraction (3D)
# ------------------------------------------------------------------

def extract_centres_from_mask3d(
    mask: np.ndarray,
    min_volume: int = 500,
    max_volume: int = 500_000
) -> np.ndarray:
    """
    Extract cell centroids from a 3D integer instance-segmentation mask.

    Args:
        mask: (D, H, W) integer array; background = 0, each cell = unique ID > 0.
        min_volume: Minimum voxel volume to accept a cell.
        max_volume: Maximum voxel volume to accept a cell.

    Returns:
        centres: (N, 3) float32 array in (z, y, x) order.
    """
    unique_ids = np.unique(mask)
    unique_ids = unique_ids[unique_ids > 0]

    if len(unique_ids) == 0:
        return np.empty((0, 3), dtype=np.float32)

    centres = []
    for cell_id in unique_ids:
        cell_mask = mask == cell_id
        volume = int(cell_mask.sum())
        if volume < min_volume or volume > max_volume:
            continue
        centroid = _ndimage.center_of_mass(cell_mask)  # (z, y, x) float
        centres.append(centroid)

    if len(centres) == 0:
        return np.empty((0, 3), dtype=np.float32)

    return np.array(centres, dtype=np.float32)


# ------------------------------------------------------------------
# Pair discovery
# ------------------------------------------------------------------

def discover_pairs(data_dir: Path):
    """
    Return a list of tuples:
        (t0, t1, pos_key, raw0_path, raw1_path, mask0_path, mask1_path)
    where pos_key = (tile, z_range, y_range, x_range) and t1 == t0 + 1.
    """
    # Build lookup: (timepoint, tile, z_range, y_range, x_range) -> raw_path
    raw_lookup: dict = {}
    for p in data_dir.glob("timepoint_*_tile_*_z*_y*_x*.tif"):
        parsed = _parse_raw(p.name)
        if parsed is not None:
            raw_lookup[parsed] = p

    # Build lookup: (timepoint, tile, z_range, y_range, x_range) -> mask_path
    mask_lookup: dict = {}
    for p in data_dir.glob("timepoint_*_tile_*_z*_y*_x*_masks.tif"):
        parsed = _parse_mask(p.name)
        if parsed is not None:
            mask_lookup[parsed] = p

    # Group by spatial position -> sorted list of timepoint values
    pos_to_times: dict = defaultdict(list)
    for (tp, tile, z_range, y_range, x_range) in mask_lookup:
        pos_to_times[(tile, z_range, y_range, x_range)].append(tp)

    for key in pos_to_times:
        pos_to_times[key].sort()

    # Find consecutive (t, t+1) pairs where both raw + mask exist
    pairs = []
    for (tile, z_range, y_range, x_range), times in pos_to_times.items():
        t_set = set(times)
        for t0 in times:
            t1 = t0 + 1
            if t1 not in t_set:
                continue
            key0 = (t0, tile, z_range, y_range, x_range)
            key1 = (t1, tile, z_range, y_range, x_range)
            if key0 not in raw_lookup or key1 not in raw_lookup:
                continue
            if key0 not in mask_lookup or key1 not in mask_lookup:
                continue
            pairs.append((
                t0, t1,
                (tile, z_range, y_range, x_range),
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
    to prevent leakage of the same location between splits.
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
# Cropping 256³ -> multiple 128³ patches
# ------------------------------------------------------------------

def extract_128_patches_from_256(
    volume: np.ndarray,
    centres: np.ndarray,
    z_start: int = 24,
    patch_size: int = 128,
    patch_stride: int = 128,
    border_margin: int = 10
):
    """
    Crop a 256³ volume into 128³ patches.
    
    Strategy (matching frame1_3d):
      - Z: crop [z_start : z_start+128]
      - Y/X: tile with stride (default 128 → 2×2 = 4 patches)
    
    For each patch:
      - Adjust centres to patch coordinates
      - Filter centres that fall outside or too close to border
    
    Returns:
        List of dicts: [{"patch": (128,128,128), "centres": (N,3)}, ...]
    """
    D, H, W = volume.shape
    assert D >= z_start + patch_size, "Volume too small in Z"
    assert H >= patch_size and W >= patch_size, "Volume too small in Y or X"
    
    # Crop Z slab
    volume_slab = volume[z_start : z_start + patch_size, :, :]  # (128, 256, 256)
    
    # Adjust centres: subtract z_start, filter out those outside the slab
    centres_slab = centres.copy()
    centres_slab[:, 0] -= z_start  # adjust z coordinate
    valid_z = (centres_slab[:, 0] >= 0) & (centres_slab[:, 0] < patch_size)
    centres_slab = centres_slab[valid_z]
    
    patches = []
    
    # Tile Y × X
    for y_start in range(0, H, patch_stride):
        if y_start + patch_size > H:
            continue
        for x_start in range(0, W, patch_stride):
            if x_start + patch_size > W:
                continue
            
            # Extract patch
            patch = volume_slab[:, y_start:y_start+patch_size, x_start:x_start+patch_size]
            
            # Adjust centres to patch coordinates
            patch_centres = centres_slab.copy()
            patch_centres[:, 1] -= y_start  # adjust y
            patch_centres[:, 2] -= x_start  # adjust x
            
            # Filter centres within patch bounds
            valid = (
                (patch_centres[:, 0] >= 0) & (patch_centres[:, 0] < patch_size) &
                (patch_centres[:, 1] >= 0) & (patch_centres[:, 1] < patch_size) &
                (patch_centres[:, 2] >= 0) & (patch_centres[:, 2] < patch_size)
            )
            patch_centres = patch_centres[valid]
            
            # Filter border cells (3D version)
            if len(patch_centres) > 0:
                valid_border = (
                    (patch_centres[:, 0] >= border_margin) & (patch_centres[:, 0] < patch_size - border_margin) &
                    (patch_centres[:, 1] >= border_margin) & (patch_centres[:, 1] < patch_size - border_margin) &
                    (patch_centres[:, 2] >= border_margin) & (patch_centres[:, 2] < patch_size - border_margin)
                )
                patch_centres = patch_centres[valid_border]
            
            patches.append({
                "patch": patch,
                "centres": patch_centres
            })
    
    return patches


# ------------------------------------------------------------------
# Processing a single pair
# ------------------------------------------------------------------

def process_pair(
    raw0_path: Path, raw1_path: Path,
    mask0_path: Path, mask1_path: Path,
    z_start: int = 24,
    patch_size: int = 128,
    min_cells: int = 5,
    min_cell_volume: int = 500,
    max_cell_volume: int = 500_000
):
    """
    Load, validate, and preprocess one temporal pair.
    
    Returns a list of dicts (one per patch):
        [
            {
                "vol_t0": (128,128,128) float32 [-1,1],
                "vol_t1": (128,128,128) float32 [-1,1],
                "centres_t0": (N,3) float32 (z,y,x),
                "centres_t1": (M,3) float32 (z,y,x)
            },
            ...
        ]
    or empty list if no valid patches.
    """
    # Load 256³ volumes
    vol0 = np.squeeze(tifffile.imread(raw0_path).astype(np.float32))
    vol1 = np.squeeze(tifffile.imread(raw1_path).astype(np.float32))
    msk0 = np.squeeze(tifffile.imread(mask0_path))
    msk1 = np.squeeze(tifffile.imread(mask1_path))
    
    # Extract centres from full 256³ masks
    centres0_full = extract_centres_from_mask3d(
        msk0, min_volume=min_cell_volume, max_volume=max_cell_volume
    )
    centres1_full = extract_centres_from_mask3d(
        msk1, min_volume=min_cell_volume, max_volume=max_cell_volume
    )
    
    # Normalize volumes
    vol0 = to_minus_one_one(normalize_raw_image(vol0))
    vol1 = to_minus_one_one(normalize_raw_image(vol1))
    
    # Extract 128³ patches from both timepoints
    patches0 = extract_128_patches_from_256(vol0, centres0_full, z_start=z_start, patch_size=patch_size)
    patches1 = extract_128_patches_from_256(vol1, centres1_full, z_start=z_start, patch_size=patch_size)
    
    # Pair up patches (assuming same tiling strategy)
    assert len(patches0) == len(patches1), "Patch counts must match between t0 and t1"
    
    results = []
    for p0, p1 in zip(patches0, patches1):
        # Filter by minimum cell count
        if len(p0["centres"]) < min_cells or len(p1["centres"]) < min_cells:
            continue
        
        results.append({
            "vol_t0":      p0["patch"].astype(np.float32),
            "vol_t1":      p1["patch"].astype(np.float32),
            "centres_t0":  p0["centres"].astype(np.float32),
            "centres_t1":  p1["centres"].astype(np.float32),
        })
    
    return results


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def build_split(pairs, out_dir: Path, split_name: str, 
                z_start: int = 24, patch_size: int = 128,
                min_cells: int = 5, min_cell_volume: int = 500,
                max_cell_volume: int = 500_000):
    """Process a list of pairs and save to out_dir/split_name/."""
    split_dir = out_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    skipped_pairs = 0
    total_pairs = len(pairs)
    log_every = max(1, total_pairs // 20)

    logger.info(f"  [{split_name}] processing {total_pairs} tile pairs ...")
    for i, (t0, t1, pos, raw0, raw1, msk0, msk1) in enumerate(pairs):
        results = process_pair(
            raw0, raw1, msk0, msk1,
            z_start=z_start,
            patch_size=patch_size,
            min_cells=min_cells,
            min_cell_volume=min_cell_volume,
            max_cell_volume=max_cell_volume
        )
        
        if len(results) == 0:
            skipped_pairs += 1
        else:
            # Save each patch from this pair
            for patch_data in results:
                prefix = split_dir / f"pair3d_{saved:05d}"
                np.save(f"{prefix}_vol_t0.npy",      patch_data["vol_t0"])
                np.save(f"{prefix}_vol_t1.npy",      patch_data["vol_t1"])
                np.save(f"{prefix}_centres_t0.npy",  patch_data["centres_t0"])
                np.save(f"{prefix}_centres_t1.npy",  patch_data["centres_t1"])
                saved += 1

        if (i + 1) % log_every == 0 or (i + 1) == total_pairs:
            pct = 100.0 * (i + 1) / total_pairs
            logger.info(
                f"  [{split_name}]  {i+1}/{total_pairs} pairs ({pct:.0f}%)  "
                f"→ {saved} patches saved, {skipped_pairs} pairs skipped"
            )

    logger.info(
        f"  [{split_name}] done — {saved} patches saved from {total_pairs - skipped_pairs} valid pairs "
        f"({skipped_pairs} pairs had no valid patches)"
    )
    return saved


def build_temporal_dataset3d(config_path: str = "configs/frame2_3d.yaml", pair_index: int = None):
    """
    Build 3D temporal pair dataset from consecutive-timepoint volumes.
    
    Args:
        config_path: Path to YAML config file.
        pair_index: Optional 0-based index of specific pair to process (for array jobs).
                   If None, processes all pairs and creates train/val/test splits.
                   If specified, only processes that pair and saves to the correct split.
    """
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    
    cfg_td = cfg.get('temporal_dataset', {})
    cfg_prep = cfg.get('preprocessing', {})
    seed_default = cfg.get('random_seed', 42)

    # Resolve parameters from config
    raw_dir    = Path(cfg_td.get('raw_dir', 'data_live_node1_3d'))
    out_dir    = Path(cfg_td.get('pairs_dir', 'data_live_node1_3d/pairs'))
    train_frac = cfg_td.get('train_frac', 0.8)
    val_frac   = cfg_td.get('val_frac', 0.1)
    seed       = cfg_td.get('split_seed', seed_default)
    min_cells  = cfg_td.get('min_cells', 5)
    z_start    = cfg_prep.get('z_start', 24)
    
    patch_size       = cfg_prep.get('patch_size', 128)
    min_cell_volume  = cfg_prep.get('min_cell_volume', 500)
    max_cell_volume  = cfg_prep.get('max_cell_volume', 500_000)

    logger.info(f"Scanning raw tiles:  {raw_dir}")
    logger.info(f"Output directory:    {out_dir}")
    logger.info(f"Z crop start:        {z_start}")
    logger.info(f"Patch size:          {patch_size}³")

    all_pairs = discover_pairs(raw_dir)
    logger.info(f"Found {len(all_pairs)} consecutive-timepoint tile pairs")

    if len(all_pairs) == 0:
        logger.error("No pairs found — check paths and filename patterns.")
        return

    # ── Array job mode: process single pair ─────────────────────────────────
    if pair_index is not None:
        if pair_index < 0 or pair_index >= len(all_pairs):
            logger.error(f"Invalid pair_index {pair_index}. Valid range: 0-{len(all_pairs)-1}")
            return
        
        # Determine split assignment deterministically (same logic as split_by_position)
        unique_pos = sorted(set(p[2] for p in all_pairs))
        rng = np.random.default_rng(seed)
        rng.shuffle(unique_pos)
        n = len(unique_pos)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        
        train_pos = set(unique_pos[:n_train])
        val_pos = set(unique_pos[n_train: n_train + n_val])
        test_pos = set(unique_pos[n_train + n_val:])
        
        # Get the position key for this pair
        pair = all_pairs[pair_index]
        pos_key = pair[2]  # (tile, z_range, y_range, x_range)
        
        if pos_key in train_pos:
            split_assignment = 'train'
        elif pos_key in val_pos:
            split_assignment = 'val'
        else:
            split_assignment = 'test'
        
        logger.info(f"ARRAY JOB MODE: Processing pair {pair_index}")
        logger.info(f"  Position: {pos_key}")
        logger.info(f"  Split assignment: {split_assignment}")
        
        # Process this single pair
        t0, t1, pos, raw0, raw1, msk0, msk1 = pair
        results = process_pair(
            raw0, raw1, msk0, msk1,
            z_start=z_start,
            patch_size=patch_size,
            min_cells=min_cells,
            min_cell_volume=min_cell_volume,
            max_cell_volume=max_cell_volume
        )
        
        if len(results) == 0:
            logger.warning(f"Pair {pair_index} produced no valid patches (skipped)")
            return
        
        # Save directly to assigned split folder
        split_dir = out_dir / split_assignment
        split_dir.mkdir(parents=True, exist_ok=True)
        
        for patch_idx, patch_data in enumerate(results):
            prefix = split_dir / f"pair3d_p{pair_index:05d}_{patch_idx:02d}"
            np.save(f"{prefix}_vol_t0.npy",      patch_data["vol_t0"])
            np.save(f"{prefix}_vol_t1.npy",      patch_data["vol_t1"])
            np.save(f"{prefix}_centres_t0.npy",  patch_data["centres_t0"])
            np.save(f"{prefix}_centres_t1.npy",  patch_data["centres_t1"])
        
        logger.info(f"Saved {len(results)} patches from pair {pair_index} → {split_dir}")
        return

    # ── Normal mode: process all pairs with splits ──────────────────────────
    train, val, test = split_by_position(
        all_pairs,
        train_frac=train_frac,
        val_frac=val_frac,
        seed=seed,
    )
    logger.info(
        f"Split (by spatial position): "
        f"train={len(train)}  val={len(val)}  test={len(test)}"
    )

    build_split(train, out_dir, "train", z_start=z_start, patch_size=patch_size,
                min_cells=min_cells, min_cell_volume=min_cell_volume, max_cell_volume=max_cell_volume)
    build_split(val,   out_dir, "val",   z_start=z_start, patch_size=patch_size,
                min_cells=min_cells, min_cell_volume=min_cell_volume, max_cell_volume=max_cell_volume)
    build_split(test,  out_dir, "test",  z_start=z_start, patch_size=patch_size,
                min_cells=min_cells, min_cell_volume=min_cell_volume, max_cell_volume=max_cell_volume)

    logger.info("Done.")


def main():
    """CLI entry point with argparse support."""
    parser = argparse.ArgumentParser(description="Build 3D temporal pair dataset.")
    parser.add_argument(
        "--config",
        default="configs/frame2_3d.yaml",
        help="Path to YAML config (e.g. configs/frame2_3d.yaml)",
    )
    parser.add_argument(
        "--pair_index",
        type=int,
        default=None,
        help="Optional: process only this pair index (for array jobs)"
    )
    args = parser.parse_args()
    
    build_temporal_dataset3d(args.config, pair_index=args.pair_index)


if __name__ == "__main__":
    main()
