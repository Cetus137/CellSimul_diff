"""
Build 3D training dataset from a folder of 256x256x256 cube volumes.

Expected input layout:
    cubes_dir/
        cube_0000.tif           ← raw fluorescence cube (256, 256, 256) uint16
        cube_0000_masks.tif     ← instance segmentation cube (256, 256, 256) int
        cube_0001.tif
        cube_0001_masks.tif
        ...

Each TIF is expected to be a multi-page tifffile readable as (D, H, W).

Pipeline per cube:
    1. Load raw cube + mask cube via tifffile.
    2. Extract 3D centre coordinates from the mask
       (scipy.ndimage.center_of_mass — ND-capable).
    3. Extract 128³ patches:
       - crop z[z_start : z_start+128]
       - tile Y × X with stride (default 128 → 2×2 = 4 patches per cube)
    4. Save patches as .npy files split into train / val / test.
"""

import numpy as np
import tifffile
from pathlib import Path
from typing import List, Dict
import yaml
from tqdm import tqdm
import logging

import sys
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.extract_patches3d import extract_patches3d, save_patch_dataset3d
from preprocessing.extract_centres import filter_border_cells

# Reuse the ND-capable centre extractor from the 2D module
from scipy import ndimage as _ndimage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Centre extraction (3D)
# ──────────────────────────────────────────────────────────────────────────────

def extract_centres_from_mask3d(
    mask: np.ndarray,
    min_volume: int = 500,
    max_volume: int = 500_000
) -> np.ndarray:
    """
    Extract cell centroids from a 3D integer instance-segmentation mask.

    Args:
        mask: (D, H, W) integer array; background = 0, each cell = unique ID > 0.
        min_volume: Minimum voxel volume to accept a cell (default 500).
        max_volume: Maximum voxel volume to accept a cell (default 500 000).

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


# ──────────────────────────────────────────────────────────────────────────────
# Split helper (reuse logic from build_dataset.py)
# ──────────────────────────────────────────────────────────────────────────────

def create_splits(
    num_samples: int,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    random_seed: int = 42
) -> Dict[str, np.ndarray]:
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, \
        "Fractions must sum to 1"
    np.random.seed(random_seed)
    indices = np.random.permutation(num_samples)
    n_train = int(num_samples * train_frac)
    n_val   = int(num_samples * val_frac)
    return {
        'train': indices[:n_train],
        'val':   indices[n_train:n_train + n_val],
        'test':  indices[n_train + n_val:],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main build function
# ──────────────────────────────────────────────────────────────────────────────

def build_dataset3d(config_path: str = "configs/frame1_3d.yaml", file_index: int = None) -> None:
    """
    Build 3D patch dataset from raw cube TIFs and their instance-mask TIFs.

    Args:
        config_path: Path to YAML config file (see configs/frame1_3d.yaml).
        file_index: Optional 0-based index of specific cube file to process (for array jobs).
                   If None, processes all files and creates train/val/test splits.
                   If specified, only processes that file and saves to all_patches/ directory (no split).
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    cubes_dir  = Path(config['raw_data']['cubes_dir'])
    output_dir = Path(config['processed_data']['patches_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    cube_suffix = config['raw_data'].get('cube_suffix', '.tif')
    mask_suffix = config['raw_data'].get('mask_suffix', '_masks.tif')

    patch_size  = config['preprocessing']['patch_size']
    stride      = config['preprocessing']['patch_stride']
    z_start     = config['preprocessing']['z_start']
    min_cells   = config['preprocessing']['min_cells_per_patch']
    min_vol     = config['preprocessing'].get('min_cell_volume', 500)
    max_vol     = config['preprocessing'].get('max_cell_volume', 500_000)

    # Discover cube files (exclude mask files)
    cube_files = sorted(
        f for f in cubes_dir.glob(f"*{cube_suffix}")
        if not f.name.endswith(mask_suffix)
    )

    if not cube_files:
        logger.error(f"No cube files found in {cubes_dir} with suffix '{cube_suffix}'")
        return

    logger.info(f"Found {len(cube_files)} cubes in {cubes_dir}")

    # Determine split assignment for array job mode
    split_assignment = None
    if file_index is not None:
        if file_index < 0 or file_index >= len(cube_files):
            logger.error(f"Invalid file_index {file_index}. Valid range: 0-{len(cube_files)-1}")
            return
        
        # Calculate which split this file belongs to (deterministic)
        n_files = len(cube_files)
        train_frac = config['splits']['train_fraction']
        val_frac = config['splits']['val_fraction']
        random_seed = config['splits']['random_seed']
        
        # Create deterministic file->split assignment
        np.random.seed(random_seed)
        file_indices = np.random.permutation(n_files)
        n_train = int(n_files * train_frac)
        n_val = int(n_files * val_frac)
        
        if file_index in file_indices[:n_train]:
            split_assignment = 'train'
        elif file_index in file_indices[n_train:n_train + n_val]:
            split_assignment = 'val'
        else:
            split_assignment = 'test'
        
        cube_files = [cube_files[file_index]]
        logger.info(f"ARRAY JOB MODE: Processing file {file_index}: {cube_files[0].name}")
        logger.info(f"Split assignment: {split_assignment}")

    all_patches: List[Dict] = []
    n_no_mask = 0
    n_no_centres = 0
    n_skipped_total = 0

    for cube_path in tqdm(cube_files, desc="Processing cubes"):
        # Locate corresponding mask file
        base = cube_path.name[: -len(cube_suffix)]   # strip suffix
        mask_path = cubes_dir / (base + mask_suffix)

        if not mask_path.exists():
            logger.warning(f"Mask not found for {cube_path.name}, skipping")
            n_no_mask += 1
            continue

        # Load volumes
        raw_volume  = np.squeeze(tifffile.imread(str(cube_path)))   # (D, H, W)
        mask_volume = np.squeeze(tifffile.imread(str(mask_path)))   # (D, H, W)

        if raw_volume.ndim != 3:
            logger.warning(f"{cube_path.name}: unexpected shape {raw_volume.shape}, skipping")
            continue

        # Extract centres from the mask
        centres = extract_centres_from_mask3d(mask_volume, min_vol, max_vol)

        if len(centres) == 0:
            logger.warning(f"No valid centres in {cube_path.name}, skipping")
            n_no_centres += 1
            continue

        # Extract 128³ patches
        patches, n_skipped = extract_patches3d(
            raw_volume, centres,
            patch_size=patch_size,
            stride=stride,
            z_start=z_start,
            min_cells=min_cells
        )
        n_skipped_total += n_skipped

        for patch in patches:
            patch['source_cube'] = cube_path.stem

        all_patches.extend(patches)

    # ── Summary ────────────────────────────────────────────────────────────────
    n_total = len(cube_files)
    logger.info("=" * 60)
    logger.info("PREPROCESSING SUMMARY (3D)")
    logger.info("=" * 60)
    logger.info(f"  Cubes found              : {n_total}")
    logger.info(f"  Skipped (no mask)        : {n_no_mask}")
    logger.info(f"  Skipped (no centres)     : {n_no_centres}")
    logger.info(f"  Patches kept             : {len(all_patches)}")
    logger.info(f"  Patches skipped (min_cells): {n_skipped_total}")
    logger.info("=" * 60)

    if not all_patches:
        logger.error("No patches extracted! Check data and config.")
        return

    # ── Split and save ─────────────────────────────────────────────────────────
    
    # Array job mode: save patches directly to assigned split folder
    if file_index is not None:
        split_dir = output_dir / split_assignment
        split_dir.mkdir(parents=True, exist_ok=True)
        
        prefix = f"patch3d_f{file_index:04d}"
        save_patch_dataset3d(all_patches, str(split_dir), prefix=prefix)
        logger.info(f"Saved {len(all_patches)} patches → {split_dir}")
        return
    
    # Normal mode: Create train/val/test splits
    splits = create_splits(
        len(all_patches),
        train_frac=config['splits']['train_fraction'],
        val_frac=config['splits']['val_fraction'],
        test_frac=config['splits']['test_fraction'],
        random_seed=config['splits']['random_seed']
    )

    # Save patches to train/val/test directories
    for split_name, indices in splits.items():
        split_patches = [all_patches[i] for i in indices]
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        save_patch_dataset3d(split_patches, str(split_dir), prefix="patch3d")
        logger.info(f"  {split_name}: {len(split_patches)} patches → {split_dir}")


if __name__ == "__main__":
    import sys
    cfg = sys.argv[1] if len(sys.argv) > 1 else "configs/frame1_3d.yaml"
    file_idx = int(sys.argv[2]) if len(sys.argv) > 2 else None
    build_dataset3d(cfg, file_index=file_idx)
