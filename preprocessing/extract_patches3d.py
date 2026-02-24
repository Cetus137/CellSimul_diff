"""
3D patch extraction from 256x256x256 cube volumes.

Strategy:
- Crop a single 128-slice slab in Z (configurable z_start).
- Tile Y and X with a sliding window (default: stride=128 → 2×2 grid → 4 patches per cube).
- Filter patches by minimum cell count.

Each saved patch is (128, 128, 128) with centres in (z, y, x) format relative to the patch.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_patches3d(
    image: np.ndarray,
    centres: np.ndarray,
    patch_size: int = 128,
    stride: int = 128,
    z_start: int = 0,
    min_cells: int = 5
) -> Tuple[List[Dict], int]:
    """
    Extract 3D patches from a 256x256x256 volume.

    Z axis is NOT tiled: a single slab of depth `patch_size` is cropped
    at `z_start`. Y and X are tiled with `stride`.

    Args:
        image: Volume array of shape (D, H, W), any dtype.
        centres: Cell centre coordinates of shape (N, 3) in (z, y, x) order,
            in *global* volume coordinates.
        patch_size: Spatial size of each cubic patch (default 128).
        stride: Stride for Y and X sliding window (default 128 = no overlap).
        z_start: First z-slice of the slab to crop (default 0).
        min_cells: Minimum cells required to keep a patch (default 5).

    Returns:
        patches: List of dicts, each with keys:
            - 'image':           np.ndarray (patch_size, patch_size, patch_size)
            - 'centres':         np.ndarray (M, 3) in patch-relative (z, y, x)
            - 'global_position': (z_start, top, left) tuple
            - 'num_cells':       int
        n_skipped: Number of candidate patches dropped due to min_cells filter.
    """
    D, H, W = image.shape

    # ── Z crop ─────────────────────────────────────────────────────────────────
    z_end = z_start + patch_size
    if z_end > D:
        raise ValueError(
            f"z_start={z_start} + patch_size={patch_size} exceeds depth D={D}"
        )
    slab = image[z_start:z_end]  # (patch_size, H, W)

    # Filter centres to those inside the z slab
    if len(centres) > 0:
        z_mask = (centres[:, 0] >= z_start) & (centres[:, 0] < z_end)
        slab_centres = centres[z_mask].copy()
        slab_centres[:, 0] -= z_start  # make z relative to slab
    else:
        slab_centres = np.empty((0, 3), dtype=np.float32)

    # ── Y × X tiling ───────────────────────────────────────────────────────────
    patches: List[Dict] = []
    n_skipped = 0

    for top in range(0, H - patch_size + 1, stride):
        for left in range(0, W - patch_size + 1, stride):
            bottom = top + patch_size
            right = left + patch_size

            # Crop patch
            patch_image = slab[:, top:bottom, left:right]  # (patch_size, patch_size, patch_size)

            # Filter and re-origin centres
            if len(slab_centres) > 0:
                yx_mask = (
                    (slab_centres[:, 1] >= top) & (slab_centres[:, 1] < bottom) &
                    (slab_centres[:, 2] >= left) & (slab_centres[:, 2] < right)
                )
                patch_centres = slab_centres[yx_mask].copy()
                patch_centres[:, 1] -= top
                patch_centres[:, 2] -= left
            else:
                patch_centres = np.empty((0, 3), dtype=np.float32)

            if len(patch_centres) < min_cells:
                n_skipped += 1
                continue

            patches.append({
                'image': patch_image,
                'centres': patch_centres.astype(np.float32),
                'global_position': (z_start, top, left),
                'num_cells': len(patch_centres),
            })

    return patches, n_skipped


def save_patch_dataset3d(
    patches: List[Dict],
    output_dir: str,
    prefix: str = "patch3d"
) -> None:
    """
    Save extracted 3D patches to disk as .npy files.

    Saved files per patch:
        {prefix}_{i:05d}_image.npy    — (128, 128, 128) array
        {prefix}_{i:05d}_centres.npy  — (M, 3) array in (z, y, x) patch-relative coords
        {prefix}_{i:05d}_meta.npz     — global_position, num_cells

    Args:
        patches: List of dicts returned by extract_patches3d.
        output_dir: Directory to save files to (created if absent).
        prefix: Filename prefix (default 'patch3d').
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for i, patch in enumerate(patches):
        stem = f"{prefix}_{i:05d}"
        np.save(out / f"{stem}_image.npy", patch['image'])
        np.save(out / f"{stem}_centres.npy", patch['centres'])
        np.savez(
            out / f"{stem}_meta.npz",
            global_position=np.array(patch['global_position']),
            num_cells=np.array(patch['num_cells'])
        )


def load_patch_dataset3d(
    patches_dir: str,
    indices: Optional[List[int]] = None
) -> List[Dict]:
    """
    Load previously saved 3D patches from disk.

    Args:
        patches_dir: Directory containing saved .npy files.
        indices: Optional subset of patch indices to load.

    Returns:
        List of patch dicts with keys 'image', 'centres', 'global_position', 'num_cells'.
    """
    d = Path(patches_dir)
    image_files = sorted(d.glob("*_image.npy"))

    if indices is not None:
        image_files = [image_files[i] for i in indices]

    patches = []
    for img_file in image_files:
        base = img_file.stem.replace('_image', '')
        centres = np.load(d / f"{base}_centres.npy")
        meta = np.load(d / f"{base}_meta.npz")
        patches.append({
            'image': np.load(img_file),
            'centres': centres,
            'global_position': tuple(meta['global_position'].tolist()),
            'num_cells': int(meta['num_cells']),
        })

    return patches


if __name__ == "__main__":
    # Quick smoke test
    np.random.seed(0)
    vol = np.random.randint(0, 65535, (256, 256, 256), dtype=np.uint16)
    centres = np.random.rand(50, 3) * np.array([256, 256, 256])

    patches, n_skip = extract_patches3d(vol, centres, patch_size=128, stride=128,
                                         z_start=64, min_cells=1)
    print(f"Extracted {len(patches)} patches, skipped {n_skip}")
    for p in patches:
        print(f"  pos={p['global_position']}  cells={p['num_cells']}  shape={p['image'].shape}")
