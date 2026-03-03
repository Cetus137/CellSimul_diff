"""
3D temporal pair dataset for two-frame 3D diffusion training.

Each sample provides:
    images      : V_{t+1} in [-1, 1],  shape (1, D, H, W)  — the **target** to generate
    conditioning: 3-channel tensor,     shape (3, D, H, W)
                    channels 0-1: C_{t+1} conditioning maps (heatmap, distance)
                    channel  2  : V_t    (previous frame, clean, in [-1, 1])

The Trainer consumes (images, conditioning) tuples without modification.

Augmentation is applied **synchronously** to both frames and their
conditioning maps so that spatial alignment is preserved.
"""

import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from preprocessing.generate_condition_maps3d import generate_conditioning_maps3d


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _list_patches(split_dir: Path) -> List[Tuple[int, int]]:
    """Return sorted (pair_idx, patch_idx) tuples from files named pair3d_pNNNNN_NN_vol_t0.npy."""
    patches = set()
    for p in split_dir.glob("pair3d_p*_*_vol_t0.npy"):
        tok = p.stem.split("_")  # ['pair3d', 'pNNNNN', 'NN', 'vol', 't0']
        if len(tok) >= 3:
            try:
                pair_idx = int(tok[1][1:])  # Strip 'p' prefix
                patch_idx = int(tok[2])
                patches.add((pair_idx, patch_idx))
            except ValueError:
                pass
    return sorted(patches)


def _apply_augmentation_3d(
    vol0: np.ndarray,
    vol1: np.ndarray,
    centres0: np.ndarray,
    centres1: np.ndarray,
    patch_d: int,
    patch_h: int,
    patch_w: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply the same random 3D flip / 90° rotation to both frames.

    Modifies volumes in-place via numpy ops and adjusts centre coordinates
    so that generate_conditioning_maps3d remains accurate.

    Note: z/D axis is NOT flipped to preserve the tissue depth gradient
          (signal attenuates with depth — flipping would confuse the model).

    Returns: (vol0, vol1, centres0, centres1) — all augmented.
    """
    # Random Y flip
    if random.random() < 0.5:
        vol0 = np.flip(vol0, axis=1).copy()
        vol1 = np.flip(vol1, axis=1).copy()
        if len(centres0):
            centres0 = centres0.copy()
            centres0[:, 1] = (patch_h - 1) - centres0[:, 1]
        if len(centres1):
            centres1 = centres1.copy()
            centres1[:, 1] = (patch_h - 1) - centres1[:, 1]

    # Random X flip
    if random.random() < 0.5:
        vol0 = np.flip(vol0, axis=2).copy()
        vol1 = np.flip(vol1, axis=2).copy()
        if len(centres0):
            centres0 = centres0.copy()
            centres0[:, 2] = (patch_w - 1) - centres0[:, 2]
        if len(centres1):
            centres1 = centres1.copy()
            centres1[:, 2] = (patch_w - 1) - centres1[:, 2]

    # Random 90° rotation in XY plane (axes 1,2) for cubic patches
    if patch_h == patch_w:
        k_xy = random.randint(0, 3)
        if k_xy > 0:
            vol0 = np.rot90(vol0, k=k_xy, axes=(1, 2)).copy()
            vol1 = np.rot90(vol1, k=k_xy, axes=(1, 2)).copy()
            # Rotate centres in YX: each 90° maps (y, x) -> (x, H-1-y)
            for _ in range(k_xy):
                if len(centres0):
                    centres0 = centres0.copy()
                    centres0[:, 1], centres0[:, 2] = (
                        centres0[:, 2].copy(),
                        (patch_h - 1) - centres0[:, 1].copy(),
                    )
                if len(centres1):
                    centres1 = centres1.copy()
                    centres1[:, 1], centres1[:, 2] = (
                        centres1[:, 2].copy(),
                        (patch_h - 1) - centres1[:, 1].copy(),
                    )

    return vol0, vol1, centres0, centres1


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

class TemporalPairDataset3D(Dataset):
    """
    Loads preprocessed 3D temporal pairs saved by build_temporal_dataset3d.py.

    Args:
        split_dir: Path to e.g. data_live_node1_3d/pairs/train/
        augment: Whether to apply random flip/rotation augmentation.
        heatmap_sigma: Passed to generate_conditioning_maps3d.
        distance_percentile: Passed to generate_conditioning_maps3d.
        overfit_n: If > 0, restrict dataset to first *overfit_n* pairs
                   (useful for the debug_overfit sanity check).
        active_channels: Dict of {channel_name: bool} controlling which of
                         the two geometry channels to generate (heatmap,
                         distance). None means both active.
    """

    def __init__(
        self,
        split_dir: str,
        augment: bool = True,
        heatmap_sigma: float = 3.0,
        distance_percentile: float = 95.0,
        overfit_n: int = 0,
        active_channels: Optional[dict] = None,
    ):
        self.split_dir = Path(split_dir)
        self.augment = augment
        self.heatmap_sigma = heatmap_sigma
        self.distance_percentile = distance_percentile
        self.active_channels = active_channels

        self.patches = _list_patches(self.split_dir)
        if not self.patches:
            raise ValueError(
                f"No pair3d_p*_*_vol_t0.npy files found in {self.split_dir}. "
                "Run preprocessing.build_temporal_dataset3d first."
            )

        if overfit_n > 0:
            self.patches = self.patches[:overfit_n]

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            images      : (1, D, H, W) float32 — V_{t+1} in [-1, 1]
            conditioning: (3, D, H, W) float32
                            [0:2] C_{t+1} maps in [0, 1] (heatmap, distance)
                            [2]   V_t        in [-1, 1]
        """
        pair_idx, patch_idx = self.patches[idx]
        pfx = self.split_dir / f"pair3d_p{pair_idx:05d}_{patch_idx:02d}"

        vol0     = np.load(f"{pfx}_vol_t0.npy")          # (D, H, W) float32 [-1,1]
        vol1     = np.load(f"{pfx}_vol_t1.npy")          # (D, H, W) float32 [-1,1]
        centres0 = np.load(f"{pfx}_centres_t0.npy")      # (N0, 3) float32 (z,y,x)
        centres1 = np.load(f"{pfx}_centres_t1.npy")      # (N1, 3) float32 (z,y,x)

        patch_d, patch_h, patch_w = vol0.shape

        # ---- Synchronized augmentation --------------------------------
        if self.augment:
            vol0, vol1, centres0, centres1 = _apply_augmentation_3d(
                vol0, vol1, centres0, centres1, patch_d, patch_h, patch_w
            )

        # ---- Generate conditioning maps for t+1 -----------------------
        cond_t1 = generate_conditioning_maps3d(
            centres1,
            volume_shape=(patch_d, patch_h, patch_w),
            heatmap_sigma=self.heatmap_sigma,
            distance_percentile=self.distance_percentile,
            active_channels=self.active_channels,
        )  # (n_geom, D, H, W) float32 in [0, 1]  — default n_geom=2

        # ---- Assemble 3-channel conditioning --------------------------
        #  [0:2] = C_{t+1} maps  (geometry of next frame: heatmap + distance) in [0, 1]
        #  [2]   = V_t           normalised to [0, 1] so all cond channels share the
        #                        same range — avoids conditioning range violation warnings
        #                        and gives the UNet a consistent input domain.
        vol0_01 = np.clip(vol0 * 0.5 + 0.5, 0.0, 1.0)  # [-1,1] → [0,1]
        vol0_ch = vol0_01[np.newaxis]  # (1, D, H, W)
        conditioning = np.concatenate(
            [cond_t1, vol0_ch], axis=0
        ).astype(np.float32)  # (3, D, H, W)

        # ---- Convert to tensors ----------------------------------------
        images = torch.from_numpy(vol1[np.newaxis])      # (1, D, H, W)
        conditioning = torch.from_numpy(conditioning)    # (3, D, H, W)

        return images, conditioning


# ------------------------------------------------------------------
# DataLoader factory (mirrors get_dataloader from centre_condition_dataset)
# ------------------------------------------------------------------

def get_temporal_dataloader_3d(
    split_dir,          # str | List[str]
    batch_size: int = 2,
    num_workers: int = 4,
    augment: bool = True,
    heatmap_sigma: float = 3.0,
    distance_percentile: float = 95.0,
    overfit_n: int = 0,
    shuffle: bool = True,
    pin_memory: bool = True,
    active_channels: Optional[dict] = None,
    p_uncond: float = 0.0,
) -> DataLoader:
    """
    Convenience wrapper returning a DataLoader for TemporalPairDataset3D.

    Args:
        split_dir           : Path to split directory (train / val / test).
        batch_size          : Batch size (default 2 for 3D).
        num_workers         : DataLoader worker count.
        augment             : Enable random flip/rotation (disable for val/test).
        heatmap_sigma       : Gaussian sigma for centre heatmap (voxels).
        distance_percentile : Percentile for distance map normalisation.
        overfit_n           : Restrict to first N pairs (debug_overfit mode).
        shuffle             : Shuffle each epoch (should be False for val/test).
        pin_memory          : Pin memory for GPU transfers.
        active_channels     : Dict controlling which geometry channels to generate.        p_uncond        : Probability of replacing the full conditioning tensor
                          with zeros for a given sample (CFG training dropout).
                          Set to 0.0 for val/test loaders.
    Returns:
        DataLoader yielding (images, conditioning) batches.
    """
    # Accept a single directory or a list of directories.
    # When multiple directories are given, datasets are concatenated.
    if isinstance(split_dir, (str, Path)):
        split_dirs = [split_dir]
    else:
        split_dirs = list(split_dir)

    if len(split_dirs) == 1:
        dataset = TemporalPairDataset3D(
            split_dir=split_dirs[0],
            augment=augment,
            heatmap_sigma=heatmap_sigma,
            distance_percentile=distance_percentile,
            overfit_n=overfit_n,
            active_channels=active_channels,
        )
    else:
        from torch.utils.data import ConcatDataset
        parts = [
            TemporalPairDataset3D(
                split_dir=d,
                augment=augment,
                heatmap_sigma=heatmap_sigma,
                distance_percentile=distance_percentile,
                overfit_n=overfit_n,
                active_channels=active_channels,
            )
            for d in split_dirs
        ]
        dataset = ConcatDataset(parts)
        import logging as _log
        _log.getLogger(__name__).info(
            "Combined %d datasets: %s patches total",
            len(parts), len(dataset),
        )

    # CFG conditioning dropout — wrap dataset so that with probability p_uncond
    # the entire conditioning tensor (heatmap + distance + V_t) is zeroed out.
    # This teaches the model what "unconditional" sampling looks like and is
    # required for classifier-free guidance at inference time.
    if p_uncond > 0.0:
        from datasets.centre_condition_dataset3d import ConditionalDropoutDataset3D
        dataset = ConditionalDropoutDataset3D(dataset, p_uncond=p_uncond)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(shuffle),  # drop last incomplete batch during training
    )
