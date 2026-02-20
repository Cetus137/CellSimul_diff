"""
Temporal pair dataset for two-frame diffusion training.

Each sample provides:
    images      : I_{t+1} in [-1, 1],  shape (1, H, W)  — the **target** to generate
    conditioning: 4-channel tensor,     shape (4, H, W)
                    channels 0-2: C_{t+1} conditioning maps (heatmap, distance, boundary)
                    channel  3  : I_t    (previous frame, clean, in [-1, 1])

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

from preprocessing.generate_condition_maps import generate_conditioning_maps


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _list_pairs(split_dir: Path) -> List[int]:
    """Return sorted pair indices from files named pair_NNNNN_img_t0.npy."""
    indices = set()
    for p in split_dir.glob("pair_*_img_t0.npy"):
        tok = p.stem.split("_")  # ['pair', 'NNNNN', 'img', 't0']
        if len(tok) >= 2:
            try:
                indices.add(int(tok[1]))
            except ValueError:
                pass
    return sorted(indices)


def _apply_augmentation(
    img0: np.ndarray,
    img1: np.ndarray,
    centres0: np.ndarray,
    centres1: np.ndarray,
    patch_h: int,
    patch_w: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply the same random flip / 90° rotation to both frames.

    Modifies images in-place via numpy ops and adjusts centre coordinates
    so that generate_conditioning_maps remains accurate.

    Returns: (img0, img1, centres0, centres1) — all augmented.
    """
    # Random horizontal flip
    if random.random() < 0.5:
        img0 = np.fliplr(img0).copy()
        img1 = np.fliplr(img1).copy()
        if len(centres0):
            centres0 = centres0.copy()
            centres0[:, 1] = (patch_w - 1) - centres0[:, 1]
        if len(centres1):
            centres1 = centres1.copy()
            centres1[:, 1] = (patch_w - 1) - centres1[:, 1]

    # Random vertical flip
    if random.random() < 0.5:
        img0 = np.flipud(img0).copy()
        img1 = np.flipud(img1).copy()
        if len(centres0):
            centres0 = centres0.copy()
            centres0[:, 0] = (patch_h - 1) - centres0[:, 0]
        if len(centres1):
            centres1 = centres1.copy()
            centres1[:, 0] = (patch_h - 1) - centres1[:, 0]

    # Random 90° rotation (0, 90, 180, 270)
    k = random.randint(0, 3)
    if k > 0:
        img0 = np.rot90(img0, k=k).copy()
        img1 = np.rot90(img1, k=k).copy()
        # Rotate centres: each 90° maps (y, x) -> (x, H-1-y) for a square patch
        # For non-square, we skip centre rotation (conservative)
        if patch_h == patch_w:
            for _ in range(k):
                if len(centres0):
                    centres0 = centres0.copy()
                    centres0[:, 0], centres0[:, 1] = (
                        centres0[:, 1].copy(),
                        (patch_h - 1) - centres0[:, 0].copy(),
                    )
                if len(centres1):
                    centres1 = centres1.copy()
                    centres1[:, 0], centres1[:, 1] = (
                        centres1[:, 1].copy(),
                        (patch_h - 1) - centres1[:, 0].copy(),
                    )

    return img0, img1, centres0, centres1


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

class TemporalPairDataset(Dataset):
    """
    Loads preprocessed temporal pairs saved by build_temporal_dataset.py.

    Args:
        split_dir: Path to e.g. data_multiz/processed/pairs/train/
        augment: Whether to apply random flip/rotation augmentation.
        heatmap_sigma: Passed to generate_conditioning_maps.
        boundary_sigma: Passed to generate_conditioning_maps.
        overfit_n: If > 0, restrict dataset to first *overfit_n* pairs
                   (useful for the debug_overfit sanity check).
        active_channels: Dict of {channel_name: bool} controlling which of
                         the three geometry channels to generate (heatmap,
                         distance, boundary).  None means all three active.
    """

    def __init__(
        self,
        split_dir: str,
        augment: bool = True,
        heatmap_sigma: float = 3.0,
        boundary_sigma: float = 2.0,
        overfit_n: int = 0,
        active_channels: Optional[dict] = None,
    ):
        self.split_dir = Path(split_dir)
        self.augment = augment
        self.heatmap_sigma = heatmap_sigma
        self.boundary_sigma = boundary_sigma
        self.active_channels = active_channels

        self.indices = _list_pairs(self.split_dir)
        if not self.indices:
            raise ValueError(
                f"No pair_*_img_t0.npy files found in {self.split_dir}. "
                "Run preprocessing.build_temporal_dataset first."
            )

        if overfit_n > 0:
            self.indices = self.indices[:overfit_n]

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            images      : (1, H, W) float32 — I_{t+1} in [-1, 1]
            conditioning: (4, H, W) float32
                            [0:3] C_{t+1} maps in [0, 1]
                            [3]   I_t        in [-1, 1]
        """
        n = self.indices[idx]
        pfx = self.split_dir / f"pair_{n:05d}"

        img0     = np.load(f"{pfx}_img_t0.npy").squeeze()   # (H, W) float32 [-1,1]
        img1     = np.load(f"{pfx}_img_t1.npy").squeeze()   # (H, W) float32 [-1,1]
        centres0 = np.load(f"{pfx}_centres_t0.npy")         # (N0, 2) float32
        centres1 = np.load(f"{pfx}_centres_t1.npy")         # (N1, 2) float32

        patch_h, patch_w = img0.shape

        # ---- Synchronized augmentation --------------------------------
        if self.augment:
            img0, img1, centres0, centres1 = _apply_augmentation(
                img0, img1, centres0, centres1, patch_h, patch_w
            )

        # ---- Generate conditioning maps for t+1 -----------------------
        cond_t1 = generate_conditioning_maps(
            centres1,
            image_shape=(patch_h, patch_w),
            heatmap_sigma=self.heatmap_sigma,
            boundary_sigma=self.boundary_sigma,
            active_channels=self.active_channels,
        )  # (n_geom, H, W) float32 in [0, 1]

        # ---- Assemble 4-channel conditioning --------------------------
        #  [0:3] = C_{t+1} maps  (geometry of next frame)
        #  [3]   = I_t           (appearance anchor from previous frame)
        img0_ch = img0[np.newaxis]  # (1, H, W)
        conditioning = np.concatenate(
            [cond_t1, img0_ch], axis=0
        ).astype(np.float32)  # (4, H, W)

        # ---- Convert to tensors ----------------------------------------
        images = torch.from_numpy(img1[np.newaxis])     # (1, H, W)
        conditioning = torch.from_numpy(conditioning)    # (4, H, W)

        return images, conditioning


# ------------------------------------------------------------------
# DataLoader factory (mirrors get_dataloader from centre_condition_dataset)
# ------------------------------------------------------------------

def get_temporal_dataloader(
    split_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    augment: bool = True,
    heatmap_sigma: float = 3.0,
    boundary_sigma: float = 2.0,
    overfit_n: int = 0,
    shuffle: bool = True,
    pin_memory: bool = True,
    active_channels: Optional[dict] = None,
) -> DataLoader:
    """
    Convenience wrapper returning a DataLoader for TemporalPairDataset.

    Args:
        split_dir      : Path to split directory (train / val / test).
        batch_size     : Batch size.
        num_workers    : DataLoader worker count.
        augment        : Enable random flip/rotation (disable for val/test).
        overfit_n      : Restrict to first N pairs (debug_overfit mode).
        shuffle        : Shuffle each epoch (should be False for val/test).
        pin_memory     : Pin memory for GPU transfers.
        active_channels: Dict controlling which geometry channels to generate.

    Returns:
        DataLoader yielding (images, conditioning) batches.
    """
    dataset = TemporalPairDataset(
        split_dir=split_dir,
        augment=augment,
        heatmap_sigma=heatmap_sigma,
        boundary_sigma=boundary_sigma,
        overfit_n=overfit_n,
        active_channels=active_channels,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(shuffle),  # drop last incomplete batch during training
    )
