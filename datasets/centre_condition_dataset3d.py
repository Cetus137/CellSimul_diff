"""
PyTorch Dataset for 3D centre-conditioned diffusion model training.

Mirrors datasets/centre_condition_dataset.py but for 3D (128, 128, 128) patches
with (N, 3) centres in (z, y, x) order.

Returns:
    image:        (1, D, H, W) float32 tensor in [-1, 1]
    conditioning: (C, D, H, W) float32 tensor in [0, 1]
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

import sys
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.generate_condition_maps3d import generate_conditioning_maps3d
from utils.normalization import normalize_raw_image, to_minus_one_one

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CentreConditionDataset3D(Dataset):
    """
    Dataset for training the 3D centre-conditioned diffusion model.

    Loads saved (128, 128, 128) patches and (N, 3) centre files, then
    generates conditioning maps on-the-fly.

    Returns:
        image:        (1, D, H, W) float32 in [-1, 1]
        conditioning: (C, D, H, W) float32 in [0, 1]
            C=2 when both heatmap and distance are active (default).
    """

    def __init__(
        self,
        patches_dir: str,
        split: str = 'train',
        heatmap_sigma: float = 3.0,
        normalize_images: bool = True,
        augment: bool = False,
        active_channels: Optional[dict] = None
    ):
        """
        Args:
            patches_dir: Root directory containing split subdirectories.
            split: 'train', 'val', or 'test'.
            heatmap_sigma: Gaussian sigma for heatmap blobs (voxels).
            normalize_images: Normalize to [-1, 1] (default True).
            augment: Apply random flips / 90° Z-plane rotations (train only).
            active_channels: e.g. {'heatmap': True, 'distance': True}.
                None → both active.
        """
        self.patches_dir = Path(patches_dir) / split
        self.split = split
        self.heatmap_sigma = heatmap_sigma
        self.normalize_images = normalize_images
        self.augment = augment and (split == 'train')
        self.active_channels = active_channels  # None = use all available

        self.patch_files = sorted(self.patches_dir.glob("*_image.npy"))

        if not self.patch_files:
            raise ValueError(f"No 3D patches found in {self.patches_dir}")

        logger.info(f"[3D Dataset] {split}: {len(self.patch_files)} patches in {self.patches_dir}")

    def __len__(self) -> int:
        return len(self.patch_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_file = self.patch_files[idx]
        base_name = image_file.stem.replace('_image', '')
        centres_file = self.patches_dir / f"{base_name}_centres.npy"

        # Load raw patch (D, H, W) — keep original dtype for normalization
        image = np.load(image_file)    # (D, H, W)
        centres = np.load(centres_file).astype(np.float32)  # (N, 3)

        # Add channel dim: (1, D, H, W)
        image = image[np.newaxis, ...]

        if self.normalize_images:
            image = self._normalize_image(image)

        volume_shape = image.shape[1:]  # (D, H, W)

        conditioning = generate_conditioning_maps3d(
            centres,
            volume_shape,
            heatmap_sigma=self.heatmap_sigma,
            distance_percentile=95.0,
            active_channels=self.active_channels
        )  # (C, D, H, W)

        if self.augment:
            image, conditioning = self._augment(image, conditioning)

        image       = torch.from_numpy(np.ascontiguousarray(image)).float()
        conditioning = torch.from_numpy(np.ascontiguousarray(conditioning)).float()

        return image, conditioning

    # ── helpers ────────────────────────────────────────────────────────────────

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """(1, D, H, W) raw → (1, D, H, W) float32 in [-1, 1]."""
        image = normalize_raw_image(image)   # → [0, 1]
        image = to_minus_one_one(image)      # → [-1, 1]
        return image

    def _augment(
        self,
        image: np.ndarray,
        conditioning: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply random volume augmentations (geometric only).

        Augmentations act identically on image and conditioning:
          - Random flip along H axis (axis 2)
          - Random flip along W axis (axis 3)
          - Random k×90° rotation in the H-W plane (axes 2,3)
        Note: z/D axis is NOT flipped to preserve the tissue depth gradient
              (signal attenuates with depth — flipping would confuse the model).
        """
        def _flip_pair(ax):
            return (np.flip(image, axis=ax).copy(),
                    np.flip(conditioning, axis=ax).copy())

        if np.random.rand() > 0.5:
            image, conditioning = _flip_pair(2)   # y
        if np.random.rand() > 0.5:
            image, conditioning = _flip_pair(3)   # x

        # Random 90-degree rotation in the y-x plane
        k = np.random.randint(0, 4)
        if k > 0:
            image       = np.rot90(image,       k, axes=(2, 3)).copy()
            conditioning = np.rot90(conditioning, k, axes=(2, 3)).copy()

        return image, conditioning


class ConditionalDropoutDataset3D(Dataset):
    """
    Wraps CentreConditionDataset3D and randomly zeros conditioning tensors
    for classifier-free guidance training.

    Args:
        dataset: Underlying CentreConditionDataset3D.
        p_uncond: Probability of replacing conditioning with zeros.
    """

    def __init__(self, dataset: CentreConditionDataset3D, p_uncond: float = 0.1):
        self.dataset = dataset
        self.p_uncond = p_uncond

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, conditioning = self.dataset[idx]
        if np.random.rand() < self.p_uncond:
            conditioning = torch.zeros_like(conditioning)
        return image, conditioning


def get_dataloader3d(
    patches_dir,    # str | List[str]
    split: str,
    batch_size: int,
    num_workers: int = 4,
    heatmap_sigma: float = 3.0,
    augment: bool = False,
    p_uncond: float = 0.0,
    shuffle: Optional[bool] = None,
    active_channels: Optional[dict] = None
) -> DataLoader:
    """
    Create a DataLoader for the 3D centre-conditioned dataset.

    Args:
        patches_dir: Root directory with split subdirectories.
        split: 'train', 'val', or 'test'.
        batch_size: Batch size.
        num_workers: DataLoader worker count.
        heatmap_sigma: Gaussian sigma for heatmap blobs.
        augment: Enable training augmentations.
        p_uncond: CFG conditioning dropout probability.
        shuffle: Override shuffle behaviour (default: True for train).
        active_channels: Conditioning channel selection dict.

    Returns:
        DataLoader yielding (image, conditioning) batches.
    """
    if isinstance(patches_dir, (str, Path)):
        patches_dirs = [patches_dir]
    else:
        patches_dirs = list(patches_dir)

    if len(patches_dirs) == 1:
        ds = CentreConditionDataset3D(
            patches_dir=patches_dirs[0],
            split=split,
            heatmap_sigma=heatmap_sigma,
            normalize_images=True,
            augment=augment,
            active_channels=active_channels
        )
    else:
        from torch.utils.data import ConcatDataset
        parts = [
            CentreConditionDataset3D(
                patches_dir=d,
                split=split,
                heatmap_sigma=heatmap_sigma,
                normalize_images=True,
                augment=augment,
                active_channels=active_channels,
            )
            for d in patches_dirs
        ]
        ds = ConcatDataset(parts)
        logger.info(
            "Combined %d datasets for split '%s': %d patches total",
            len(parts), split, len(ds),
        )

    if p_uncond > 0.0:
        ds = ConditionalDropoutDataset3D(ds, p_uncond=p_uncond)

    if shuffle is None:
        shuffle = (split == 'train')

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )
