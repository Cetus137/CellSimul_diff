"""
Unified 3D dataset for autoregressive diffusion model training.

Combines single-frame patches (CentreConditionDataset3D) and temporal pairs
(TemporalPairDataset3D) into one dataset with consistent 2-channel conditioning:

    Channel 0: heatmap of target-frame centres        in [0, 1]
    Channel 1: V_{t-1} (previous frame, or zeros)     in [0, 1]

For single-frame items  channel 1 is all-zeros (no previous frame available).
For temporal pair items channel 1 is V_{t-1} re-ranged to [0,1], and is zeroed
independently with probability p_prev_drop to teach the model the frame-0 case.

NOTE: distance conditioning is intentionally excluded — only the heatmap is used.

Usage:
    from datasets.unified_dataset3d import get_unified_dataloader3d

    train_loader = get_unified_dataloader3d(
        patches_dirs=["data_live_node1_3d/", "data_live_node2_3d/"],
        pairs_dirs=["data_live_node1_3d/pairs/", "data_live_node2_3d/pairs/"],
        split='train',
        batch_size=2,
        p_uncond=0.1,
        pair_sample_weight=1.0,
        p_prev_drop=0.15,
    )
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

import sys
sys.path.append(str(Path(__file__).parent.parent))

from datasets.centre_condition_dataset3d import (
    CentreConditionDataset3D,
    ConditionalDropoutDataset3D,
)
from datasets.temporal_pair_dataset3d import TemporalPairDataset3D

logger = logging.getLogger(__name__)

# Default conditioning channels for the unified model (heatmap only).
# Override by passing active_channels to UnifiedFrame3DDataset / get_unified_dataloader3d.
_UNIFIED_ACTIVE_CHANNELS_DEFAULT = {"heatmap": True, "distance": False}


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class UnifiedFrame3DDataset(Dataset):
    """
    Joint dataset mixing single-frame patches and temporal pairs.

    Both item types return (image, conditioning) with:
        image        : (1, D, H, W) float32 in [-1, 1]
        conditioning : (2, D, H, W) float32
                          ch 0: heatmap of target-frame centres in [0, 1]
                          ch 1: previous frame V_{t-1} in [0, 1]  (zeros for single-frame)

    Args:
        patches_dirs       : Root directories for single-frame data
                             (subdirs train/ val/ test/ are appended using `split`).
        pairs_dirs         : Root pair directories (subdirs <split>/ appended).
        split              : 'train', 'val', or 'test'.
        pair_sample_weight : Relative sampling weight given to each pair item relative
                             to single-frame items (default 1.0).
        p_prev_drop        : Probability of zeroing channel 1 for pair items, teaching
                             the model how to handle frame-0 (no previous frame).
        augment            : Enable random flip/rotation augmentation.
        heatmap_sigma      : Gaussian sigma for centre heatmap (voxels).
        distance_percentile: Passed through (unused, kept for API consistency).
        overfit_n          : If > 0, restrict each sub-dataset to first N items.
    """

    def __init__(
        self,
        patches_dirs: List[str],
        pairs_dirs: List[str],
        split: str = "train",
        pair_sample_weight: float = 1.0,
        p_prev_drop: float = 0.15,
        augment: bool = True,
        heatmap_sigma: float = 3.0,
        distance_percentile: float = 95.0,
        overfit_n: int = 0,
        active_channels: Optional[dict] = None,
    ):
        self.split = split
        self.pair_sample_weight = pair_sample_weight
        self.p_prev_drop = p_prev_drop
        self.augment = augment and (split == "train")
        _active = active_channels if active_channels is not None else _UNIFIED_ACTIVE_CHANNELS_DEFAULT

        # ── Single-frame sub-datasets ──────────────────────────────────────────
        sf_parts = [
            CentreConditionDataset3D(
                patches_dir=d,
                split=split,
                heatmap_sigma=heatmap_sigma,
                normalize_images=True,
                augment=self.augment,
                active_channels=_active,
            )
            for d in patches_dirs
        ]
        # Optionally limit each part for debug/overfit
        if overfit_n > 0:
            for part in sf_parts:
                part.patch_files = part.patch_files[:overfit_n]

        self._sf_dataset: Dataset = (
            sf_parts[0] if len(sf_parts) == 1 else ConcatDataset(sf_parts)
        )
        self._n_sf = len(self._sf_dataset)

        # ── Temporal pair sub-datasets ─────────────────────────────────────────
        pair_parts = [
            TemporalPairDataset3D(
                split_dir=str(Path(d) / split),
                augment=self.augment,
                heatmap_sigma=heatmap_sigma,
                distance_percentile=distance_percentile,
                overfit_n=overfit_n,
                active_channels=_active,
            )
            for d in pairs_dirs
        ]
        self._pair_dataset: Dataset = (
            pair_parts[0] if len(pair_parts) == 1 else ConcatDataset(pair_parts)
        )
        self._n_pair = len(self._pair_dataset)

        logger.info(
            "[UnifiedFrame3DDataset] split=%s  single-frame=%d  pairs=%d  "
            "pair_sample_weight=%.2f  p_prev_drop=%.2f",
            split,
            self._n_sf,
            self._n_pair,
            pair_sample_weight,
            p_prev_drop,
        )

    # ── Length & indexing ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self._n_sf + self._n_pair

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx < self._n_sf:
            return self._get_single_frame(idx)
        else:
            return self._get_pair(idx - self._n_sf)

    # ── Internal fetchers ──────────────────────────────────────────────────────

    def _get_single_frame(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (image, conditioning) for a single-frame patch.
        The sub-dataset returns n_geom geometry channels; we append exactly
        1 zero channel for the (absent) previous frame, regardless of n_geom.
        """
        image, cond_geom = self._sf_dataset[idx]  # cond_geom: (n_geom, D, H, W)

        # Always append exactly 1 zero prev-frame channel (not zeros_like which
        # would mirror the full n_geom channels when n_geom > 1).
        zero_prev = torch.zeros(1, *cond_geom.shape[1:], dtype=cond_geom.dtype)
        conditioning = torch.cat([cond_geom, zero_prev], dim=0)  # (n_geom+1, D, H, W)

        return image, conditioning

    def _get_pair(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (image, conditioning) for a temporal pair.
        TemporalPairDataset3D returns (image, (n_geom+1, D, H, W)) where the
        last channel is V_{t-1}.  We optionally zero that last channel with
        probability p_prev_drop to teach the model the frame-0 case.
        """
        image, conditioning = self._pair_dataset[idx]  # conditioning: (n_geom+1, D, H, W)

        # Independent prev-frame dropout — prev-frame is always the LAST channel.
        if self.p_prev_drop > 0.0 and np.random.rand() < self.p_prev_drop:
            conditioning = conditioning.clone()
            conditioning[-1] = 0.0

        return image, conditioning

    # ── Sampling weights ───────────────────────────────────────────────────────

    def get_sample_weights(self) -> List[float]:
        """
        Returns a weight for every item in the dataset.
        Single-frame items get weight 1.0; pair items get pair_sample_weight.
        Pass to torch.utils.data.WeightedRandomSampler.
        """
        weights = [1.0] * self._n_sf + [self.pair_sample_weight] * self._n_pair
        return weights


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────────────────────────────────────

def get_unified_dataloader3d(
    patches_dirs: Union[str, List[str]],
    pairs_dirs: Union[str, List[str]],
    split: str,
    batch_size: int = 2,
    num_workers: int = 4,
    augment: bool = True,
    heatmap_sigma: float = 3.0,
    distance_percentile: float = 95.0,
    pair_sample_weight: float = 1.0,
    p_prev_drop: float = 0.15,
    p_uncond: float = 0.0,
    overfit_n: int = 0,
    pin_memory: bool = True,
    active_channels: Optional[dict] = None,
) -> DataLoader:
    """
    Create a DataLoader for the unified 3D autoregressive diffusion model.

    Uses WeightedRandomSampler to mix single-frame and pair items according
    to pair_sample_weight.  CFG conditioning dropout (p_uncond) wraps the
    entire dataset so both item types can be unconditionally sampled.

    Args:
        patches_dirs       : Root directories for single-frame data.
        pairs_dirs         : Root pair directories (train/val/test appended).
        split              : 'train', 'val', or 'test'.
        batch_size         : Batch size (default 2 for 128³ patches on A100).
        num_workers        : DataLoader worker count.
        augment            : Enable augmentation (auto-disabled for val/test).
        heatmap_sigma      : Gaussian sigma for centre heatmap (voxels).
        distance_percentile: Passed through to TemporalPairDataset3D.
        pair_sample_weight : Weight for pair items relative to single-frame items.
        p_prev_drop        : Prob. of zeroing prev-frame channel for pair items.
        p_uncond           : Prob. of zeroing all conditioning (CFG training).
        overfit_n          : If > 0 restrict each sub-dataset to first N items.
        pin_memory         : Pin memory for GPU transfers.

    Returns:
        DataLoader yielding (image, conditioning) batches with conditioning shape
        (B, 2, D, H, W).
    """
    if isinstance(patches_dirs, (str, Path)):
        patches_dirs = [patches_dirs]
    if isinstance(pairs_dirs, (str, Path)):
        pairs_dirs = [pairs_dirs]

    is_train = split == "train"

    dataset = UnifiedFrame3DDataset(
        patches_dirs=list(patches_dirs),
        pairs_dirs=list(pairs_dirs),
        split=split,
        pair_sample_weight=pair_sample_weight,
        p_prev_drop=p_prev_drop if is_train else 0.0,
        augment=augment,
        heatmap_sigma=heatmap_sigma,
        distance_percentile=distance_percentile,
        overfit_n=overfit_n,
        active_channels=active_channels,
    )

    # CFG dropout: zero full conditioning tensor with probability p_uncond.
    # Applied only during training; val/test loaders always use real conditioning.
    inner_dataset = dataset
    if p_uncond > 0.0 and is_train:
        dataset = ConditionalDropoutDataset3D(dataset, p_uncond=p_uncond)

    if is_train:
        # WeightedRandomSampler for mixing single-frame / pair items.
        weights = inner_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )
    else:
        # Use the same WeightedRandomSampler as train so that val loss is
        # computed on the same single-frame / pair mix, making the metric
        # directly comparable to (and not artificially lower than) train loss.
        weights = inner_dataset.get_sample_weights()
        val_sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
