"""
PyTorch Dataset for centre-conditioned microscopy image synthesis.

Loads patches and generates conditioning maps on-the-fly or from cache.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

import sys
sys.path.append(str(Path(__file__).parent.parent / 'preprocessing'))
from generate_condition_maps import generate_conditioning_maps

# Import normalization utilities
sys.path.append(str(Path(__file__).parent.parent))
from utils.normalization import normalize_raw_image, to_minus_one_one

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CentreConditionDataset(Dataset):
    """
    Dataset for training centre-conditioned diffusion models.
    
    Returns:
        image: Normalized image tensor of shape (1, H, W) in [-1, 1]
        conditioning: Conditioning maps tensor of shape (3, H, W)
            [0]: Centre heatmap
            [1]: Distance-to-nearest-centre map
            [2]: Boundary likelihood map
    
    Note:
        Conditioning maps are generated on-the-fly from stored centres.
        This allows for flexible experimentation with conditioning parameters.
    """
    
    def __init__(
        self,
        patches_dir: str,
        split: str = 'train',
        heatmap_sigma: float = 3.0,
        boundary_sigma: float = 2.0,
        normalize_images: bool = True,
        augment: bool = False,
        active_channels: Optional[dict] = None
    ):
        """
        Initialize dataset.
        
        Args:
            patches_dir: Root directory containing processed patches
            split: One of 'train', 'val', 'test'
            heatmap_sigma: Sigma for centre heatmap Gaussians
            boundary_sigma: Temperature for boundary soft-assignment
            normalize_images: Normalize images to [-1, 1]
            augment: Apply random augmentations (flips, rotations)
            active_channels: Dict of booleans controlling which conditioning
                channels to generate, e.g. {'heatmap': True, 'distance': True,
                'boundary': False}.  None = all three active (default).
        """
        self.patches_dir = Path(patches_dir) / split
        self.split = split
        self.heatmap_sigma = heatmap_sigma
        self.boundary_sigma = boundary_sigma
        self.normalize_images = normalize_images
        self.augment = augment and (split == 'train')  # Only augment training
        self.active_channels = active_channels  # None = use all three
        
        # Find all patches
        self.patch_files = sorted(self.patches_dir.glob("*_image.npy"))
        
        if len(self.patch_files) == 0:
            raise ValueError(f"No patches found in {self.patches_dir}")
        
        logger.info(f"Loaded {len(self.patch_files)} patches from {split} split")
    
    def __len__(self) -> int:
        return len(self.patch_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            image: Tensor of shape (1, H, W)
            conditioning: Tensor of shape (3, H, W)
        """
        # Load patch data
        image_file = self.patch_files[idx]
        base_name = image_file.stem.replace('_image', '')
        centres_file = self.patches_dir / f"{base_name}_centres.npy"
        
        # Load image and centres
        # CRITICAL: Do NOT convert image dtype here!
        # normalize_raw_image() needs to see the original dtype (uint16)
        # to apply correct normalization strategy
        image = np.load(image_file)  # Keep original dtype
        centres = np.load(centres_file).astype(np.float32)
        
        # Handle single-channel images
        if image.ndim == 2:
            image = image[np.newaxis, ...]  # (1, H, W)
        elif image.ndim == 3:
            # Assume (H, W, C), transpose to (C, H, W)
            image = image.transpose(2, 0, 1)
        
        # Normalize image to [-1, 1]
        if self.normalize_images:
            image = self._normalize_image(image)
        
        # Generate conditioning maps
        # Note: centres are in (y, x) format, image is (C, H, W)
        image_shape = (image.shape[1], image.shape[2])  # (H, W)
        
        conditioning = generate_conditioning_maps(
            centres,
            image_shape,
            heatmap_sigma=self.heatmap_sigma,
            boundary_sigma=self.boundary_sigma,
            boundary_method='entropy',  # Robust soft-assignment method
            distance_percentile=95.0,   # CRITICAL: Percentile normalization for distance map
            active_channels=self.active_channels
        )
        
        # Apply augmentations if enabled
        if self.augment:
            image, conditioning = self._augment(image, conditioning)
        
        # Convert to tensors
        image = torch.from_numpy(image).float()
        conditioning = torch.from_numpy(conditioning).float()
        
        return image, conditioning
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image to [-1, 1] range for diffusion training.
        
        NORMALIZATION PIPELINE FOR UINT16 MICROSCOPY IMAGES:
        
        Step 1: Raw uint16 → [0, 1] using PERCENTILE normalization
            - Computes 1st and 99th percentiles per image
            - Clips values to [p1, p99] to remove outliers
            - Scales to [0, 1]: (img - p1) / (p99 - p1)
            - WHY: Microscopy images rarely use full uint16 range
                   Dividing by 65535 would compress signal dramatically
        
        Step 2: [0, 1] → [-1, 1] for diffusion
            - Apply: x = 2*x - 1
            - WHY: DDPM models typically train on [-1, 1] range
                   This centers the data around zero
        
        Conditioning maps stay in [0, 1] range throughout.
        Only images are converted to [-1, 1].
        
        Args:
            image: Raw image array (uint8, uint16, or float)
        
        Returns:
            Normalized image in [-1, 1] as float32
        """
        # Step 1: Normalize raw image to [0, 1] based on dtype
        # For uint16: uses percentile-based normalization (see utils/normalization.py)
        image = normalize_raw_image(image)
        
        # Step 2: Convert [0, 1] → [-1, 1] for diffusion training
        image = to_minus_one_one(image)
        
        return image
    
    def _augment(
        self,
        image: np.ndarray,
        conditioning: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply random augmentations to image and conditioning.
        
        Augmentations:
            - Random horizontal flip
            - Random vertical flip
            - Random 90-degree rotation
        
        Note:
            These are geometric transformations that preserve the
            relationship between image and conditioning maps.
        """
        # Random horizontal flip
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=2).copy()
            conditioning = np.flip(conditioning, axis=2).copy()
        
        # Random vertical flip
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=1).copy()
            conditioning = np.flip(conditioning, axis=1).copy()
        
        # Random 90-degree rotation (0, 90, 180, 270 degrees)
        k = np.random.randint(0, 4)
        if k > 0:
            # Rotate in the (H, W) plane (axes 1, 2)
            image = np.rot90(image, k=k, axes=(1, 2)).copy()
            conditioning = np.rot90(conditioning, k=k, axes=(1, 2)).copy()
        
        return image, conditioning


class ConditionalDropoutDataset(Dataset):
    """
    Wrapper dataset that randomly drops conditioning for classifier-free guidance.
    
    During training, conditioning is replaced with zeros with probability p_uncond.
    This enables classifier-free guidance during sampling.
    """
    
    def __init__(
        self,
        base_dataset: CentreConditionDataset,
        p_uncond: float = 0.1
    ):
        """
        Args:
            base_dataset: Underlying dataset
            p_uncond: Probability of dropping conditioning
        """
        self.base_dataset = base_dataset
        self.p_uncond = p_uncond
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, conditioning = self.base_dataset[idx]
        
        # Randomly drop conditioning
        if np.random.rand() < self.p_uncond:
            # Replace with zeros (unconditional)
            conditioning = torch.zeros_like(conditioning)
        
        return image, conditioning


def get_dataloader(
    patches_dir: str,
    split: str,
    batch_size: int,
    num_workers: int = 4,
    heatmap_sigma: float = 3.0,
    boundary_sigma: float = 2.0,
    augment: bool = False,
    p_uncond: float = 0.0,
    shuffle: Optional[bool] = None,
    active_channels: Optional[dict] = None
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for the specified split.
    
    Args:
        patches_dir: Root directory containing processed patches
        split: One of 'train', 'val', 'test'
        batch_size: Batch size
        num_workers: Number of data loading workers
        heatmap_sigma: Sigma for centre heatmap
        boundary_sigma: Temperature for boundary map
        augment: Apply data augmentation
        p_uncond: Probability of dropping conditioning (for CFG)
        shuffle: Whether to shuffle data (defaults to True for train, False otherwise)
        active_channels: Dict of booleans for which conditioning channels to use.
            None = all three active (default).
    
    Returns:
        dataloader: PyTorch DataLoader
    """
    # Create base dataset
    dataset = CentreConditionDataset(
        patches_dir=patches_dir,
        split=split,
        heatmap_sigma=heatmap_sigma,
        boundary_sigma=boundary_sigma,
        normalize_images=True,
        augment=augment,
        active_channels=active_channels
    )
    
    # Wrap with conditional dropout if p_uncond > 0
    if p_uncond > 0:
        dataset = ConditionalDropoutDataset(dataset, p_uncond=p_uncond)
    
    # Default shuffle behavior
    if shuffle is None:
        shuffle = (split == 'train')
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')  # Drop last incomplete batch for training
    )
    
    return dataloader


if __name__ == "__main__":
    # Test dataset loading
    import matplotlib.pyplot as plt
    
    # Create dataset
    dataset = CentreConditionDataset(
        patches_dir="data/processed/patches",
        split="train",
        augment=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Load a sample
    image, conditioning = dataset[0]
    
    print(f"Image shape: {image.shape}")
    print(f"Conditioning shape: {conditioning.shape}")
    print(f"Image range: [{image.min():.3f}, {image.max():.3f}]")
    
    # Visualize
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(image[0], cmap='gray')
    axes[0].set_title('Image')
    axes[0].axis('off')
    
    axes[1].imshow(conditioning[0], cmap='hot')
    axes[1].set_title('Centre Heatmap')
    axes[1].axis('off')
    
    axes[2].imshow(conditioning[1], cmap='viridis')
    axes[2].set_title('Distance Map')
    axes[2].axis('off')
    
    axes[3].imshow(conditioning[2], cmap='plasma')
    axes[3].set_title('Boundary Map')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig('dataset_sample.png', dpi=150, bbox_inches='tight')
    print("Saved sample to dataset_sample.png")
