"""
Normalization utilities for consistent data range handling across training and visualization.

CRITICAL: Diffusion models are sensitive to input data range.
This module ensures consistency between:
- Training (typically [-1, 1] for images)
- Preprocessing (raw images in [0, 1] or [0, 255])
- Visualization (display in [0, 1])
- Conditioning (always [0, 1])
"""

import torch
import numpy as np
from typing import Union


def to_minus_one_one(x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    Convert images from [0, 1] to [-1, 1] range (for training).
    
    Args:
        x: Images in [0, 1] range
    
    Returns:
        Images in [-1, 1] range
    
    Note:
        Use this when loading raw data for training.
        Assumes input is already in [0, 1] (not [0, 255]).
    """
    return 2.0 * x - 1.0


def to_zero_one(x: Union[torch.Tensor, np.ndarray], 
                data_min: float = -1.0, 
                data_max: float = 1.0) -> Union[torch.Tensor, np.ndarray]:
    """
    Convert images from [data_min, data_max] to [0, 1] range (for visualization).
    
    Args:
        x: Images in [data_min, data_max] range
        data_min: Minimum value in input range (default: -1.0)
        data_max: Maximum value in input range (default: 1.0)
    
    Returns:
        Images in [0, 1] range
    
    Note:
        Use this for visualization when training uses [-1, 1].
    """
    return (x - data_min) / (data_max - data_min)


def normalize_raw_image(img: np.ndarray, 
                        percentile_low: float = 1.0,
                        percentile_high: float = 99.0) -> np.ndarray:
    """
    Normalize raw microscopy image to [0, 1] range.
    
    Args:
        img: Raw image (uint8, uint16, or float)
        percentile_low: Lower percentile for clipping (default: 1.0)
        percentile_high: Upper percentile for clipping (default: 99.0)
    
    Returns:
        Normalized image in [0, 1] as float32
    
    Note:
        For uint16 images, we use PERCENTILE NORMALIZATION instead of dividing by 65535.
        
        WHY PERCENTILES?
        - Microscopy images rarely use the full 16-bit range [0, 65535]
        - Most pixels cluster in a narrow band (e.g., [500, 8000])
        - Dividing by 65535 would compress signal into ~12% of [0,1]
        - This causes:
            (a) Loss of dynamic range for diffusion
            (b) Vanishing gradients in early layers
            (c) Poor discrimination between cell/background
        - Percentile normalization:
            (a) Stretches actual signal to full [0,1]
            (b) Clips outliers (hot pixels, noise)
            (c) Per-image adaptation to intensity distribution
        
        Example:
            Raw uint16: min=200, p1=450, p99=7800, max=12500
            After percentile norm: p1→0.0, p99→1.0, outliers clipped
            After /65535: entire signal compressed to [0.007, 0.191]
    """
    # Convert to float32 first
    img_float = img.astype(np.float32)
    
    if img.dtype == np.uint8:
        # For 8-bit, simple division is usually fine
        return img_float / 255.0
    
    elif img.dtype == np.uint16:
        # For 16-bit: PERCENTILE-BASED NORMALIZATION
        # Compute percentiles per image
        p_low = np.percentile(img_float, percentile_low)
        p_high = np.percentile(img_float, percentile_high)
        
        # Avoid division by zero
        if p_high - p_low < 1e-6:
            # Uniform image, map to middle gray
            return np.full_like(img_float, 0.5)
        
        # Clip to percentile range
        img_clipped = np.clip(img_float, p_low, p_high)
        
        # Scale to [0, 1]
        img_normalized = (img_clipped - p_low) / (p_high - p_low)
        
        return img_normalized
    
    elif img.dtype in [np.float32, np.float64]:
        # Assume already normalized, clip to [0, 1]
        return np.clip(img_float, 0.0, 1.0)
    
    else:
        raise ValueError(f"Unsupported image dtype: {img.dtype}")


def check_range(x: torch.Tensor, 
                expected_min: float, 
                expected_max: float, 
                name: str = "tensor",
                epsilon: float = 0.1) -> None:
    """
    Check if tensor values are within expected range and warn if not.
    
    Args:
        x: Tensor to check
        expected_min: Expected minimum value
        expected_max: Expected maximum value
        name: Name for logging
        epsilon: Tolerance for range violation
    
    Raises:
        Warning if values are outside expected range by more than epsilon
    """
    actual_min = x.min().item()
    actual_max = x.max().item()
    
    if actual_min < expected_min - epsilon or actual_max > expected_max + epsilon:
        import warnings
        warnings.warn(
            f"{name} range [{actual_min:.3f}, {actual_max:.3f}] is outside "
            f"expected [{expected_min:.3f}, {expected_max:.3f}]. "
            f"This may cause training instability."
        )


if __name__ == "__main__":
    # Test utilities
    import matplotlib.pyplot as plt
    
    # Test [0,1] -> [-1,1] -> [0,1] roundtrip
    x_01 = np.random.rand(10, 10).astype(np.float32)
    x_11 = to_minus_one_one(x_01)
    x_01_back = to_zero_one(x_11)
    
    assert np.allclose(x_01, x_01_back), "Roundtrip failed!"
    print("✓ Normalization roundtrip test passed")
    
    # Test range checking
    x = torch.randn(10, 10) * 0.5  # Should be in [-1, 1] range
    check_range(x, -1.0, 1.0, "test_tensor")
    print("✓ Range check test passed")
