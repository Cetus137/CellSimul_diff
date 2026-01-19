"""
Boundary profile analysis.

Analyzes intensity profiles across cell boundaries to assess
whether generated images have realistic cell-cell interfaces.
"""

import numpy as np
from scipy import ndimage
from typing import List, Tuple
import matplotlib.pyplot as plt


def extract_boundary_profiles(
    image: np.ndarray,
    centres: np.ndarray,
    profile_length: int = 50,
    num_samples_per_cell: int = 8
) -> List[np.ndarray]:
    """
    Extract intensity profiles across cell boundaries.
    
    Samples radial profiles from each cell centre outward,
    capturing the transition from cell interior to boundary to exterior.
    
    Args:
        image: 2D image (H, W)
        centres: Cell centres (N, 2) in (y, x) format
        profile_length: Length of each profile in pixels
        num_samples_per_cell: Number of radial samples per cell
    
    Returns:
        profiles: List of 1D intensity profiles
    """
    h, w = image.shape
    profiles = []
    
    for cy, cx in centres:
        # Sample radial profiles at different angles
        angles = np.linspace(0, 2*np.pi, num_samples_per_cell, endpoint=False)
        
        for angle in angles:
            # Create line coordinates
            dx = np.cos(angle)
            dy = np.sin(angle)
            
            # Sample points along the line
            t = np.linspace(0, profile_length, profile_length)
            xs = cx + dx * t
            ys = cy + dy * t
            
            # Check bounds
            valid = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
            
            if valid.sum() < profile_length // 2:
                continue  # Skip if too many points are out of bounds
            
            # Interpolate intensity values
            # Use map_coordinates for sub-pixel sampling
            coords = np.array([ys, xs])
            profile = ndimage.map_coordinates(image, coords, order=1, mode='nearest')
            
            profiles.append(profile)
    
    return profiles


def compute_average_profile(profiles: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute average and standard deviation of profiles.
    
    Args:
        profiles: List of 1D profiles
    
    Returns:
        mean_profile: Average profile
        std_profile: Standard deviation
    """
    if len(profiles) == 0:
        return np.array([]), np.array([])
    
    # Ensure all profiles have the same length
    min_len = min(len(p) for p in profiles)
    profiles_arr = np.array([p[:min_len] for p in profiles])
    
    mean_profile = profiles_arr.mean(axis=0)
    std_profile = profiles_arr.std(axis=0)
    
    return mean_profile, std_profile


def compare_boundary_profiles(
    real_images: List[np.ndarray],
    real_centres_list: List[np.ndarray],
    generated_images: List[np.ndarray],
    generated_centres_list: List[np.ndarray],
    save_path: str = None
) -> float:
    """
    Compare boundary profiles of real and generated images.
    
    Args:
        real_images: List of real images
        real_centres_list: List of centres for each real image
        generated_images: List of generated images
        generated_centres_list: List of centres for each generated image
        save_path: Optional path to save comparison plot
    
    Returns:
        distance: L2 distance between average profiles
    """
    # Extract all profiles
    real_profiles = []
    for img, centres in zip(real_images, real_centres_list):
        profiles = extract_boundary_profiles(img, centres)
        real_profiles.extend(profiles)
    
    gen_profiles = []
    for img, centres in zip(generated_images, generated_centres_list):
        profiles = extract_boundary_profiles(img, centres)
        gen_profiles.extend(profiles)
    
    # Compute average profiles
    real_mean, real_std = compute_average_profile(real_profiles)
    gen_mean, gen_std = compute_average_profile(gen_profiles)
    
    # Ensure same length
    min_len = min(len(real_mean), len(gen_mean))
    real_mean = real_mean[:min_len]
    gen_mean = gen_mean[:min_len]
    real_std = real_std[:min_len]
    gen_std = gen_std[:min_len]
    
    # Compute distance
    distance = np.sqrt(np.mean((real_mean - gen_mean) ** 2))
    
    # Plot comparison
    if save_path:
        plt.figure(figsize=(10, 5))
        
        x = np.arange(len(real_mean))
        
        plt.plot(x, real_mean, 'b-', label='Real', linewidth=2)
        plt.fill_between(x, real_mean - real_std, real_mean + real_std,
                         alpha=0.3, color='b')
        
        plt.plot(x, gen_mean, 'r-', label='Generated', linewidth=2)
        plt.fill_between(x, gen_mean - gen_std, gen_mean + gen_std,
                         alpha=0.3, color='r')
        
        plt.xlabel('Distance from centre (pixels)')
        plt.ylabel('Normalized intensity')
        plt.legend()
        plt.title(f'Boundary Profile Comparison (L2: {distance:.4f})')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return distance


def analyze_boundary_sharpness(profiles: List[np.ndarray]) -> float:
    """
    Measure boundary sharpness from profiles.
    
    Computes the average gradient magnitude at the steepest point,
    which corresponds to the cell boundary.
    
    Args:
        profiles: List of intensity profiles
    
    Returns:
        sharpness: Average maximum gradient magnitude
    """
    sharpnesses = []
    
    for profile in profiles:
        # Compute gradient
        gradient = np.gradient(profile)
        
        # Find maximum absolute gradient (boundary location)
        max_grad = np.abs(gradient).max()
        sharpnesses.append(max_grad)
    
    return np.mean(sharpnesses)


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    
    size = 256
    
    # Create synthetic cell-like image
    def create_cell_image(centres):
        from scipy.spatial import distance_matrix
        
        y, x = np.ogrid[:size, :size]
        coords = np.stack([y.ravel(), x.ravel()], axis=1)
        
        # Distance to nearest centre
        dists = distance_matrix(coords, centres)
        min_dists = dists.min(axis=1).reshape(size, size)
        
        # Intensity decreases with distance
        image = 1.0 / (1.0 + min_dists / 20)
        
        # Add noise
        image += 0.1 * np.random.randn(size, size)
        
        return image
    
    # Create test images
    real_centres = [np.random.rand(15, 2) * size for _ in range(5)]
    real_images = [create_cell_image(c) for c in real_centres]
    
    gen_centres = [np.random.rand(15, 2) * size for _ in range(5)]
    gen_images = [create_cell_image(c) for c in gen_centres]
    
    # Compare
    distance = compare_boundary_profiles(
        real_images, real_centres,
        gen_images, gen_centres,
        save_path='boundary_profiles_test.png'
    )
    
    print(f"Boundary profile distance: {distance:.4f}")
    
    # Test sharpness
    profiles = extract_boundary_profiles(real_images[0], real_centres[0])
    sharpness = analyze_boundary_sharpness(profiles)
    print(f"Boundary sharpness: {sharpness:.4f}")
    
    print("✓ Boundary profile test passed!")
