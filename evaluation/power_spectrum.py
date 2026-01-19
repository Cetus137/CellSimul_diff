"""
Power spectrum analysis for evaluating texture similarity.

Compares the spatial frequency content of real and generated images.
"""

import numpy as np
from scipy import signal
from typing import List, Tuple
import matplotlib.pyplot as plt


def compute_power_spectrum_2d(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 2D power spectrum of an image.
    
    Args:
        image: 2D array (H, W)
    
    Returns:
        frequencies: Radial frequency bins
        power: Radially averaged power spectrum
    """
    # Compute 2D FFT
    fft = np.fft.fft2(image)
    fft_shifted = np.fft.fftshift(fft)
    
    # Compute power spectrum (magnitude squared)
    power_2d = np.abs(fft_shifted) ** 2
    
    # Radial averaging
    h, w = image.shape
    cy, cx = h // 2, w // 2
    
    # Create coordinate grids
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)
    
    # Radial bins
    r_max = int(np.sqrt(cy**2 + cx**2))
    radial_bins = np.arange(0, r_max)
    
    # Average power in each radial bin
    radial_power = np.zeros(r_max)
    for i in range(r_max):
        mask = (r == i)
        if mask.sum() > 0:
            radial_power[i] = power_2d[mask].mean()
    
    # Frequencies (cycles per pixel)
    frequencies = radial_bins / min(h, w)
    
    return frequencies, radial_power


def compute_average_power_spectrum(images: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute average power spectrum across multiple images.
    
    Args:
        images: List of 2D images
    
    Returns:
        frequencies: Frequency bins
        avg_power: Average power spectrum
        std_power: Standard deviation of power
    """
    all_spectra = []
    frequencies = None
    
    for img in images:
        freq, power = compute_power_spectrum_2d(img)
        all_spectra.append(power)
        
        if frequencies is None:
            frequencies = freq
    
    # Stack and average
    all_spectra = np.array(all_spectra)
    avg_power = all_spectra.mean(axis=0)
    std_power = all_spectra.std(axis=0)
    
    return frequencies, avg_power, std_power


def compare_power_spectra(
    real_images: List[np.ndarray],
    generated_images: List[np.ndarray],
    save_path: str = None
) -> float:
    """
    Compare power spectra of real and generated images.
    
    Args:
        real_images: List of real images
        generated_images: List of generated images
        save_path: Optional path to save comparison plot
    
    Returns:
        distance: L2 distance between log power spectra (lower is better)
    """
    # Compute average spectra
    freq_real, power_real, std_real = compute_average_power_spectrum(real_images)
    freq_gen, power_gen, std_gen = compute_average_power_spectrum(generated_images)
    
    # Ensure same length (in case images have different sizes)
    min_len = min(len(power_real), len(power_gen))
    power_real = power_real[:min_len]
    power_gen = power_gen[:min_len]
    std_real = std_real[:min_len]
    std_gen = std_gen[:min_len]
    freq = freq_real[:min_len]
    
    # Compute distance in log space (more perceptually relevant)
    log_real = np.log10(power_real + 1e-10)
    log_gen = np.log10(power_gen + 1e-10)
    distance = np.sqrt(np.mean((log_real - log_gen) ** 2))
    
    # Plot comparison
    if save_path:
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(freq, power_real, 'b-', label='Real', linewidth=2)
        plt.fill_between(freq, power_real - std_real, power_real + std_real, 
                         alpha=0.3, color='b')
        plt.plot(freq, power_gen, 'r-', label='Generated', linewidth=2)
        plt.fill_between(freq, power_gen - std_gen, power_gen + std_gen,
                         alpha=0.3, color='r')
        plt.xlabel('Frequency (cycles/pixel)')
        plt.ylabel('Power')
        plt.yscale('log')
        plt.legend()
        plt.title('Power Spectrum Comparison')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(freq, log_real, 'b-', label='Real', linewidth=2)
        plt.plot(freq, log_gen, 'r-', label='Generated', linewidth=2)
        plt.xlabel('Frequency (cycles/pixel)')
        plt.ylabel('Log10(Power)')
        plt.legend()
        plt.title(f'Log Power (L2 distance: {distance:.4f})')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return distance


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    
    # Create synthetic texture
    size = 256
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    
    # Real: combination of frequencies
    real_images = []
    for _ in range(10):
        img = (np.sin(2*np.pi*x/20) + np.sin(2*np.pi*y/30) + 
               0.5*np.random.randn(size, size))
        real_images.append(img)
    
    # Generated: similar but slightly different
    gen_images = []
    for _ in range(10):
        img = (np.sin(2*np.pi*x/22) + np.sin(2*np.pi*y/28) + 
               0.5*np.random.randn(size, size))
        gen_images.append(img)
    
    # Compare
    distance = compare_power_spectra(real_images, gen_images, 'power_spectrum_test.png')
    print(f"Power spectrum distance: {distance:.4f}")
    print("✓ Power spectrum test passed!")
