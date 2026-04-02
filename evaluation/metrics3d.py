"""
Pixel-level 3D realism metrics.

All functions accept (D, H, W) float32 volumes in any intensity range
(normalisation is applied internally where needed).

Public API
----------
compute_snr_3d              — foreground SNR via Otsu threshold
compute_radial_power_3d     — radially-averaged 3D power spectrum
compare_power_spectra_3d    — mean ± std band comparison, L2 of log-power curves
compare_intensity_histograms — Wasserstein-1 + KL on voxel intensity distributions
compare_all_metrics         — runs all of the above, returns flat summary dict
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import wasserstein_distance
from skimage.filters import threshold_otsu

logger = logging.getLogger(__name__)


# ─── SNR ──────────────────────────────────────────────────────────────────────

def compute_snr_3d(volume: np.ndarray) -> float:
    """
    Signal-to-noise ratio of a single 3D volume.

    Internally normalised to [0, 1]; Otsu threshold separates foreground (signal)
    from background (noise floor).  Returns mean(foreground) / std(background).

    A higher value means a cleaner signal relative to background noise.
    Returns NaN if the background is flat or too small to estimate std.

    Args:
        volume: (D, H, W) float32, any range.

    Returns:
        SNR as a float, or NaN on degenerate inputs.
    """
    v = volume.astype(np.float32)
    v_range = v.max() - v.min()
    if v_range < 1e-8:
        return float("nan")
    v = (v - v.min()) / v_range   # → [0, 1]

    try:
        thr = threshold_otsu(v)
    except Exception:
        thr = 0.5

    fg = v[v >= thr]
    bg = v[v <  thr]

    if len(bg) < 10 or bg.std() < 1e-8:
        return float("nan")

    return float(fg.mean() / bg.std())


# ─── 3D Power spectrum ────────────────────────────────────────────────────────

def compute_radial_power_3d(
    volume: np.ndarray,
    n_bins: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Radially-averaged 3D power spectrum via real FFT.

    Frequency magnitude is computed as sqrt(fz² + fy² + fx²) and binned
    into `n_bins` equal-width bins from 0 to 0.5 (Nyquist).

    Args:
        volume: (D, H, W) float32, any range.
        n_bins: Number of radial bins.

    Returns:
        freq_centres: (n_bins,)  normalised spatial frequency in [0, 0.5]
        radial_power: (n_bins,)  mean spectral power per bin
    """
    D, H, W = volume.shape
    fft = np.fft.rfftn(volume.astype(np.float32))
    power = np.abs(fft) ** 2

    fz = np.fft.fftfreq(D)[:, np.newaxis, np.newaxis]
    fy = np.fft.fftfreq(H)[np.newaxis, :, np.newaxis]
    fx = np.fft.rfftfreq(W)[np.newaxis, np.newaxis, :]
    freq_mag = np.sqrt(fz ** 2 + fy ** 2 + fx ** 2)

    bins = np.linspace(0.0, 0.5, n_bins + 1)
    freq_centres = 0.5 * (bins[:-1] + bins[1:])
    radial_power = np.zeros(n_bins, dtype=np.float64)

    for i in range(n_bins):
        mask = (freq_mag >= bins[i]) & (freq_mag < bins[i + 1])
        if mask.any():
            radial_power[i] = float(power[mask].mean())

    return freq_centres, radial_power


def compare_power_spectra_3d(
    real_vols: List[np.ndarray],
    syn_vols: List[np.ndarray],
    save_path: str = None,
    n_bins: int = 64,
) -> Dict:
    """
    Compare radial power spectra between real and synthetic volume sets.

    Computes mean ± std bands across each set and returns the L2 distance
    between mean log-power curves (lower = better spectral match).

    Args:
        real_vols:  List of (D, H, W) float32 real volumes.
        syn_vols:   List of (D, H, W) float32 synthetic volumes.
        save_path:  If set, saves a comparison PNG to this path.
        n_bins:     Number of radial frequency bins.

    Returns:
        dict with keys:
            l2_log_distance   — L2 between mean log₁₀ power curves
            real_mean         — (n_bins,) mean power, real
            syn_mean          — (n_bins,) mean power, synthetic
    """
    freqs, _ = compute_radial_power_3d(real_vols[0], n_bins)

    def _stack_power(vols):
        return np.stack([compute_radial_power_3d(v, n_bins)[1] for v in vols], axis=0)

    real_p = _stack_power(real_vols)   # (N_r, n_bins)
    syn_p  = _stack_power(syn_vols)    # (N_s, n_bins)

    real_mean, real_std = real_p.mean(0), real_p.std(0)
    syn_mean,  syn_std  = syn_p.mean(0),  syn_p.std(0)

    eps = 1e-12
    l2 = float(np.linalg.norm(
        np.log10(real_mean + eps) - np.log10(syn_mean + eps)
    ))

    if save_path:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.fill_between(
            freqs,
            np.log10(np.maximum(real_mean - real_std, eps)),
            np.log10(real_mean + real_std + eps),
            alpha=0.25, color="#2196F3", label=None,
        )
        ax.plot(freqs, np.log10(real_mean + eps), color="#2196F3", lw=2, label="real")
        ax.fill_between(
            freqs,
            np.log10(np.maximum(syn_mean - syn_std, eps)),
            np.log10(syn_mean + syn_std + eps),
            alpha=0.25, color="#FF5722", label=None,
        )
        ax.plot(freqs, np.log10(syn_mean + eps), color="#FF5722", lw=2, label="synthetic")
        ax.set_xlabel("Normalised spatial frequency")
        ax.set_ylabel("log₁₀ power")
        ax.set_title(f"3D Radial Power Spectrum  (L2={l2:.4f})")
        ax.legend(framealpha=0.9)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved power spectrum plot → %s", save_path)

    return {
        "l2_log_distance": round(l2, 6),
        "real_mean": real_mean,
        "syn_mean":  syn_mean,
    }


# ─── Intensity histograms ─────────────────────────────────────────────────────

def compare_intensity_histograms(
    real_vols: List[np.ndarray],
    syn_vols: List[np.ndarray],
    save_path: str = None,
    n_bins: int = 64,
    max_voxels: int = 200_000,
) -> Dict:
    """
    Compare per-voxel intensity distributions via Wasserstein-1 and KL divergence.

    Voxels are pooled across all volumes and down-sampled to `max_voxels` for
    speed (uniformly random without replacement).

    Args:
        real_vols, syn_vols: Lists of (D, H, W) float32 volumes (any range).
        save_path:           If set, saves a comparison PNG.
        n_bins:              Histogram bins for KL and plotting.
        max_voxels:          Max voxels sampled per set for Wasserstein.

    Returns:
        dict with keys:
            wasserstein_1  — W1 distance between voxel distributions
            kl_divergence  — KL(real || synthetic) on histograms
    """
    def _pool_and_normalise(vols, n):
        flat = np.concatenate(
            [v.astype(np.float32).ravel() for v in vols]
        )
        # normalise to [0, 1] so comparisons are scale-invariant
        v_min, v_max = flat.min(), flat.max()
        flat = (flat - v_min) / (v_max - v_min + 1e-8)
        if len(flat) > n:
            flat = np.random.choice(flat, n, replace=False)
        return flat

    r_flat = _pool_and_normalise(real_vols, max_voxels)
    s_flat = _pool_and_normalise(syn_vols,  max_voxels)

    w1 = float(wasserstein_distance(r_flat, s_flat))

    # KL divergence from histograms
    lo, hi = min(r_flat.min(), s_flat.min()), max(r_flat.max(), s_flat.max())
    bin_edges = np.linspace(lo, hi, n_bins + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    r_hist, _ = np.histogram(r_flat, bins=bin_edges, density=True)
    s_hist, _ = np.histogram(s_flat, bins=bin_edges, density=True)
    eps = 1e-8
    kl = float(np.sum(r_hist * np.log((r_hist + eps) / (s_hist + eps))))

    # Also collect per-volume global stats
    r_stats = np.array([[v.mean(), v.std(), stats.skew(v.ravel()), stats.kurtosis(v.ravel())]
                         for v in real_vols])
    s_stats = np.array([[v.mean(), v.std(), stats.skew(v.ravel()), stats.kurtosis(v.ravel())]
                         for v in syn_vols])

    if save_path:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(bin_centres, r_hist, color="#2196F3", lw=1.5, label="real")
        ax.plot(bin_centres, s_hist, color="#FF5722", lw=1.5, label="synthetic")
        ax.set_xlabel("Normalised voxel intensity [0, 1]")
        ax.set_ylabel("Density")
        ax.set_title(f"Voxel intensity distribution  W1={w1:.4f}  KL={kl:.4f}")
        ax.legend(framealpha=0.9)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved intensity histogram → %s", save_path)

    return {
        "wasserstein_1":       round(w1, 6),
        "kl_divergence":       round(kl, 6),
        "real_mean_intensity": round(float(r_stats[:, 0].mean()), 4),
        "syn_mean_intensity":  round(float(s_stats[:, 0].mean()), 4),
        "real_std_intensity":  round(float(r_stats[:, 1].mean()), 4),
        "syn_std_intensity":   round(float(s_stats[:, 1].mean()), 4),
        "real_skewness":       round(float(r_stats[:, 2].mean()), 4),
        "syn_skewness":        round(float(s_stats[:, 2].mean()), 4),
        "real_kurtosis":       round(float(r_stats[:, 3].mean()), 4),
        "syn_kurtosis":        round(float(s_stats[:, 3].mean()), 4),
    }


# ─── Master comparison ────────────────────────────────────────────────────────

def compare_all_metrics(
    real_vols: List[np.ndarray],
    syn_vols: List[np.ndarray],
    out_dir: str = ".",
) -> Dict:
    """
    Run all pixel-level metrics and return a flat summary dict.

    Saves plots to `out_dir`:
        power_spectrum_3d.png
        intensity_histogram_3d.png

    Args:
        real_vols: List of (D, H, W) float32 real volumes.
        syn_vols:  List of (D, H, W) float32 synthetic volumes.
        out_dir:   Directory for output plots (created if absent).

    Returns:
        Flat dict of scalar metrics (serialisable to YAML).
    """
    import os
    os.makedirs(out_dir, exist_ok=True)

    logger.info("Computing SNR ...")
    snr_real = np.nanmean([compute_snr_3d(v) for v in real_vols])
    snr_syn  = np.nanmean([compute_snr_3d(v) for v in syn_vols])

    logger.info("Computing 3D power spectra (%d real, %d synthetic) ...",
                len(real_vols), len(syn_vols))
    ps = compare_power_spectra_3d(
        real_vols, syn_vols,
        save_path=str(Path(out_dir) / "power_spectrum_3d.png"),
    )

    logger.info("Computing intensity histograms ...")
    ih = compare_intensity_histograms(
        real_vols, syn_vols,
        save_path=str(Path(out_dir) / "intensity_histogram_3d.png"),
    )

    summary = {
        "n_real":                     len(real_vols),
        "n_synthetic":                len(syn_vols),
        "snr_real":                   round(float(snr_real), 4),
        "snr_synthetic":              round(float(snr_syn),  4),
        "snr_ratio_syn_over_real":    round(float(snr_syn / (snr_real + 1e-8)), 4),
        "power_spectrum_l2":          ps["l2_log_distance"],
        **ih,
    }

    logger.info("─" * 50)
    logger.info("Pixel-level metric summary:")
    for k, v in summary.items():
        if isinstance(v, float):
            logger.info("  %-38s = %.4f", k, v)
        else:
            logger.info("  %-38s = %s", k, v)
    logger.info("─" * 50)

    return summary
