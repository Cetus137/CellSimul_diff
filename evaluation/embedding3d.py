"""
Embedding-based realism evaluation utilities.

Public API
----------
build_joint_input   — (image_np, centres_np) → (image_tensor, cond_tensor)
extract_embeddings  — batch-encode a list of (image_np, centres_np) pairs
frechet_distance    — FD between two (N, D) embedding arrays
linear_probe        — balanced accuracy of logistic regression on embeddings
plot_umap           — coloured 2D UMAP scatter saved to file
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
import umap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

from preprocessing.generate_condition_maps3d import generate_conditioning_maps3d
from utils.normalization import normalize_raw_image, to_minus_one_one

logger = logging.getLogger(__name__)


# ─── Input builder ────────────────────────────────────────────────────────────

def build_joint_input(
    image_np: np.ndarray,
    centres_np: np.ndarray,
    heatmap_sigma: float = 3.0,
    distance_percentile: float = 95.0,
    active_channels: Optional[dict] = None,
    prev_frame: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build the encoder's input tensors from raw numpy data.

    Args:
        image_np:            (D, H, W) or (1, D, H, W) raw image (any range).
        centres_np:          (N, 3) float32 cell centres in (z, y, x) patch coords.
        heatmap_sigma:       Gaussian sigma for heatmap blobs (voxels).
        distance_percentile: Percentile for distance map normalisation.
        active_channels:     Dict controlling which conditioning channels are built
                             (e.g. {"heatmap": True, "distance": False}).  When None
                             defaults to heatmap+distance (both True) for back-compat.
        prev_frame:          If True, append a zeros channel for the prev-frame
                             slot (frame-0 case for unified models).

    Returns:
        image_t: (1, D, H, W) float32 tensor in [-1, 1]
        cond_t:  (C, D, H, W) float32 tensor matching active_channels (+1 if prev_frame)
    """
    if active_channels is None:
        active_channels = {"heatmap": True, "distance": True}
    if image_np.ndim == 4:
        image_np = image_np[0]

    volume_shape = image_np.shape   # (D, H, W)

    # Normalise image
    img = normalize_raw_image(image_np.astype(np.float32))  # → [0, 1]
    img = to_minus_one_one(img)                              # → [-1, 1]

    # Generate conditioning channels matching the model's training config
    cond = generate_conditioning_maps3d(
        centres_np,
        volume_shape,
        heatmap_sigma=heatmap_sigma,
        distance_percentile=distance_percentile,
        active_channels=active_channels,
    )  # (C, D, H, W) float32 in [0, 1]

    image_t = torch.from_numpy(img[np.newaxis]).float()     # (1, D, H, W)
    cond_t  = torch.from_numpy(cond).float()

    if prev_frame:
        # Unified model: append zeros for the prev-frame channel (frame-0 case)
        zeros = torch.zeros(1, *volume_shape, dtype=torch.float32)
        cond_t = torch.cat([cond_t, zeros], dim=0)

    return image_t, cond_t


# ─── Batch embedding extraction ────────────────────────────────────────────────

def extract_embeddings(
    encoder,
    data_tuples: List[Tuple[np.ndarray, np.ndarray]],
    device: str = "cpu",
    batch_size: int = 4,
    heatmap_sigma: float = 3.0,
    distance_percentile: float = 95.0,
    active_channels: Optional[dict] = None,
    prev_frame: bool = False,
) -> np.ndarray:
    """
    Encode a list of (image_np, centres_np) pairs through CellEncoder3D.

    Args:
        encoder:    CellEncoder3D instance (already on `device`).
        data_tuples: List of (image_np, centres_np) — raw numpy arrays.
        device:     Target device.
        batch_size: Volumes per forward pass (reduce if OOM).
        heatmap_sigma, distance_percentile: Passed to build_joint_input.
        prev_frame: If True, append a zeros prev-frame channel (unified models).

    Returns:
        (N, latent_dim) float32 numpy array, L2-normalised rows.
    """
    encoder.eval()
    all_z = []
    img_batch, cond_batch = [], []

    def _flush():
        with torch.no_grad():
            z = encoder.encode(
                torch.stack(img_batch).to(device),
                torch.stack(cond_batch).to(device),
            )
        all_z.append(z.cpu().numpy())
        img_batch.clear()
        cond_batch.clear()

    for i, (image_np, centres_np) in enumerate(data_tuples):
        img_t, cond_t = build_joint_input(
            image_np, centres_np, heatmap_sigma, distance_percentile,
            active_channels, prev_frame,
        )
        img_batch.append(img_t)
        cond_batch.append(cond_t)

        if len(img_batch) == batch_size:
            _flush()

    if img_batch:
        _flush()

    embeddings = np.concatenate(all_z, axis=0)
    logger.info("Extracted %d embeddings (dim=%d)", len(embeddings), embeddings.shape[1])
    return embeddings


# ─── Fréchet Distance ─────────────────────────────────────────────────────────

def frechet_distance(Z_real: np.ndarray, Z_syn: np.ndarray) -> float:
    """
    Fréchet Distance between two sets of embeddings.

    Both arrays should be (N, D) float32.  N should be ≥ D for a reliable
    covariance estimate; a warning is logged if N < 2*D.

    Returns:
        Scalar FD value (lower = more similar distributions).
    """
    if len(Z_real) < 2 * Z_real.shape[1]:
        logger.warning(
            "Real set has only %d samples for %d-dim embeddings — "
            "FD covariance estimate may be unreliable.  "
            "Generate more samples for a robust score.",
            len(Z_real), Z_real.shape[1],
        )
    if len(Z_syn) < 2 * Z_syn.shape[1]:
        logger.warning(
            "Synthetic set has only %d samples for %d-dim embeddings — "
            "consider generating more synthetic volumes.",
            len(Z_syn), Z_syn.shape[1],
        )

    mu_r = Z_real.mean(axis=0)
    mu_s = Z_syn.mean(axis=0)
    cov_r = np.cov(Z_real, rowvar=False)
    cov_s = np.cov(Z_syn,  rowvar=False)

    diff = mu_r - mu_s

    # Matrix square root of (cov_r @ cov_s)
    covmean, _ = sqrtm(cov_r @ cov_s, disp=False)
    if np.iscomplexobj(covmean):
        if np.abs(covmean.imag).max() > 1e-3:
            logger.warning("sqrtm produced large imaginary component — FD may be inaccurate")
        covmean = covmean.real

    fd = float(diff @ diff + np.trace(cov_r + cov_s - 2.0 * covmean))
    return fd


# ─── Linear probe ─────────────────────────────────────────────────────────────

def linear_probe(
    Z_real: np.ndarray,
    Z_syn: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> float:
    """
    Train a logistic regression classifier on pooled real + synthetic embeddings.

    Returns balanced accuracy on a held-out split:
        ~0.50 = indistinguishable from random (good generation)
        ~1.00 = perfectly separable (poor generation)
    """
    Z = np.concatenate([Z_real, Z_syn], axis=0)
    y = np.array([0] * len(Z_real) + [1] * len(Z_syn))

    X_tr, X_te, y_tr, y_te = train_test_split(
        Z, y, test_size=test_size, stratify=y, random_state=random_state
    )

    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=random_state)
    clf.fit(X_tr, y_tr)
    acc = balanced_accuracy_score(y_te, clf.predict(X_te))
    logger.info(
        "Linear probe balanced accuracy: %.3f  "
        "(0.50 = indistinguishable, 1.00 = perfectly separated)",
        acc,
    )
    return float(acc)


# ─── t-SNE visualisation ──────────────────────────────────────────────────────

def plot_umap(
    Z_real: np.ndarray,
    Z_syn: np.ndarray,
    save_path: str,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> None:
    """
    Compute 2D UMAP of pooled embeddings and save a coloured scatter plot.

    Unlike t-SNE, UMAP preserves global structure so cluster positions and
    inter-cluster distances are meaningful.
    Real points are shown in blue, synthetic in orange.
    """
    Z = np.concatenate([Z_real, Z_syn], axis=0)
    labels = np.array(["real"] * len(Z_real) + ["synthetic"] * len(Z_syn))

    # n_neighbors must be < n_samples
    n_neighbors = min(n_neighbors, len(Z) - 1)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        metric="euclidean",
    )
    Z2 = reducer.fit_transform(Z)

    fig, ax = plt.subplots(figsize=(7, 6))
    for lbl, colour, marker in [
        ("real",      "#2196F3", "o"),
        ("synthetic", "#FF5722", "^"),
    ]:
        mask = labels == lbl
        ax.scatter(
            Z2[mask, 0], Z2[mask, 1],
            c=colour, label=lbl,
            alpha=0.6, s=20, marker=marker, linewidths=0,
        )

    ax.legend(framealpha=0.9)
    ax.set_title(
        f"UMAP of encoder embeddings — real vs synthetic\n"
        f"(real n={len(Z_real)}, synthetic n={len(Z_syn)}, n_neighbors={n_neighbors})"
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved UMAP plot → %s", save_path)
