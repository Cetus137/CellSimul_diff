"""
3-D Training Visualizer.

Shows central XY / XZ / YZ orthogonal slices of all relevant volumes
at each training epoch.  Mirrors TrainingVisualizer (visualizer.py) but
accepts 5-D tensors of shape (B, C, D, H, W).

Layout
------
  Rows  :  num_samples × 3  (three orthogonal views per sample)
  Cols  :  Image | Noisy | Heatmap | Distance | Pred.Noise | Denoised
            (geometry columns vary with n_geom_channels)

Slice conventions
-----------------
  XY view  : vol[D//2, :,    :]    central z-slice        → (H, W)
  XZ view  : vol[:,    H//2, :]    central y-slice        → (D, W)
  YZ view  : vol[:,    :,    W//2] central x-slice        → (D, H)
"""

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VIEW_LABELS = ["XY  (z=mid)", "XZ  (y=mid)", "YZ  (x=mid)"]
_GEOM_META   = [("Heatmap", "hot"), ("Distance", "viridis"), ("Boundary", "plasma")]


def _to_zero_one(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Linearly map [vmin, vmax] → [0, 1], then clip."""
    out = (arr - vmin) / (vmax - vmin + 1e-8)
    return np.clip(out, 0.0, 1.0)


def _central_slices(vol: np.ndarray) -> List[np.ndarray]:
    """
    Extract three orthogonal 2-D slices through the centre of a 3-D volume.

    Parameters
    ----------
    vol : ndarray, shape (D, H, W)

    Returns
    -------
    [xy, xz, yz]  each a 2-D ndarray
    """
    if vol.ndim != 3:
        raise ValueError(f"Expected 3-D volume (D,H,W), got shape {vol.shape}")
    D, H, W = vol.shape
    xy = vol[D // 2,  :,     :]       # (H, W)
    xz = vol[:,       H // 2, :]      # (D, W)
    yz = vol[:,       :,      W // 2] # (D, H)
    return [xy, xz, yz]


# ---------------------------------------------------------------------------
# Visualizer class
# ---------------------------------------------------------------------------

class TrainingVisualizer3D:
    """
    Visualizer for 3-D diffusion model training.

    Usage
    -----
    Identical public API to TrainingVisualizer; swap class name only.
    """

    def __init__(self, save_dir: str = "visualizations", dpi: int = 150):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

        self.train_losses: List[float]                  = []
        self.val_losses:   List[tuple]                  = []  # (step, loss) tuples
        self.epochs:       List[int]                    = []
        self.train_steps:  List[int]                    = []

    # ------------------------------------------------------------------
    # Metric tracking
    # ------------------------------------------------------------------

    def add_metrics(
        self,
        epoch:      int,
        train_loss: float,
        step:       int = 0,
    ):
        """Append per-epoch train metrics."""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.train_steps.append(step)

    def add_val_metric(self, step: int, val_loss: float):
        """
        Record a validation measurement at a given global step.

        Called every validate_every steps for sub-epoch resolution.
        """
        self.val_losses.append((step, val_loss))

    def get_history(self) -> dict:
        """Return loss history as a plain dict for checkpoint persistence."""
        return {
            'epochs':       list(self.epochs),
            'train_losses': list(self.train_losses),
            'train_steps':  list(self.train_steps),
            'val_losses':   list(self.val_losses),
        }

    def set_history(self, history: dict):
        """Restore loss history from a checkpoint (enables cross-run plots)."""
        self.epochs       = history.get('epochs',       [])
        self.train_losses = history.get('train_losses', [])
        self.train_steps  = history.get('train_steps',  [])
        self.val_losses   = history.get('val_losses',   [])

    def plot_loss_curves(self, save_name: str = "loss_curves.png"):
        """Plot and save train/val loss curves (x-axis = global step)."""
        if len(self.epochs) == 0 and len(self.val_losses) == 0:
            return

        fig, ax = plt.subplots(figsize=(8, 5))

        if self.train_steps:
            ax.plot(self.train_steps, self.train_losses, 'b-o',
                    label="Train", linewidth=2, markersize=4)
        else:
            ax.plot(self.epochs, self.train_losses, 'b-o',
                    label="Train", linewidth=2, markersize=4)

        if self.val_losses:
            vs, vl = zip(*self.val_losses)
            ax.plot(vs, vl, 'r-o', label="Val", linewidth=2, markersize=4)

        # Log scale when dynamic range > 10×
        all_losses = list(self.train_losses)
        if self.val_losses:
            all_losses += [l for _, l in self.val_losses]
        if all_losses:
            lo, hi = min(all_losses), max(all_losses)
            if lo > 0 and hi / lo > 10:
                ax.set_yscale("log")

        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss – 3-D model", fontweight="bold")
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=self.dpi, bbox_inches="tight")
        plt.close()

    # ------------------------------------------------------------------
    # Main per-epoch visualisation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def visualize_model_output(
        self,
        model,
        images:      torch.Tensor,    # (B, 1, D, H, W)  clean, in model data range
        conditioning: torch.Tensor,   # (B, C, D, H, W)  in [0, 1]
        epoch:       int,
        device,
        num_samples: int = 2,
        save_name:   Optional[str] = None,
        n_geom_channels: Optional[int] = None,
        geom_channel_names: Optional[List[str]] = None,
    ):
        """
        Forward a batch through the (noise-added) model and save an orthogonal-
        slice montage showing Image / Noisy / Conditioning / PredNoise / Denoised.

        Parameters
        ----------
        model : DDPM3D instance
        images : (B, 1, D, H, W) clean training images
        conditioning : (B, C, D, H, W) conditioning maps
        epoch : current epoch number
        device : torch device
        num_samples : how many batch entries to visualise (≤ B)
        save_name : optional filename; defaults to ``epoch_XXXX.png``
        n_geom_channels : number of geometry conditioning channels
        geom_channel_names : display names for each geometry channel
        """
        model.eval()

        # ------------------------------------------------------------------ #
        # 1. Prepare tensors                                                  #
        # ------------------------------------------------------------------ #
        B = min(num_samples, images.shape[0])
        images       = images[:B].to(device)
        conditioning = conditioning[:B].to(device)

        # Unwrap DataParallel if needed
        mdl = model.module if hasattr(model, "module") else model

        data_min: float = getattr(mdl, "data_min", -1.0)
        data_max: float = getattr(mdl, "data_max",  1.0)
        T: int          = mdl.timesteps

        # Sample an early timestep (low noise → better visual comparison)
        timesteps  = torch.randint(0, max(1, T // 4), (B,), device=device)
        noisy      = mdl.q_sample(images, timesteps)
        pred_noise = mdl.model(noisy, timesteps, conditioning)   # UNet forward
        denoised   = mdl.predict_start_from_noise(noisy, timesteps, pred_noise)

        # ------------------------------------------------------------------ #
        # 2. Move to CPU float numpy                                          #
        # ------------------------------------------------------------------ #
        imgs_np     = images.cpu().float().numpy()       # (B,1,D,H,W)
        noisy_np    = noisy.cpu().float().numpy()
        cond_np     = conditioning.cpu().float().numpy() # (B,C,D,H,W)
        pnoise_np   = pred_noise.cpu().float().numpy()
        denoised_np = denoised.cpu().float().numpy()
        t_np        = timesteps.cpu().numpy()

        # Normalise image-type arrays to [0, 1] for display
        imgs_d     = _to_zero_one(imgs_np,     data_min, data_max)
        noisy_d    = _to_zero_one(noisy_np,    data_min, data_max)
        denoised_d = _to_zero_one(denoised_np, data_min, data_max)

        # ------------------------------------------------------------------ #
        # 3. Conditioning channel layout                                      #
        # ------------------------------------------------------------------ #
        n_cond_total = cond_np.shape[1]
        n_geom = n_geom_channels if n_geom_channels is not None \
                                  else min(n_cond_total, len(_GEOM_META))

        if geom_channel_names is not None:
            cmaps_list = [cm for (_, cm) in _GEOM_META]
            geom_channels = list(
                zip(geom_channel_names, cmaps_list[:len(geom_channel_names)])
            )
        else:
            geom_channels = _GEOM_META[:n_geom]

        # Frame-2 mode: last conditioning channel is V_t (previous frame)
        is_frame2 = n_cond_total > n_geom

        # ------------------------------------------------------------------ #
        # 4. Build figure                                                     #
        # ------------------------------------------------------------------ #
        n_views = 3  # XY, XZ, YZ
        nrows   = B * n_views
        # cols: [V_t] | V_{t+1}/Image | Noisy | geom... | PredNoise | Denoised
        ncols   = (1 if is_frame2 else 0) + 2 + n_geom + 2
        cell_w  = 2.4

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(cell_w * ncols, cell_w * nrows),
            squeeze=False,
        )

        for i in range(B):
            for vi, vlabel in enumerate(_VIEW_LABELS):
                row = i * n_views + vi

                def slc(arr5d: np.ndarray, ch: int = 0) -> np.ndarray:
                    """Return 2-D view `vi` for sample `i`, channel `ch`."""
                    vol = arr5d[i, ch]              # (D, H, W)
                    return _central_slices(vol)[vi]

                col = 0

                # -- V_t (frame-2 only): previous frame -------------------
                if is_frame2:
                    # Last cond channel is V_t already in [0, 1] (rescaled in dataset)
                    vt_vol = cond_np[i, -1]          # (D, H, W)
                    vt_slice = _central_slices(vt_vol)[vi]
                    axes[row, col].imshow(vt_slice, cmap="gray", vmin=0, vmax=1)
                    axes[row, col].set_title("V_t (prev)", fontsize=8)
                    axes[row, col].axis("off")
                    col += 1

                # -- Target image ------------------------------------------
                img_title = "V_{t+1} (target)" if is_frame2 else "Image (target)"
                axes[row, col].imshow(slc(imgs_d),    cmap="gray", vmin=0, vmax=1)
                axes[row, col].set_title(img_title, fontsize=8)
                axes[row, col].axis("off")
                col += 1

                # -- Noisy image -------------------------------------------
                axes[row, col].imshow(slc(noisy_d),   cmap="gray", vmin=0, vmax=1)
                axes[row, col].set_title(f"Noisy  t={t_np[i]}", fontsize=8)
                axes[row, col].axis("off")
                col += 1

                # -- Geometry conditioning maps ----------------------------
                for ch_idx, (ch_name, ch_cmap) in enumerate(geom_channels):
                    axes[row, col].imshow(
                        slc(cond_np, ch=ch_idx),
                        cmap=ch_cmap, vmin=0, vmax=1,
                    )
                    axes[row, col].set_title(ch_name, fontsize=8)
                    axes[row, col].axis("off")
                    col += 1

                # -- Predicted noise ---------------------------------------
                axes[row, col].imshow(slc(pnoise_np), cmap="gray", vmin=-3, vmax=3)
                axes[row, col].set_title("Pred. Noise", fontsize=8)
                axes[row, col].axis("off")
                col += 1

                # -- Denoised ----------------------------------------------
                axes[row, col].imshow(slc(denoised_d), cmap="gray", vmin=0, vmax=1)
                axes[row, col].set_title("Denoised", fontsize=8)
                axes[row, col].axis("off")

                # -- Row label (view + sample index) -----------------------
                axes[row, 0].set_ylabel(
                    f"S{i}  {vlabel}", fontsize=8, labelpad=3, rotation=0,
                    ha="right", va="center",
                )
                axes[row, 0].yaxis.set_visible(True)

        plt.suptitle(
            f"3-D Model Outputs – Epoch {epoch}",
            fontsize=12, fontweight="bold", y=1.002,
        )
        plt.tight_layout()

        if save_name is None:
            save_name = f"epoch_{epoch:04d}.png"

        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        logger.info("Saved 3-D model output visualization to %s", save_path)
        model.train()

    # ------------------------------------------------------------------
    def clear_history(self):
        """Clear all stored metrics."""
        self.train_losses.clear()
        self.val_losses.clear()
        self.epochs.clear()
