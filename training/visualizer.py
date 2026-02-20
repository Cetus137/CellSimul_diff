"""
Visualization utilities for training monitoring.

Generates visual aids including:
- Loss curves
- Model outputs vs inputs
- Conditioning maps visualization
"""

import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging
import sys

# Import normalization utilities
sys.path.append(str(Path(__file__).parent.parent))
from utils.normalization import to_zero_one

logger = logging.getLogger(__name__)


class TrainingVisualizer:
    """
    Visualizer for training progress.
    
    Creates plots for:
    - Loss curves (train/val)
    - Sample model outputs with inputs
    """
    
    def __init__(
        self,
        save_dir: str = 'visualizations',
        dpi: int = 100
    ):
        """
        Args:
            save_dir: Directory to save visualizations
            dpi: DPI for saved figures
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        
        # Loss history
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        
    def add_metrics(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None
    ):
        """
        Add metrics for the current epoch.

        Args:
            epoch: Current epoch number
            train_loss: Training loss
            val_loss: Validation loss (optional)
        """
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append((epoch, val_loss))
    
    def plot_loss_curves(self, save_name: str = 'loss_curves.png'):
        """
        Plot training and validation loss curves.
        
        Args:
            save_name: Filename for saved plot
        """
        if len(self.epochs) == 0:
            logger.warning("No metrics to plot")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Plot train loss
        plt.plot(self.epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        
        # Plot val loss
        if len(self.val_losses) > 0:
            ve, vl = zip(*self.val_losses)
            plt.plot(ve, vl, 'r-', label='Val Loss', linewidth=2)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Progress', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Use log scale if losses vary significantly
        if len(self.train_losses) > 0:
            all_losses = list(self.train_losses)
            if self.val_losses: all_losses += [l for _, l in self.val_losses]
            if max(all_losses) / (min(all_losses) + 1e-10) > 10:
                plt.yscale('log')
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved loss curves to {save_path}")
    
    @torch.no_grad()
    def visualize_model_output(
        self,
        model,
        images: torch.Tensor,
        conditioning: torch.Tensor,
        epoch: int,
        device: str = 'cpu',
        num_samples: int = 4,
        save_name: Optional[str] = None,
        n_geom_channels: Optional[int] = None,
        geom_channel_names: Optional[list] = None,
    ):
        """
        Visualize model inputs and outputs.
        
        Shows:
        - Original image
        - Noisy image (at random timestep)
        - Conditioning maps (heatmap, distance, boundary)
        - Predicted noise
        - Denoised image
        
        Args:
            model: Trained diffusion model
            images: Batch of images [B, C, H, W]
            conditioning: Batch of conditioning maps [B, 3, H, W]
            epoch: Current epoch
            device: Device for computation
            num_samples: Number of samples to visualize
            save_name: Custom filename (defaults to epoch_X.png)
        """
        model.eval()
        
        # Unwrap model if it's wrapped in DataParallel or similar
        if hasattr(model, 'module'):
            diffusion_model = model.module
        else:
            diffusion_model = model
        
        # Limit to num_samples
        images = images[:num_samples].to(device)
        conditioning = conditioning[:num_samples].to(device)
        
        batch_size = images.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, diffusion_model.timesteps // 4, (batch_size,), device=device)  # Use early timesteps for visibility
        
        # Add noise to images
        noise = torch.randn_like(images)
        noisy_images = diffusion_model.q_sample(images, t, noise)
        
        # Predict noise
        predicted_noise = diffusion_model.predict_noise(noisy_images, t, conditioning)
        
        # Compute denoised image (x0 = (x_t - sqrt(1-alpha_bar) * noise) / sqrt(alpha_bar))
        sqrt_alpha_bar = diffusion_model.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha_bar = diffusion_model.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        denoised = (noisy_images - sqrt_one_minus_alpha_bar * predicted_noise) / sqrt_alpha_bar
        
        # Get data range for proper visualization
        data_min = getattr(diffusion_model, 'data_min', -1.0)
        data_max = getattr(diffusion_model, 'data_max', 1.0)
        
        # Convert to numpy for plotting
        images_np = images.cpu().numpy()
        noisy_np = noisy_images.cpu().numpy()
        conditioning_np = conditioning.cpu().numpy()
        predicted_noise_np = predicted_noise.cpu().numpy()
        denoised_np = denoised.cpu().numpy()
        timesteps_np = t.cpu().numpy()
        
        # CRITICAL: Convert images/noisy/denoised from training range to display range
        # Training uses [-1, 1], visualization needs [0, 1]
        # Formula: (x - data_min) / (data_max - data_min)
        # For [-1, 1] → [0, 1]: (x + 1) / 2
        images_display = to_zero_one(images_np, data_min, data_max)
        noisy_display = to_zero_one(noisy_np, data_min, data_max)
        denoised_display = to_zero_one(denoised_np, data_min, data_max)
        # Conditioning is already in [0, 1], no conversion needed
        
        # Detect frame-2 mode: last channel is I_t (previous frame image).
        # frame2 conditioning = [heatmap?, distance?, boundary?, I_t]
        # We identify it by checking if the model has condition_channels > n_geom_only,
        # but simplest heuristic: frame2 always has I_t as the LAST channel, stored
        # separately.  We infer it from shape vs known geom-channel names.
        _ALL_GEOM = [
            ('Heatmap',  'hot'),
            ('Distance', 'viridis'),
            ('Boundary', 'plasma'),
        ]
        total_cond = conditioning_np.shape[1]

        # Determine how many channels are geometry maps vs image (I_t).
        # If n_geom_channels is supplied (by frame-2 trainer), use it directly.
        # Otherwise fall back to heuristic: geometry ≤ 3, extras are I_t.
        if n_geom_channels is not None:
            n_geom = n_geom_channels
        else:
            n_geom = min(total_cond, len(_ALL_GEOM))
        is_frame2 = total_cond > n_geom  # has one or more image channels appended

        # Build per-channel (name, colormap) list for the active geometry channels
        if geom_channel_names is not None:
            cmaps = [cm for (_, cm) in _ALL_GEOM]
            GEOM_CHANNELS = list(zip(geom_channel_names, cmaps[:len(geom_channel_names)]))
        else:
            GEOM_CHANNELS = _ALL_GEOM[:n_geom]

        # ncols = [I_t] + image + noisy + geom_channels + pred_noise + denoised
        ncols = (1 if is_frame2 else 0) + 2 + n_geom + 2
        fig, axes = plt.subplots(batch_size, ncols, figsize=(2.5 * ncols, 2.5 * batch_size))
        if batch_size == 1:
            axes = axes[np.newaxis, :]

        for i in range(batch_size):
            col = 0

            # --- Column 0 (frame-2 only): I_t  previous frame --------
            if is_frame2:
                it_display = to_zero_one(conditioning_np[i, -1:], data_min, data_max)
                axes[i, col].imshow(it_display[0], cmap='gray', vmin=0, vmax=1)
                axes[i, col].set_title('I_t (prev)', fontsize=10)
                axes[i, col].axis('off')
                col += 1

            # --- Target image -----------------------------------------
            img_title = 'I_{t+1} (target)' if is_frame2 else 'Image (target)'
            axes[i, col].imshow(images_display[i, 0], cmap='gray', vmin=0, vmax=1)
            axes[i, col].set_title(img_title, fontsize=10)
            axes[i, col].axis('off')
            col += 1

            # --- Noisy ------------------------------------------------
            axes[i, col].imshow(noisy_display[i, 0], cmap='gray', vmin=0, vmax=1)
            axes[i, col].set_title(f'Noisy (t={timesteps_np[i]})', fontsize=10)
            axes[i, col].axis('off')
            col += 1

            # --- Geometry conditioning maps (however many are active) --
            for ch_idx, (ch_name, ch_cmap) in enumerate(GEOM_CHANNELS):
                axes[i, col].imshow(conditioning_np[i, ch_idx], cmap=ch_cmap, vmin=0, vmax=1)
                axes[i, col].set_title(ch_name, fontsize=10)
                axes[i, col].axis('off')
                col += 1

            # --- Predicted noise --------------------------------------
            axes[i, col].imshow(predicted_noise_np[i, 0], cmap='gray', vmin=-3, vmax=3)
            axes[i, col].set_title('Pred. Noise', fontsize=10)
            axes[i, col].axis('off')
            col += 1

            # --- Denoised ---------------------------------------------
            axes[i, col].imshow(denoised_display[i, 0], cmap='gray', vmin=0, vmax=1)
            axes[i, col].set_title('Denoised', fontsize=10)
            axes[i, col].axis('off')
        
        plt.suptitle(f'Model Outputs - Epoch {epoch}', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_name is None:
            save_name = f'epoch_{epoch:04d}.png'
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved model output visualization to {save_path}")
        
        model.train()
    
    def clear_history(self):
        """Clear all stored metrics."""
        self.train_losses.clear()
        self.val_losses.clear()
        self.epochs.clear()
