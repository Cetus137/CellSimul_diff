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
            self.val_losses.append(val_loss)
    
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
        
        # Plot val loss if available
        if len(self.val_losses) > 0:
            plt.plot(self.epochs, self.val_losses, 'r-', label='Val Loss', linewidth=2)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Progress', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Use log scale if losses vary significantly
        if len(self.train_losses) > 0:
            max_loss = max(self.train_losses)
            min_loss = min(self.train_losses)
            if max_loss / min_loss > 10:
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
        save_name: Optional[str] = None
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
        
        # Limit to num_samples
        images = images[:num_samples].to(device)
        conditioning = conditioning[:num_samples].to(device)
        
        batch_size = images.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, model.timesteps // 4, (batch_size,), device=device)  # Use early timesteps for visibility
        
        # Add noise to images
        noise = torch.randn_like(images)
        noisy_images = model.q_sample(images, t, noise)
        
        # Predict noise
        predicted_noise = model.predict_noise(noisy_images, t, conditioning)
        
        # Compute denoised image (x0 = (x_t - sqrt(1-alpha_bar) * noise) / sqrt(alpha_bar))
        sqrt_alpha_bar = model.sqrt_alpha_bar[t][:, None, None, None]
        sqrt_one_minus_alpha_bar = model.sqrt_one_minus_alpha_bar[t][:, None, None, None]
        denoised = (noisy_images - sqrt_one_minus_alpha_bar * predicted_noise) / sqrt_alpha_bar
        
        # Convert to numpy for plotting
        images_np = images.cpu().numpy()
        noisy_np = noisy_images.cpu().numpy()
        conditioning_np = conditioning.cpu().numpy()
        predicted_noise_np = predicted_noise.cpu().numpy()
        denoised_np = denoised.cpu().numpy()
        timesteps_np = t.cpu().numpy()
        
        # Create figure
        fig, axes = plt.subplots(batch_size, 7, figsize=(18, 2.5 * batch_size))
        if batch_size == 1:
            axes = axes[np.newaxis, :]
        
        for i in range(batch_size):
            # Original image
            axes[i, 0].imshow(images_np[i, 0], cmap='gray')
            axes[i, 0].set_title(f'Original', fontsize=10)
            axes[i, 0].axis('off')
            
            # Noisy image
            axes[i, 1].imshow(noisy_np[i, 0], cmap='gray')
            axes[i, 1].set_title(f'Noisy (t={timesteps_np[i]})', fontsize=10)
            axes[i, 1].axis('off')
            
            # Conditioning - heatmap
            axes[i, 2].imshow(conditioning_np[i, 0], cmap='hot')
            axes[i, 2].set_title('Heatmap', fontsize=10)
            axes[i, 2].axis('off')
            
            # Conditioning - distance
            axes[i, 3].imshow(conditioning_np[i, 1], cmap='viridis')
            axes[i, 3].set_title('Distance', fontsize=10)
            axes[i, 3].axis('off')
            
            # Conditioning - boundary
            axes[i, 4].imshow(conditioning_np[i, 2], cmap='plasma')
            axes[i, 4].set_title('Boundary', fontsize=10)
            axes[i, 4].axis('off')
            
            # Predicted noise
            axes[i, 5].imshow(predicted_noise_np[i, 0], cmap='gray', vmin=-3, vmax=3)
            axes[i, 5].set_title('Pred. Noise', fontsize=10)
            axes[i, 5].axis('off')
            
            # Denoised
            axes[i, 6].imshow(np.clip(denoised_np[i, 0], 0, 1), cmap='gray')
            axes[i, 6].set_title('Denoised', fontsize=10)
            axes[i, 6].axis('off')
        
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
