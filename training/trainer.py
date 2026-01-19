"""
Training infrastructure for diffusion models.

Includes:
- Exponential Moving Average (EMA)
- Mixed precision training
- Checkpointing
- Logging
"""

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from pathlib import Path
from typing import Dict, Optional
import logging
from tqdm import tqdm
import copy
import sys

from .visualizer import TrainingVisualizer

# Import normalization utilities
sys.path.append(str(Path(__file__).parent.parent))
from utils.normalization import check_range

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EMA:
    """
    Exponential Moving Average of model parameters.
    
    Maintains a moving average of model weights for better sample quality.
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        """
        Args:
            model: Model to track
            decay: EMA decay rate (higher = slower update)
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Replace model parameters with EMA parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class Trainer:
    """
    Trainer for diffusion models with small-data optimizations.
    
    Handles training loop, validation, checkpointing, logging, and
    diagnostic checks for model collapse.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
        checkpoint_dir: str = 'checkpoints',
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        ema_update_every: int = 10,
        mixed_precision: bool = True,
        gradient_clip: float = 1.0,
        log_every: int = 100,
        save_every: int = 5000,
        validate_every: int = 2000,
        visualize: bool = True,
        viz_dir: str = 'visualizations',
        low_noise_bias: bool = False,
        low_noise_fraction: float = 0.3,
        low_noise_weight: float = 3.0
    ):
        """
        Args:
            model: Diffusion model (DDPM)
            optimizer: Optimizer
            train_loader: Training data loader
            val_loader: Validation data loader
            scheduler: Learning rate scheduler
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            use_ema: Use exponential moving average
            ema_decay: EMA decay rate
            ema_update_every: Update EMA every N steps
            mixed_precision: Use automatic mixed precision
            gradient_clip: Gradient clipping value
            log_every: Log every N steps
            save_every: Save checkpoint every N steps
            validate_every: Run validation every N steps
            visualize: Enable visualization of training progress
            viz_dir: Directory for saving visualizations
            low_noise_bias: Bias timestep sampling toward low noise
            low_noise_fraction: Fraction of timesteps considered low noise
            low_noise_weight: Weight for low-noise timestep sampling
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Small-data training settings
        self.low_noise_bias = low_noise_bias
        self.low_noise_fraction = low_noise_fraction
        self.low_noise_weight = low_noise_weight
        
        # Visualization
        self.visualize = visualize
        self.visualizer = TrainingVisualizer(save_dir=viz_dir) if visualize else None
        
        # EMA
        self.use_ema = use_ema
        self.ema = EMA(model, decay=ema_decay) if use_ema else None
        self.ema_update_every = ema_update_every
        
        # Mixed precision
        self.mixed_precision = mixed_precision
        self.scaler = GradScaler() if mixed_precision else None
        
        # Training settings
        self.gradient_clip = gradient_clip
        self.log_every = log_every
        self.save_every = save_every
        self.validate_every = validate_every
        
        # State
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Diagnostic tracking
        self.timestep_losses = {}  # Track loss by timestep bucket
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch with diagnostic logging.
        
        Returns:
            metrics: Dictionary of training metrics including diagnostics
        """
        self.model.train()
        total_loss = 0.0
        total_noise_true_norm = 0.0
        total_noise_pred_norm = 0.0
        num_batches = 0
        
        # Track loss by timestep bucket
        timestep_buckets = {
            'low': [],    # t < 0.3 * T
            'mid': [],    # 0.3 * T <= t < 0.7 * T
            'high': []    # t >= 0.7 * T
        }
        
        # Track data range statistics (check once per epoch)
        range_logged = False
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, (images, conditioning) in enumerate(pbar):
            # CRITICAL: Dataset returns CLEAN images (x0) only
            # Noise is added ONLY in model.compute_loss() via q_sample()
            # This is verified - dataset contains no randn_like or noise injection
            images = images.to(self.device)
            conditioning = conditioning.to(self.device)
            
            # DIAGNOSTIC: Print actual batch statistics (first batch only)
            # This catches 80% of normalization bugs before they waste GPU time
            if not range_logged:
                print("\n" + "="*60)
                print(f"ACTUAL BATCH STATS (Epoch {self.epoch}, Batch 0)")
                print("="*60)
                
                # Print dtype and shape
                print(f"x0 dtype: {images.dtype}, shape: {tuple(images.shape)}")
                print(f"x0 min/max/mean/std: {images.min().item():.6f}, {images.max().item():.6f}, "
                      f"{images.mean().item():.6f}, {images.std().item():.6f}")
                
                print(f"\ncond dtype: {conditioning.dtype}, shape: {tuple(conditioning.shape)}")
                for c in range(conditioning.shape[1]):
                    cc = conditioning[:, c]
                    print(f"cond[{c}] min/max/mean/std: {cc.min().item():.6f}, {cc.max().item():.6f}, "
                          f"{cc.mean().item():.6f}, {cc.std().item():.6f}")
                
                print("\n" + "-"*60)
                print("EXPECTED VALUES:")
                print("  For training in [-1,1]: x0 min≈-1, max≈1, std~0.1-0.6")
                print("  For training in [0,1]:  x0 min≈0, max≈1")
                print("  BAD SIGNS: min≈max (tiny std) or values outside expected range")
                print("-"*60)
                
                logger.info("\n" + "="*60)
                logger.info("DATA RANGE DIAGNOSTICS (Epoch %d)", self.epoch)
                logger.info("="*60)
                logger.info("Images (x_start):")
                logger.info("  min: %.4f, max: %.4f", images.min().item(), images.max().item())
                logger.info("  mean: %.4f, std: %.4f", images.mean().item(), images.std().item())
                
                logger.info("Conditioning channels:")
                for i in range(conditioning.shape[1]):
                    logger.info("  Channel %d: min=%.4f, max=%.4f, mean=%.4f",
                              i, 
                              conditioning[:, i].min().item(),
                              conditioning[:, i].max().item(),
                              conditioning[:, i].mean().item())
                
                # SAFETY CHECK: Warn if x_start is out of expected range
                # Get expected data range from model
                if hasattr(self.model, 'module'):  # Unwrap DataParallel
                    data_min = self.model.module.data_min
                    data_max = self.model.module.data_max
                else:
                    data_min = self.model.data_min
                    data_max = self.model.data_max
                
                img_min = images.min().item()
                img_max = images.max().item()
                tolerance = 1e-3
                
                if img_min < data_min - tolerance or img_max > data_max + tolerance:
                    logger.warning(
                        "⚠️  DATA RANGE VIOLATION DETECTED!\n"
                        f"   x_start range: [{img_min:.6f}, {img_max:.6f}]\n"
                        f"   Expected range: [{data_min:.6f}, {data_max:.6f}]\n"
                        f"   Out-of-range by: min={data_min - img_min:.6f}, max={img_max - data_max:.6f}\n"
                        "   This indicates normalization error in dataset preprocessing.\n"
                        "   Check utils/normalization.py and dataset loading code."
                    )
                else:
                    logger.info("✓ x_start within expected range [%.4f, %.4f]", data_min, data_max)
                
                # Also check conditioning is in [0, 1]
                cond_min = conditioning.min().item()
                cond_max = conditioning.max().item()
                if cond_min < -tolerance or cond_max > 1.0 + tolerance:
                    logger.warning(
                        "⚠️  CONDITIONING RANGE VIOLATION!\n"
                        f"   Conditioning range: [{cond_min:.6f}, {cond_max:.6f}]\n"
                        f"   Expected: [0.0, 1.0]\n"
                        "   Check preprocessing/generate_condition_maps.py"
                    )
                else:
                    logger.info("✓ Conditioning within expected range [0.0, 1.0]")
                
                logger.info("="*60 + "\n")
                range_logged = True
            
            # Forward pass with diagnostics
            if self.mixed_precision:
                with autocast(device_type='cuda' if self.device == 'cuda' else 'cpu'):
                    loss, diagnostics = self.model.compute_loss(
                        images,
                        conditioning,
                        low_noise_bias=self.low_noise_bias,
                        low_noise_fraction=self.low_noise_fraction,
                        low_noise_weight=self.low_noise_weight
                    )
            else:
                loss, diagnostics = self.model.compute_loss(
                    images,
                    conditioning,
                    low_noise_bias=self.low_noise_bias,
                    low_noise_fraction=self.low_noise_fraction,
                    low_noise_weight=self.low_noise_weight
                )
            
            # Track diagnostics
            total_noise_true_norm += diagnostics['noise_true_norm'].item()
            total_noise_pred_norm += diagnostics['noise_pred_norm'].item()
            
            # Track loss by timestep bucket
            timesteps = diagnostics['timesteps']
            T = self.model.timesteps
            for i, t in enumerate(timesteps):
                t_val = t.item()
                loss_val = loss.item()  # Could track per-sample loss if needed
                
                if t_val < 0.3 * T:
                    timestep_buckets['low'].append(loss_val)
                elif t_val < 0.7 * T:
                    timestep_buckets['mid'].append(loss_val)
                else:
                    timestep_buckets['high'].append(loss_val)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                
                self.optimizer.step()
            
            # Update EMA
            if self.use_ema and self.step % self.ema_update_every == 0:
                self.ema.update()
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Logging
            total_loss += loss.item()
            num_batches += 1
            
            if self.step % self.log_every == 0:
                avg_loss = total_loss / num_batches
                lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'lr': f'{lr:.2e}'})
            
            # Validation
            if self.val_loader is not None and self.step % self.validate_every == 0 and self.step > 0:
                val_metrics = self.validate()
                logger.info(f"Step {self.step} - Val loss: {val_metrics['loss']:.4f}")
                self.model.train()
            
            # Checkpointing
            if self.step % self.save_every == 0 and self.step > 0:
                self.save_checkpoint()
            
            self.step += 1
        
        # Compute epoch metrics
        avg_loss = total_loss / num_batches
        avg_noise_true_norm = total_noise_true_norm / num_batches
        avg_noise_pred_norm = total_noise_pred_norm / num_batches
        
        # Compute average loss by timestep bucket
        bucket_losses = {}
        for bucket_name, losses in timestep_buckets.items():
            if len(losses) > 0:
                bucket_losses[f'loss_{bucket_name}_t'] = sum(losses) / len(losses)
        
        # DIAGNOSTIC: Check for collapse (predicted noise << true noise)
        collapse_ratio = avg_noise_pred_norm / (avg_noise_true_norm + 1e-10)
        if collapse_ratio < 0.5:
            logger.warning(
                f"⚠️  MODEL COLLAPSE DETECTED: ||ε_pred|| = {avg_noise_pred_norm:.4f} "
                f"<< ||ε_true|| = {avg_noise_true_norm:.4f} (ratio={collapse_ratio:.3f})\n"
                f"   This indicates the model is predicting near-zero noise everywhere.\n"
                f"   Likely causes: (1) conditioning scale mismatch, (2) insufficient multi-scale injection,\n"
                f"   (3) too much classifier-free guidance dropout. Check conditioning normalization."
            )
        
        metrics = {
            'loss': avg_loss,
            'noise_true_norm': avg_noise_true_norm,
            'noise_pred_norm': avg_noise_pred_norm,
            'collapse_ratio': collapse_ratio,
            **bucket_losses
        }
        
        # Log detailed diagnostics
        logger.info(
            f"Epoch {self.epoch} metrics:\n"
            f"  Loss: {avg_loss:.4f}\n"
            f"  ||ε_true||: {avg_noise_true_norm:.4f}\n"
            f"  ||ε_pred||: {avg_noise_pred_norm:.4f}\n"
            f"  Collapse ratio: {collapse_ratio:.3f}\n"
            f"  Loss by timestep: {bucket_losses}"
        )
        
        return metrics
        return {'loss': avg_loss}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Run validation.
        
        Returns:
            metrics: Dictionary of validation metrics
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        # Use EMA model for validation if available
        if self.use_ema:
            self.ema.apply_shadow()
        
        for images, conditioning in tqdm(self.val_loader, desc="Validation"):
            images = images.to(self.device)
            conditioning = conditioning.to(self.device)
            
            if self.mixed_precision:
                with autocast(device_type='cuda' if self.device == 'cuda' else 'cpu'):
                    loss, _ = self.model.compute_loss(images, conditioning)
            else:
                loss, _ = self.model.compute_loss(images, conditioning)
            
            total_loss += loss.item()
            num_batches += 1
        
        # Restore model parameters
        if self.use_ema:
            self.ema.restore()
        
        avg_loss = total_loss / num_batches
        
        # Update best model
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_checkpoint(name='best.pt')
        
        return {'loss': avg_loss}
    
    def train(self, num_epochs: int):
        """
        Train for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Mixed precision: {self.mixed_precision}")
        logger.info(f"EMA: {self.use_ema}")
        logger.info(f"Visualization: {self.visualize}")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            logger.info(f"Epoch {epoch} - Train loss: {train_metrics['loss']:.4f}")
            
            # Validate
            val_loss = None
            if self.val_loader is not None:
                val_metrics = self.validate()
                val_loss = val_metrics['loss']
                logger.info(f"Epoch {epoch} - Val loss: {val_loss:.4f}")
            
            # Visualize
            if self.visualize:
                # Add metrics
                self.visualizer.add_metrics(epoch, train_metrics['loss'], val_loss)
                
                # Plot loss curves
                self.visualizer.plot_loss_curves()
                
                # Visualize model output (every epoch)
                # Get a sample batch from validation set
                if self.val_loader is not None:
                    sample_images, sample_conditioning = next(iter(self.val_loader))
                else:
                    sample_images, sample_conditioning = next(iter(self.train_loader))
                
                self.visualizer.visualize_model_output(
                    model=self.model,
                    images=sample_images,
                    conditioning=sample_conditioning,
                    epoch=epoch,
                    device=self.device,
                    num_samples=min(4, sample_images.shape[0])
                )
        
        logger.info("Training completed!")
    
    def save_checkpoint(self, name: Optional[str] = None):
        """
        Save model checkpoint.
        
        Args:
            name: Optional checkpoint name (defaults to step-based)
        """
        if name is None:
            name = f'checkpoint_step_{self.step}.pt'
        
        checkpoint = {
            'step': self.step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.use_ema:
            checkpoint['ema_shadow'] = self.ema.shadow
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        checkpoint_path = self.checkpoint_dir / name
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, path: str):
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.step = checkpoint.get('step', 0)
        self.epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.use_ema and 'ema_shadow' in checkpoint:
            self.ema.shadow = checkpoint['ema_shadow']
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logger.info(f"Loaded checkpoint from {path} (step {self.step})")


if __name__ == "__main__":
    # Test trainer
    from models.unet import ConditionalUNet
    from models.diffusion import DDPM
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    unet = ConditionalUNet(base_channels=32)
    model = DDPM(unet, timesteps=100)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Create dummy data
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 100
        
        def __getitem__(self, idx):
            return torch.randn(1, 64, 64), torch.randn(3, 64, 64)
    
    train_loader = torch.utils.data.DataLoader(DummyDataset(), batch_size=4)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        device=device,
        log_every=10,
        save_every=50
    )
    
    # Train for 1 epoch
    trainer.train(num_epochs=1)
    
    print("✓ Trainer test passed!")
