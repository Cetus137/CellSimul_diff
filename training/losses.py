"""
Loss functions for training diffusion models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionLoss(nn.Module):
    """
    Simple diffusion loss wrapper.
    
    The actual loss computation is handled by DDPM.compute_loss(),
    but this class can be extended with additional regularization terms.
    """
    
    def __init__(self, loss_type: str = 'l2'):
        super().__init__()
        self.loss_type = loss_type
    
    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss between predicted and target noise.
        
        Args:
            predicted: Predicted noise from model
            target: Ground truth noise
        
        Returns:
            loss: Scalar loss value
        """
        if self.loss_type == 'l1':
            return F.l1_loss(predicted, target)
        elif self.loss_type == 'l2':
            return F.mse_loss(predicted, target)
        elif self.loss_type == 'huber':
            return F.smooth_l1_loss(predicted, target)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


class WeightedDiffusionLoss(nn.Module):
    """
    Diffusion loss with timestep-dependent weighting.
    
    Some works suggest weighting losses differently at different timesteps,
    e.g., emphasizing later timesteps (less noisy) or earlier ones (more noisy).
    """
    
    def __init__(
        self,
        loss_type: str = 'l2',
        weighting: str = 'uniform'  # 'uniform', 'snr', 'truncated_snr'
    ):
        super().__init__()
        self.loss_type = loss_type
        self.weighting = weighting
    
    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        t: torch.Tensor,
        alphas_cumprod: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weighted loss.
        
        Args:
            predicted: Predicted noise
            target: Target noise
            t: Timesteps
            alphas_cumprod: Cumulative alphas for SNR calculation
        
        Returns:
            loss: Weighted loss
        """
        # Base loss
        if self.loss_type == 'l1':
            loss = F.l1_loss(predicted, target, reduction='none')
        elif self.loss_type == 'l2':
            loss = F.mse_loss(predicted, target, reduction='none')
        elif self.loss_type == 'huber':
            loss = F.smooth_l1_loss(predicted, target, reduction='none')
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Compute weights
        if self.weighting == 'uniform':
            weights = torch.ones_like(loss)
        elif self.weighting == 'snr':
            # SNR = alpha_t / (1 - alpha_t)
            alpha_t = alphas_cumprod.gather(-1, t)
            snr = alpha_t / (1.0 - alpha_t + 1e-8)
            weights = snr.view(-1, 1, 1, 1)
        elif self.weighting == 'truncated_snr':
            # Truncated SNR weighting (from Imagen paper)
            alpha_t = alphas_cumprod.gather(-1, t)
            snr = alpha_t / (1.0 - alpha_t + 1e-8)
            weights = torch.clamp(snr, max=5.0).view(-1, 1, 1, 1)
        else:
            raise ValueError(f"Unknown weighting: {self.weighting}")
        
        # Apply weights and reduce
        weighted_loss = loss * weights
        return weighted_loss.mean()


if __name__ == "__main__":
    # Test losses
    predicted = torch.randn(4, 1, 64, 64)
    target = torch.randn(4, 1, 64, 64)
    t = torch.randint(0, 1000, (4,))
    alphas_cumprod = torch.linspace(0.99, 0.01, 1000)
    
    # Test simple loss
    loss_fn = DiffusionLoss(loss_type='l2')
    loss = loss_fn(predicted, target)
    print(f"L2 loss: {loss.item():.4f}")
    
    # Test weighted loss
    weighted_loss_fn = WeightedDiffusionLoss(loss_type='l2', weighting='snr')
    weighted_loss = weighted_loss_fn(predicted, target, t, alphas_cumprod)
    print(f"Weighted loss: {weighted_loss.item():.4f}")
    
    print("✓ Loss tests passed!")
