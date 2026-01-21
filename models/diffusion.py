"""
DDPM (Denoising Diffusion Probabilistic Model) implementation.

Implements the forward diffusion process (adding noise) and
reverse diffusion process (denoising) for image generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict
from tqdm import tqdm


def linear_beta_schedule(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
    """Linear noise schedule."""
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def quadratic_beta_schedule(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
    """Quadratic noise schedule."""
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


class DDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model.
    
    Implements:
    - Forward process: q(x_t | x_0) - adding noise
    - Reverse process: p(x_{t-1} | x_t) - denoising
    - Training objective: predict noise ε
    """
    
    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        beta_schedule: str = 'linear',
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        prediction_type: str = 'epsilon',
        loss_type: str = 'l2',
        data_min: float = -1.0,
        data_max: float = 1.0
    ):
        """
        Args:
            model: Noise prediction model (e.g., U-Net)
            timesteps: Number of diffusion steps
            beta_schedule: Type of noise schedule ('linear', 'cosine', 'quadratic')
            beta_start: Starting beta value
            beta_end: Ending beta value
            prediction_type: What the model predicts ('epsilon' for noise)
            loss_type: Loss function ('l1', 'l2', 'huber')
            data_min: Minimum value of input data range (default: -1.0)
            data_max: Maximum value of input data range (default: 1.0)
        
        Note on data_min/data_max:
            These parameters define the expected range of input data.
            - Images are normalized to [data_min, data_max] during preprocessing
            - Default [-1, 1] is standard for diffusion models
            - All clamping operations use these bounds to ensure stability
            - Visualization must convert back from [data_min, data_max] to [0, 1]
            - CRITICAL: Mismatched ranges cause blank outputs or numerical instability
        """
        super().__init__()
        
        self.model = model
        self.timesteps = timesteps
        self.prediction_type = prediction_type
        self.loss_type = loss_type
        self.data_min = data_min
        self.data_max = data_max
        
        # Create noise schedule
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == 'quadratic':
            betas = quadratic_beta_schedule(timesteps, beta_start, beta_end)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Pre-compute diffusion constants
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Register buffers (moved to device automatically)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # Calculations for diffusion q(x_t | x_0)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped',
                           torch.log(torch.clamp(posterior_variance, min=1e-20)))
        
        # Calculations for p(x_{t-1} | x_t)
        self.register_buffer('posterior_mean_coef1',
                           betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                           (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))
    
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward diffusion process: add noise to x_0 to get x_t.
        
        q(x_t | x_0) = N(x_t; sqrt(α_t) * x_0, (1 - α_t) * I)
        
        Args:
            x_start: Clean images of shape (B, C, H, W)
            t: Timesteps of shape (B,)
            noise: Optional pre-sampled noise
        
        Returns:
            x_t: Noisy images at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Extract coefficients for timesteps
        sqrt_alpha_prod = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha_prod = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        # Apply noise: x_t = sqrt(α_t) * x_0 + sqrt(1 - α_t) * ε
        return sqrt_alpha_prod * x_start + sqrt_one_minus_alpha_prod * noise
    
    def predict_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        conditioning: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict noise in x_t using the model.
        
        Args:
            x_t: Noisy images of shape (B, C, H, W)
            t: Timesteps of shape (B,)
            conditioning: Conditioning maps of shape (B, condition_channels, H, W)
        
        Returns:
            predicted_noise: Predicted noise ε of shape (B, C, H, W)
        """
        return self.model(x_t, t, conditioning)
    
    def predict_start_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict x_0 from x_t and predicted noise.
        
        x_0 = (x_t - sqrt(1 - α_t) * ε) / sqrt(α_t)
        
        Args:
            x_t: Noisy images
            t: Timesteps
            noise: Predicted noise
        
        Returns:
            x_0: Predicted clean images
        """
        sqrt_alpha_prod = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alpha_prod = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        
        return (x_t - sqrt_one_minus_alpha_prod * noise) / sqrt_alpha_prod
    
    def p_mean_variance(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        conditioning: torch.Tensor,
        clip_denoised: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute mean and variance for reverse process p(x_{t-1} | x_t).
        
        Args:
            x_t: Current noisy images
            t: Current timesteps
            conditioning: Conditioning maps
            clip_denoised: Clip predicted x_0 to [-1, 1]
        
        Returns:
            mean: Predicted mean of shape (B, C, H, W)
            variance: Variance of shape (B, C, H, W)
        """
        # Predict noise
        predicted_noise = self.predict_noise(x_t, t, conditioning)
        
        # Predict x_0
        x_start = self.predict_start_from_noise(x_t, t, predicted_noise)
        
        if clip_denoised:
            x_start = torch.clamp(x_start, self.data_min, self.data_max)
        
        # Compute posterior mean
        mean_coef1 = self._extract(self.posterior_mean_coef1, t, x_t.shape)
        mean_coef2 = self._extract(self.posterior_mean_coef2, t, x_t.shape)
        mean = mean_coef1 * x_start + mean_coef2 * x_t
        
        # Get variance
        variance = self._extract(self.posterior_variance, t, x_t.shape)
        
        return mean, variance
    
    @torch.no_grad()
    def p_sample(
        self,
        x_t: torch.Tensor,
        t: int,
        conditioning: torch.Tensor,
        clip_denoised: bool = True
    ) -> torch.Tensor:
        """
        Sample x_{t-1} from x_t (single denoising step).
        
        Args:
            x_t: Current noisy images
            t: Current timestep (scalar)
            conditioning: Conditioning maps
            clip_denoised: Clip predicted x_0
        
        Returns:
            x_{t-1}: Denoised images at previous timestep
        """
        batch_size = x_t.shape[0]
        t_tensor = torch.full((batch_size,), t, device=x_t.device, dtype=torch.long)
        
        # Predict mean and variance
        mean, variance = self.p_mean_variance(x_t, t_tensor, conditioning, clip_denoised)
        
        # Sample from N(mean, variance)
        noise = torch.randn_like(x_t)
        
        # No noise at t=0 (use tensor-based masking)
        nonzero_mask = (t_tensor != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        
        return mean + nonzero_mask * torch.sqrt(variance) * noise
    
    @torch.no_grad()
    def sample(
        self,
        conditioning: torch.Tensor,
        shape: Optional[Tuple[int, ...]] = None,
        clip_denoised: bool = True,
        return_intermediates: bool = False
    ) -> torch.Tensor:
        """
        Generate samples from random noise.
        
        Args:
            conditioning: Conditioning maps of shape (B, condition_channels, H, W)
            shape: Optional shape (B, C, H, W), inferred from conditioning if None
            clip_denoised: Clip intermediate predictions
            return_intermediates: Return all intermediate steps
        
        Returns:
            samples: Generated images of shape (B, C, H, W)
                    Or list of all steps if return_intermediates=True
        """
        device = conditioning.device
        
        # Infer shape from conditioning if not provided
        if shape is None:
            batch_size, _, h, w = conditioning.shape
            shape = (batch_size, 1, h, w)  # Assume single-channel output
        
        # Start from random noise
        x_t = torch.randn(shape, device=device)
        
        intermediates = [x_t] if return_intermediates else []
        
        # Reverse diffusion
        for t in tqdm(reversed(range(self.timesteps)), desc='Sampling', total=self.timesteps):
            x_t = self.p_sample(x_t, t, conditioning, clip_denoised)
            
            if return_intermediates:
                intermediates.append(x_t)
        
        if return_intermediates:
            return intermediates
        
        return x_t
    
    @torch.no_grad()
    def sample_with_cfg(
        self,
        conditioning: torch.Tensor,
        guidance_scale: float = 3.0,
        shape: Optional[Tuple[int, ...]] = None,
        clip_denoised: bool = True
    ) -> torch.Tensor:
        """
        Generate samples using classifier-free guidance.
        
        ε = ε_uncond + w * (ε_cond - ε_uncond)
        
        Args:
            conditioning: Conditioning maps
            guidance_scale: Guidance strength (w)
            shape: Optional output shape
            clip_denoised: Clip predictions
        
        Returns:
            samples: Generated images
        """
        device = conditioning.device
        
        # Infer shape
        if shape is None:
            batch_size, _, h, w = conditioning.shape
            shape = (batch_size, 1, h, w)
        
        # Start from noise
        x_t = torch.randn(shape, device=device)
        
        # Create unconditional conditioning (zeros)
        uncond = torch.zeros_like(conditioning)
        
        # Reverse diffusion with CFG
        for t in tqdm(reversed(range(self.timesteps)), desc='Sampling (CFG)', total=self.timesteps):
            batch_size = x_t.shape[0]
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Predict noise with and without conditioning
            noise_cond = self.predict_noise(x_t, t_tensor, conditioning)
            noise_uncond = self.predict_noise(x_t, t_tensor, uncond)
            
            # Apply guidance
            noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            
            # Predict x_0
            x_start = self.predict_start_from_noise(x_t, t_tensor, noise)
            
            if clip_denoised:
                x_start = torch.clamp(x_start, self.data_min, self.data_max)
            
            # Compute posterior mean
            mean_coef1 = self._extract(self.posterior_mean_coef1, t_tensor, x_t.shape)
            mean_coef2 = self._extract(self.posterior_mean_coef2, t_tensor, x_t.shape)
            mean = mean_coef1 * x_start + mean_coef2 * x_t
            
            # Get variance
            variance = self._extract(self.posterior_variance, t_tensor, x_t.shape)
            
            # Sample
            noise_sample = torch.randn_like(x_t)
            # No noise at t=0 (use tensor-based masking)
            nonzero_mask = (t_tensor != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
            
            x_t = mean + nonzero_mask * torch.sqrt(variance) * noise_sample
        
        return x_t
    

    @torch.no_grad()
    def reverse_from_xT(
        self,
        x_T: torch.Tensor,
        conditioning: torch.Tensor,
        guidance_scale: float = 0.0,
        clip_denoised: bool = True,
        uncond_conditioning: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Run the reverse diffusion starting from provided x_T.
        - x_T: tensor shape (B, C, H, W) starting noise (can be shared between runs)
        - conditioning: (B, cond_ch, H, W)
        - guidance_scale: classifier-free guidance scale (0 => no guidance)
        - uncond_conditioning: if None, uses zeros_like(conditioning)
        Returns:
            x_0: final denoised sample tensor (B, C, H, W)
        """
        device = x_T.device
        x_t = x_T.clone()

        if uncond_conditioning is None:
            uncond_conditioning = torch.zeros_like(conditioning)

        for t in tqdm(reversed(range(self.timesteps)), desc='Reverse from x_T', total=self.timesteps):
            batch_size = x_t.shape[0]
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

            if guidance_scale != 0.0:
                # Predict noise with and without conditioning
                noise_cond = self.predict_noise(x_t, t_tensor, conditioning)
                noise_uncond = self.predict_noise(x_t, t_tensor, uncond_conditioning)
                noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            else:
                noise = self.predict_noise(x_t, t_tensor, conditioning)

            # Predict x_start and mean/variance using the chosen noise (same as sample_with_cfg)
            x_start = self.predict_start_from_noise(x_t, t_tensor, noise)
            if clip_denoised:
                x_start = torch.clamp(x_start, self.data_min, self.data_max)

            mean_coef1 = self._extract(self.posterior_mean_coef1, t_tensor, x_t.shape)
            mean_coef2 = self._extract(self.posterior_mean_coef2, t_tensor, x_t.shape)
            mean = mean_coef1 * x_start + mean_coef2 * x_t

            variance = self._extract(self.posterior_variance, t_tensor, x_t.shape)
            # sample noise for the step
            noise_sample = torch.randn_like(x_t)
            nonzero_mask = (t_tensor != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
            x_t = mean + nonzero_mask * torch.sqrt(variance) * noise_sample

        return x_t
    
    def compute_loss(
        self,
        x_start: torch.Tensor,
        conditioning: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        low_noise_bias: bool = False,
        low_noise_fraction: float = 0.3,
        low_noise_weight: float = 3.0
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute training loss with optional timestep biasing for small datasets.
        
        Args:
            x_start: Clean images
            conditioning: Conditioning maps
            t: Optional timesteps (sampled if None)
            low_noise_bias: If True, bias sampling toward low-noise timesteps
            low_noise_fraction: Fraction of timesteps considered "low noise"
            low_noise_weight: Weight multiplier for low-noise sampling probability
        
        Returns:
            loss: Scalar loss
            diagnostics: Dict with keys:
                - 'noise_true_norm': Mean L2 norm of true noise
                - 'noise_pred_norm': Mean L2 norm of predicted noise
                - 'timesteps': Sampled timesteps (for logging)
        
        Note:
            Low-noise bias is CRITICAL for small datasets (~100-1000 patches).
            It helps the model learn structure before tackling high-noise denoising.
            Typical usage: enable for first 50-100 epochs, then disable.
        """
        batch_size = x_start.shape[0]
        device = x_start.device
        
        # Sample timesteps if not provided
        if t is None:
            if low_noise_bias:
                # Biased sampling: favor t in [0, low_noise_fraction * T]
                low_t_threshold = int(self.timesteps * low_noise_fraction)
                
                # Create probability weights
                # Low noise region: weight * uniform, High noise region: uniform
                low_prob = low_noise_weight / (low_t_threshold + (self.timesteps - low_t_threshold) / low_noise_weight)
                high_prob = low_prob / low_noise_weight
                
                # Sample with bias
                probs = torch.ones(self.timesteps, device=device)
                probs[:low_t_threshold] = low_prob
                probs[low_t_threshold:] = high_prob
                probs = probs / probs.sum()
                
                t = torch.multinomial(probs, batch_size, replacement=True)
            else:
                # Uniform sampling
                t = torch.randint(0, self.timesteps, (batch_size,), device=device, dtype=torch.long)
        
        # Sample noise
        noise = torch.randn_like(x_start)
        
        # Forward diffusion (add noise)
        x_t = self.q_sample(x_start, t, noise)
        
        # Predict noise
        predicted_noise = self.predict_noise(x_t, t, conditioning)
        
        # Compute loss
        if self.loss_type == 'l1':
            loss = F.l1_loss(predicted_noise, noise)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(predicted_noise, noise)
        elif self.loss_type == 'huber':
            loss = F.smooth_l1_loss(predicted_noise, noise)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Compute diagnostics
        noise_true_norm = torch.norm(noise.view(batch_size, -1), dim=1).mean()
        noise_pred_norm = torch.norm(predicted_noise.view(batch_size, -1), dim=1).mean()
        
        diagnostics = {
            'noise_true_norm': noise_true_norm.detach(),
            'noise_pred_norm': noise_pred_norm.detach(),
            'timesteps': t.detach()
        }
        
        return loss, diagnostics
    
    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
        """
        Extract coefficients at specified timesteps and reshape for broadcasting.
        
        Args:
            a: Tensor of coefficients
            t: Timesteps
            x_shape: Shape to broadcast to
        
        Returns:
            Extracted and reshaped coefficients
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


if __name__ == "__main__":
    # Test DDPM
    from unet import ConditionalUNet
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    unet = ConditionalUNet(
        in_channels=1,
        out_channels=1,
        condition_channels=3,
        base_channels=64
    )
    
    ddpm = DDPM(
        model=unet,
        timesteps=1000,
        beta_schedule='linear'
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in ddpm.parameters()):,}")
    
    # Test training step
    x = torch.randn(2, 1, 64, 64).to(device)
    conditioning = torch.randn(2, 3, 64, 64).to(device)
    
    loss, diagnostics = ddpm.compute_loss(x, conditioning)
    print(f"Loss: {loss.item():.4f}")
    print(f"||ε_true||: {diagnostics['noise_true_norm'].item():.4f}")
    print(f"||ε_pred||: {diagnostics['noise_pred_norm'].item():.4f}")
    
    # Test sampling (only a few steps for speed)
    ddpm.timesteps = 10
    with torch.no_grad():
        samples = ddpm.sample(conditioning[:1], shape=(1, 1, 64, 64))
    
    print(f"Sample shape: {samples.shape}")
    print("✓ DDPM test passed!")
