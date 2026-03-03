"""
DDPM implementation for 3D volumetric diffusion models.

This is a dimension-generic version of models/diffusion.py.

The only functional difference is that the four sampling methods
(`sample`, `sample_with_cfg`, `sample_correlated`, `sample_pair_same_noise`)
infer the output shape from conditioning using

    batch_size, _, *spatial = conditioning.shape
    shape = (batch_size, 1, *spatial)

instead of the 2D-specific `batch_size, _, h, w = conditioning.shape`.
This makes the class work identically for 2D (H, W) and 3D (D, H, W) inputs.

All other logic — noise schedules, forward process, loss, reverse process — is
unchanged from the original diffusion.py.

The 2D models/diffusion.py is NOT modified.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict
from tqdm import tqdm


def linear_beta_schedule(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def quadratic_beta_schedule(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


class DDPM3D(nn.Module):
    """
    Denoising Diffusion Probabilistic Model — dimension-generic.

    Works for any spatial dimensionality (2D or 3D) as long as the wrapped
    model accepts tensors of the corresponding shape.

    Implements:
    - Forward process: q(x_t | x_0) — adding noise
    - Reverse process: p(x_{t-1} | x_t) — denoising
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
        super().__init__()

        self.model = model
        self.timesteps = timesteps
        self.prediction_type = prediction_type
        self.loss_type = loss_type
        self.data_min = data_min
        self.data_max = data_max

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == 'quadratic':
            betas = quadratic_beta_schedule(timesteps, beta_start, beta_end)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

    # ── Forward process ────────────────────────────────────────────────────────

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_prod = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus  = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alpha_prod * x_start + sqrt_one_minus * noise

    # ── Model call ─────────────────────────────────────────────────────────────

    def predict_noise(self, x_t: torch.Tensor, t: torch.Tensor,
                      conditioning: torch.Tensor) -> torch.Tensor:
        return self.model(x_t, t, conditioning)

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor,
                                  noise: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_m = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return (x_t - sqrt_one_m * noise) / sqrt_alpha

    # ── Reverse process ────────────────────────────────────────────────────────

    def p_mean_variance(self, x_t: torch.Tensor, t: torch.Tensor,
                        conditioning: torch.Tensor,
                        clip_denoised: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        predicted_noise = self.predict_noise(x_t, t, conditioning)
        x_start = self.predict_start_from_noise(x_t, t, predicted_noise)
        if clip_denoised:
            x_start = torch.clamp(x_start, self.data_min, self.data_max)
        coef1 = self._extract(self.posterior_mean_coef1, t, x_t.shape)
        coef2 = self._extract(self.posterior_mean_coef2, t, x_t.shape)
        mean  = coef1 * x_start + coef2 * x_t
        var   = self._extract(self.posterior_variance, t, x_t.shape)
        return mean, var

    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: int,
                 conditioning: torch.Tensor,
                 clip_denoised: bool = True) -> torch.Tensor:
        B = x_t.shape[0]
        t_tensor = torch.full((B,), t, device=x_t.device, dtype=torch.long)
        mean, variance = self.p_mean_variance(x_t, t_tensor, conditioning, clip_denoised)
        noise = torch.randn_like(x_t)
        nonzero = (t_tensor != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        return mean + nonzero * torch.sqrt(variance) * noise

    # ── Sampling methods (dimension-generic) ───────────────────────────────────

    @torch.no_grad()
    def sample(self, conditioning: torch.Tensor,
               shape: Optional[Tuple[int, ...]] = None,
               clip_denoised: bool = True,
               return_intermediates: bool = False) -> torch.Tensor:
        """
        Generate samples from random noise.

        Args:
            conditioning: (B, C_cond, *spatial) — works for 2D or 3D.
            shape: Optional explicit output shape. Inferred from conditioning if None.
        """
        device = conditioning.device
        if shape is None:
            batch_size, _, *spatial = conditioning.shape
            shape = (batch_size, 1, *spatial)

        x_t = torch.randn(shape, device=device)
        intermediates = [x_t] if return_intermediates else []

        for t in tqdm(reversed(range(self.timesteps)), desc='Sampling', total=self.timesteps):
            x_t = self.p_sample(x_t, t, conditioning, clip_denoised)
            if return_intermediates:
                intermediates.append(x_t)

        return intermediates if return_intermediates else x_t

    @torch.no_grad()
    def sample_with_cfg(self, conditioning: torch.Tensor,
                        guidance_scale: float = 3.0,
                        shape: Optional[Tuple[int, ...]] = None,
                        clip_denoised: bool = True) -> torch.Tensor:
        """Classifier-free guidance sampling (dimension-generic)."""
        device = conditioning.device
        if shape is None:
            batch_size, _, *spatial = conditioning.shape
            shape = (batch_size, 1, *spatial)

        x_t = torch.randn(shape, device=device)
        uncond = torch.zeros_like(conditioning)

        for t in tqdm(reversed(range(self.timesteps)), desc='Sampling (CFG)', total=self.timesteps):
            B = x_t.shape[0]
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

            noise_cond   = self.predict_noise(x_t, t_tensor, conditioning)
            noise_uncond = self.predict_noise(x_t, t_tensor, uncond)
            noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

            x_start = self.predict_start_from_noise(x_t, t_tensor, noise)
            if clip_denoised:
                x_start = torch.clamp(x_start, self.data_min, self.data_max)

            coef1 = self._extract(self.posterior_mean_coef1, t_tensor, x_t.shape)
            coef2 = self._extract(self.posterior_mean_coef2, t_tensor, x_t.shape)
            mean  = coef1 * x_start + coef2 * x_t
            var   = self._extract(self.posterior_variance, t_tensor, x_t.shape)

            noise_s = torch.randn_like(x_t)
            nz_mask = (t_tensor != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
            x_t = mean + nz_mask * torch.sqrt(var) * noise_s

        return x_t

    # ── DDIM sampling (fast inference, same weights as DDPM) ───────────────────

    @torch.no_grad()
    def sample_ddim(self, conditioning: torch.Tensor,
                    ddim_steps: int = 200,
                    eta: float = 0.0,
                    shape: Optional[Tuple[int, ...]] = None,
                    clip_denoised: bool = True) -> torch.Tensor:
        """
        DDIM sampling — deterministic reverse process over a sub-sequence of
        timesteps (Song et al., 2020, arXiv:2010.02502).

        Typical speedups vs DDPM-1000:
            ddim_steps=200  →  ~5x   (recommended default)
            ddim_steps=100  →  ~10x
            ddim_steps=50   →  ~20x

        Args:
            conditioning: (B, C_cond, *spatial) — works for 2D or 3D.
            ddim_steps: Number of denoising steps.
            eta: Stochasticity coefficient — 0 = deterministic, 1 ≈ DDPM.
            shape: Optional explicit output shape. Inferred from conditioning if None.
            clip_denoised: Clip predicted x₀ to [data_min, data_max].
        """
        device = conditioning.device
        if shape is None:
            batch_size, _, *spatial = conditioning.shape
            shape = (batch_size, 1, *spatial)

        # Build uniformly-spaced sub-sequence τ over [0, T), then reverse
        T = self.timesteps
        step_size = max(1, T // ddim_steps)
        tau = list(range(0, T, step_size))
        if tau[-1] != T - 1:
            tau.append(T - 1)
        tau = list(reversed(tau))  # descending, e.g. [999, 994, …, 4, 0]

        x_t = torch.randn(shape, device=device)

        for idx in tqdm(range(len(tau)), desc=f'DDIM ({len(tau)} steps)', total=len(tau)):
            t      = tau[idx]
            t_prev = tau[idx + 1] if idx + 1 < len(tau) else -1

            B = x_t.shape[0]
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

            eps = self.predict_noise(x_t, t_tensor, conditioning)

            # Predicted x₀ from current noisy sample and predicted noise
            sqrt_acp = self._extract(self.sqrt_alphas_cumprod, t_tensor, x_t.shape)
            sqrt_1m  = self._extract(self.sqrt_one_minus_alphas_cumprod, t_tensor, x_t.shape)
            x0_pred  = (x_t - sqrt_1m * eps) / sqrt_acp
            if clip_denoised:
                x0_pred = torch.clamp(x0_pred, self.data_min, self.data_max)

            if t_prev < 0:
                x_t = x0_pred
                continue

            acp_t         = self._extract(self.alphas_cumprod, t_tensor, x_t.shape)
            t_prev_tensor = torch.full((B,), t_prev, device=device, dtype=torch.long)
            acp_prev      = self._extract(self.alphas_cumprod, t_prev_tensor, x_t.shape)

            # DDIM σ — zero when eta=0 (fully deterministic)
            sigma  = eta * torch.sqrt(
                (1.0 - acp_prev) / (1.0 - acp_t) * (1.0 - acp_t / acp_prev)
            )
            dir_xt = torch.sqrt(torch.clamp(1.0 - acp_prev - sigma ** 2, min=0.0)) * eps
            noise  = torch.randn_like(x_t) if eta > 0 else torch.zeros_like(x_t)
            x_t    = torch.sqrt(acp_prev) * x0_pred + dir_xt + sigma * noise

        return x_t

    @torch.no_grad()
    def sample_ddim_with_cfg(self, conditioning: torch.Tensor,
                              guidance_scale: float = 3.0,
                              ddim_steps: int = 200,
                              eta: float = 0.0,
                              shape: Optional[Tuple[int, ...]] = None,
                              clip_denoised: bool = True) -> torch.Tensor:
        """DDIM sampling with classifier-free guidance (dimension-generic)."""
        device = conditioning.device
        if shape is None:
            batch_size, _, *spatial = conditioning.shape
            shape = (batch_size, 1, *spatial)

        T = self.timesteps
        step_size = max(1, T // ddim_steps)
        tau = list(range(0, T, step_size))
        if tau[-1] != T - 1:
            tau.append(T - 1)
        tau = list(reversed(tau))

        x_t    = torch.randn(shape, device=device)
        uncond = torch.zeros_like(conditioning)

        for idx in tqdm(range(len(tau)), desc=f'DDIM CFG ({len(tau)} steps)', total=len(tau)):
            t      = tau[idx]
            t_prev = tau[idx + 1] if idx + 1 < len(tau) else -1

            B = x_t.shape[0]
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

            eps_cond   = self.predict_noise(x_t, t_tensor, conditioning)
            eps_uncond = self.predict_noise(x_t, t_tensor, uncond)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            sqrt_acp = self._extract(self.sqrt_alphas_cumprod, t_tensor, x_t.shape)
            sqrt_1m  = self._extract(self.sqrt_one_minus_alphas_cumprod, t_tensor, x_t.shape)
            x0_pred  = (x_t - sqrt_1m * eps) / sqrt_acp
            if clip_denoised:
                x0_pred = torch.clamp(x0_pred, self.data_min, self.data_max)

            if t_prev < 0:
                x_t = x0_pred
                continue

            acp_t         = self._extract(self.alphas_cumprod, t_tensor, x_t.shape)
            t_prev_tensor = torch.full((B,), t_prev, device=device, dtype=torch.long)
            acp_prev      = self._extract(self.alphas_cumprod, t_prev_tensor, x_t.shape)

            sigma  = eta * torch.sqrt(
                (1.0 - acp_prev) / (1.0 - acp_t) * (1.0 - acp_t / acp_prev)
            )
            dir_xt = torch.sqrt(torch.clamp(1.0 - acp_prev - sigma ** 2, min=0.0)) * eps
            noise  = torch.randn_like(x_t) if eta > 0 else torch.zeros_like(x_t)
            x_t    = torch.sqrt(acp_prev) * x0_pred + dir_xt + sigma * noise

        return x_t

    @torch.no_grad()
    def reverse_from_xT(self, x_T: torch.Tensor, conditioning: torch.Tensor,
                         guidance_scale: float = 0.0,
                         clip_denoised: bool = True,
                         uncond_conditioning: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run reverse diffusion from a provided x_T (dimension-generic)."""
        device  = x_T.device
        x_t = x_T.clone()
        if uncond_conditioning is None:
            uncond_conditioning = torch.zeros_like(conditioning)

        for t in tqdm(reversed(range(self.timesteps)), desc='Reverse from x_T', total=self.timesteps):
            B = x_t.shape[0]
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

            if guidance_scale != 0.0:
                noise_cond   = self.predict_noise(x_t, t_tensor, conditioning)
                noise_uncond = self.predict_noise(x_t, t_tensor, uncond_conditioning)
                noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            else:
                noise = self.predict_noise(x_t, t_tensor, conditioning)

            x_start = self.predict_start_from_noise(x_t, t_tensor, noise)
            if clip_denoised:
                x_start = torch.clamp(x_start, self.data_min, self.data_max)

            coef1 = self._extract(self.posterior_mean_coef1, t_tensor, x_t.shape)
            coef2 = self._extract(self.posterior_mean_coef2, t_tensor, x_t.shape)
            mean  = coef1 * x_start + coef2 * x_t
            var   = self._extract(self.posterior_variance, t_tensor, x_t.shape)

            noise_s = torch.randn_like(x_t)
            nz_mask = (t_tensor != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
            x_t = mean + nz_mask * torch.sqrt(var) * noise_s

        return x_t

    @torch.no_grad()
    def sample_correlated(self, conditioning: torch.Tensor, x_T: torch.Tensor,
                           seed_shared: int, seed_unique: int, rho: float = 0.95,
                           temporal_smoothness: float = 0.0,
                           unique_noise_prev: Optional[torch.Tensor] = None,
                           guidance_scale: float = 0.0,
                           clip_denoised: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate temporally correlated samples (dimension-generic).

        See models/diffusion.py for full documentation.
        """
        device = conditioning.device
        batch_size, _, *spatial = conditioning.shape
        out_shape = x_T.shape

        gen_shared = torch.Generator(device=device).manual_seed(int(seed_shared))
        beta = float(temporal_smoothness)

        if unique_noise_prev is not None and beta > 0:
            prev_seed_contrib = int(unique_noise_prev.abs().mean().item() * 1e6) % 1_000_000
            blended_seed = int(beta * prev_seed_contrib + (1 - beta) * seed_unique)
            gen_unique = torch.Generator(device=device).manual_seed(blended_seed)
        else:
            gen_unique = torch.Generator(device=device).manual_seed(int(seed_unique))

        sqrt_rho         = torch.sqrt(torch.tensor(max(0.0, float(rho))))
        sqrt_1_minus_rho = torch.sqrt(torch.tensor(max(0.0, 1.0 - float(rho))))
        sqrt_1_minus_b2  = torch.sqrt(torch.tensor(max(0.0, 1.0 - beta**2)))

        if unique_noise_prev is not None and beta > 0:
            innov = torch.randn(out_shape, device=device, generator=gen_unique)
            initial_unique_base = beta * unique_noise_prev + sqrt_1_minus_b2 * innov
        else:
            initial_unique_base = None

        x = x_T.clone()
        uncond = torch.zeros_like(conditioning) if guidance_scale != 0.0 else None

        unique_noise_state_for_next = None
        first_step_done = False

        for t in reversed(range(self.timesteps)):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

            if guidance_scale != 0.0:
                eps_c = self.predict_noise(x, t_tensor, conditioning)
                eps_u = self.predict_noise(x, t_tensor, uncond)
                eps = eps_u + guidance_scale * (eps_c - eps_u)
            else:
                eps = self.predict_noise(x, t_tensor, conditioning)

            x_start = self.predict_start_from_noise(x, t_tensor, eps)
            if clip_denoised:
                x_start = torch.clamp(x_start, self.data_min, self.data_max)

            coef1 = self._extract(self.posterior_mean_coef1, t_tensor, x.shape)
            coef2 = self._extract(self.posterior_mean_coef2, t_tensor, x.shape)
            mean  = coef1 * x_start + coef2 * x
            var   = self._extract(self.posterior_variance, t_tensor, x.shape)

            if t > 0:
                noise_shared = torch.randn(out_shape, device=device, generator=gen_shared)
                if not first_step_done and initial_unique_base is not None:
                    noise_unique = initial_unique_base
                    unique_noise_state_for_next = noise_unique.clone()
                    first_step_done = True
                else:
                    noise_unique = torch.randn(out_shape, device=device, generator=gen_unique)
                    if not first_step_done:
                        unique_noise_state_for_next = noise_unique.clone()
                        first_step_done = True
                noise = sqrt_rho * noise_shared + sqrt_1_minus_rho * noise_unique
                x = mean + torch.sqrt(var) * noise
            else:
                x = mean

        return x, unique_noise_state_for_next

    @torch.no_grad()
    def sample_pair_same_noise(self, cond_A: torch.Tensor, cond_B: torch.Tensor,
                                guidance_scale: float = 0.0, out_channels: int = 1,
                                seed: int = 0,
                                clip_denoised: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a matched pair of samples with identical per-step noise (dimension-generic).

        See models/diffusion.py for full documentation.
        """
        device = cond_A.device
        assert cond_A.shape[0] == cond_B.shape[0]
        B, _, *spatial = cond_A.shape
        out_shape = (B, out_channels, *spatial)

        x_T_base = torch.randn((B, 1, *spatial), device=device)
        x_T = x_T_base.repeat(1, out_channels, *([1] * len(spatial)))

        uncond = torch.zeros_like(cond_A)

        def _reverse(xT, conditioning, seed_local):
            gen = torch.Generator(device=device).manual_seed(int(seed_local))
            x = xT.clone()
            for t in reversed(range(self.timesteps)):
                t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
                if guidance_scale != 0.0:
                    eps_c = self.predict_noise(x, t_tensor, conditioning)
                    eps_u = self.predict_noise(x, t_tensor, uncond)
                    eps = eps_u + guidance_scale * (eps_c - eps_u)
                else:
                    eps = self.predict_noise(x, t_tensor, conditioning)
                x_start = self.predict_start_from_noise(x, t_tensor, eps)
                if clip_denoised:
                    x_start = torch.clamp(x_start, self.data_min, self.data_max)
                c1  = self._extract(self.posterior_mean_coef1, t_tensor, x.shape)
                c2  = self._extract(self.posterior_mean_coef2, t_tensor, x.shape)
                mean = c1 * x_start + c2 * x
                var  = self._extract(self.posterior_variance, t_tensor, x.shape)
                ns   = torch.randn(out_shape, device=device, generator=gen)
                nz   = (t_tensor != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
                x = mean + nz * torch.sqrt(var) * ns
            return x

        frameA = _reverse(x_T, cond_A, seed)
        frameB = _reverse(x_T, cond_B, seed)
        return frameA, frameB

    # ── Training loss ──────────────────────────────────────────────────────────

    def compute_loss(self, x_start: torch.Tensor, conditioning: torch.Tensor,
                     t: Optional[torch.Tensor] = None,
                     low_noise_bias: bool = False,
                     low_noise_fraction: float = 0.3,
                     low_noise_weight: float = 3.0) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute training loss (identical to DDPM.compute_loss in diffusion.py)."""
        batch_size = x_start.shape[0]
        device = x_start.device

        if t is None:
            if low_noise_bias:
                low_t = int(self.timesteps * low_noise_fraction)
                low_p = low_noise_weight / (low_t + (self.timesteps - low_t) / low_noise_weight)
                hi_p  = low_p / low_noise_weight
                probs = torch.ones(self.timesteps, device=device)
                probs[:low_t]  = low_p
                probs[low_t:]  = hi_p
                probs /= probs.sum()
                t = torch.multinomial(probs, batch_size, replacement=True)
            else:
                t = torch.randint(0, self.timesteps, (batch_size,), device=device, dtype=torch.long)

        noise = torch.randn_like(x_start)
        x_t   = self.q_sample(x_start, t, noise)
        pred  = self.predict_noise(x_t, t, conditioning)

        if self.loss_type == 'l1':
            loss = F.l1_loss(pred, noise)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(pred, noise)
        elif self.loss_type == 'huber':
            loss = F.smooth_l1_loss(pred, noise)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        diag = {
            'noise_true_norm': torch.norm(noise.view(batch_size, -1), dim=1).mean().detach(),
            'noise_pred_norm': torch.norm(pred.view(batch_size, -1), dim=1).mean().detach(),
            'timesteps': t.detach(),
        }
        return loss, diag

    # ── Utility ────────────────────────────────────────────────────────────────

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
        """Gather buffer values at timesteps t and broadcast to x_shape."""
        out = a.gather(-1, t)
        return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))


if __name__ == "__main__":
    from models.unet3d import ConditionalUNet3D

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unet = ConditionalUNet3D(base_channels=16, channel_multipliers=[1, 2, 4, 8])
    ddpm = DDPM3D(model=unet, timesteps=100).to(device)

    x   = torch.randn(1, 1, 128, 128, 128, device=device)
    c   = torch.randn(1, 2, 128, 128, 128, device=device)
    loss, _ = ddpm.compute_loss(x, c)
    print(f"Loss: {loss.item():.4f}")

    ddpm.timesteps = 5
    sample = ddpm.sample(c)
    print(f"Sample shape: {sample.shape}")
    print("✓ DDPM3D OK")
