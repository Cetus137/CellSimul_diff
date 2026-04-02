"""
Frozen 3D encoder for realism evaluation.

Wraps the encoder half of a trained ConditionalUNet3D (from a DDPM3D
checkpoint) as a feature extractor, discarding the decoder entirely.

Input:
    x:           (B, 1, D, H, W)  — image in [-1, 1]
    conditioning: (B, 2, D, H, W) — heatmap + distance map in [0, 1]

Output: (B, latent_dim) — L2-normalised embedding on the unit sphere.

Both channels in `conditioning` match the trained model's conditioning
contract exactly (heatmap + distance), requiring zero weight surgery on the
loaded checkpoint.

The diffusion timestep is fixed to t=0 (zero time embedding) so the
residual blocks behave as standard ResNets and produce deterministic features
independent of any particular noise level.

Usage
-----
encoder = CellEncoder3D.from_ddpm_checkpoint(
    ckpt_path="checkpoints/frame1_3d/best.pt",
    config_path="configs/frame1_3d.yaml",
    device="cuda",
)
z = encoder.encode(image_tensor, cond_tensor)   # (B, 128)
"""

from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from models.unet3d import ConditionalUNet3D, Downsample3D, ResidualBlock3D


class CellEncoder3D(nn.Module):
    """
    Frozen encoder extracted from a trained ConditionalUNet3D.

    Architecture mirrors the UNet3D encoder + bottleneck, then:
        global average pool → Linear(256→latent_dim) → ReLU → Linear(latent_dim→latent_dim)
        → L2 normalise

    The UNet decoder is never instantiated — weights are not loaded for it,
    keeping peak memory low.

    Args:
        unet:       A ConditionalUNet3D instance with loaded weights.
        latent_dim: Output embedding dimensionality (default 128).
    """

    def __init__(self, unet: ConditionalUNet3D, latent_dim: int = 128):
        super().__init__()

        # Freeze and store the UNet encoder components only
        self.time_embedding = unet.time_embedding
        self.init_conv       = unet.init_conv
        self.encoder         = unet.encoder
        self.bottleneck      = unet.bottleneck
        self._n_encoder_levels = unet.num_levels

        for p in self.parameters():
            p.requires_grad_(False)

        # Infer bottleneck channels from the bottleneck's second block output
        # = base_channels * channel_multipliers[-1] = 32 * 8 = 256 by default
        bottleneck_ch = unet.bottleneck[-1].conv2.out_channels

        # Small trainable projection head (not frozen — can be fine-tuned later)
        self.proj = nn.Sequential(
            nn.Linear(bottleneck_ch, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim),
        )

        self.latent_dim = latent_dim

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract L2-normalised embedding.

        Args:
            x:           (B, 1, D, H, W)  image in [-1, 1]
            conditioning: (B, 2, D, H, W) heatmap + distance in [0, 1]

        Returns:
            z: (B, latent_dim) float32, unit-norm
        """
        B = x.shape[0]
        device = x.device

        # Fixed t=0 time embedding — deterministic features, no timestep variance
        t_fixed = torch.zeros(B, dtype=torch.long, device=device)
        time_emb = self.time_embedding(t_fixed)  # (B, time_emb_dim)

        # Initial conv — same as UNet forward: cat image + full-res conditioning
        h = self.init_conv(torch.cat([x, conditioning], dim=1))

        # Encoder path — mirrors ConditionalUNet3D.forward() exactly
        for level_idx, blocks in enumerate(self.encoder):
            cond_scaled = F.interpolate(
                conditioning,
                size=h.shape[2:],
                mode="trilinear",
                align_corners=False,
            )
            for block_idx, block in enumerate(blocks):
                if isinstance(block, ResidualBlock3D) and block_idx == 0:
                    h = torch.cat([h, cond_scaled], dim=1)
                    h = block(h, time_emb)
                elif isinstance(block, ResidualBlock3D):
                    h = block(h, time_emb)
                elif isinstance(block, Downsample3D):
                    h = block(h)
                # (skip_connections not needed — encoder only)

        # Bottleneck — same injection pattern as UNet forward
        cond_bn = F.interpolate(
            conditioning,
            size=h.shape[2:],
            mode="trilinear",
            align_corners=False,
        )
        for idx, block in enumerate(self.bottleneck):
            if idx == 0:
                h = torch.cat([h, cond_bn], dim=1)
            h = block(h, time_emb)
        # h: (B, bottleneck_ch, d, h_s, w_s)  e.g. (B, 256, 8, 8, 8) for 128³ input

        # Global average pool → flat vector
        h = h.mean(dim=[2, 3, 4])   # (B, bottleneck_ch)

        z = self.proj(h)             # (B, latent_dim)
        return F.normalize(z, dim=-1)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        return self.encode(x, conditioning)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_ddpm_checkpoint(
        cls,
        ckpt_path: str,
        config_path: str,
        device: str = "cpu",
        latent_dim: int = 128,
    ) -> "CellEncoder3D":
        """
        Build a CellEncoder3D from a trained DDPM3D frame1_3d checkpoint.

        Only the UNet encoder + bottleneck weights are used; the decoder
        weights in the checkpoint are loaded into the full UNet but then
        discarded when the encoder wraps only the encoder half.

        Args:
            ckpt_path:   Path to .pt checkpoint (model_state_dict or bare).
            config_path: Path to the corresponding YAML config.
            device:      Target device string.
            latent_dim:  Embedding dimensionality.

        Returns:
            CellEncoder3D in eval() mode with frozen UNet weights.
        """
        from sampling.sample_from_centres3d import load_model3d

        ddpm = load_model3d(ckpt_path, config_path, device=device)
        unet: ConditionalUNet3D = ddpm.model

        encoder = cls(unet=unet, latent_dim=latent_dim).to(device)
        encoder.eval()
        return encoder
