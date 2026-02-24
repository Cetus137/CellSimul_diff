"""
3D Conditional U-Net for diffusion models.

Direct 3D equivalent of models/unet.py with:
  - All Conv2d → Conv3d
  - bilinear → trilinear interpolation
  - time embedding broadcast: [:, :, None, None] → [:, :, None, None, None]
  - No attention blocks (memory cost prohibitive at 128³)

Multi-scale conditioning injection is preserved at every encoder level,
the bottleneck, and every decoder level — identical strategy to the 2D model.

Recommended defaults for a single A100-40 GB:
    base_channels=32, channel_multipliers=[1,2,4,8]
    → feature channels = [32, 64, 128, 256]
    → bottleneck spatial at 128³ input: 8³ (4 downsamples)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────────────────────

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional encoding for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ResidualBlock3D(nn.Module):
    """
    3D residual block with GroupNorm and time embedding injection.

    Identical logic to the 2D ResidualBlock but uses Conv3d and broadcasts
    the time embedding over (D, H, W) spatial dims.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_groups: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()

        groups_in = num_groups
        while in_channels % groups_in != 0 and groups_in > 1:
            groups_in -= 1

        groups_out = num_groups
        while out_channels % groups_out != 0 and groups_out > 1:
            groups_out -= 1

        self.norm1 = nn.GroupNorm(groups_in, in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        self.norm2 = nn.GroupNorm(groups_out, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)

        self.dropout = nn.Dropout(dropout)

        self.residual_conv = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, D, H, W)
            time_emb: (B, time_emb_dim)
        Returns:
            (B, out_channels, D, H, W)
        """
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # Broadcast time embedding over D, H, W
        t = self.time_mlp(time_emb)
        h = h + t[:, :, None, None, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.residual_conv(x)


class Downsample3D(nn.Module):
    """Halve spatial resolution with a stride-2 Conv3d."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample3D(nn.Module):
    """Double spatial resolution with trilinear nearest + Conv3d."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        return self.conv(x)


# ──────────────────────────────────────────────────────────────────────────────
# 3D Conditional U-Net
# ──────────────────────────────────────────────────────────────────────────────

class ConditionalUNet3D(nn.Module):
    """
    3D Conditional U-Net with multi-scale conditioning injection.

    Conditioning (e.g. heatmap + distance maps) is injected at every
    encoder level, the bottleneck, and every decoder level via channel
    concatenation — identical to the 2D ConditionalUNet.

    No attention blocks are included (memory cost at 128³ is prohibitive).

    Args:
        in_channels: Input image channels (default 1 for single-channel fluorescence).
        out_channels: Output channels (default 1).
        condition_channels: Number of conditioning map channels (e.g. 2).
        base_channels: Feature channels at the shallowest level.
        channel_multipliers: Multipliers per encoder/decoder level.
        num_res_blocks: ResidualBlock3D count per level.
        time_emb_dim: Sinusoidal time embedding size.
        num_groups: GroupNorm groups (auto-adjusted for divisibility).
        dropout: Dropout probability inside residual blocks.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        condition_channels: int = 2,
        base_channels: int = 32,
        channel_multipliers: List[int] = [1, 2, 4, 8],
        num_res_blocks: int = 2,
        time_emb_dim: int = 256,
        num_groups: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.condition_channels = condition_channels
        self.num_levels = len(channel_multipliers)

        # ── Time embedding ────────────────────────────────────────────────────
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        channels = [base_channels * m for m in channel_multipliers]

        # ── Initial convolution ───────────────────────────────────────────────
        # Concatenate image + conditioning at full resolution
        self.init_conv = nn.Conv3d(
            in_channels + condition_channels,
            base_channels,
            kernel_size=3,
            padding=1
        )

        # ── Encoder ───────────────────────────────────────────────────────────
        self.encoder = nn.ModuleList()
        in_ch = base_channels

        for level, ch in enumerate(channels):
            blocks = nn.ModuleList()

            # First block injects conditioning
            blocks.append(ResidualBlock3D(
                in_ch + condition_channels, ch, time_emb_dim, num_groups, dropout
            ))
            in_ch = ch

            for _ in range(num_res_blocks - 1):
                blocks.append(ResidualBlock3D(
                    in_ch, ch, time_emb_dim, num_groups, dropout
                ))
                in_ch = ch

            if level < len(channels) - 1:
                blocks.append(Downsample3D(ch))

            self.encoder.append(blocks)

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.bottleneck = nn.ModuleList([
            ResidualBlock3D(
                channels[-1] + condition_channels,
                channels[-1],
                time_emb_dim, num_groups, dropout
            ),
            ResidualBlock3D(
                channels[-1],
                channels[-1],
                time_emb_dim, num_groups, dropout
            ),
        ])

        # ── Decoder ───────────────────────────────────────────────────────────
        self.decoder = nn.ModuleList()

        for level in reversed(range(len(channels))):
            blocks = nn.ModuleList()
            out_ch = channels[level]

            for i in range(num_res_blocks + 1):
                if i == 0:
                    if level < len(channels) - 1:
                        in_ch = channels[level + 1] + channels[level] + condition_channels
                    else:
                        in_ch = channels[level] * 2 + condition_channels
                else:
                    in_ch = out_ch

                blocks.append(ResidualBlock3D(
                    in_ch, out_ch, time_emb_dim, num_groups, dropout
                ))

            if level > 0:
                blocks.append(Upsample3D(out_ch))

            self.decoder.append(blocks)

        # ── Final convolution ─────────────────────────────────────────────────
        final_groups = num_groups
        while base_channels % final_groups != 0 and final_groups > 1:
            final_groups -= 1

        self.final_conv = nn.Sequential(
            nn.GroupNorm(final_groups, base_channels),
            nn.SiLU(),
            nn.Conv3d(base_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        conditioning: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with multi-scale 3D conditioning injection.

        Args:
            x:           (B, 1, D, H, W) noisy volume in [-1, 1]
            t:           (B,) integer timesteps
            conditioning: (B, C_cond, D, H, W) conditioning maps in [0, 1]

        Returns:
            (B, 1, D, H, W) predicted noise
        """
        time_emb = self.time_embedding(t)

        # Input concat + initial conv
        h = torch.cat([x, conditioning], dim=1)
        h = self.init_conv(h)

        # ── Encoder ───────────────────────────────────────────────────────────
        skip_connections = []

        for level_idx, blocks in enumerate(self.encoder):
            cond_scaled = F.interpolate(
                conditioning,
                size=h.shape[2:],
                mode='trilinear',
                align_corners=False
            )

            for block_idx, block in enumerate(blocks):
                if isinstance(block, ResidualBlock3D) and block_idx == 0:
                    h = torch.cat([h, cond_scaled], dim=1)
                    h = block(h, time_emb)
                elif isinstance(block, ResidualBlock3D):
                    h = block(h, time_emb)
                elif isinstance(block, Downsample3D):
                    skip_connections.append(h)
                    h = block(h)
                # (no attention blocks)

            if level_idx == len(self.encoder) - 1:
                skip_connections.append(h)

        # ── Bottleneck ────────────────────────────────────────────────────────
        cond_bn = F.interpolate(
            conditioning, size=h.shape[2:], mode='trilinear', align_corners=False
        )
        for idx, block in enumerate(self.bottleneck):
            if idx == 0:
                h = torch.cat([h, cond_bn], dim=1)
            h = block(h, time_emb)

        # ── Decoder ───────────────────────────────────────────────────────────
        skip_connections = skip_connections[::-1]
        skip_idx = 0

        for level_idx, blocks in enumerate(self.decoder):
            cond_scaled = F.interpolate(
                conditioning,
                size=h.shape[2:],
                mode='trilinear',
                align_corners=False
            )

            for i, block in enumerate(blocks):
                if isinstance(block, ResidualBlock3D) and i == 0 and skip_idx < len(skip_connections):
                    skip = skip_connections[skip_idx]
                    skip_idx += 1
                    if skip.shape[2:] != h.shape[2:]:
                        skip = F.interpolate(skip, size=h.shape[2:], mode='nearest')
                    h = torch.cat([h, skip, cond_scaled], dim=1)
                    h = block(h, time_emb)
                elif isinstance(block, ResidualBlock3D):
                    h = block(h, time_emb)
                elif isinstance(block, Upsample3D):
                    h = block(h)

        return self.final_conv(h)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ConditionalUNet3D(
        in_channels=1,
        out_channels=1,
        condition_channels=2,
        base_channels=32,
        channel_multipliers=[1, 2, 4, 8],
        num_res_blocks=2,
        time_emb_dim=256,
        dropout=0.1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    B = 1
    x   = torch.randn(B, 1, 128, 128, 128, device=device)
    t   = torch.randint(0, 1000, (B,), device=device)
    c   = torch.randn(B, 2, 128, 128, 128, device=device)

    with torch.no_grad():
        out = model(x, t, c)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    assert out.shape == (B, 1, 128, 128, 128), "Shape mismatch!"
    print("✓ 3D UNet forward pass OK")
