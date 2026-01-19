"""
Conditional U-Net for diffusion models.

Architecture:
- Encoder-decoder with skip connections
- Time embedding via sinusoidal encoding
- Conditioning via channel concatenation
- Self-attention at specified resolutions
- GroupNorm and residual blocks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional
from einops import rearrange


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal positional encoding for timesteps.
    
    Maps integer timesteps to continuous embeddings using sine/cosine functions.
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Timesteps of shape (B,)
        
        Returns:
            embeddings: Time embeddings of shape (B, dim)
        """
        device = t.device
        half_dim = self.dim // 2
        
        # Compute frequency factors
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        
        # Compute embeddings
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        return emb


class ResidualBlock(nn.Module):
    """
    Residual block with GroupNorm and time embedding injection.
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
        
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, in_channels, H, W)
            time_emb: Time embedding of shape (B, time_emb_dim)
        
        Returns:
            output: Tensor of shape (B, out_channels, H, W)
        """
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        # Residual connection
        return h + self.residual_conv(x)


class AttentionBlock(nn.Module):
    """
    Self-attention block with multi-head attention.
    """
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        num_groups: int = 32
    ):
        super().__init__()
        
        self.channels = channels
        self.num_heads = num_heads
        
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.norm = nn.GroupNorm(num_groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            output: Tensor of shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Normalize
        h = self.norm(x)
        
        # Compute Q, K, V
        qkv = self.qkv(h)  # (B, 3*C, H, W)
        qkv = rearrange(qkv, 'b (three heads c) h w -> three b heads (h w) c',
                       three=3, heads=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        scale = (C // self.num_heads) ** -0.5
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        
        # Apply attention to values
        h = torch.matmul(attn, v)
        
        # Reshape back
        h = rearrange(h, 'b heads (h w) c -> b (heads c) h w', h=H, w=W)
        
        # Project
        h = self.proj(h)
        
        # Residual
        return x + h


class Downsample(nn.Module):
    """Downsampling by 2x using stride-2 convolution."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Upsampling by 2x using nearest-neighbor + convolution."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class ConditionalUNet(nn.Module):
    """
    U-Net for conditional image generation.
    
    Conditioning is injected via channel concatenation at the input.
    Time embedding is injected into each residual block.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        condition_channels: int = 3,
        base_channels: int = 128,
        channel_multipliers: List[int] = [1, 2, 3, 4],
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = [16, 8],
        num_heads: int = 8,
        time_emb_dim: int = 512,
        num_groups: int = 32,
        dropout: float = 0.1
    ):
        """
        Args:
            in_channels: Number of input image channels
            out_channels: Number of output image channels
            condition_channels: Number of conditioning map channels
            base_channels: Base number of channels
            channel_multipliers: Channel multipliers for each resolution level
            num_res_blocks: Number of residual blocks per level
            attention_resolutions: Resolutions at which to apply attention
            num_heads: Number of attention heads
            time_emb_dim: Dimension of time embeddings
            num_groups: Number of groups for GroupNorm
            dropout: Dropout probability
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.condition_channels = condition_channels
        
        # Time embedding
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Initial convolution (concatenate image + conditioning)
        self.init_conv = nn.Conv2d(
            in_channels + condition_channels,
            base_channels,
            kernel_size=3,
            padding=1
        )
        
        # Compute channel dimensions for each level
        channels = [base_channels * mult for mult in channel_multipliers]
        
        # Encoder (downsampling path)
        self.encoder = nn.ModuleList()
        in_ch = base_channels
        
        for level, ch in enumerate(channels):
            blocks = nn.ModuleList()
            
            for _ in range(num_res_blocks):
                blocks.append(ResidualBlock(
                    in_ch, ch, time_emb_dim, num_groups, dropout
                ))
                in_ch = ch
                
                # Add attention if at specified resolution
                # Resolution = image_size / 2^level (approximately)
                if level < len(attention_resolutions):
                    expected_res = attention_resolutions[level]
                    # We don't know the exact resolution here, so we add attention
                    # at each level and rely on the user to configure correctly
                    if level >= len(channels) - len(attention_resolutions):
                        blocks.append(AttentionBlock(ch, num_heads, num_groups))
            
            # Add downsampling (except last level)
            if level < len(channels) - 1:
                blocks.append(Downsample(ch))
            
            self.encoder.append(blocks)
        
        # Bottleneck
        self.bottleneck = nn.ModuleList([
            ResidualBlock(channels[-1], channels[-1], time_emb_dim, num_groups, dropout),
            AttentionBlock(channels[-1], num_heads, num_groups),
            ResidualBlock(channels[-1], channels[-1], time_emb_dim, num_groups, dropout)
        ])
        
        # Decoder (upsampling path)
        self.decoder = nn.ModuleList()
        
        for level in reversed(range(len(channels))):
            blocks = nn.ModuleList()
            out_ch = channels[level]
            
            # Account for skip connections (concatenation)
            for i in range(num_res_blocks + 1):
                # First block at each level receives skip connection
                # Skip connection doubles the input channels
                if i == 0:
                    if level < len(channels) - 1:
                        # After upsampling from previous level + skip connection
                        in_ch = channels[level + 1] + channels[level]
                    else:
                        # First decoder level, coming from bottleneck + skip
                        in_ch = channels[level] * 2
                else:
                    in_ch = out_ch
                
                blocks.append(ResidualBlock(
                    in_ch, out_ch, time_emb_dim, num_groups, dropout
                ))
                
                # Add attention if at specified resolution
                if level >= len(channels) - len(attention_resolutions):
                    blocks.append(AttentionBlock(out_ch, num_heads, num_groups))
            
            # Add upsampling (except first level)
            if level > 0:
                blocks.append(Upsample(out_ch))
            
            self.decoder.append(blocks)
        
        # Final convolution
        self.final_conv = nn.Sequential(
            nn.GroupNorm(num_groups, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        conditioning: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Noisy input image of shape (B, in_channels, H, W)
            t: Timesteps of shape (B,)
            conditioning: Conditioning maps of shape (B, condition_channels, H, W)
        
        Returns:
            output: Predicted noise/image of shape (B, out_channels, H, W)
        """
        # Time embedding
        time_emb = self.time_embedding(t)
        
        # Concatenate input and conditioning
        h = torch.cat([x, conditioning], dim=1)
        
        # Initial convolution
        h = self.init_conv(h)
        
        # Encoder
        skip_connections = []
        for level_idx, blocks in enumerate(self.encoder):
            for block in blocks:
                if isinstance(block, ResidualBlock):
                    h = block(h, time_emb)
                elif isinstance(block, Downsample):
                    # Save skip connection before downsampling
                    skip_connections.append(h)
                    h = block(h)
                else:
                    h = block(h)
            
            # Save skip for the last encoder level (no downsampling)
            if level_idx == len(self.encoder) - 1:
                skip_connections.append(h)
        
        # Bottleneck
        for block in self.bottleneck:
            if isinstance(block, ResidualBlock):
                h = block(h, time_emb)
            else:
                h = block(h)
        
        # Decoder
        skip_connections = skip_connections[::-1]  # Reverse order
        skip_idx = 0
        
        for blocks in self.decoder:
            for i, block in enumerate(blocks):
                # Add skip connection at the start of each level
                if isinstance(block, ResidualBlock) and i == 0 and skip_idx < len(skip_connections):
                    skip = skip_connections[skip_idx]
                    skip_idx += 1
                    
                    # Resize skip if needed
                    if skip.shape != h.shape:
                        skip = F.interpolate(skip, size=h.shape[2:], mode='nearest')
                    
                    h = torch.cat([h, skip], dim=1)
                
                if isinstance(block, ResidualBlock):
                    h = block(h, time_emb)
                else:
                    h = block(h)
        
        # Final convolution
        output = self.final_conv(h)
        
        return output


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ConditionalUNet(
        in_channels=1,
        out_channels=1,
        condition_channels=3,
        base_channels=64,  # Smaller for testing
        channel_multipliers=[1, 2, 2, 2],
        num_res_blocks=2
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 1, 256, 256).to(device)
    t = torch.randint(0, 1000, (batch_size,)).to(device)
    conditioning = torch.randn(batch_size, 3, 256, 256).to(device)
    
    with torch.no_grad():
        output = model(x, t, conditioning)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("✓ Model test passed!")
