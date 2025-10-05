"""Building blocks for the TFSWA-UNet model."""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .attention import (
    FrequencySequenceAttention,
    ShiftedWindowAttention,
    TemporalSequenceAttention,
)


class TFSWABlock(nn.Module):
    """
    Temporal-Frequency and Shifted Window Attention Block.
    
    Combines three attention mechanisms:
    1. TSA (Temporal Sequence Attention) - captures temporal dependencies
    2. FSA (Frequency Sequence Attention) - captures frequency relationships
    3. Shifted Window Attention - captures local spatial correlations
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        window_size: Window size for shifted window attention
        shift_size: Shift size for window attention (0 for W-MSA, >0 for SW-MSA)
        num_heads: Number of attention heads
        dropout: Dropout rate (default: 0.0)
        mlp_ratio: Ratio of MLP hidden dim to embedding dim (default: 4.0)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        window_size: int,
        shift_size: int,
        num_heads: int,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        
        # Input projection to match dimensions
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )
        
        # Temporal Sequence Attention
        self.tsa = TemporalSequenceAttention(
            dim=out_channels,
            num_heads=num_heads,
            dropout=dropout,
            mlp_ratio=mlp_ratio,
        )
        
        # Frequency Sequence Attention
        self.fsa = FrequencySequenceAttention(
            dim=out_channels,
            num_heads=num_heads,
            dropout=dropout,
            mlp_ratio=mlp_ratio,
        )
        
        # Shifted Window Attention
        self.swa = ShiftedWindowAttention(
            dim=out_channels,
            window_size=window_size,
            num_heads=num_heads,
            shift_size=shift_size,
            dropout=dropout,
            mlp_ratio=mlp_ratio,
        )
        
        # Feature fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        
        # Skip connection projection (if channels don't match)
        self.skip_proj = None
        if in_channels != out_channels:
            self.skip_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        skip: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through TFSWA block.
        
        Args:
            x: Input tensor of shape (B, C_in, H, W)
            skip: Optional skip connection from encoder (B, C_in, H, W)
            
        Returns:
            Output tensor of shape (B, C_out, H, W)
        """
        # Store input for residual connection
        identity = x
        
        # Input projection
        x = self.input_proj(x)
        
        # Apply three types of attention in parallel
        tsa_out = self.tsa(x)  # Temporal attention
        fsa_out = self.fsa(x)  # Frequency attention
        swa_out = self.swa(x)  # Shifted window attention
        
        # Concatenate attention outputs
        combined = torch.cat([tsa_out, fsa_out, swa_out], dim=1)
        
        # Fuse features
        features = self.fusion(combined)
        
        # Add residual connection
        if self.skip_proj is not None:
            identity = self.skip_proj(identity)
        features = features + identity
        
        # Add skip connection from encoder (if provided)
        if skip is not None:
            if skip.shape != features.shape:
                # Adjust skip connection if dimensions don't match
                skip = nn.functional.interpolate(
                    skip, 
                    size=features.shape[2:], 
                    mode='bilinear', 
                    align_corners=False
                )
                if skip.shape[1] != features.shape[1]:
                    skip_conv = nn.Conv2d(skip.shape[1], features.shape[1], kernel_size=1).to(features.device)
                    skip = skip_conv(skip)
            features = features + skip
            
        return features


class DownsampleBlock(nn.Module):
    """Downsampling block for encoder path."""
    
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample(x)


class UpsampleBlock(nn.Module):
    """Upsampling block for decoder path."""
    
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsample(x)
