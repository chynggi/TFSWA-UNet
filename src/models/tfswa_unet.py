"""Core TFSWA-UNet architecture definition."""
from __future__ import annotations

from typing import List

import torch
from torch import nn

from .blocks import DownsampleBlock, TFSWABlock, UpsampleBlock


class TFSWAUNet(nn.Module):
    """
    Temporal-Frequency and Shifted Window Attention U-Net for music source separation.
    
    Architecture:
        - Encoder: 4 stages with progressively increasing channels [32, 64, 128, 256]
        - Decoder: 3 stages with progressively decreasing channels [128, 64, 32]
        - Each stage contains multiple TFSWA blocks
        - Skip connections from encoder to decoder
        
    Args:
        in_channels: Number of input channels (typically 2 for complex spectrogram)
        out_channels: Number of output channels (typically 2 for magnitude masks)
        depths: Number of TFSWA blocks at each stage (e.g., [2, 2, 6, 2])
        dims: Channel dimensions at each stage (e.g., [32, 64, 128, 256])
        window_size: Window size for shifted window attention
        shift_size: Shift size for window attention
        num_heads: Number of attention heads
        dropout: Dropout rate (default: 0.0)
        mlp_ratio: MLP expansion ratio (default: 4.0)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depths: List[int],
        dims: List[int],
        window_size: int,
        shift_size: int,
        num_heads: int,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        
        assert len(depths) == len(dims), "depths and dims must have the same length"
        assert len(depths) == 4, "Expected 4 stages (3 encoder + 1 bottleneck)"
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depths = depths
        self.dims = dims
        self.num_stages = len(depths)
        
        # Stem: Initial convolution
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(dims[0]),
            nn.GELU(),
        )
        
        # ===== ENCODER =====
        self.encoder_stages = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        
        for stage_idx in range(self.num_stages - 1):  # 3 encoder stages
            # Create TFSWA blocks for this stage
            blocks = nn.ModuleList()
            for block_idx in range(depths[stage_idx]):
                # Alternate between W-MSA (shift_size=0) and SW-MSA (shift_size>0)
                shift = 0 if block_idx % 2 == 0 else shift_size
                
                blocks.append(
                    TFSWABlock(
                        in_channels=dims[stage_idx],
                        out_channels=dims[stage_idx],
                        window_size=window_size,
                        shift_size=shift,
                        num_heads=num_heads,
                        dropout=dropout,
                        mlp_ratio=mlp_ratio,
                    )
                )
            self.encoder_stages.append(blocks)
            
            # Downsampling layer (except after last encoder stage)
            self.downsample_layers.append(
                DownsampleBlock(dims[stage_idx], dims[stage_idx + 1])
            )
        
        # ===== BOTTLENECK =====
        bottleneck_blocks = nn.ModuleList()
        for block_idx in range(depths[-1]):  # Last depth is for bottleneck
            shift = 0 if block_idx % 2 == 0 else shift_size
            bottleneck_blocks.append(
                TFSWABlock(
                    in_channels=dims[-1],
                    out_channels=dims[-1],
                    window_size=window_size,
                    shift_size=shift,
                    num_heads=num_heads,
                    dropout=dropout,
                    mlp_ratio=mlp_ratio,
                )
            )
        self.bottleneck = bottleneck_blocks
        
        # ===== DECODER =====
        self.upsample_layers = nn.ModuleList()
        self.decoder_stages = nn.ModuleList()
        
        for stage_idx in range(self.num_stages - 2, -1, -1):  # 3 decoder stages (reverse order)
            # Upsampling layer
            self.upsample_layers.append(
                UpsampleBlock(dims[stage_idx + 1], dims[stage_idx])
            )
            
            # Create TFSWA blocks for this decoder stage
            blocks = nn.ModuleList()
            for block_idx in range(depths[stage_idx]):
                shift = 0 if block_idx % 2 == 0 else shift_size
                
                blocks.append(
                    TFSWABlock(
                        in_channels=dims[stage_idx],
                        out_channels=dims[stage_idx],
                        window_size=window_size,
                        shift_size=shift,
                        num_heads=num_heads,
                        dropout=dropout,
                        mlp_ratio=mlp_ratio,
                    )
                )
            self.decoder_stages.append(blocks)
        
        # ===== OUTPUT HEAD =====
        self.output_head = nn.Sequential(
            nn.Conv2d(dims[0], dims[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.GELU(),
            nn.Conv2d(dims[0], out_channels, kernel_size=1),
            nn.Sigmoid(),  # Output masks in [0, 1] range
        )
        
        self._init_weights()
        
    def _init_weights(self) -> None:
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through TFSWA-UNet.
        
        Args:
            x: Input spectrogram of shape (B, C, T, F)
               where B=batch, C=channels (2 for real/imag or magnitude/phase),
               T=time frames, F=frequency bins
               
        Returns:
            Output mask of shape (B, out_channels, T, F)
        """
        # Stem
        x = self.stem(x)
        
        # ===== ENCODER PATH =====
        skip_connections = []
        
        for stage_idx, (encoder_blocks, downsample) in enumerate(
            zip(self.encoder_stages, self.downsample_layers)
        ):
            # Apply TFSWA blocks
            for block in encoder_blocks:
                x = block(x)
            
            # Store for skip connection
            skip_connections.append(x)
            
            # Downsample
            x = downsample(x)
        
        # ===== BOTTLENECK =====
        for block in self.bottleneck:
            x = block(x)
        
        # ===== DECODER PATH =====
        for stage_idx, (upsample, decoder_blocks) in enumerate(
            zip(self.upsample_layers, self.decoder_stages)
        ):
            # Upsample
            x = upsample(x)
            
            # Get corresponding skip connection (in reverse order)
            skip = skip_connections[-(stage_idx + 1)]
            
            # Match skip connection size if needed
            if x.shape[2:] != skip.shape[2:]:
                x = nn.functional.interpolate(
                    x,
                    size=skip.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
            
            # Apply TFSWA blocks with skip connection
            for block_idx, block in enumerate(decoder_blocks):
                # Add skip connection to first block of each decoder stage
                if block_idx == 0:
                    x = block(x, skip=skip)
                else:
                    x = block(x)
        
        # ===== OUTPUT HEAD =====
        output = self.output_head(x)
        
        return output
    
    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> dict:
        """Return model configuration and statistics."""
        return {
            "architecture": "TFSWA-UNet",
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "depths": self.depths,
            "dims": self.dims,
            "num_parameters": self.get_num_parameters(),
            "num_stages": self.num_stages,
        }
