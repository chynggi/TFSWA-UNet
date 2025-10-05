"""Attention layers used in TFSWA blocks."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention helper."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = dim ** -0.5

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, v)
        return output, weights


class MultiHeadAttention(nn.Module):
    """Multi-head attention module for TSA/FSA."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (B, N, C) where N is sequence length
            mask: Optional attention mask
            
        Returns:
            output: Attention output of shape (B, N, C)
            weights: Attention weights of shape (B, num_heads, N, N)
        """
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
            
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        
        # Apply attention to values
        output = torch.matmul(weights, v)
        output = output.transpose(1, 2).reshape(B, N, C)
        output = self.proj(output)
        
        return output, weights


class TemporalSequenceAttention(nn.Module):
    """
    Temporal Sequence Attention (TSA) module.
    
    Applies multi-head self-attention along the temporal dimension
    of the spectrogram to capture long-range temporal dependencies.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        
        # MLP (Feed-forward network)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, T, F) where
               B = batch, C = channels, T = time, F = frequency
               
        Returns:
            Output tensor of same shape (B, C, T, F)
        """
        B, C, T, F = x.shape
        
        # Process each frequency bin independently
        # Reshape to (B*F, T, C) to apply temporal attention
        x_reshaped = x.permute(0, 3, 2, 1).reshape(B * F, T, C)
        
        # Self-attention with residual connection
        normed = self.norm1(x_reshaped)
        attn_out, _ = self.attn(normed)
        x_reshaped = x_reshaped + attn_out
        
        # MLP with residual connection
        x_reshaped = x_reshaped + self.mlp(self.norm2(x_reshaped))
        
        # Reshape back to (B, C, T, F)
        output = x_reshaped.reshape(B, F, T, C).permute(0, 3, 2, 1)
        
        return output


class FrequencySequenceAttention(nn.Module):
    """
    Frequency Sequence Attention (FSA) module.
    
    Applies multi-head self-attention along the frequency dimension
    of the spectrogram to capture frequency-domain relationships.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        
        # MLP (Feed-forward network)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, T, F) where
               B = batch, C = channels, T = time, F = frequency
               
        Returns:
            Output tensor of same shape (B, C, T, F)
        """
        B, C, T, F = x.shape
        
        # Process each time frame independently
        # Reshape to (B*T, F, C) to apply frequency attention
        x_reshaped = x.permute(0, 2, 3, 1).reshape(B * T, F, C)
        
        # Self-attention with residual connection
        normed = self.norm1(x_reshaped)
        attn_out, _ = self.attn(normed)
        x_reshaped = x_reshaped + attn_out
        
        # MLP with residual connection
        x_reshaped = x_reshaped + self.mlp(self.norm2(x_reshaped))
        
        # Reshape back to (B, C, T, F)
        output = x_reshaped.reshape(B, T, F, C).permute(0, 3, 1, 2)
        
        return output


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Partition input into non-overlapping windows.
    
    Args:
        x: Input tensor of shape (B, C, H, W)
        window_size: Window size
        
    Returns:
        windows: Tensor of shape (num_windows*B, window_size, window_size, C)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """
    Reverse window partition to get original tensor.
    
    Args:
        windows: Window tensor of shape (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image
        
    Returns:
        x: Tensor of shape (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    C = windows.shape[-1]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, C)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    x = x.view(B, C, H, W)
    return x


class ShiftedWindowAttention(nn.Module):
    """
    Shifted Window based Multi-head Self-Attention (SW-MSA) module.
    
    Implements the shifted window mechanism from Swin Transformer,
    adapted for 2D spectrogram inputs to capture local correlations.
    """

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        shift_size: int = 0,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.shift_size = shift_size
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        
        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
        )
        
        # Create attention mask for shifted window
        if self.shift_size > 0:
            # Calculate attention mask for SW-MSA
            H = W = self.window_size * 8  # Assume reasonable feature map size
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
                    
            mask_windows = window_partition(img_mask.permute(0, 3, 1, 2), self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
            attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
            self.register_buffer("attn_mask", attn_mask)
        else:
            self.attn_mask = None
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor of same shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Ensure dimensions are divisible by window_size
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            _, _, H_pad, W_pad = x.shape
        else:
            H_pad, W_pad = H, W
            
        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        else:
            shifted_x = x
            
        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # (num_windows*B, window_size, window_size, C)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # Attention
        normed = self.norm1(x_windows)
        
        # Apply attention without mask (simplified for now)
        # TODO: Properly implement attention mask for shifted windows
        attn_out, _ = self.attn(normed)
            
        x_windows = x_windows + attn_out
        
        # MLP
        x_windows = x_windows + self.mlp(self.norm2(x_windows))
        
        # Merge windows
        x_windows = x_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(x_windows, self.window_size, H_pad, W_pad)
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        else:
            x = shifted_x
            
        # Remove padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H, :W]
            
        return x
