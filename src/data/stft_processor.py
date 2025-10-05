"""STFT/ISTFT preprocessing pipeline for spectrograms."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torchaudio.transforms as T


class STFTProcessor(nn.Module):
    """
    STFT/ISTFT processor for converting between waveform and spectrogram.
    
    Supports both magnitude-phase and real-imaginary representations.
    
    Args:
        n_fft: FFT size (default: 4096)
        hop_length: Hop length in samples (default: 1024)
        win_length: Window length (default: None, uses n_fft)
        window: Window function (default: 'hann')
        center: Whether to center the window (default: True)
        normalized: Whether to normalize STFT (default: False)
        onesided: Whether to use one-sided FFT (default: True)
        return_complex: Whether to return complex tensor (default: True)
        sample_rate: Audio sample rate (default: 44100)
    """
    
    def __init__(
        self,
        n_fft: int = 4096,
        hop_length: int = 1024,
        win_length: Optional[int] = None,
        window: str = 'hann',
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
        return_complex: bool = True,
        sample_rate: int = 44100,
    ) -> None:
        super().__init__()
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.window = window
        self.center = center
        self.normalized = normalized
        self.onesided = onesided
        self.return_complex = return_complex
        self.sample_rate = sample_rate
        
        # Create STFT transform
        self.stft_transform = T.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=self.win_length,
            window_fn=self._get_window_fn(),
            power=None,  # Return complex spectrogram
            normalized=normalized,
            center=center,
            onesided=onesided,
        )
        
        # Create inverse STFT
        self.istft_transform = T.InverseSpectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=self.win_length,
            window_fn=self._get_window_fn(),
            normalized=normalized,
            center=center,
            onesided=onesided,
        )
        
    def _get_window_fn(self):
        """Get window function."""
        if self.window == 'hann':
            return torch.hann_window
        elif self.window == 'hamming':
            return torch.hamming_window
        elif self.window == 'blackman':
            return torch.blackman_window
        else:
            raise ValueError(f"Unknown window function: {self.window}")
    
    def stft(
        self,
        waveform: torch.Tensor,
        return_magnitude_phase: bool = False,
    ) -> torch.Tensor:
        """
        Compute STFT.
        
        Args:
            waveform: (B, channels, samples) or (channels, samples)
            return_magnitude_phase: If True, return magnitude and phase separately
            
        Returns:
            If return_magnitude_phase=False:
                complex_spec: (B, channels, freq_bins, time_frames) complex tensor
            If return_magnitude_phase=True:
                magnitude: (B, channels, freq_bins, time_frames)
                phase: (B, channels, freq_bins, time_frames)
        """
        # Handle 2D input
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False
        
        B, C, T = waveform.shape
        
        # Compute STFT for each channel
        specs = []
        for b in range(B):
            channel_specs = []
            for c in range(C):
                spec = self.stft_transform(waveform[b, c])
                channel_specs.append(spec)
            specs.append(torch.stack(channel_specs, dim=0))
        
        complex_spec = torch.stack(specs, dim=0)  # (B, C, freq_bins, time_frames)
        
        if squeeze_batch:
            complex_spec = complex_spec.squeeze(0)
        
        if return_magnitude_phase:
            magnitude = torch.abs(complex_spec)
            phase = torch.angle(complex_spec)
            return magnitude, phase
        
        return complex_spec
    
    def istft(
        self,
        complex_spec: Optional[torch.Tensor] = None,
        magnitude: Optional[torch.Tensor] = None,
        phase: Optional[torch.Tensor] = None,
        length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute inverse STFT.
        
        Args:
            complex_spec: (B, channels, freq_bins, time_frames) complex tensor
            magnitude: (B, channels, freq_bins, time_frames) if using mag-phase
            phase: (B, channels, freq_bins, time_frames) if using mag-phase
            length: Target length of output waveform
            
        Returns:
            waveform: (B, channels, samples)
        """
        # Handle magnitude-phase input
        if complex_spec is None:
            if magnitude is None or phase is None:
                raise ValueError("Either complex_spec or (magnitude, phase) must be provided")
            complex_spec = magnitude * torch.exp(1j * phase)
        
        # Handle 3D input
        if complex_spec.ndim == 3:
            complex_spec = complex_spec.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False
        
        B, C, F, T = complex_spec.shape
        
        # Compute ISTFT for each channel
        waveforms = []
        for b in range(B):
            channel_waveforms = []
            for c in range(C):
                waveform = self.istft_transform(complex_spec[b, c], length=length)
                channel_waveforms.append(waveform)
            waveforms.append(torch.stack(channel_waveforms, dim=0))
        
        waveform = torch.stack(waveforms, dim=0)  # (B, C, samples)
        
        if squeeze_batch:
            waveform = waveform.squeeze(0)
        
        return waveform
    
    def to_model_input(self, complex_spec: torch.Tensor) -> torch.Tensor:
        """
        Convert complex spectrogram to model input format.
        
        Args:
            complex_spec: (B, channels, freq_bins, time_frames) complex tensor
            
        Returns:
            model_input: (B, 2*channels, freq_bins, time_frames) 
                        where 2*channels = [real, imag] for each channel
        """
        real = torch.real(complex_spec)
        imag = torch.imag(complex_spec)
        
        # Interleave real and imaginary parts
        # (B, C, F, T) -> (B, 2*C, F, T)
        model_input = torch.cat([real, imag], dim=1)
        
        return model_input
    
    def from_model_output(
        self,
        model_output: torch.Tensor,
        mixture_spec: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert model output (masks) to complex spectrogram.
        
        Args:
            model_output: (B, n_stems*channels, freq_bins, time_frames) masks in [0, 1]
            mixture_spec: (B, channels, freq_bins, time_frames) complex mixture spectrogram
            
        Returns:
            separated_specs: (B, n_stems, channels, freq_bins, time_frames) complex tensors
        """
        B, _, F, T = model_output.shape
        n_channels = mixture_spec.shape[1]
        n_stems = model_output.shape[1] // n_channels
        
        # Reshape masks: (B, n_stems*C, F, T) -> (B, n_stems, C, F, T)
        masks = model_output.view(B, n_stems, n_channels, F, T)
        
        # Apply masks to mixture
        separated_specs = []
        for stem_idx in range(n_stems):
            stem_mask = masks[:, stem_idx]  # (B, C, F, T)
            stem_spec = mixture_spec * stem_mask  # Element-wise multiplication
            separated_specs.append(stem_spec)
        
        separated_specs = torch.stack(separated_specs, dim=1)  # (B, n_stems, C, F, T)
        
        return separated_specs


class SpectrogramNormalizer(nn.Module):
    """
    Normalize spectrograms for training.
    
    Supports instance normalization per frequency bin.
    
    Args:
        mode: Normalization mode ('instance', 'batch', 'none')
        eps: Small constant for numerical stability
    """
    
    def __init__(self, mode: str = 'instance', eps: float = 1e-8) -> None:
        super().__init__()
        self.mode = mode
        self.eps = eps
        
    def forward(
        self,
        spec: torch.Tensor,
        return_stats: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Normalize spectrogram.
        
        Args:
            spec: (B, C, F, T) spectrogram
            return_stats: Whether to return mean and std for denormalization
            
        Returns:
            normalized_spec: (B, C, F, T)
            mean: (B, C, F, 1) if return_stats=True
            std: (B, C, F, 1) if return_stats=True
        """
        if self.mode == 'none':
            if return_stats:
                return spec, torch.zeros_like(spec[:, :, :, :1]), torch.ones_like(spec[:, :, :, :1])
            return spec
        
        if self.mode == 'instance':
            # Normalize per frequency bin across time
            mean = spec.mean(dim=-1, keepdim=True)  # (B, C, F, 1)
            std = spec.std(dim=-1, keepdim=True) + self.eps  # (B, C, F, 1)
        elif self.mode == 'batch':
            # Normalize across batch
            mean = spec.mean(dim=(0, 1, 2, 3), keepdim=True)  # (1, 1, 1, 1)
            std = spec.std(dim=(0, 1, 2, 3), keepdim=True) + self.eps  # (1, 1, 1, 1)
        else:
            raise ValueError(f"Unknown normalization mode: {self.mode}")
        
        normalized_spec = (spec - mean) / std
        
        if return_stats:
            return normalized_spec, mean, std
        return normalized_spec
    
    def denormalize(
        self,
        normalized_spec: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        """
        Denormalize spectrogram.
        
        Args:
            normalized_spec: (B, C, F, T)
            mean: (B, C, F, 1)
            std: (B, C, F, 1)
            
        Returns:
            spec: (B, C, F, T)
        """
        return normalized_spec * std + mean
