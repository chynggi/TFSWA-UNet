"""Loss functions for music source separation."""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class L1SpectrogramLoss(nn.Module):
    """
    L1 loss on magnitude spectrograms.
    
    Args:
        reduction: Reduction method ('mean', 'sum', 'none')
    """
    
    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        pred_spec: torch.Tensor,
        target_spec: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute L1 loss.
        
        Args:
            pred_spec: (B, F, T) or (B, C, F, T) predicted spectrogram (can be complex)
            target_spec: (B, F, T) or (B, C, F, T) target spectrogram (can be complex)
            
        Returns:
            loss: Scalar or (B,) tensor depending on reduction
        """
        # Compute magnitude if complex
        if torch.is_complex(pred_spec):
            pred_spec = torch.abs(pred_spec)
        if torch.is_complex(target_spec):
            target_spec = torch.abs(target_spec)
        
        # Ensure tensors are float (not half during backprop)
        pred_spec = pred_spec.float()
        target_spec = target_spec.float()
        
        loss = F.l1_loss(pred_spec, target_spec, reduction=self.reduction)
        return loss


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss.
    
    Computes STFT loss at multiple resolutions to capture both
    fine-grained and coarse-grained spectral information.
    
    Args:
        fft_sizes: List of FFT sizes (default: [2048, 1024, 512])
        hop_sizes: List of hop sizes (default: [512, 256, 128])
        win_lengths: List of window lengths (default: [2048, 1024, 512])
        magnitude_weight: Weight for magnitude loss (default: 1.0)
        log_magnitude_weight: Weight for log magnitude loss (default: 1.0)
    """
    
    def __init__(
        self,
        fft_sizes: List[int] = [2048, 1024, 512],
        hop_sizes: List[int] = [512, 256, 128],
        win_lengths: List[int] = [2048, 1024, 512],
        magnitude_weight: float = 1.0,
        log_magnitude_weight: float = 1.0,
    ) -> None:
        super().__init__()
        
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths), \
            "fft_sizes, hop_sizes, and win_lengths must have the same length"
        
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.magnitude_weight = magnitude_weight
        self.log_magnitude_weight = log_magnitude_weight
        
    def _stft(
        self,
        x: torch.Tensor,
        fft_size: int,
        hop_size: int,
        win_length: int,
    ) -> torch.Tensor:
        """Compute STFT."""
        # x: (B, samples)
        window = torch.hann_window(win_length, device=x.device)
        
        spec = torch.stft(
            x,
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=win_length,
            window=window,
            center=True,
            return_complex=True,
        )
        
        return spec
    
    def _magnitude_loss(
        self,
        pred_mag: torch.Tensor,
        target_mag: torch.Tensor,
    ) -> torch.Tensor:
        """Compute magnitude loss."""
        return F.l1_loss(pred_mag, target_mag)
    
    def _log_magnitude_loss(
        self,
        pred_mag: torch.Tensor,
        target_mag: torch.Tensor,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        """Compute log magnitude loss."""
        pred_log_mag = torch.log(pred_mag + eps)
        target_log_mag = torch.log(target_mag + eps)
        return F.l1_loss(pred_log_mag, target_log_mag)
    
    def forward(
        self,
        pred_audio: torch.Tensor,
        target_audio: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute multi-resolution STFT loss.
        
        Args:
            pred_audio: (B, channels, samples) predicted audio
            target_audio: (B, channels, samples) target audio
            
        Returns:
            loss: Scalar loss
        """
        total_loss = 0.0
        
        # Flatten channels into batch for STFT
        B, C, T = pred_audio.shape
        pred_audio = pred_audio.reshape(B * C, T)
        target_audio = target_audio.reshape(B * C, T)
        
        for fft_size, hop_size, win_length in zip(
            self.fft_sizes, self.hop_sizes, self.win_lengths
        ):
            # Compute STFT
            pred_spec = self._stft(pred_audio, fft_size, hop_size, win_length)
            target_spec = self._stft(target_audio, fft_size, hop_size, win_length)
            
            # Compute magnitudes
            pred_mag = torch.abs(pred_spec)
            target_mag = torch.abs(target_spec)
            
            # Magnitude loss
            if self.magnitude_weight > 0:
                mag_loss = self._magnitude_loss(pred_mag, target_mag)
                total_loss += self.magnitude_weight * mag_loss
            
            # Log magnitude loss
            if self.log_magnitude_weight > 0:
                log_mag_loss = self._log_magnitude_loss(pred_mag, target_mag)
                total_loss += self.log_magnitude_weight * log_mag_loss
        
        # Average over resolutions
        total_loss = total_loss / len(self.fft_sizes)
        
        return total_loss


class SourceSeparationLoss(nn.Module):
    """
    Combined loss for source separation.
    
    Combines L1 spectrogram loss and multi-resolution STFT loss.
    
    Args:
        l1_weight: Weight for L1 spectrogram loss (default: 1.0)
        mrstft_weight: Weight for multi-resolution STFT loss (default: 1.0)
        use_l1: Whether to use L1 loss (default: True)
        use_mrstft: Whether to use multi-resolution STFT loss (default: True)
        fft_sizes: FFT sizes for multi-resolution STFT
        hop_sizes: Hop sizes for multi-resolution STFT
        win_lengths: Window lengths for multi-resolution STFT
    """
    
    def __init__(
        self,
        l1_weight: float = 1.0,
        mrstft_weight: float = 0.5,
        use_l1: bool = True,
        use_mrstft: bool = True,
        fft_sizes: List[int] = [2048, 1024, 512],
        hop_sizes: List[int] = [512, 256, 128],
        win_lengths: List[int] = [2048, 1024, 512],
    ) -> None:
        super().__init__()
        
        self.l1_weight = l1_weight
        self.mrstft_weight = mrstft_weight
        self.use_l1 = use_l1
        self.use_mrstft = use_mrstft
        
        if use_l1:
            self.l1_loss = L1SpectrogramLoss()
        
        if use_mrstft:
            self.mrstft_loss = MultiResolutionSTFTLoss(
                fft_sizes=fft_sizes,
                hop_sizes=hop_sizes,
                win_lengths=win_lengths,
            )
    
    def forward(
        self,
        pred_specs: Dict[str, torch.Tensor],
        target_specs: Dict[str, torch.Tensor],
        pred_audios: Optional[Dict[str, torch.Tensor]] = None,
        target_audios: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            pred_specs: Dict of stem_name -> (B, C, F, T) predicted spectrograms
            target_specs: Dict of stem_name -> (B, C, F, T) target spectrograms
            pred_audios: Dict of stem_name -> (B, C, samples) predicted audio (for MRSTFT loss)
            target_audios: Dict of stem_name -> (B, C, samples) target audio (for MRSTFT loss)
            
        Returns:
            loss_dict: Dict containing 'total_loss' and individual losses
        """
        loss_dict = {}
        total_loss = 0.0
        
        # L1 spectrogram loss
        if self.use_l1:
            l1_loss_sum = 0.0
            for stem_name in pred_specs:
                stem_loss = self.l1_loss(pred_specs[stem_name], target_specs[stem_name])
                l1_loss_sum += stem_loss
                loss_dict[f'l1_{stem_name}'] = stem_loss
            
            l1_loss_avg = l1_loss_sum / len(pred_specs)
            loss_dict['l1_loss'] = l1_loss_avg
            total_loss += self.l1_weight * l1_loss_avg
        
        # Multi-resolution STFT loss
        if self.use_mrstft and pred_audios is not None and target_audios is not None:
            mrstft_loss_sum = 0.0
            for stem_name in pred_audios:
                stem_loss = self.mrstft_loss(pred_audios[stem_name], target_audios[stem_name])
                mrstft_loss_sum += stem_loss
                loss_dict[f'mrstft_{stem_name}'] = stem_loss
            
            mrstft_loss_avg = mrstft_loss_sum / len(pred_audios)
            loss_dict['mrstft_loss'] = mrstft_loss_avg
            total_loss += self.mrstft_weight * mrstft_loss_avg
        
        loss_dict['total_loss'] = total_loss
        
        return loss_dict


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using pre-trained features (optional).
    
    This is a placeholder for future implementation.
    Could use VGGish or other pre-trained audio models.
    """
    
    def __init__(self) -> None:
        super().__init__()
        # TODO: Implement perceptual loss with pre-trained model
        raise NotImplementedError("Perceptual loss not yet implemented")
    
    def forward(
        self,
        pred_audio: torch.Tensor,
        target_audio: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError
