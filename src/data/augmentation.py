"""Data augmentation for music source separation."""
from __future__ import annotations

import random
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torchaudio.transforms as T


class AudioAugmentation(nn.Module):
    """
    Audio augmentation pipeline for training.
    
    Supports:
    - Time stretching
    - Pitch shifting
    - Volume scaling
    - Frequency masking
    - Time masking
    
    Args:
        sample_rate: Audio sample rate
        time_stretch_range: (min, max) tempo change ratio (default: (0.9, 1.1))
        pitch_shift_range: (min, max) pitch shift in semitones (default: (-2, 2))
        volume_range: (min, max) volume scaling (default: (0.7, 1.3))
        freq_mask_param: Maximum frequency bands to mask (default: 80)
        time_mask_param: Maximum time frames to mask (default: 40)
        apply_prob: Probability of applying each augmentation (default: 0.5)
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        time_stretch_range: Tuple[float, float] = (0.9, 1.1),
        pitch_shift_range: Tuple[float, float] = (-2, 2),
        volume_range: Tuple[float, float] = (0.7, 1.3),
        freq_mask_param: int = 80,
        time_mask_param: int = 40,
        apply_prob: float = 0.5,
    ) -> None:
        super().__init__()
        
        self.sample_rate = sample_rate
        self.time_stretch_range = time_stretch_range
        self.pitch_shift_range = pitch_shift_range
        self.volume_range = volume_range
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.apply_prob = apply_prob
        
    def time_stretch(
        self,
        waveform: torch.Tensor,
        rate: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Apply time stretching (tempo change).
        
        Args:
            waveform: (channels, samples)
            rate: Stretch rate (1.0 = no change, >1.0 = faster, <1.0 = slower)
            
        Returns:
            stretched: (channels, new_samples)
        """
        if rate is None:
            rate = random.uniform(*self.time_stretch_range)
        
        if rate == 1.0:
            return waveform
        
        # Simple resampling-based time stretch
        # For better quality, consider using librosa.effects.time_stretch
        original_length = waveform.shape[-1]
        new_length = int(original_length / rate)
        
        stretched = torch.nn.functional.interpolate(
            waveform.unsqueeze(0),
            size=new_length,
            mode='linear',
            align_corners=False,
        ).squeeze(0)
        
        # Pad or trim to original length
        if stretched.shape[-1] < original_length:
            pad = original_length - stretched.shape[-1]
            stretched = torch.nn.functional.pad(stretched, (0, pad))
        elif stretched.shape[-1] > original_length:
            stretched = stretched[..., :original_length]
        
        return stretched
    
    def pitch_shift(
        self,
        waveform: torch.Tensor,
        n_steps: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Apply pitch shifting.
        
        Args:
            waveform: (channels, samples)
            n_steps: Number of semitones to shift (positive = up, negative = down)
            
        Returns:
            shifted: (channels, samples)
        """
        if n_steps is None:
            n_steps = random.uniform(*self.pitch_shift_range)
        
        if n_steps == 0:
            return waveform
        
        # Use torchaudio's pitch shift
        shift_transform = T.PitchShift(
            sample_rate=self.sample_rate,
            n_steps=n_steps,
        )
        
        shifted = torch.stack([
            shift_transform(waveform[ch]) for ch in range(waveform.shape[0])
        ], dim=0)
        
        return shifted
    
    def volume_scale(
        self,
        waveform: torch.Tensor,
        gain: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Apply volume scaling.
        
        Args:
            waveform: (channels, samples)
            gain: Volume gain multiplier
            
        Returns:
            scaled: (channels, samples)
        """
        if gain is None:
            gain = random.uniform(*self.volume_range)
        
        return waveform * gain
    
    def frequency_mask(
        self,
        spectrogram: torch.Tensor,
        mask_param: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Apply frequency masking to spectrogram.
        
        Args:
            spectrogram: (channels, freq_bins, time_frames)
            mask_param: Maximum number of consecutive frequency bins to mask
            
        Returns:
            masked: (channels, freq_bins, time_frames)
        """
        if mask_param is None:
            mask_param = self.freq_mask_param
        
        freq_mask = T.FrequencyMasking(freq_mask_param=mask_param)
        
        masked = torch.stack([
            freq_mask(spectrogram[ch]) for ch in range(spectrogram.shape[0])
        ], dim=0)
        
        return masked
    
    def time_mask(
        self,
        spectrogram: torch.Tensor,
        mask_param: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Apply time masking to spectrogram.
        
        Args:
            spectrogram: (channels, freq_bins, time_frames)
            mask_param: Maximum number of consecutive time frames to mask
            
        Returns:
            masked: (channels, freq_bins, time_frames)
        """
        if mask_param is None:
            mask_param = self.time_mask_param
        
        time_mask = T.TimeMasking(time_mask_param=mask_param)
        
        masked = torch.stack([
            time_mask(spectrogram[ch]) for ch in range(spectrogram.shape[0])
        ], dim=0)
        
        return masked
    
    def forward_waveform(
        self,
        waveform: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply waveform-level augmentations.
        
        Args:
            waveform: (channels, samples)
            
        Returns:
            augmented: (channels, samples)
        """
        # Time stretching
        if random.random() < self.apply_prob:
            waveform = self.time_stretch(waveform)
        
        # Pitch shifting
        if random.random() < self.apply_prob:
            waveform = self.pitch_shift(waveform)
        
        # Volume scaling
        if random.random() < self.apply_prob:
            waveform = self.volume_scale(waveform)
        
        return waveform
    
    def forward_spectrogram(
        self,
        spectrogram: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply spectrogram-level augmentations.
        
        Args:
            spectrogram: (channels, freq_bins, time_frames)
            
        Returns:
            augmented: (channels, freq_bins, time_frames)
        """
        # Frequency masking
        if random.random() < self.apply_prob:
            spectrogram = self.frequency_mask(spectrogram)
        
        # Time masking
        if random.random() < self.apply_prob:
            spectrogram = self.time_mask(spectrogram)
        
        return spectrogram
    
    def forward(
        self,
        waveform: torch.Tensor,
        apply_waveform_aug: bool = True,
        apply_spectrogram_aug: bool = False,
        spectrogram: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply augmentations.
        
        Args:
            waveform: (channels, samples)
            apply_waveform_aug: Whether to apply waveform augmentations
            apply_spectrogram_aug: Whether to apply spectrogram augmentations
            spectrogram: (channels, freq_bins, time_frames) if apply_spectrogram_aug=True
            
        Returns:
            If apply_spectrogram_aug=False:
                augmented_waveform: (channels, samples)
            If apply_spectrogram_aug=True:
                augmented_waveform: (channels, samples)
                augmented_spectrogram: (channels, freq_bins, time_frames)
        """
        if apply_waveform_aug:
            waveform = self.forward_waveform(waveform)
        
        if apply_spectrogram_aug:
            if spectrogram is None:
                raise ValueError("spectrogram must be provided when apply_spectrogram_aug=True")
            spectrogram = self.forward_spectrogram(spectrogram)
            return waveform, spectrogram
        
        return waveform


class MixupAugmentation(nn.Module):
    """
    Mixup augmentation for source separation.
    
    Mixes different tracks with random weights.
    
    Args:
        alpha: Beta distribution parameter (default: 0.4)
        apply_prob: Probability of applying mixup (default: 0.5)
    """
    
    def __init__(self, alpha: float = 0.4, apply_prob: float = 0.5) -> None:
        super().__init__()
        self.alpha = alpha
        self.apply_prob = apply_prob
    
    def forward(
        self,
        mixture1: torch.Tensor,
        targets1: Dict[str, torch.Tensor],
        mixture2: torch.Tensor,
        targets2: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply mixup between two samples.
        
        Args:
            mixture1, mixture2: (channels, samples)
            targets1, targets2: Dict of stem_name -> (channels, samples)
            
        Returns:
            mixed_mixture: (channels, samples)
            mixed_targets: Dict of stem_name -> (channels, samples)
        """
        if random.random() > self.apply_prob:
            return mixture1, targets1
        
        # Sample mixing weight
        lam = random.betavariate(self.alpha, self.alpha)
        
        # Mix mixtures
        mixed_mixture = lam * mixture1 + (1 - lam) * mixture2
        
        # Mix targets
        mixed_targets = {}
        for stem_name in targets1:
            mixed_targets[stem_name] = lam * targets1[stem_name] + (1 - lam) * targets2[stem_name]
        
        return mixed_mixture, mixed_targets
