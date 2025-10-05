"""
Inference pipeline for music source separation.

Handles full-track processing with overlap-add and gradient checkpointing.
"""

import torch
import torch.nn as nn
import torchaudio
from typing import Dict, Optional, Tuple, List
import numpy as np
from pathlib import Path
import warnings

from ..data.stft_processor import STFTProcessor, SpectrogramNormalizer
from ..models.tfswa_unet import TFSWAUNet


class SourceSeparator:
    """
    High-level interface for source separation inference.
    """
    
    def __init__(
        self,
        model: nn.Module,
        stft_processor: STFTProcessor,
        normalizer: Optional[SpectrogramNormalizer] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        use_amp: bool = True,
        segment_length: float = 10.0,
        overlap: float = 0.25,
    ):
        """
        Args:
            model: Trained separation model
            stft_processor: STFT/ISTFT processor
            normalizer: Spectrogram normalizer (optional)
            device: Device to run inference on
            use_amp: Whether to use automatic mixed precision
            segment_length: Length of segments in seconds
            overlap: Overlap ratio between segments (0-1)
        """
        self.model = model.to(device)
        self.model.eval()
        
        self.stft_processor = stft_processor
        self.normalizer = normalizer
        self.device = device
        self.use_amp = use_amp
        
        self.segment_length = segment_length
        self.overlap = overlap
        
        # Compute segment parameters
        self.sample_rate = stft_processor.sample_rate
        self.segment_samples = int(segment_length * self.sample_rate)
        self.hop_samples = int(self.segment_samples * (1 - overlap))
    
    @torch.no_grad()
    def separate(
        self,
        audio: torch.Tensor,
        stem_names: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Separate sources from mixture audio.
        
        Args:
            audio: Input mixture [channels, time] or [time]
            stem_names: Names of output stems (default: ['vocals', 'other'])
            
        Returns:
            Dictionary {stem_name: separated_audio [channels, time]}
        """
        if stem_names is None:
            stem_names = ['vocals', 'other']
        
        # Ensure 2D tensor [channels, time]
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # [1, time]
        
        # Convert to mono if needed (average channels)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)  # [1, time]
        
        # Move to device
        audio = audio.to(self.device)
        
        # Process full track with overlap-add
        if audio.shape[1] > self.segment_samples:
            separated = self._separate_long(audio, stem_names)
        else:
            separated = self._separate_segment(audio, stem_names)
        
        return separated
    
    def _separate_segment(
        self,
        audio: torch.Tensor,
        stem_names: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Separate a single segment.
        
        Args:
            audio: Input mixture [1, time]
            stem_names: Names of output stems
            
        Returns:
            Dictionary {stem_name: separated_audio [1, time]}
        """
        # STFT - audio is [1, time] -> complex_spec is [1, freq, time] (no channel dim!)
        complex_spec = self.stft_processor.stft(audio)  # complex [1, freq, time]
        
        # Add channel dimension [1, 1, freq, time]
        if complex_spec.dim() == 3:
            complex_spec = complex_spec.unsqueeze(1)
        
        # Convert to model input [1, 2, freq, time] where 2 = [real, imag]
        model_input = self.stft_processor.to_model_input(complex_spec)
        
        # Normalize
        if self.normalizer is not None:
            model_input, mean, std = self.normalizer.forward(model_input, return_stats=True)
        
        # Inference with AMP
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            masks = self.model(model_input)  # [1, num_stems, freq, time]
        
        # Denormalize masks (if needed)
        if self.normalizer is not None:
            masks = self.normalizer.denormalize(masks, mean, std)
        
        # Apply masks and reconstruct
        separated = {}
        num_stems = masks.shape[1]
        
        for i, stem_name in enumerate(stem_names[:num_stems]):
            # Get mask [1, 1, freq, time]
            mask = masks[:, i:i+1]
            
            # Apply mask to complex spectrogram
            # Expand mask to match complex_spec channels
            masked_spec = complex_spec * mask  # [1, 1, freq, time]
            
            # Reconstruct audio
            separated_audio = self.stft_processor.istft(
                complex_spec=masked_spec
            )  # [1, 1, time]
            
            # Remove channel dimension [1, time]
            separated_audio = separated_audio.squeeze(1)
            
            separated[stem_name] = separated_audio
        
        return separated
    
    def _separate_long(
        self,
        audio: torch.Tensor,
        stem_names: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Separate long audio with overlap-add.
        
        Args:
            audio: Input mixture [1, time]
            stem_names: Names of output stems
            
        Returns:
            Dictionary {stem_name: separated_audio [1, time]}
        """
        total_length = audio.shape[1]
        
        # Initialize output buffers
        separated = {
            name: torch.zeros(1, total_length, device=self.device)
            for name in stem_names
        }
        normalization = torch.zeros(total_length, device=self.device)
        
        # Create window for overlap-add
        window = self._create_window(self.segment_samples)
        
        # Process segments
        num_segments = (total_length - self.segment_samples) // self.hop_samples + 1
        
        for i in range(num_segments):
            start = i * self.hop_samples
            end = start + self.segment_samples
            
            # Handle last segment
            if end > total_length:
                end = total_length
                start = max(0, end - self.segment_samples)
            
            # Extract segment
            segment = audio[:, start:end]
            
            # Pad if needed
            if segment.shape[1] < self.segment_samples:
                pad_length = self.segment_samples - segment.shape[1]
                segment = torch.nn.functional.pad(segment, (0, pad_length))
            
            # Separate segment
            separated_segment = self._separate_segment(segment, stem_names)
            
            # Apply window and accumulate
            actual_length = min(end - start, separated_segment[stem_names[0]].shape[1])
            segment_window = window[:actual_length].to(self.device)
            
            for stem_name in stem_names:
                separated[stem_name][:, start:start+actual_length] += (
                    separated_segment[stem_name][:, :actual_length] * segment_window
                )
            
            normalization[start:start+actual_length] += segment_window
        
        # Normalize
        normalization = torch.clamp(normalization, min=1e-8)
        for stem_name in stem_names:
            separated[stem_name] = separated[stem_name] / normalization.unsqueeze(0)
        
        return separated
    
    def _create_window(self, length: int) -> torch.Tensor:
        """
        Create Hann window for overlap-add.
        
        Args:
            length: Window length
            
        Returns:
            Window tensor [length]
        """
        return torch.hann_window(length)
    
    def separate_file(
        self,
        input_path: str,
        output_dir: str,
        stem_names: Optional[List[str]] = None,
        save_mixture: bool = False
    ) -> Dict[str, str]:
        """
        Separate sources from audio file.
        
        Args:
            input_path: Path to input audio file
            output_dir: Directory to save separated sources
            stem_names: Names of output stems
            save_mixture: Whether to save mixture copy
            
        Returns:
            Dictionary {stem_name: output_path}
        """
        # Load audio
        audio, sr = torchaudio.load(input_path)
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)
        
        # Separate
        separated = self.separate(audio, stem_names)
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get input filename
        input_name = Path(input_path).stem
        
        # Save separated sources
        output_paths = {}
        for stem_name, stem_audio in separated.items():
            output_path = output_dir / f"{input_name}_{stem_name}.wav"
            torchaudio.save(
                str(output_path),
                stem_audio.cpu(),
                self.sample_rate
            )
            output_paths[stem_name] = str(output_path)
        
        # Save mixture if requested
        if save_mixture:
            mixture_path = output_dir / f"{input_name}_mixture.wav"
            torchaudio.save(
                str(mixture_path),
                audio,
                self.sample_rate
            )
            output_paths['mixture'] = str(mixture_path)
        
        return output_paths


def load_separator_from_checkpoint(
    checkpoint_path: str,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    **separator_kwargs
) -> SourceSeparator:
    """
    Load separator from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        **separator_kwargs: Additional arguments for SourceSeparator
        
    Returns:
        Initialized SourceSeparator
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model config
    if 'config' in checkpoint:
        model_config = checkpoint['config']['model']
    else:
        # Default config
        model_config = {
            'in_channels': 2,
            'out_channels': 2,
            'depths': [2, 2, 6, 2],
            'dims': [32, 64, 128, 256],
            'window_size': 8,
            'shift_size': 4,
            'num_heads': 8,
        }
    
    # Create model
    model = TFSWAUNet(**model_config)
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Create STFT processor
    stft_config = checkpoint.get('config', {}).get('data', {})
    stft_processor = STFTProcessor(
        n_fft=stft_config.get('n_fft', 2048),
        hop_length=stft_config.get('hop_length', 512),
        window_fn=torch.hann_window,
        sample_rate=stft_config.get('sample_rate', 44100)
    )
    
    # Create normalizer
    normalizer = SpectrogramNormalizer(mode='instance')
    
    # Create separator
    separator = SourceSeparator(
        model=model,
        stft_processor=stft_processor,
        normalizer=normalizer,
        device=device,
        **separator_kwargs
    )
    
    return separator


class BatchSeparator:
    """
    Batch processing for multiple audio files.
    """
    
    def __init__(
        self,
        separator: SourceSeparator,
        batch_size: int = 1
    ):
        """
        Args:
            separator: Initialized SourceSeparator
            batch_size: Number of files to process in parallel
        """
        self.separator = separator
        self.batch_size = batch_size
    
    def separate_files(
        self,
        input_paths: List[str],
        output_dir: str,
        stem_names: Optional[List[str]] = None,
        verbose: bool = True
    ) -> Dict[str, Dict[str, str]]:
        """
        Separate multiple audio files.
        
        Args:
            input_paths: List of input audio file paths
            output_dir: Directory to save separated sources
            stem_names: Names of output stems
            verbose: Whether to show progress
            
        Returns:
            Dictionary {input_path: {stem_name: output_path}}
        """
        from tqdm import tqdm
        
        all_output_paths = {}
        
        iterator = tqdm(input_paths) if verbose else input_paths
        
        for input_path in iterator:
            if verbose:
                iterator.set_description(f"Processing {Path(input_path).name}")
            
            try:
                output_paths = self.separator.separate_file(
                    input_path=input_path,
                    output_dir=output_dir,
                    stem_names=stem_names
                )
                all_output_paths[input_path] = output_paths
            except Exception as e:
                warnings.warn(f"Failed to process {input_path}: {e}")
                all_output_paths[input_path] = None
        
        return all_output_paths
