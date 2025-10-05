"""MUSDB18 dataset loader with flexible stem selection."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

import musdb


class MUSDB18Dataset(Dataset):
    """
    MUSDB18 dataset for music source separation.
    
    Supports flexible stem selection:
    - All stems: ['vocals', 'drums', 'bass', 'other']
    - Binary: ['vocals', 'other'] where 'other' = drums + bass + other
    - Custom combinations
    
    Args:
        root: Path to MUSDB18 dataset
        split: 'train', 'valid', or 'test'
        target_stems: List of stems to separate. 
                     If ['vocals', 'other'], will combine drums+bass+other
        sample_rate: Target sample rate (default: 44100)
        segment_seconds: Length of audio segments in seconds (default: 10)
        random_segments: Whether to randomly sample segments (train) or sequential (test)
        overlap: Overlap ratio for sequential segments (default: 0.25)
    """
    
    AVAILABLE_STEMS = ['vocals', 'drums', 'bass', 'other']
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        target_stems: Optional[List[str]] = None,
        sample_rate: int = 44100,
        segment_seconds: float = 10.0,
        random_segments: bool = True,
        overlap: float = 0.25,
    ) -> None:
        super().__init__()
        
        if musdb is None:
            raise ImportError(
                "musdb package is required. Install with: pip install musdb"
            )
        
        self.root = Path(root)
        self.split = split
        self.sample_rate = sample_rate
        self.segment_samples = int(segment_seconds * sample_rate)
        self.random_segments = random_segments
        self.overlap = overlap
        
        # Target stems configuration
        if target_stems is None:
            target_stems = ['vocals', 'other']  # Binary separation by default
        
        self.target_stems = target_stems
        self._validate_stems()
        
        # Load MUSDB18 tracks
        self.mus = musdb.DB(root=str(self.root), subsets=[split], split='train' if split == 'train' else 'valid')
        self.tracks = self.mus.tracks
        
        print(f"Loaded {len(self.tracks)} tracks from MUSDB18 {split} split")
        print(f"Target stems: {self.target_stems}")
        
    def _validate_stems(self) -> None:
        """Validate target stems."""
        for stem in self.target_stems:
            if stem not in self.AVAILABLE_STEMS:
                raise ValueError(
                    f"Invalid stem '{stem}'. Available stems: {self.AVAILABLE_STEMS}"
                )
    
    def _get_combined_stem(self, track: musdb.MultiTrack, stem_name: str) -> np.ndarray:
        """
        Get individual stem or combined stem.
        
        If stem_name is 'other' and not in target_stems as individual,
        combine all non-target stems.
        """
        # If requesting 'other' in binary mode (vocals vs rest)
        if stem_name == 'other' and len(self.target_stems) == 2 and 'vocals' in self.target_stems:
            # Combine drums + bass + other
            audio = np.zeros_like(track.audio)
            for source in ['drums', 'bass', 'other']:
                audio += track.targets[source].audio
            return audio
        
        # Otherwise return individual stem
        return track.targets[stem_name].audio
    
    def _load_audio_segment(
        self,
        track: musdb.MultiTrack,
        start_sample: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Load audio segment from track.
        
        Returns:
            mixture: (2, samples) stereo mixture
            targets: Dict of stem_name -> (2, samples) stereo audio
        """
        # Get track length
        track_length = track.audio.shape[0]
        
        # Determine segment start
        if start_sample is None:
            if self.random_segments:
                max_start = max(0, track_length - self.segment_samples)
                start_sample = random.randint(0, max_start) if max_start > 0 else 0
            else:
                start_sample = 0
        
        # Ensure we don't exceed track length
        end_sample = min(start_sample + self.segment_samples, track_length)
        actual_samples = end_sample - start_sample
        
        # Load mixture
        mixture = track.audio[start_sample:end_sample].T  # (samples, 2) -> (2, samples)
        
        # Load target stems
        targets = {}
        for stem_name in self.target_stems:
            stem_audio = self._get_combined_stem(track, stem_name)
            stem_audio = stem_audio[start_sample:end_sample].T  # (samples, 2) -> (2, samples)
            targets[stem_name] = torch.from_numpy(stem_audio).float()
        
        mixture = torch.from_numpy(mixture).float()
        
        # Pad if necessary (for last segment)
        if actual_samples < self.segment_samples:
            pad_samples = self.segment_samples - actual_samples
            mixture = torch.nn.functional.pad(mixture, (0, pad_samples))
            for stem_name in targets:
                targets[stem_name] = torch.nn.functional.pad(
                    targets[stem_name], (0, pad_samples)
                )
        
        return mixture, targets
    
    def __len__(self) -> int:
        """
        Return number of segments.
        
        For random sampling: use number of tracks (each epoch samples differently)
        For sequential: calculate total number of segments across all tracks
        """
        if self.random_segments:
            return len(self.tracks)
        else:
            # Calculate total segments with overlap
            total_segments = 0
            for track in self.tracks:
                track_length = track.audio.shape[0]
                hop = int(self.segment_samples * (1 - self.overlap))
                n_segments = max(1, (track_length - self.segment_samples) // hop + 1)
                total_segments += n_segments
            return total_segments
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get a single training sample.
        
        Returns:
            mixture: (2, samples) stereo mixture
            targets: Dict of stem_name -> (2, samples) stereo audio
        """
        if self.random_segments:
            # Random sampling: idx is track index
            track = self.tracks[idx]
            mixture, targets = self._load_audio_segment(track)
        else:
            # Sequential sampling: find track and segment
            track_idx = 0
            segment_idx = idx
            hop = int(self.segment_samples * (1 - self.overlap))
            
            for track in self.tracks:
                track_length = track.audio.shape[0]
                n_segments = max(1, (track_length - self.segment_samples) // hop + 1)
                
                if segment_idx < n_segments:
                    start_sample = segment_idx * hop
                    mixture, targets = self._load_audio_segment(track, start_sample)
                    break
                
                segment_idx -= n_segments
                track_idx += 1
        
        return mixture, targets
    
    def get_full_track(
        self,
        track_idx: int,
        return_name: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Optional[str]]:
        """
        Get full track without segmentation (for evaluation).
        
        Args:
            track_idx: Track index
            return_name: Whether to return track name
            
        Returns:
            mixture: (2, full_length) full track mixture
            targets: Dict of stem_name -> (2, full_length) full track audio
            name: Track name (if return_name=True)
        """
        track = self.tracks[track_idx]
        
        # Load full track
        mixture = torch.from_numpy(track.audio.T).float()  # (samples, 2) -> (2, samples)
        
        targets = {}
        for stem_name in self.target_stems:
            stem_audio = self._get_combined_stem(track, stem_name)
            targets[stem_name] = torch.from_numpy(stem_audio.T).float()
        
        if return_name:
            return mixture, targets, track.name
        return mixture, targets, None


def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Collate function for DataLoader.
    
    Args:
        batch: List of (mixture, targets) tuples
        
    Returns:
        mixtures: (B, 2, samples) batch of mixtures
        targets: Dict of stem_name -> (B, 2, samples) batch of stems
    """
    mixtures = []
    all_targets = {}
    
    for mixture, targets in batch:
        mixtures.append(mixture)
        
        for stem_name, stem_audio in targets.items():
            if stem_name not in all_targets:
                all_targets[stem_name] = []
            all_targets[stem_name].append(stem_audio)
    
    # Stack into batches
    mixtures = torch.stack(mixtures, dim=0)
    
    for stem_name in all_targets:
        all_targets[stem_name] = torch.stack(all_targets[stem_name], dim=0)
    
    return mixtures, all_targets
