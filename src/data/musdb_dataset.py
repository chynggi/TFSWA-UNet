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
        is_wav: Force dataset format (True for MUSDB18-HQ wav files). If None, attempt auto-detection.
        max_segments_per_track: Limit on sequential segments per track (validation). None uses full coverage.
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
    is_wav: Optional[bool] = None,
    max_segments_per_track: Optional[int] = None,
    ) -> None:
        super().__init__()
        
        if musdb is None:
            raise ImportError(
                "musdb package is required. Install with: pip install musdb"
            )
        
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(
                f"MUSDB18 root '{self.root}' does not exist. Set --data_root to the dataset directory."
            )
        self.split = split
        self.sample_rate = sample_rate
        self.segment_samples = int(segment_seconds * sample_rate)
        self.random_segments = random_segments
        self.overlap = overlap
        self.max_segments_per_track = max_segments_per_track
        if self.max_segments_per_track is not None and self.max_segments_per_track <= 0:
            raise ValueError("max_segments_per_track must be positive when provided")
        
        # Target stems configuration
        if target_stems is None:
            target_stems = ['vocals', 'other']  # Binary separation by default
        
        self.target_stems = target_stems
        self._validate_stems()

        # Determine file format for MUSDB18 (HQ uses wav files)
        self.is_wav = is_wav if is_wav is not None else self._infer_is_wav()
        
        # Load MUSDB18 tracks (musdb expects validation split with subsets=['train'])
        if split not in {'train', 'valid', 'test'}:
            raise ValueError("split must be one of {'train', 'valid', 'test'}")

        if split == 'train':
            subsets = ['train']
            split_arg: Optional[str] = None
        elif split == 'valid':
            subsets = ['train']
            split_arg = 'valid'
        else:  # split == 'test'
            subsets = ['test']
            split_arg = None

        self.mus = musdb.DB(
            root=str(self.root),
            subsets=subsets,
            split=split_arg,
            is_wav=self.is_wav,
        )
        self.tracks = self.mus.tracks
        self._segment_index: Optional[List[Tuple[int, int]]] = None

        if not self.random_segments:
            self._build_sequential_index()

        if len(self.tracks) == 0:
            raise RuntimeError(
                "No MUSDB18 tracks found. Ensure the dataset is downloaded and located under "
                f"'{self.root}'. Refer to https://sigsep.github.io/datasets/musdb.html for download instructions."
            )
        
        print(f"Loaded {len(self.tracks)} tracks from MUSDB18 {split} split")
        print(f"Target stems: {self.target_stems}")
        
    def _validate_stems(self) -> None:
        """Validate target stems."""
        for stem in self.target_stems:
            if stem not in self.AVAILABLE_STEMS:
                raise ValueError(
                    f"Invalid stem '{stem}'. Available stems: {self.AVAILABLE_STEMS}"
                )

    def _infer_is_wav(self) -> bool:
        """Infer whether the dataset uses wav (MUSDB18-HQ) or stem files."""
        # Prefer stem files if they exist; otherwise fall back to wav tracks.
        for subset in ('train', 'test', 'valid'):
            subset_dir = self.root / subset
            if not subset_dir.exists():
                continue

            stem_sample = next(subset_dir.rglob('*.stem.mp4'), None)
            if stem_sample is not None:
                return False

            wav_sample = next(subset_dir.rglob('*.wav'), None)
            if wav_sample is not None:
                return True

        raise RuntimeError(
            "Could not determine MUSDB18 file format. Ensure the dataset contains either "
            "'.stem.mp4' (original) or '.wav' (MUSDB18-HQ) files."
        )
    
    def _get_combined_stem(self, track: musdb.MultiTrack, stem_name: str) -> np.ndarray:
        """
        Get individual stem or combined stem.
        
        If stem_name is 'other' and not in target_stems as individual,
        combine all non-target stems.
        """
        available_sources = track.targets.keys()

        # Direct match first
        if stem_name in available_sources:
            return track.targets[stem_name].audio

        # Handle binary separation when dataset only provides accompaniment stem
        if stem_name == 'other' and len(self.target_stems) == 2 and 'vocals' in self.target_stems:
            # MUSDB18-HQ exposes "accompaniment" instead of individual drums/bass/other
            if 'accompaniment' in available_sources:
                return track.targets['accompaniment'].audio

            # Fall back to summing every non-vocal source that exists
            audio = np.zeros_like(track.audio)
            combined_sources = [name for name in available_sources if name != 'vocals']
            if not combined_sources:
                raise KeyError(
                    "No accompaniment sources found to construct 'other'. Available sources: "
                    f"{sorted(available_sources)}"
                )
            for source in combined_sources:
                audio += track.targets[source].audio
            return audio

        raise KeyError(
            f"Stem '{stem_name}' not available in track '{track.name}'. Available sources: {sorted(available_sources)}"
        )
    
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
    
    def _build_sequential_index(self) -> None:
        """Precompute segment start positions for sequential sampling."""
        self._segment_index = []

        for track_idx, track in enumerate(self.tracks):
            track_length = track.audio.shape[0]

            if track_length <= self.segment_samples:
                starts = [0]
            else:
                hop = max(1, int(self.segment_samples * (1 - self.overlap)))
                n_segments = max(1, (track_length - self.segment_samples) // hop + 1)

                if self.max_segments_per_track is not None and n_segments > self.max_segments_per_track:
                    max_start = track_length - self.segment_samples
                    # Evenly spaced segment starting points across track
                    starts = [int(round(x)) for x in np.linspace(0, max_start, self.max_segments_per_track)]
                else:
                    starts = [min(track_length - self.segment_samples, i * hop) for i in range(n_segments)]

            for start in starts:
                self._segment_index.append((track_idx, start))

    def __len__(self) -> int:
        """Return dataset length."""
        if self.random_segments:
            return len(self.tracks)
        assert self._segment_index is not None
        return len(self._segment_index)
    
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
            assert self._segment_index is not None
            track_idx, start_sample = self._segment_index[idx]
            track = self.tracks[track_idx]
            mixture, targets = self._load_audio_segment(track, start_sample)

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
