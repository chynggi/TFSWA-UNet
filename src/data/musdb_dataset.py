"""MUSDB18 dataset loader with flexible stem selection and memory optimization."""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import soundfile as sf
from torch.utils.data import Dataset

import musdb


def load_chunk(path: str, length: int, chunk_size: int, offset: Optional[int] = None) -> np.ndarray:
    """
    Load audio chunk efficiently using soundfile.
    
    Args:
        path: Path to audio file
        length: Total length of audio in samples
        chunk_size: Size of chunk to load
        offset: Starting offset (random if None)
        
    Returns:
        audio: (2, chunk_size) stereo audio chunk
    """
    if chunk_size <= length:
        if offset is None:
            offset = np.random.randint(length - chunk_size + 1)
        # Load only the required chunk
        x = sf.read(path, dtype='float32', start=offset, frames=chunk_size)[0]
    else:
        # If chunk is larger than file, load full file and pad
        x = sf.read(path, dtype='float32')[0]
        if len(x.shape) == 1:
            pad = np.zeros((chunk_size - length))
        else:
            pad = np.zeros([chunk_size - length, x.shape[-1]])
        x = np.concatenate([x, pad], axis=0)
    
    # Ensure stereo format (2, samples)
    if len(x.shape) == 1:
        x = np.expand_dims(x, axis=1)
    return x.T


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
        use_efficient_loading: bool = True,
        min_mean_abs: float = 0.0,
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
        
        # Memory optimization options
        self.use_efficient_loading = use_efficient_loading
        self.min_mean_abs = min_mean_abs  # Minimum absolute mean to filter silent chunks
        self._track_cache = {}  # Cache for track paths
        
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
    
    def _load_audio_segment_efficient(
        self,
        track: musdb.MultiTrack,
        start_sample: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Load audio segment efficiently using chunk-based loading.
        
        This method loads only the required audio chunk instead of the entire track.
        
        Returns:
            mixture: (2, samples) stereo mixture
            targets: Dict of stem_name -> (2, samples) stereo audio
        """
        # MUSDB18-HQ structure: root/train|test/track_name/stem.wav
        # track.path points to the track directory
        track_path = Path(track.path)
        track_length = track.samples
        
        # Debug: Print track path for first load
        if not hasattr(self, '_debug_printed'):
            print(f"\nDEBUG: Track path structure:")
            print(f"  track.path: {track.path}")
            print(f"  track.name: {track.name}")
            print(f"  track_path exists: {track_path.exists()}")
            if track_path.exists():
                print(f"  Contents: {list(track_path.iterdir())[:5]}")
            self._debug_printed = True
        
        # Determine segment start
        if start_sample is None:
            if self.random_segments:
                max_start = max(0, track_length - self.segment_samples)
                start_sample = random.randint(0, max_start) if max_start > 0 else 0
            else:
                start_sample = 0
        
        # Load stems efficiently using soundfile
        targets = {}
        stem_audios = []
        
        for stem_name in self.target_stems:
            # Find the stem file path - MUSDB18-HQ stores as: track_path/stem.wav
            if stem_name == 'other' and len(self.target_stems) == 2 and 'vocals' in self.target_stems:
                # For binary separation, combine non-vocal stems
                other_audio = None
                for other_stem in ['drums', 'bass', 'other']:
                    stem_path = track_path / f"{other_stem}.wav"
                    if not stem_path.exists():
                        # Try without track subfolder (some formats store differently)
                        stem_path = Path(track.path).parent / f"{track.name}_{other_stem}.wav"
                    
                    if stem_path.exists():
                        try:
                            chunk = load_chunk(str(stem_path), track_length, 
                                             self.segment_samples, start_sample)
                            if other_audio is None:
                                other_audio = chunk
                            else:
                                other_audio = other_audio + chunk
                        except Exception as e:
                            print(f"Warning: Failed to load {stem_path}: {e}")
                            continue
                
                if other_audio is None:
                    # Fallback: load from musdb if file loading fails
                    print(f"Warning: Could not load 'other' stem files, using fallback method")
                    return self._load_audio_segment(track, start_sample)
                    
                targets[stem_name] = torch.from_numpy(other_audio).float()
                stem_audios.append(other_audio)
            else:
                stem_path = track_path / f"{stem_name}.wav"
                if not stem_path.exists():
                    # Try alternative path format
                    stem_path = Path(track.path).parent / f"{track.name}_{stem_name}.wav"
                
                if stem_path.exists():
                    try:
                        chunk = load_chunk(str(stem_path), track_length, 
                                         self.segment_samples, start_sample)
                    except Exception as e:
                        print(f"Warning: Failed to load {stem_path}: {e}")
                        chunk = np.zeros((2, self.segment_samples), dtype=np.float32)
                else:
                    # Fallback to loading full track
                    print(f"Warning: Stem file not found: {stem_path}, using fallback")
                    return self._load_audio_segment(track, start_sample)
                    
                targets[stem_name] = torch.from_numpy(chunk).float()
                stem_audios.append(chunk)
        
        # Create mixture by summing all stems
        if len(stem_audios) > 0:
            mixture = np.sum(stem_audios, axis=0)
            mixture = torch.from_numpy(mixture).float()
        else:
            # If no stems loaded, use fallback
            return self._load_audio_segment(track, start_sample)
        
        return mixture, targets
    
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
        # Use efficient loading if enabled
        if self.use_efficient_loading and self.is_wav:
            try:
                result = self._load_audio_segment_efficient(track, start_sample)
                # Verify data is not all zeros
                mixture, targets = result
                if mixture.abs().sum() > 0:
                    return result
                else:
                    if not hasattr(self, '_warned_zero_data'):
                        print(f"Warning: Efficient loading returned zero data, using fallback method")
                        self._warned_zero_data = True
            except Exception as e:
                # Fallback to original method if efficient loading fails
                if not hasattr(self, '_warned_efficient_fail'):
                    print(f"Warning: Efficient loading failed ({e}), using fallback method")
                    self._warned_efficient_fail = True
        
        # Original loading method (loads entire track into memory)
        # Debug: Print track info on first call
        if not hasattr(self, '_debug_fallback_printed'):
            print(f"\nDEBUG: Using fallback loading method")
            print(f"  Track name: {track.name}")
            print(f"  Track audio shape: {track.audio.shape}")
            print(f"  Track audio range: [{track.audio.min():.4f}, {track.audio.max():.4f}]")
            print(f"  Available targets: {list(track.targets.keys())}")
            self._debug_fallback_printed = True
        
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
        
        # Debug: Verify data is not zero
        if not hasattr(self, '_debug_data_verified'):
            print(f"\nDEBUG: Verifying loaded data")
            print(f"  Mixture range: [{mixture.min():.4f}, {mixture.max():.4f}], mean: {mixture.mean():.4f}")
            for k, v in targets.items():
                print(f"  {k} range: [{v.min():.4f}, {v.max():.4f}], mean: {v.mean():.4f}")
            self._debug_data_verified = True
        
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
