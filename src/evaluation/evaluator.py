"""
Evaluation system for MUSDB18 benchmark.

Evaluates trained models on standard test set with official metrics.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import warnings

try:
    import musdb
    import museval
    MUSDB_AVAILABLE = True
except ImportError:
    MUSDB_AVAILABLE = False
    warnings.warn("musdb/museval not available. Install with: pip install musdb museval")

from .inference import SourceSeparator
from .metrics import compute_musdb_metrics, MetricsCalculator


class MUSDB18Evaluator:
    """
    Evaluator for MUSDB18 dataset.
    """
    
    def __init__(
        self,
        separator: SourceSeparator,
        data_root: str,
        subset: str = 'test',
        output_dir: Optional[str] = None,
        save_estimates: bool = False,
        use_museval: bool = True
    ):
        """
        Args:
            separator: Initialized source separator
            data_root: Path to MUSDB18 dataset
            subset: Dataset subset ('train', 'test')
            output_dir: Directory to save results
            save_estimates: Whether to save separated audio
            use_museval: Whether to use official museval metrics
        """
        if not MUSDB_AVAILABLE:
            raise ImportError("musdb/museval required for evaluation")
        
        self.separator = separator
        self.subset = subset
        self.output_dir = Path(output_dir) if output_dir else None
        self.save_estimates = save_estimates
        self.use_museval = use_museval
        
        # Load MUSDB
        self.mus = musdb.DB(root=data_root, subsets=subset)
        
        # Create output directory
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics calculator
        self.metrics_calculator = MetricsCalculator(
            sample_rate=separator.sample_rate,
            segment_length=separator.sample_rate * 10  # 10-second segments
        )
    
    def evaluate(
        self,
        stem_names: Optional[List[str]] = None,
        num_tracks: Optional[int] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Evaluate on full test set.
        
        Args:
            stem_names: Stems to evaluate (default: ['vocals', 'other'])
            num_tracks: Number of tracks to evaluate (None = all)
            verbose: Whether to show progress
            
        Returns:
            Dictionary with evaluation results
        """
        if stem_names is None:
            stem_names = ['vocals', 'other']
        
        # Select tracks
        tracks = self.mus.tracks[:num_tracks] if num_tracks else self.mus.tracks
        
        # Evaluate each track
        all_results = []
        
        iterator = tqdm(tracks, desc="Evaluating") if verbose else tracks
        
        for track in iterator:
            if verbose:
                iterator.set_description(f"Evaluating {track.name}")
            
            try:
                results = self.evaluate_track(track, stem_names)
                all_results.append(results)
            except Exception as e:
                warnings.warn(f"Failed to evaluate {track.name}: {e}")
        
        # Aggregate results
        aggregated = self._aggregate_results(all_results, stem_names)
        
        # Save results
        if self.output_dir:
            self._save_results(aggregated, all_results)
        
        # Print summary
        if verbose:
            self._print_summary(aggregated)
        
        return aggregated
    
    def evaluate_track(
        self,
        track: 'musdb.MultiTrack',
        stem_names: List[str]
    ) -> Dict:
        """
        Evaluate single track.
        
        Args:
            track: MUSDB track
            stem_names: Stems to evaluate
            
        Returns:
            Dictionary with track results
        """
        # Get mixture audio
        mixture = torch.from_numpy(track.audio.T).float()  # [channels, time]
        
        # Separate sources
        separated = self.separator.separate(mixture, stem_names)
        
        # Prepare references
        references = {}
        if 'vocals' in stem_names:
            references['vocals'] = track.targets['vocals'].audio.T
        if 'drums' in stem_names:
            references['drums'] = track.targets['drums'].audio.T
        if 'bass' in stem_names:
            references['bass'] = track.targets['bass'].audio.T
        if 'other' in stem_names:
            if 'other' in track.targets:
                references['other'] = track.targets['other'].audio.T
            else:
                # Combine other stems
                references['other'] = (
                    track.targets['drums'].audio.T +
                    track.targets['bass'].audio.T +
                    track.targets['other'].audio.T
                ) if len(stem_names) == 2 else track.targets['other'].audio.T
        
        # Convert to numpy for evaluation
        estimates = {
            name: audio.cpu().numpy()[0]  # Take first channel
            for name, audio in separated.items()
        }
        
        references = {
            name: audio[0]  # Take first channel
            for name, audio in references.items()
        }
        
        # Compute metrics
        if self.use_museval:
            # Use official museval
            metrics = self._compute_museval_metrics(
                track, estimates, stem_names
            )
        else:
            # Use custom metrics
            metrics = compute_musdb_metrics(
                estimates, references, self.separator.sample_rate
            )
        
        # Save estimates if requested
        if self.save_estimates and self.output_dir:
            self._save_estimates(track.name, estimates)
        
        return {
            'track_name': track.name,
            'duration': track.duration,
            'metrics': metrics
        }
    
    def _compute_museval_metrics(
        self,
        track: 'musdb.MultiTrack',
        estimates: Dict[str, np.ndarray],
        stem_names: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics using official museval.
        
        Args:
            track: MUSDB track
            estimates: Estimated sources
            stem_names: Stem names
            
        Returns:
            Dictionary {stem_name: {metric_name: value}}
        """
        # Prepare estimates dict
        estimates_dict = {
            name: np.expand_dims(audio, axis=1)  # Add channel dimension
            for name, audio in estimates.items()
        }
        
        # Evaluate
        scores = museval.eval_mus_track(
            track,
            estimates_dict,
            output_dir=str(self.output_dir) if self.output_dir else None
        )
        
        # Extract metrics
        metrics = {}
        for stem_name in stem_names:
            if stem_name in scores.targets:
                stem_scores = scores.targets[stem_name]
                metrics[stem_name] = {
                    'sdr': float(np.nanmedian(stem_scores.metrics['SDR'])),
                    'sir': float(np.nanmedian(stem_scores.metrics['SIR'])),
                    'sar': float(np.nanmedian(stem_scores.metrics['SAR'])),
                    'isr': float(np.nanmedian(stem_scores.metrics['ISR'])),
                }
        
        return metrics
    
    def _save_estimates(
        self,
        track_name: str,
        estimates: Dict[str, np.ndarray]
    ):
        """Save separated audio estimates."""
        import soundfile as sf
        
        track_dir = self.output_dir / track_name
        track_dir.mkdir(parents=True, exist_ok=True)
        
        for stem_name, audio in estimates.items():
            output_path = track_dir / f"{stem_name}.wav"
            sf.write(
                str(output_path),
                audio,
                self.separator.sample_rate
            )
    
    def _aggregate_results(
        self,
        all_results: List[Dict],
        stem_names: List[str]
    ) -> Dict:
        """
        Aggregate results across tracks.
        
        Args:
            all_results: List of track results
            stem_names: Stem names
            
        Returns:
            Aggregated results
        """
        aggregated = {
            'num_tracks': len(all_results),
            'total_duration': sum(r['duration'] for r in all_results),
            'per_stem': {}
        }
        
        # Aggregate per stem
        for stem_name in stem_names:
            stem_metrics = []
            for result in all_results:
                if stem_name in result['metrics']:
                    stem_metrics.append(result['metrics'][stem_name])
            
            if stem_metrics:
                # Compute mean and std
                metrics_dict = {}
                for metric_name in stem_metrics[0].keys():
                    values = [m[metric_name] for m in stem_metrics]
                    metrics_dict[metric_name] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'median': float(np.median(values)),
                    }
                
                aggregated['per_stem'][stem_name] = metrics_dict
        
        # Compute overall average
        all_sdrs = []
        for stem_name in stem_names:
            if stem_name in aggregated['per_stem']:
                all_sdrs.append(
                    aggregated['per_stem'][stem_name]['sdr']['mean']
                )
        
        if all_sdrs:
            aggregated['overall_sdr'] = {
                'mean': float(np.mean(all_sdrs)),
                'std': float(np.std(all_sdrs))
            }
        
        aggregated['per_track'] = all_results
        
        return aggregated
    
    def _save_results(
        self,
        aggregated: Dict,
        all_results: List[Dict]
    ):
        """Save evaluation results to JSON."""
        # Save aggregated results
        agg_path = self.output_dir / 'evaluation_results.json'
        with open(agg_path, 'w') as f:
            json.dump(aggregated, f, indent=2)
        
        # Save detailed results
        detailed_path = self.output_dir / 'detailed_results.json'
        with open(detailed_path, 'w') as f:
            json.dump(all_results, f, indent=2)
    
    def _print_summary(self, aggregated: Dict):
        """Print evaluation summary."""
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Number of tracks: {aggregated['num_tracks']}")
        print(f"Total duration: {aggregated['total_duration']:.2f} seconds")
        print()
        
        for stem_name, metrics in aggregated['per_stem'].items():
            print(f"{stem_name.upper()}:")
            for metric_name, values in metrics.items():
                print(f"  {metric_name.upper()}: "
                      f"{values['mean']:.3f} ± {values['std']:.3f} dB "
                      f"(median: {values['median']:.3f} dB)")
            print()
        
        if 'overall_sdr' in aggregated:
            print(f"OVERALL SDR: "
                  f"{aggregated['overall_sdr']['mean']:.3f} ± "
                  f"{aggregated['overall_sdr']['std']:.3f} dB")
        
        print("="*60 + "\n")


class CustomDatasetEvaluator:
    """
    Evaluator for custom datasets (non-MUSDB).
    """
    
    def __init__(
        self,
        separator: SourceSeparator,
        output_dir: Optional[str] = None
    ):
        """
        Args:
            separator: Initialized source separator
            output_dir: Directory to save results
        """
        self.separator = separator
        self.output_dir = Path(output_dir) if output_dir else None
        
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_calculator = MetricsCalculator(
            sample_rate=separator.sample_rate,
            segment_length=separator.sample_rate * 10
        )
    
    def evaluate_pairs(
        self,
        mixture_paths: List[str],
        reference_paths: Dict[str, List[str]],
        stem_names: List[str],
        verbose: bool = True
    ) -> Dict:
        """
        Evaluate on mixture-reference pairs.
        
        Args:
            mixture_paths: List of mixture audio paths
            reference_paths: Dict {stem_name: [reference_audio_paths]}
            stem_names: Stem names to evaluate
            verbose: Whether to show progress
            
        Returns:
            Evaluation results
        """
        import torchaudio
        
        all_results = []
        
        iterator = tqdm(mixture_paths) if verbose else mixture_paths
        
        for i, mixture_path in enumerate(iterator):
            if verbose:
                iterator.set_description(f"Evaluating {Path(mixture_path).name}")
            
            # Load mixture
            mixture, sr = torchaudio.load(mixture_path)
            if sr != self.separator.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.separator.sample_rate)
                mixture = resampler(mixture)
            
            # Separate
            separated = self.separator.separate(mixture, stem_names)
            
            # Load references
            references = {}
            for stem_name in stem_names:
                ref_path = reference_paths[stem_name][i]
                ref_audio, ref_sr = torchaudio.load(ref_path)
                if ref_sr != self.separator.sample_rate:
                    resampler = torchaudio.transforms.Resample(
                        ref_sr, self.separator.sample_rate
                    )
                    ref_audio = resampler(ref_audio)
                references[stem_name] = ref_audio.numpy()[0]
            
            # Compute metrics
            estimates = {
                name: audio.cpu().numpy()[0]
                for name, audio in separated.items()
            }
            
            metrics = compute_musdb_metrics(
                estimates, references, self.separator.sample_rate
            )
            
            all_results.append({
                'mixture_path': mixture_path,
                'metrics': metrics
            })
        
        # Aggregate
        aggregated = self._aggregate_custom_results(all_results, stem_names)
        
        # Save
        if self.output_dir:
            results_path = self.output_dir / 'custom_evaluation_results.json'
            with open(results_path, 'w') as f:
                json.dump(aggregated, f, indent=2)
        
        return aggregated
    
    def _aggregate_custom_results(
        self,
        all_results: List[Dict],
        stem_names: List[str]
    ) -> Dict:
        """Aggregate custom dataset results."""
        aggregated = {
            'num_samples': len(all_results),
            'per_stem': {}
        }
        
        for stem_name in stem_names:
            stem_metrics = []
            for result in all_results:
                if stem_name in result['metrics']:
                    stem_metrics.append(result['metrics'][stem_name])
            
            if stem_metrics:
                metrics_dict = {}
                for metric_name in stem_metrics[0].keys():
                    values = [m[metric_name] for m in stem_metrics]
                    metrics_dict[metric_name] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'median': float(np.median(values)),
                    }
                
                aggregated['per_stem'][stem_name] = metrics_dict
        
        return aggregated
