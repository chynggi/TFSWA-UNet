"""
Source separation evaluation metrics.

Implements SDR, SIR, SAR, and ISR metrics for evaluating 
music source separation quality.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import warnings


def _safe_db(num: torch.Tensor, den: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    Convert ratio to dB scale safely.
    
    Args:
        num: Numerator tensor
        den: Denominator tensor
        eps: Small constant to avoid log(0)
        
    Returns:
        dB value
    """
    ratio = torch.clamp(num / (den + eps), min=eps)
    return 10 * torch.log10(ratio)


def sdr(
    estimate: torch.Tensor,
    reference: torch.Tensor,
    eps: float = 1e-10
) -> torch.Tensor:
    """
    Signal-to-Distortion Ratio (SDR).
    
    Measures overall separation quality.
    
    Args:
        estimate: Estimated source [batch, time] or [time]
        reference: Ground truth source [batch, time] or [time]
        eps: Small constant for numerical stability
        
    Returns:
        SDR value in dB [batch] or scalar
    """
    # Ensure same shape
    assert estimate.shape == reference.shape, \
        f"Shape mismatch: {estimate.shape} vs {reference.shape}"
    
    # Handle batch dimension
    if estimate.dim() == 1:
        estimate = estimate.unsqueeze(0)
        reference = reference.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    
    # Compute signal and noise power
    signal_power = torch.sum(reference ** 2, dim=-1)
    noise_power = torch.sum((estimate - reference) ** 2, dim=-1)
    
    # Compute SDR
    sdr_value = _safe_db(signal_power, noise_power, eps)
    
    if squeeze:
        sdr_value = sdr_value.squeeze(0)
    
    return sdr_value


def si_sdr(
    estimate: torch.Tensor,
    reference: torch.Tensor,
    eps: float = 1e-10
) -> torch.Tensor:
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR).
    
    Similar to SDR but invariant to scaling of estimate.
    
    Args:
        estimate: Estimated source [batch, time] or [time]
        reference: Ground truth source [batch, time] or [time]
        eps: Small constant for numerical stability
        
    Returns:
        SI-SDR value in dB [batch] or scalar
    """
    # Ensure same shape
    assert estimate.shape == reference.shape
    
    # Handle batch dimension
    if estimate.dim() == 1:
        estimate = estimate.unsqueeze(0)
        reference = reference.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    
    # Zero-mean normalization
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    reference = reference - reference.mean(dim=-1, keepdim=True)
    
    # Compute scaling factor
    dot_product = torch.sum(estimate * reference, dim=-1, keepdim=True)
    ref_energy = torch.sum(reference ** 2, dim=-1, keepdim=True)
    scale = dot_product / (ref_energy + eps)
    
    # Compute scaled target and error
    scaled_target = scale * reference
    error = estimate - scaled_target
    
    # Compute SI-SDR
    target_power = torch.sum(scaled_target ** 2, dim=-1)
    error_power = torch.sum(error ** 2, dim=-1)
    
    si_sdr_value = _safe_db(target_power, error_power, eps)
    
    if squeeze:
        si_sdr_value = si_sdr_value.squeeze(0)
    
    return si_sdr_value


def sir(
    estimate: torch.Tensor,
    reference: torch.Tensor,
    sources: torch.Tensor,
    eps: float = 1e-10
) -> torch.Tensor:
    """
    Signal-to-Interference Ratio (SIR).
    
    Measures suppression of other sources.
    
    Args:
        estimate: Estimated source [batch, time] or [time]
        reference: Ground truth target source [batch, time] or [time]
        sources: All ground truth sources [batch, num_sources, time] or [num_sources, time]
        eps: Small constant for numerical stability
        
    Returns:
        SIR value in dB [batch] or scalar
    """
    # Handle batch dimension
    if estimate.dim() == 1:
        estimate = estimate.unsqueeze(0)
        reference = reference.unsqueeze(0)
        sources = sources.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    
    batch_size, num_sources, time_steps = sources.shape
    
    # Find target source index
    # Compute correlation with each source
    reference_expanded = reference.unsqueeze(1)  # [batch, 1, time]
    correlations = torch.sum(reference_expanded * sources, dim=-1)  # [batch, num_sources]
    target_idx = torch.argmax(correlations, dim=1)  # [batch]
    
    # Compute interference (sum of other sources)
    interference = torch.zeros_like(reference)
    for b in range(batch_size):
        for s in range(num_sources):
            if s != target_idx[b]:
                interference[b] += sources[b, s]
    
    # Project estimate onto reference and interference
    ref_projection = _project(estimate, reference, eps)
    interference_projection = _project(estimate, interference, eps)
    
    # Compute SIR
    signal_power = torch.sum(ref_projection ** 2, dim=-1)
    interference_power = torch.sum(interference_projection ** 2, dim=-1)
    
    sir_value = _safe_db(signal_power, interference_power, eps)
    
    if squeeze:
        sir_value = sir_value.squeeze(0)
    
    return sir_value


def sar(
    estimate: torch.Tensor,
    reference: torch.Tensor,
    sources: torch.Tensor,
    eps: float = 1e-10
) -> torch.Tensor:
    """
    Signal-to-Artifacts Ratio (SAR).
    
    Measures artifacts introduced by separation algorithm.
    
    Args:
        estimate: Estimated source [batch, time] or [time]
        reference: Ground truth target source [batch, time] or [time]
        sources: All ground truth sources [batch, num_sources, time] or [num_sources, time]
        eps: Small constant for numerical stability
        
    Returns:
        SAR value in dB [batch] or scalar
    """
    # Handle batch dimension
    if estimate.dim() == 1:
        estimate = estimate.unsqueeze(0)
        reference = reference.unsqueeze(0)
        sources = sources.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    
    # Compute allowed distortion (projection onto all sources)
    allowed_distortion = torch.zeros_like(reference)
    for s in range(sources.shape[1]):
        allowed_distortion += _project(estimate, sources[:, s], eps)
    
    # Artifacts are everything else
    artifacts = estimate - allowed_distortion
    
    # Compute SAR
    signal_power = torch.sum(allowed_distortion ** 2, dim=-1)
    artifact_power = torch.sum(artifacts ** 2, dim=-1)
    
    sar_value = _safe_db(signal_power, artifact_power, eps)
    
    if squeeze:
        sar_value = sar_value.squeeze(0)
    
    return sar_value


def _project(
    estimate: torch.Tensor,
    reference: torch.Tensor,
    eps: float = 1e-10
) -> torch.Tensor:
    """
    Project estimate onto reference.
    
    Args:
        estimate: Signal to project [batch, time]
        reference: Reference signal [batch, time]
        eps: Small constant for numerical stability
        
    Returns:
        Projection [batch, time]
    """
    dot_product = torch.sum(estimate * reference, dim=-1, keepdim=True)
    ref_energy = torch.sum(reference ** 2, dim=-1, keepdim=True)
    scale = dot_product / (ref_energy + eps)
    return scale * reference


def bss_eval(
    estimate: torch.Tensor,
    reference: torch.Tensor,
    sources: torch.Tensor,
    eps: float = 1e-10
) -> Dict[str, torch.Tensor]:
    """
    Compute all BSS Eval metrics (SDR, SIR, SAR).
    
    Args:
        estimate: Estimated source [batch, time] or [time]
        reference: Ground truth target source [batch, time] or [time]
        sources: All ground truth sources [batch, num_sources, time] or [num_sources, time]
        eps: Small constant for numerical stability
        
    Returns:
        Dictionary with 'sdr', 'si_sdr', 'sir', 'sar' values
    """
    metrics = {
        'sdr': sdr(estimate, reference, eps),
        'si_sdr': si_sdr(estimate, reference, eps),
        'sir': sir(estimate, reference, sources, eps),
        'sar': sar(estimate, reference, sources, eps),
    }
    
    return metrics


def median_filter_metrics(
    metrics: Dict[str, torch.Tensor],
    window_size: int = 3
) -> Dict[str, torch.Tensor]:
    """
    Apply median filtering to metrics for stability.
    
    Args:
        metrics: Dictionary of metrics [batch] or [num_frames]
        window_size: Median filter window size
        
    Returns:
        Filtered metrics
    """
    filtered = {}
    
    for key, values in metrics.items():
        if values.dim() == 0:  # Scalar
            filtered[key] = values
        else:
            # Apply median filter
            values_np = values.cpu().numpy()
            from scipy.ndimage import median_filter as scipy_median
            filtered_values = scipy_median(values_np, size=window_size)
            filtered[key] = torch.from_numpy(filtered_values).to(values.device)
    
    return filtered


class MetricsCalculator:
    """
    Convenience class for computing source separation metrics.
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        segment_length: Optional[int] = None,
        eps: float = 1e-10
    ):
        """
        Args:
            sample_rate: Audio sample rate
            segment_length: Length of segments for frame-wise metrics (in samples)
            eps: Small constant for numerical stability
        """
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.eps = eps
    
    def compute(
        self,
        estimate: torch.Tensor,
        reference: torch.Tensor,
        sources: Optional[torch.Tensor] = None,
        compute_all: bool = True
    ) -> Dict[str, float]:
        """
        Compute metrics for a single audio pair.
        
        Args:
            estimate: Estimated source [time] or [batch, time]
            reference: Ground truth source [time] or [batch, time]
            sources: All sources for SIR/SAR [num_sources, time] or [batch, num_sources, time]
            compute_all: Whether to compute SIR/SAR (requires sources)
            
        Returns:
            Dictionary with metric values (scalars)
        """
        # Ensure tensors
        if not isinstance(estimate, torch.Tensor):
            estimate = torch.from_numpy(estimate).float()
        if not isinstance(reference, torch.Tensor):
            reference = torch.from_numpy(reference).float()
        if sources is not None and not isinstance(sources, torch.Tensor):
            sources = torch.from_numpy(sources).float()
        
        # Move to same device
        device = estimate.device
        reference = reference.to(device)
        if sources is not None:
            sources = sources.to(device)
        
        # Compute frame-wise or full metrics
        if self.segment_length is not None and estimate.shape[-1] > self.segment_length:
            metrics = self._compute_framewise(estimate, reference, sources, compute_all)
        else:
            metrics = self._compute_full(estimate, reference, sources, compute_all)
        
        # Convert to float
        metrics = {k: float(v.mean()) if isinstance(v, torch.Tensor) else v 
                   for k, v in metrics.items()}
        
        return metrics
    
    def _compute_full(
        self,
        estimate: torch.Tensor,
        reference: torch.Tensor,
        sources: Optional[torch.Tensor],
        compute_all: bool
    ) -> Dict[str, torch.Tensor]:
        """Compute metrics on full audio."""
        metrics = {
            'sdr': sdr(estimate, reference, self.eps),
            'si_sdr': si_sdr(estimate, reference, self.eps),
        }
        
        if compute_all and sources is not None:
            metrics['sir'] = sir(estimate, reference, sources, self.eps)
            metrics['sar'] = sar(estimate, reference, sources, self.eps)
        
        return metrics
    
    def _compute_framewise(
        self,
        estimate: torch.Tensor,
        reference: torch.Tensor,
        sources: Optional[torch.Tensor],
        compute_all: bool
    ) -> Dict[str, torch.Tensor]:
        """Compute frame-wise metrics and aggregate."""
        # Handle batch dimension
        if estimate.dim() == 1:
            estimate = estimate.unsqueeze(0)
            reference = reference.unsqueeze(0)
            if sources is not None:
                sources = sources.unsqueeze(0)
        
        batch_size = estimate.shape[0]
        total_length = estimate.shape[1]
        
        # Split into segments
        num_segments = total_length // self.segment_length
        
        all_metrics = []
        
        for i in range(num_segments):
            start = i * self.segment_length
            end = start + self.segment_length
            
            est_seg = estimate[:, start:end]
            ref_seg = reference[:, start:end]
            src_seg = sources[:, :, start:end] if sources is not None else None
            
            seg_metrics = self._compute_full(est_seg, ref_seg, src_seg, compute_all)
            all_metrics.append(seg_metrics)
        
        # Aggregate metrics (median)
        aggregated = {}
        for key in all_metrics[0].keys():
            values = torch.stack([m[key] for m in all_metrics])
            aggregated[key] = torch.median(values, dim=0)[0]
        
        return aggregated


def compute_musdb_metrics(
    estimates: Dict[str, np.ndarray],
    references: Dict[str, np.ndarray],
    sample_rate: int = 44100
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics for MUSDB-style multi-stem separation.
    
    Args:
        estimates: Dictionary {stem_name: audio_array [time]}
        references: Dictionary {stem_name: audio_array [time]}
        sample_rate: Audio sample rate
        
    Returns:
        Dictionary {stem_name: {metric_name: value}}
    """
    calculator = MetricsCalculator(sample_rate=sample_rate, segment_length=sample_rate * 10)
    
    results = {}
    
    # Collect all sources for SIR/SAR
    all_references = torch.stack([
        torch.from_numpy(ref).float() 
        for ref in references.values()
    ])  # [num_sources, time]
    
    for stem_name in estimates.keys():
        if stem_name not in references:
            warnings.warn(f"Reference for {stem_name} not found, skipping")
            continue
        
        estimate = torch.from_numpy(estimates[stem_name]).float()
        reference = torch.from_numpy(references[stem_name]).float()
        
        metrics = calculator.compute(
            estimate=estimate,
            reference=reference,
            sources=all_references,
            compute_all=True
        )
        
        results[stem_name] = metrics
    
    return results
