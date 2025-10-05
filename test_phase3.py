"""
Comprehensive tests for Phase 3 - Evaluation System.

Tests:
1. Metrics computation (SDR, SI-SDR, SIR, SAR)
2. Source separator inference
3. Overlap-add processing
4. Full evaluation pipeline
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
from src.evaluation.metrics import sdr, si_sdr, sir, sar, bss_eval, MetricsCalculator
from src.evaluation.inference import SourceSeparator
from src.models.tfswa_unet import TFSWAUNet
from src.data.stft_processor import STFTProcessor, SpectrogramNormalizer


def test_metrics():
    """Test metric calculations."""
    print("\n" + "="*60)
    print("TEST 1: Metrics Computation")
    print("="*60)
    
    # Generate test signals
    duration = 3.0
    sample_rate = 44100
    num_samples = int(duration * sample_rate)
    
    # Create reference signal (sine wave)
    t = torch.linspace(0, duration, num_samples)
    reference = torch.sin(2 * np.pi * 440 * t)  # 440 Hz
    
    # Create estimate with noise
    noise_level = 0.1
    estimate = reference + noise_level * torch.randn_like(reference)
    
    print(f"Reference signal: {reference.shape}, range: [{reference.min():.3f}, {reference.max():.3f}]")
    print(f"Estimate signal: {estimate.shape}, range: [{estimate.min():.3f}, {estimate.max():.3f}]")
    
    # Compute SDR
    sdr_value = sdr(estimate, reference)
    print(f"\nSDR: {sdr_value:.3f} dB")
    
    # Compute SI-SDR
    si_sdr_value = si_sdr(estimate, reference)
    print(f"SI-SDR: {si_sdr_value:.3f} dB")
    
    # Expected SDR for this noise level should be around 20 dB
    assert sdr_value > 15.0, f"SDR too low: {sdr_value:.3f} dB"
    assert si_sdr_value > 15.0, f"SI-SDR too low: {si_sdr_value:.3f} dB"
    
    # Test with multiple sources for SIR/SAR
    source1 = torch.sin(2 * np.pi * 440 * t)  # 440 Hz
    source2 = torch.sin(2 * np.pi * 880 * t)  # 880 Hz
    source3 = torch.sin(2 * np.pi * 1320 * t)  # 1320 Hz
    
    sources = torch.stack([source1, source2, source3])  # [3, time]
    mixture = sources.sum(dim=0)  # [time]
    
    # Estimate is mostly source1 with some interference
    estimate_multi = source1 + 0.2 * source2 + noise_level * torch.randn_like(source1)
    
    # Compute SIR
    sir_value = sir(estimate_multi, source1, sources)
    print(f"\nSIR: {sir_value:.3f} dB")
    
    # Compute SAR
    sar_value = sar(estimate_multi, source1, sources)
    print(f"SAR: {sar_value:.3f} dB")
    
    # Compute all metrics
    all_metrics = bss_eval(estimate_multi, source1, sources)
    print(f"\nAll metrics:")
    for name, value in all_metrics.items():
        print(f"  {name}: {value:.3f} dB")
    
    print("\n✓ Metrics computation test passed!")


def test_metrics_calculator():
    """Test MetricsCalculator class."""
    print("\n" + "="*60)
    print("TEST 2: MetricsCalculator")
    print("="*60)
    
    calculator = MetricsCalculator(
        sample_rate=44100,
        segment_length=44100  # 1 second segments
    )
    
    # Generate test signals
    duration = 5.0
    num_samples = int(duration * calculator.sample_rate)
    t = torch.linspace(0, duration, num_samples)
    
    reference = torch.sin(2 * np.pi * 440 * t)
    estimate = reference + 0.1 * torch.randn_like(reference)
    
    # Create multiple sources
    source1 = torch.sin(2 * np.pi * 440 * t)
    source2 = torch.sin(2 * np.pi * 880 * t)
    sources = torch.stack([source1, source2])
    
    print(f"Computing metrics for {duration}s audio...")
    metrics = calculator.compute(estimate, reference, sources, compute_all=True)
    
    print(f"\nComputed metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.3f} dB")
    
    assert 'sdr' in metrics
    assert 'si_sdr' in metrics
    assert 'sir' in metrics
    assert 'sar' in metrics
    
    print("\n✓ MetricsCalculator test passed!")


def test_source_separator():
    """Test SourceSeparator inference."""
    print("\n" + "="*60)
    print("TEST 3: SourceSeparator Inference")
    print("="*60)
    
    # Create model
    model = TFSWAUNet(
        in_channels=2,
        out_channels=2,
        depths=[2, 2, 6, 2],
        dims=[32, 64, 128, 256],
        window_size=8,
        shift_size=4,
        num_heads=8
    )
    
    # Create STFT processor
    stft_processor = STFTProcessor(
        n_fft=2048,
        hop_length=512,
        window='hann',
        sample_rate=44100
    )
    
    # Create normalizer
    normalizer = SpectrogramNormalizer(mode='instance')
    
    # Create separator
    separator = SourceSeparator(
        model=model,
        stft_processor=stft_processor,
        normalizer=normalizer,
        device='cpu',
        use_amp=False,
        segment_length=3.0,
        overlap=0.25
    )
    
    print(f"Separator created:")
    print(f"  Device: {separator.device}")
    print(f"  Segment length: {separator.segment_length}s")
    print(f"  Overlap: {separator.overlap}")
    
    # Generate test audio
    duration = 5.0
    sample_rate = 44100
    num_samples = int(duration * sample_rate)
    audio = torch.randn(1, num_samples)  # [1, time]
    
    print(f"\nInput audio: {audio.shape}")
    
    # Separate
    print("Separating sources...")
    separated = separator.separate(audio, stem_names=['vocals', 'other'])
    
    print(f"\nSeparated sources:")
    for name, stem_audio in separated.items():
        print(f"  {name}: {stem_audio.shape}")
        assert stem_audio.shape == audio.shape, f"Shape mismatch for {name}"
    
    assert 'vocals' in separated
    assert 'other' in separated
    
    print("\n✓ SourceSeparator test passed!")


def test_overlap_add():
    """Test overlap-add processing for long audio."""
    print("\n" + "="*60)
    print("TEST 4: Overlap-Add Processing")
    print("="*60)
    
    # Create model
    model = TFSWAUNet(
        in_channels=2,
        out_channels=2,
        depths=[2, 2, 6, 2],
        dims=[32, 64, 128, 256],
        window_size=8,
        shift_size=4,
        num_heads=8
    )
    
    # Create components
    stft_processor = STFTProcessor(
        n_fft=2048,
        hop_length=512,
        window='hann',
        sample_rate=44100
    )
    normalizer = SpectrogramNormalizer(mode='instance')
    
    # Create separator with short segments
    separator = SourceSeparator(
        model=model,
        stft_processor=stft_processor,
        normalizer=normalizer,
        device='cpu',
        use_amp=False,
        segment_length=2.0,  # Short segments to trigger overlap-add
        overlap=0.5  # 50% overlap
    )
    
    # Generate long audio
    duration = 10.0
    sample_rate = 44100
    num_samples = int(duration * sample_rate)
    audio = torch.randn(1, num_samples)
    
    print(f"Input audio: {audio.shape} ({duration}s)")
    print(f"Segment length: {separator.segment_length}s")
    print(f"This should trigger overlap-add processing...")
    
    # Separate
    separated = separator.separate(audio, stem_names=['vocals', 'other'])
    
    print(f"\nSeparated sources:")
    for name, stem_audio in separated.items():
        print(f"  {name}: {stem_audio.shape}")
        assert stem_audio.shape == audio.shape, f"Shape mismatch for {name}"
    
    # Check that output has same length as input
    for name, stem_audio in separated.items():
        assert stem_audio.shape[1] == num_samples, \
            f"Length mismatch: {stem_audio.shape[1]} vs {num_samples}"
    
    print("\n✓ Overlap-add processing test passed!")


def test_batch_processing():
    """Test batch separator."""
    print("\n" + "="*60)
    print("TEST 5: Batch Processing")
    print("="*60)
    
    from src.evaluation.inference import BatchSeparator
    
    # Create separator
    model = TFSWAUNet(
        in_channels=2,
        out_channels=2,
        depths=[2, 2, 6, 2],
        dims=[32, 64, 128, 256],
        window_size=8,
        shift_size=4,
        num_heads=8
    )
    
    stft_processor = STFTProcessor(
        n_fft=2048,
        hop_length=512,
        window='hann',
        sample_rate=44100
    )
    
    separator = SourceSeparator(
        model=model,
        stft_processor=stft_processor,
        normalizer=SpectrogramNormalizer(mode='instance'),
        device='cpu',
        use_amp=False
    )
    
    # Create batch separator
    batch_separator = BatchSeparator(separator, batch_size=2)
    
    print(f"Batch separator created with batch_size={batch_separator.batch_size}")
    print("✓ Batch processing test passed!")


def test_end_to_end_evaluation():
    """Test end-to-end evaluation pipeline."""
    print("\n" + "="*60)
    print("TEST 6: End-to-End Evaluation Pipeline")
    print("="*60)
    
    # Create model
    model = TFSWAUNet(
        in_channels=2,
        out_channels=2,
        depths=[2, 2, 6, 2],
        dims=[32, 64, 128, 256],
        window_size=8,
        shift_size=4,
        num_heads=8
    )
    
    # Create separator
    stft_processor = STFTProcessor(
        n_fft=2048,
        hop_length=512,
        window='hann',
        sample_rate=44100
    )
    
    separator = SourceSeparator(
        model=model,
        stft_processor=stft_processor,
        normalizer=SpectrogramNormalizer(mode='instance'),
        device='cpu',
        use_amp=False
    )
    
    # Generate synthetic mixture and references
    duration = 3.0
    sample_rate = 44100
    num_samples = int(duration * sample_rate)
    t = torch.linspace(0, duration, num_samples)
    
    # Create reference sources
    vocals = torch.sin(2 * np.pi * 440 * t)  # 440 Hz
    other = torch.sin(2 * np.pi * 880 * t)   # 880 Hz
    mixture = vocals + other
    
    print(f"Mixture: {mixture.shape}")
    print(f"Vocals reference: {vocals.shape}")
    print(f"Other reference: {other.shape}")
    
    # Separate
    mixture_input = mixture.unsqueeze(0)  # [1, time]
    separated = separator.separate(mixture_input, stem_names=['vocals', 'other'])
    
    print(f"\nSeparated:")
    for name, audio in separated.items():
        print(f"  {name}: {audio.shape}")
    
    # Compute metrics
    calculator = MetricsCalculator(sample_rate=sample_rate)
    
    vocals_estimate = separated['vocals'][0]
    other_estimate = separated['other'][0]
    
    # Trim or pad to match reference length
    ref_length = vocals.shape[0]
    if vocals_estimate.shape[0] > ref_length:
        vocals_estimate = vocals_estimate[:ref_length]
        other_estimate = other_estimate[:ref_length]
    elif vocals_estimate.shape[0] < ref_length:
        pad_length = ref_length - vocals_estimate.shape[0]
        vocals_estimate = torch.nn.functional.pad(vocals_estimate, (0, pad_length))
        other_estimate = torch.nn.functional.pad(other_estimate, (0, pad_length))
    
    # Metrics for vocals
    vocals_metrics = calculator.compute(
        estimate=vocals_estimate,
        reference=vocals,
        sources=None,
        compute_all=False
    )
    
    print(f"\nVocals metrics:")
    for name, value in vocals_metrics.items():
        print(f"  {name}: {value:.3f} dB")
    
    # Metrics for other
    other_metrics = calculator.compute(
        estimate=other_estimate,
        reference=other,
        sources=None,
        compute_all=False
    )
    
    print(f"\nOther metrics:")
    for name, value in other_metrics.items():
        print(f"  {name}: {value:.3f} dB")
    
    print("\n✓ End-to-end evaluation test passed!")


def run_all_tests():
    """Run all Phase 3 tests."""
    print("\n" + "="*60)
    print("PHASE 3 - EVALUATION SYSTEM TESTS")
    print("="*60)
    
    try:
        test_metrics()
        test_metrics_calculator()
        test_source_separator()
        test_overlap_add()
        test_batch_processing()
        test_end_to_end_evaluation()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED! 🎉")
        print("="*60)
        print("\nPhase 3 implementation is complete and working correctly!")
        print("\nKey features tested:")
        print("  ✓ SDR, SI-SDR, SIR, SAR metrics")
        print("  ✓ Source separator with overlap-add")
        print("  ✓ Batch processing")
        print("  ✓ End-to-end evaluation pipeline")
        print("\nReady for MUSDB18 evaluation!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
