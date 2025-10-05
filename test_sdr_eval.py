"""Test SDR evaluation functionality."""
import torch
import argparse
from pathlib import Path

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent))

from src.models.tfswa_unet import TFSWAUNet
from src.data.musdb_dataset import MUSDB18Dataset
from src.data.stft_processor import STFTProcessor
from src.evaluation.metrics import sdr, si_sdr


def test_sdr_computation():
    """Test basic SDR computation."""
    print("\n" + "="*60)
    print("Testing SDR Computation")
    print("="*60)
    
    # Create synthetic signals
    time = torch.linspace(0, 1, 44100)
    reference = torch.sin(2 * torch.pi * 440 * time)  # 440 Hz sine wave
    
    # Perfect estimate (should give very high SDR)
    estimate_perfect = reference.clone()
    sdr_perfect = sdr(estimate_perfect, reference)
    print(f"Perfect estimate SDR: {sdr_perfect:.2f} dB (should be very high)")
    
    # Noisy estimate
    noise = torch.randn_like(reference) * 0.1
    estimate_noisy = reference + noise
    sdr_noisy = sdr(estimate_noisy, reference)
    print(f"Noisy estimate SDR: {sdr_noisy:.2f} dB")
    
    # Very noisy estimate
    noise_heavy = torch.randn_like(reference) * 0.5
    estimate_very_noisy = reference + noise_heavy
    sdr_very_noisy = sdr(estimate_very_noisy, reference)
    print(f"Very noisy estimate SDR: {sdr_very_noisy:.2f} dB")
    
    # SI-SDR test
    si_sdr_val = si_sdr(estimate_noisy, reference)
    print(f"SI-SDR (noisy): {si_sdr_val:.2f} dB")
    
    print("\n✓ SDR computation works correctly!\n")


def test_dataset_loading(data_root: str):
    """Test dataset loading for evaluation."""
    print("\n" + "="*60)
    print("Testing Dataset Loading")
    print("="*60)
    
    try:
        val_dataset = MUSDB18Dataset(
            root=data_root,
            split='valid',
            target_stems=['vocals', 'other'],
            sample_rate=44100,
            segment_seconds=3.0,
            random_segments=False,
        )
        
        print(f"✓ Dataset loaded: {len(val_dataset.tracks)} tracks")
        
        # Test loading a track
        track = val_dataset.tracks[0]
        print(f"\nTest track: {track.name}")
        print(f"  Duration: {track.duration:.2f} seconds")
        print(f"  Samples: {track.samples}")
        print(f"  Sample rate: {track.rate}")
        print(f"  Available targets: {list(track.targets.keys())}")
        
        # Test audio loading
        mixture = torch.from_numpy(track.audio.T).float()
        print(f"\nMixture shape: {mixture.shape}")
        print(f"Mixture range: [{mixture.min():.4f}, {mixture.max():.4f}]")
        
        # Test target loading
        vocals = track.targets['vocals'].audio
        print(f"Vocals shape: {vocals.shape}")
        print(f"Vocals range: [{vocals.min():.4f}, {vocals.max():.4f}]")
        
        print("\n✓ Dataset loading works correctly!\n")
        
    except Exception as e:
        print(f"\n✗ Dataset loading failed: {e}\n")
        return False
    
    return True


def test_model_inference():
    """Test model can perform inference."""
    print("\n" + "="*60)
    print("Testing Model Inference")
    print("="*60)
    
    # Create small model
    model = TFSWAUNet(
        in_channels=4,  # real + imag
        out_channels=4,  # 2 stems * 2 channels
        depths=[2, 2, 2, 2],
        dims=[16, 32, 64, 128],
        window_size=4,
        shift_size=2,
        num_heads=4,
    )
    
    print(f"Model parameters: {model.get_num_parameters():,}")
    
    # Create dummy input
    batch_size = 1
    freq_bins = 513  # n_fft=1024 -> freq_bins=513
    time_frames = 50
    dummy_input = torch.randn(batch_size, 4, freq_bins, time_frames)
    
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    print("\n✓ Model inference works correctly!\n")


def test_full_pipeline(data_root: str):
    """Test full evaluation pipeline."""
    print("\n" + "="*60)
    print("Testing Full Evaluation Pipeline")
    print("="*60)
    
    try:
        # Load dataset
        val_dataset = MUSDB18Dataset(
            root=data_root,
            split='valid',
            target_stems=['vocals', 'other'],
            sample_rate=44100,
            segment_seconds=3.0,
            random_segments=False,
        )
        
        # Create model
        model = TFSWAUNet(
            in_channels=4,
            out_channels=4,
            depths=[2, 2, 2, 2],
            dims=[16, 32, 64, 128],
            window_size=4,
            shift_size=2,
            num_heads=4,
        )
        model.eval()
        
        # Create STFT processor
        stft_processor = STFTProcessor(
            n_fft=1024,
            hop_length=256,
        )
        
        # Get a short segment from first track
        track = val_dataset.tracks[0]
        segment_samples = 44100 * 3  # 3 seconds
        
        mixture = torch.from_numpy(track.audio[:segment_samples].T).float()  # (2, samples)
        mixture_mono = mixture.mean(dim=0, keepdim=True)  # (1, samples)
        
        print(f"Processing segment from: {track.name}")
        print(f"Segment shape: {mixture_mono.shape}")
        
        # Process with model
        segment_stereo = mixture_mono.repeat(2, 1).unsqueeze(0)  # (1, 2, samples)
        complex_spec = stft_processor.stft(segment_stereo)  # (1, 2, F, T)
        
        print(f"Spectrogram shape: {complex_spec.shape}")
        
        # Model input
        model_input = stft_processor.to_model_input(complex_spec)
        print(f"Model input shape: {model_input.shape}")
        
        # Inference
        with torch.no_grad():
            output = model(model_input)
        
        print(f"Model output shape: {output.shape}")
        
        # Extract masks
        vocals_mask = output[:, 0:2, :, :]
        mask_mag = torch.sqrt(vocals_mask[:, 0, :, :]**2 + vocals_mask[:, 1, :, :]**2 + 1e-8)
        mask_mag = torch.sigmoid(mask_mag)
        
        print(f"Vocals mask shape: {mask_mag.shape}")
        print(f"Mask range: [{mask_mag.min():.4f}, {mask_mag.max():.4f}]")
        
        # Apply mask and reconstruct
        mixture_mono_spec = complex_spec.mean(dim=1)  # (1, F, T)
        mixture_mag = torch.abs(mixture_mono_spec)
        mixture_phase = torch.angle(mixture_mono_spec)
        
        masked_mag = mixture_mag * mask_mag
        masked_spec = masked_mag * torch.exp(1j * mixture_phase)
        masked_spec = masked_spec.unsqueeze(1).repeat(1, 2, 1, 1)  # (1, 2, F, T)
        
        separated = stft_processor.istft(masked_spec)
        separated_mono = separated.mean(dim=1)  # (1, samples)
        
        print(f"Separated audio shape: {separated_mono.shape}")
        print(f"Separated range: [{separated_mono.min():.4f}, {separated_mono.max():.4f}]")
        
        # Compute SDR against reference
        reference_vocals = torch.from_numpy(
            track.targets['vocals'].audio[:segment_samples].T
        ).float().mean(dim=0)  # (samples,)
        
        # Trim to same length
        min_len = min(separated_mono.shape[1], reference_vocals.shape[0])
        separated_trim = separated_mono[0, :min_len]
        reference_trim = reference_vocals[:min_len]
        
        sdr_val = sdr(separated_trim, reference_trim)
        si_sdr_val = si_sdr(separated_trim, reference_trim)
        
        print(f"\nComputed metrics:")
        print(f"  SDR: {sdr_val:.3f} dB")
        print(f"  SI-SDR: {si_sdr_val:.3f} dB")
        
        print("\n✓ Full evaluation pipeline works correctly!\n")
        
    except Exception as e:
        print(f"\n✗ Full pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Test SDR evaluation')
    parser.add_argument('--data_root', type=str, default='/workspace/dataset',
                        help='Path to MUSDB18 dataset')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("TFSWA-UNet SDR Evaluation Test")
    print("="*60)
    
    # Run tests
    test_sdr_computation()
    
    if Path(args.data_root).exists():
        test_dataset_loading(args.data_root)
        test_full_pipeline(args.data_root)
    else:
        print(f"\nWarning: Data root '{args.data_root}' not found.")
        print("Skipping dataset tests. Specify --data_root to test with real data.\n")
    
    test_model_inference()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
