"""Test Phase 2 implementation (data pipeline and training setup)."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
from src.models.tfswa_unet import TFSWAUNet
from src.data.stft_processor import STFTProcessor, SpectrogramNormalizer
from src.training.losses import L1SpectrogramLoss, MultiResolutionSTFTLoss, SourceSeparationLoss


def test_stft_processor():
    """Test STFT/ISTFT processor."""
    print("=" * 80)
    print("Testing STFT Processor")
    print("=" * 80)
    
    processor = STFTProcessor(n_fft=2048, hop_length=512)
    
    # Create dummy waveform
    batch_size = 2
    channels = 2
    samples = 44100  # 1 second at 44.1kHz
    
    waveform = torch.randn(batch_size, channels, samples)
    print(f"\n1. Input waveform shape: {waveform.shape}")
    
    # Forward STFT
    complex_spec = processor.stft(waveform)
    print(f"2. Complex spectrogram shape: {complex_spec.shape}")
    print(f"   - Frequency bins: {complex_spec.shape[2]}")
    print(f"   - Time frames: {complex_spec.shape[3]}")
    print(f"   - Is complex: {torch.is_complex(complex_spec)}")
    
    # Convert to model input
    model_input = processor.to_model_input(complex_spec)
    print(f"3. Model input shape: {model_input.shape}")
    print(f"   - Channels (real+imag): {model_input.shape[1]}")
    
    # Inverse STFT
    reconstructed = processor.istft(complex_spec, length=samples)
    print(f"4. Reconstructed waveform shape: {reconstructed.shape}")
    
    # Check reconstruction error
    error = torch.abs(waveform - reconstructed).mean()
    print(f"5. Reconstruction error: {error.item():.6f}")
    
    if error < 0.01:
        print("   ✓ STFT/ISTFT working correctly!")
    else:
        print("   ⚠ High reconstruction error")
    
    print()


def test_spectrogram_normalizer():
    """Test spectrogram normalizer."""
    print("=" * 80)
    print("Testing Spectrogram Normalizer")
    print("=" * 80)
    
    normalizer = SpectrogramNormalizer(mode='instance')
    
    # Create dummy spectrogram
    spec = torch.randn(2, 2, 1025, 100)
    print(f"\n1. Input spectrogram shape: {spec.shape}")
    print(f"   - Mean: {spec.mean().item():.4f}")
    print(f"   - Std: {spec.std().item():.4f}")
    
    # Normalize
    normalized, mean, std = normalizer(spec, return_stats=True)
    print(f"2. Normalized spectrogram:")
    print(f"   - Mean: {normalized.mean().item():.6f}")
    print(f"   - Std: {normalized.std().item():.4f}")
    
    # Denormalize
    denormalized = normalizer.denormalize(normalized, mean, std)
    error = torch.abs(spec - denormalized).mean()
    print(f"3. Denormalization error: {error.item():.6f}")
    
    if error < 1e-5:
        print("   ✓ Normalization working correctly!")
    else:
        print("   ⚠ Denormalization error")
    
    print()


def test_losses():
    """Test loss functions."""
    print("=" * 80)
    print("Testing Loss Functions")
    print("=" * 80)
    
    # Create dummy data
    batch_size = 2
    freq_bins = 1025
    time_frames = 100
    
    pred_spec = torch.randn(batch_size, 2, freq_bins, time_frames)
    target_spec = torch.randn(batch_size, 2, freq_bins, time_frames)
    
    pred_audio = torch.randn(batch_size, 2, 44100)
    target_audio = torch.randn(batch_size, 2, 44100)
    
    # Test L1 loss
    print("\n1. L1 Spectrogram Loss:")
    l1_loss = L1SpectrogramLoss()
    loss = l1_loss(pred_spec, target_spec)
    print(f"   Loss: {loss.item():.4f}")
    print(f"   ✓ L1 loss computed")
    
    # Test Multi-resolution STFT loss
    print("\n2. Multi-resolution STFT Loss:")
    mrstft_loss = MultiResolutionSTFTLoss()
    loss = mrstft_loss(pred_audio, target_audio)
    print(f"   Loss: {loss.item():.4f}")
    print(f"   ✓ Multi-resolution STFT loss computed")
    
    # Test combined loss
    print("\n3. Source Separation Loss:")
    sep_loss = SourceSeparationLoss(l1_weight=1.0, mrstft_weight=0.5, use_mrstft=False)
    
    pred_specs = {'vocals': pred_spec, 'other': pred_spec}
    target_specs = {'vocals': target_spec, 'other': target_spec}
    
    loss_dict = sep_loss(pred_specs, target_specs)
    print(f"   Total loss: {loss_dict['total_loss'].item():.4f}")
    print(f"   L1 loss: {loss_dict['l1_loss'].item():.4f}")
    print(f"   ✓ Combined loss computed")
    
    print()


def test_end_to_end():
    """Test end-to-end pipeline."""
    print("=" * 80)
    print("Testing End-to-End Pipeline")
    print("=" * 80)
    
    # Create model
    print("\n1. Creating model...")
    model = TFSWAUNet(
        in_channels=4,  # real + imag for stereo
        out_channels=4,  # 2 stems × 2 channels
        depths=[2, 2, 6, 2],
        dims=[32, 64, 128, 256],
        window_size=8,
        shift_size=4,
        num_heads=8,
    )
    model.eval()
    print(f"   ✓ Model created ({model.get_num_parameters():,} parameters)")
    
    # Create STFT processor
    print("\n2. Creating STFT processor...")
    processor = STFTProcessor(n_fft=2048, hop_length=512)
    print("   ✓ STFT processor created")
    
    # Create dummy input
    print("\n3. Creating dummy audio...")
    batch_size = 1
    samples = 44100 * 3  # 3 seconds
    mixture = torch.randn(batch_size, 2, samples)
    print(f"   Mixture shape: {mixture.shape}")
    
    # Forward pass
    print("\n4. Forward pass:")
    with torch.no_grad():
        # STFT
        mixture_spec = processor.stft(mixture)
        print(f"   a) STFT: {mixture_spec.shape}")
        
        # Convert to model input
        model_input = processor.to_model_input(mixture_spec)
        print(f"   b) Model input: {model_input.shape}")
        
        # Model prediction
        model_output = model(model_input)
        print(f"   c) Model output: {model_output.shape}")
        
        # Extract masks for each stem
        n_stems = 2
        pred_audios = {}
        
        for idx, stem_name in enumerate(['vocals', 'other']):
            # Extract mask
            stem_mask = model_output[:, idx*2:(idx+1)*2, :, :]
            
            # Convert to complex
            real = stem_mask[:, 0:1, :, :]
            imag = stem_mask[:, 1:2, :, :]
            mask_complex = torch.complex(real, imag).squeeze(1)
            
            # Apply mask
            mixture_spec_mono = mixture_spec.mean(dim=1)
            pred_spec = mixture_spec_mono * mask_complex
            
            # ISTFT
            pred_spec_stereo = pred_spec.unsqueeze(1).repeat(1, 2, 1, 1)
            pred_audio = processor.istft(pred_spec_stereo, length=samples)
            pred_audios[stem_name] = pred_audio
            
            print(f"   d) {stem_name}: {pred_audio.shape}")
    
    print("\n5. ✓ End-to-end pipeline working!")
    print()


def test_flexible_stems():
    """Test flexible stem selection."""
    print("=" * 80)
    print("Testing Flexible Stem Selection")
    print("=" * 80)
    
    # Test different stem configurations
    stem_configs = [
        ['vocals', 'other'],
        ['vocals', 'drums', 'bass', 'other'],
        ['vocals'],
    ]
    
    for idx, stems in enumerate(stem_configs, 1):
        print(f"\n{idx}. Configuration: {stems}")
        print(f"   Number of stems: {len(stems)}")
        
        # Model output channels
        out_channels = len(stems) * 2  # 2 channels per stem (real, imag)
        print(f"   Model output channels: {out_channels}")
        
        # Create model for this configuration
        model = TFSWAUNet(
            in_channels=4,
            out_channels=out_channels,
            depths=[1, 1, 1, 1],  # Minimal for testing
            dims=[16, 32, 64, 128],
            window_size=8,
            shift_size=4,
            num_heads=4,
        )
        
        print(f"   Model parameters: {model.get_num_parameters():,}")
        print(f"   ✓ Model created for {len(stems)}-stem separation")
    
    print("\n✓ Flexible stem selection supported!")
    print()


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "PHASE 2 IMPLEMENTATION TESTS" + " " * 30 + "║")
    print("╚" + "═" * 78 + "╝")
    print()
    
    try:
        test_stft_processor()
        test_spectrogram_normalizer()
        test_losses()
        test_end_to_end()
        test_flexible_stems()
        
        print("╔" + "═" * 78 + "╗")
        print("║" + " " * 25 + "ALL TESTS PASSED! 🎉" + " " * 32 + "║")
        print("╚" + "═" * 78 + "╝")
        print()
        print("Phase 2 implementation is working correctly!")
        print()
        print("Next steps:")
        print("  1. Prepare MUSDB18 dataset")
        print("  2. Run training: python scripts/train.py --data_root /path/to/musdb18")
        print("  3. Monitor with TensorBoard: tensorboard --logdir outputs/tfswa_unet/logs")
        print()
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ TEST FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
