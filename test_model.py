"""Test script to verify TFSWA-UNet implementation."""
import torch
from src.models.tfswa_unet import TFSWAUNet


def test_model_forward():
    """Test forward pass with dummy input."""
    print("=" * 60)
    print("Testing TFSWA-UNet Forward Pass")
    print("=" * 60)
    
    # Model configuration (from config file)
    config = {
        "in_channels": 2,
        "out_channels": 2,
        "depths": [2, 2, 6, 2],
        "dims": [32, 64, 128, 256],
        "window_size": 8,
        "shift_size": 4,
        "num_heads": 8,
    }
    
    # Create model
    print("\n1. Creating model...")
    model = TFSWAUNet(**config)
    print(f"   ✓ Model created successfully")
    
    # Print model info
    info = model.get_model_info()
    print(f"\n2. Model Information:")
    print(f"   - Architecture: {info['architecture']}")
    print(f"   - Input channels: {info['in_channels']}")
    print(f"   - Output channels: {info['out_channels']}")
    print(f"   - Depths per stage: {info['depths']}")
    print(f"   - Dimensions per stage: {info['dims']}")
    print(f"   - Total parameters: {info['num_parameters']:,}")
    print(f"   - Model size: ~{info['num_parameters'] * 4 / 1024 / 1024:.2f} MB (FP32)")
    
    # Create dummy input (simulating STFT spectrogram)
    batch_size = 2
    time_frames = 256  # T dimension
    freq_bins = 512    # F dimension (for 4096 window size, ~2048 freq bins, using 512 for memory)
    
    print(f"\n3. Creating dummy input...")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Time frames: {time_frames}")
    print(f"   - Frequency bins: {freq_bins}")
    print(f"   - Input shape: ({batch_size}, {config['in_channels']}, {time_frames}, {freq_bins})")
    
    x = torch.randn(batch_size, config["in_channels"], time_frames, freq_bins)
    
    # Forward pass
    print(f"\n4. Running forward pass...")
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"   ✓ Forward pass successful!")
    print(f"   - Output shape: {tuple(output.shape)}")
    print(f"   - Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    # Check output properties
    print(f"\n5. Verifying output properties...")
    assert output.shape == (batch_size, config["out_channels"], time_frames, freq_bins), \
        f"Output shape mismatch! Expected {(batch_size, config['out_channels'], time_frames, freq_bins)}, got {output.shape}"
    print(f"   ✓ Output shape correct")
    
    assert output.min() >= 0 and output.max() <= 1, \
        f"Output should be in [0, 1] range (using Sigmoid), got [{output.min():.4f}, {output.max():.4f}]"
    print(f"   ✓ Output range correct (masked values in [0, 1])")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    

def test_gradient_flow():
    """Test if gradients flow through the model."""
    print("\n" + "=" * 60)
    print("Testing Gradient Flow")
    print("=" * 60)
    
    config = {
        "in_channels": 2,
        "out_channels": 2,
        "depths": [2, 2, 6, 2],
        "dims": [32, 64, 128, 256],
        "window_size": 8,
        "shift_size": 4,
        "num_heads": 8,
    }
    
    model = TFSWAUNet(**config)
    model.train()
    
    # Create dummy input and target
    x = torch.randn(1, 2, 128, 256, requires_grad=True)
    target = torch.randn(1, 2, 128, 256)
    
    # Forward pass
    output = model(x)
    
    # Compute loss
    loss = torch.nn.functional.mse_loss(output, target)
    print(f"\n1. Loss computed: {loss.item():.6f}")
    
    # Backward pass
    loss.backward()
    print(f"2. Backward pass successful")
    
    # Check gradients
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for _ in model.parameters())
    print(f"3. Gradients computed: {has_grad}/{total_params} parameters")
    
    assert has_grad == total_params, "Not all parameters have gradients!"
    print(f"   ✓ All parameters have gradients")
    
    # Check gradient magnitudes
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    avg_grad_norm = sum(grad_norms) / len(grad_norms)
    max_grad_norm = max(grad_norms)
    
    print(f"4. Gradient statistics:")
    print(f"   - Average gradient norm: {avg_grad_norm:.6f}")
    print(f"   - Max gradient norm: {max_grad_norm:.6f}")
    
    print("\n" + "=" * 60)
    print("✓ Gradient flow test passed!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_model_forward()
        test_gradient_flow()
        
        print("\n" + "🎉" * 20)
        print("ALL TESTS PASSED! TFSWA-UNet implementation is working correctly.")
        print("🎉" * 20)
        
    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
