"""
Phase 4 Integration Test Suite

Tests optimization utilities:
- Gradient checkpointing
- Model export (ONNX, TorchScript)
- Quantization (dynamic, static)
"""

import torch
import torch.nn as nn
import pytest
import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.tfswa_unet import TFSWAUNet
from src.optimization.gradient_checkpoint import (
    enable_gradient_checkpointing,
    checkpoint_sequential,
    GradientCheckpointWrapper,
    get_memory_stats,
    estimate_memory_savings,
)
from src.optimization.export import (
    export_to_onnx,
    export_to_torchscript,
    optimize_for_inference,
    benchmark_model,
)
from src.optimization.quantization import (
    quantize_dynamic,
    compare_models,
    benchmark_quantized_model,
)


@pytest.fixture
def small_model():
    """Create small model for testing."""
    model = TFSWAUNet(
        in_channels=2,
        out_channels=2,
        depths=[1, 1, 1, 1],
        dims=[16, 32, 64, 128],
        window_size=4,
        num_heads=4,
        mlp_ratio=2.0,
    )
    return model


@pytest.fixture
def test_input():
    """Create test input."""
    return torch.randn(1, 2, 128, 64)


@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp)


def test_gradient_checkpointing_enable(small_model, test_input):
    """Test enabling gradient checkpointing."""
    print("\n=== Test 1: Enable Gradient Checkpointing ===")
    
    # Enable checkpointing
    checkpointed_model = enable_gradient_checkpointing(small_model)
    
    # Test forward pass
    output = checkpointed_model(test_input)
    assert output.shape == (1, 2, 128, 64)
    
    # Test backward pass
    loss = output.mean()
    loss.backward()
    
    # Check gradients exist
    has_gradients = any(
        p.grad is not None for p in checkpointed_model.parameters()
    )
    assert has_gradients, "No gradients after backward pass"
    
    print("✓ Gradient checkpointing enabled successfully")


def test_checkpoint_sequential(test_input):
    """Test checkpoint_sequential function."""
    print("\n=== Test 2: Checkpoint Sequential ===")
    
    # Create sequential layers
    layers = nn.Sequential(
        nn.Conv2d(2, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 2, 3, padding=1),
    )
    
    # Apply checkpointing
    output = checkpoint_sequential(layers, 2, test_input)
    assert output.shape == test_input.shape
    
    print("✓ Sequential checkpointing works")


def test_memory_estimation(small_model):
    """Test memory estimation."""
    print("\n=== Test 3: Memory Estimation ===")
    
    savings = estimate_memory_savings(
        small_model,
        input_shape=(1, 2, 128, 64)
    )
    
    assert 'estimated_savings_mb' in savings
    assert 'estimated_savings_percent' in savings
    assert savings['estimated_savings_percent'] > 0
    
    print(f"✓ Estimated memory savings: {savings['estimated_savings_percent']:.1f}%")


def test_onnx_export(small_model, test_input, temp_dir):
    """Test ONNX export."""
    print("\n=== Test 4: ONNX Export ===")
    
    output_path = os.path.join(temp_dir, "model.onnx")
    
    success = export_to_onnx(
        small_model,
        test_input,
        output_path,
        dynamic_axes={
            'input': {0: 'batch', 2: 'freq', 3: 'time'},
            'output': {0: 'batch', 2: 'freq', 3: 'time'}
        },
        verify=True,
    )
    
    assert success, "ONNX export failed"
    assert os.path.exists(output_path), "ONNX file not created"
    
    # Check file size
    file_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"✓ ONNX export successful ({file_size:.2f} MB)")


def test_torchscript_export(small_model, test_input, temp_dir):
    """Test TorchScript export."""
    print("\n=== Test 5: TorchScript Export ===")
    
    output_path = os.path.join(temp_dir, "model.pt")
    
    scripted_model = export_to_torchscript(
        small_model,
        test_input,
        output_path,
        method='trace',
        verify=True,
    )
    
    assert scripted_model is not None, "TorchScript export failed"
    assert os.path.exists(output_path), "TorchScript file not created"
    
    # Test inference
    output = scripted_model(test_input)
    assert output.shape == (1, 2, 128, 64)
    
    print("✓ TorchScript export successful")


def test_optimize_for_inference(small_model):
    """Test inference optimization."""
    print("\n=== Test 6: Optimize for Inference ===")
    
    optimized = optimize_for_inference(small_model)
    
    # Check model is in eval mode
    assert not optimized.training
    
    # Check all batchnorm layers are fused or in eval mode
    for module in optimized.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            assert not module.training
    
    print("✓ Model optimized for inference")


def test_model_benchmark(small_model, test_input):
    """Test model benchmarking."""
    print("\n=== Test 7: Model Benchmark ===")
    
    stats = benchmark_model(
        small_model,
        test_input,
        num_iterations=10,
        warmup_iterations=2,
    )
    
    assert 'avg_time_ms' in stats
    assert 'throughput_samples_per_sec' in stats
    assert stats['avg_time_ms'] > 0
    
    print(f"✓ Benchmark complete: {stats['avg_time_ms']:.2f} ms/iter")


def test_dynamic_quantization(small_model, test_input):
    """Test dynamic quantization."""
    print("\n=== Test 8: Dynamic Quantization ===")
    
    # Quantize model
    quantized = quantize_dynamic(
        small_model,
        qconfig_spec={nn.Linear, nn.Conv2d},
        dtype=torch.qint8,
    )
    
    # Test inference
    with torch.no_grad():
        output = quantized(test_input)
    
    assert output.shape == (1, 2, 128, 64)
    
    print("✓ Dynamic quantization successful")


def test_model_comparison(small_model, test_input):
    """Test model comparison."""
    print("\n=== Test 9: Model Comparison ===")
    
    # Quantize
    quantized = quantize_dynamic(small_model)
    
    # Compare
    results = compare_models(
        small_model,
        quantized,
        test_input,
        metric='mse'
    )
    
    assert 'error' in results
    assert 'compression_ratio' in results
    assert results['compression_ratio'] > 1.0
    
    print(f"✓ Comparison complete: {results['compression_ratio']:.2f}x compression")


def test_quantization_benchmark(small_model):
    """Test quantization speedup benchmark."""
    print("\n=== Test 10: Quantization Speedup ===")
    
    # Quantize
    quantized = quantize_dynamic(small_model)
    
    # Benchmark
    results = benchmark_quantized_model(
        small_model,
        quantized,
        input_shape=(1, 2, 128, 64),
        num_iterations=20,
        device='cpu',
    )
    
    assert 'speedup' in results
    assert results['speedup'] > 0
    
    print(f"✓ Speedup: {results['speedup']:.2f}x")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PHASE 4 OPTIMIZATION TEST SUITE")
    print("=" * 70)
    
    # Create fixtures
    model = TFSWAUNet(
        in_channels=2,
        out_channels=2,
        depths=[1, 1, 1, 1],
        dims=[16, 32, 64, 128],
        window_size=4,
        num_heads=4,
        mlp_ratio=2.0,
    )
    test_input = torch.randn(1, 2, 128, 64)
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Run tests
        test_gradient_checkpointing_enable(model, test_input)
        test_checkpoint_sequential(test_input)
        test_memory_estimation(model)
        test_onnx_export(model, test_input, temp_dir)
        test_torchscript_export(model, test_input, temp_dir)
        test_optimize_for_inference(model)
        test_model_benchmark(model, test_input)
        test_dynamic_quantization(model, test_input)
        test_model_comparison(model, test_input)
        test_quantization_benchmark(model)
        
        print("\n" + "=" * 70)
        print("✅ ALL 10 TESTS PASSED")
        print("=" * 70)
        
    finally:
        shutil.rmtree(temp_dir)
