"""
Model quantization utilities for deployment.

Supports dynamic, static, and QAT (Quantization-Aware Training) quantization.

Authors: Zhenyu Yao, Yuping Su, Honghong Yang, Yumei Zhang, Xiaojun Wu
License: CC BY-NC-ND 4.0
"""

import torch
import torch.nn as nn
from torch.quantization import (
    quantize_dynamic as torch_quantize_dynamic,
    get_default_qconfig,
    prepare,
    convert,
    QuantStub,
    DeQuantStub,
)
from typing import Optional, Set, Tuple
import copy
import warnings


def quantize_dynamic(
    model: nn.Module,
    qconfig_spec: Optional[Set[type]] = None,
    dtype: torch.dtype = torch.qint8,
    inplace: bool = False
) -> nn.Module:
    """
    Apply dynamic quantization to model.
    
    Dynamic quantization converts weights to int8 and computes activations
    in int8 on-the-fly. Good for models where memory bandwidth is bottleneck.
    
    Args:
        model: Model to quantize
        qconfig_spec: Set of layer types to quantize
        dtype: Quantization dtype (qint8 or float16)
        inplace: Whether to modify model in-place
    
    Returns:
        Quantized model
    
    Example:
        >>> model = TFSWAUNet(...)
        >>> quantized = quantize_dynamic(
        ...     model,
        ...     qconfig_spec={nn.Linear, nn.Conv2d}
        ... )
    """
    if qconfig_spec is None:
        # Default: quantize Linear and Conv layers
        qconfig_spec = {nn.Linear, nn.Conv2d}
    
    if not inplace:
        model = copy.deepcopy(model)
    
    model.eval()
    
    print(f"Applying dynamic quantization (dtype={dtype})...")
    
    quantized_model = torch_quantize_dynamic(
        model,
        qconfig_spec=qconfig_spec,
        dtype=dtype
    )
    
    # Count quantized layers
    quantized_count = sum(
        1 for module in quantized_model.modules()
        if hasattr(module, 'weight') and module.weight.dtype in [torch.qint8, torch.quint8]
    )
    
    print(f"✓ {quantized_count} layers quantized")
    
    return quantized_model


def quantize_static(
    model: nn.Module,
    calibration_data: torch.utils.data.DataLoader,
    qconfig: Optional[str] = 'fbgemm',
    inplace: bool = False
) -> nn.Module:
    """
    Apply static quantization to model.
    
    Static quantization requires calibration data to determine optimal
    quantization parameters. Provides better accuracy than dynamic.
    
    Args:
        model: Model to quantize
        calibration_data: DataLoader with calibration samples
        qconfig: Quantization configuration ('fbgemm' or 'qnnpack')
        inplace: Whether to modify model in-place
    
    Returns:
        Quantized model
    
    Example:
        >>> model = TFSWAUNet(...)
        >>> quantized = quantize_static(
        ...     model,
        ...     calibration_loader,
        ...     qconfig='fbgemm'
        ... )
    """
    if not inplace:
        model = copy.deepcopy(model)
    
    model.eval()
    
    # Set quantization config
    if qconfig == 'fbgemm':
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    elif qconfig == 'qnnpack':
        model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    else:
        raise ValueError(f"Unknown qconfig: {qconfig}")
    
    print(f"Preparing model for static quantization ({qconfig})...")
    
    # Prepare model
    prepared_model = torch.quantization.prepare(model, inplace=False)
    
    # Calibrate
    print("Calibrating with calibration data...")
    num_batches = 0
    with torch.no_grad():
        for batch in calibration_data:
            if isinstance(batch, (tuple, list)):
                inputs = batch[0]
            else:
                inputs = batch
            
            prepared_model(inputs)
            num_batches += 1
            
            if num_batches >= 100:  # Limit calibration
                break
    
    print(f"Calibrated with {num_batches} batches")
    
    # Convert to quantized model
    print("Converting to quantized model...")
    quantized_model = torch.quantization.convert(prepared_model, inplace=False)
    
    print("✓ Static quantization complete")
    
    return quantized_model


def prepare_qat(
    model: nn.Module,
    qconfig: Optional[str] = 'fbgemm',
    inplace: bool = False
) -> nn.Module:
    """
    Prepare model for Quantization-Aware Training (QAT).
    
    QAT simulates quantization during training to improve accuracy
    of the final quantized model.
    
    Args:
        model: Model to prepare
        qconfig: Quantization configuration
        inplace: Whether to modify model in-place
    
    Returns:
        Model prepared for QAT
    
    Example:
        >>> model = TFSWAUNet(...)
        >>> qat_model = prepare_qat(model)
        >>> 
        >>> # Train with QAT
        >>> for epoch in range(num_epochs):
        ...     train_epoch(qat_model, ...)
        >>> 
        >>> # Convert to quantized model
        >>> quantized = torch.quantization.convert(qat_model)
    """
    if not inplace:
        model = copy.deepcopy(model)
    
    # Set quantization config
    if qconfig == 'fbgemm':
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    elif qconfig == 'qnnpack':
        model.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
    else:
        raise ValueError(f"Unknown qconfig: {qconfig}")
    
    print(f"Preparing model for QAT ({qconfig})...")
    
    # Prepare for QAT
    qat_model = torch.quantization.prepare_qat(model, inplace=False)
    
    print("✓ Model prepared for QAT")
    print("Train the model, then call torch.quantization.convert() to get quantized model")
    
    return qat_model


class QuantizableModel(nn.Module):
    """
    Wrapper to make model quantization-friendly.
    
    Adds QuantStub and DeQuantStub for proper quantization flow.
    
    Example:
        >>> base_model = TFSWAUNet(...)
        >>> quantizable = QuantizableModel(base_model)
        >>> # Now can be quantized
    """
    
    def __init__(self, model: nn.Module):
        """
        Args:
            model: Base model to wrap
        """
        super().__init__()
        self.model = model
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quant/dequant stubs."""
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x
    
    def fuse_model(self):
        """Fuse Conv-BN-ReLU patterns for better quantization."""
        # Model-specific fusion patterns
        # This needs to be customized based on architecture
        pass


def compare_models(
    original_model: nn.Module,
    quantized_model: nn.Module,
    test_input: torch.Tensor,
    metric: str = 'mse'
) -> dict:
    """
    Compare original and quantized models.
    
    Args:
        original_model: Original float model
        quantized_model: Quantized model
        test_input: Test input tensor
        metric: Comparison metric ('mse' or 'mae')
    
    Returns:
        Dictionary with comparison results
    """
    original_model.eval()
    quantized_model.eval()
    
    with torch.no_grad():
        original_output = original_model(test_input)
        quantized_output = quantized_model(test_input)
    
    if metric == 'mse':
        error = ((original_output - quantized_output) ** 2).mean().item()
        error_name = 'MSE'
    elif metric == 'mae':
        error = (original_output - quantized_output).abs().mean().item()
        error_name = 'MAE'
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Calculate model sizes
    def get_model_size(model):
        torch.save(model.state_dict(), 'temp_model.pth')
        import os
        size = os.path.getsize('temp_model.pth') / 1024 / 1024  # MB
        os.remove('temp_model.pth')
        return size
    
    original_size = get_model_size(original_model)
    quantized_size = get_model_size(quantized_model)
    
    compression_ratio = original_size / quantized_size if quantized_size > 0 else 0
    
    results = {
        'error': error,
        'error_metric': error_name,
        'original_size_mb': original_size,
        'quantized_size_mb': quantized_size,
        'compression_ratio': compression_ratio,
        'size_reduction_percent': (1 - quantized_size / original_size) * 100
    }
    
    print(f"\nModel Comparison:")
    print(f"  {error_name}: {error:.6f}")
    print(f"  Original size: {original_size:.2f} MB")
    print(f"  Quantized size: {quantized_size:.2f} MB")
    print(f"  Compression: {compression_ratio:.2f}x")
    print(f"  Size reduction: {results['size_reduction_percent']:.1f}%")
    
    return results


def benchmark_quantized_model(
    original_model: nn.Module,
    quantized_model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 2, 1025, 259),
    num_iterations: int = 100,
    device: str = 'cpu'
) -> dict:
    """
    Benchmark speedup from quantization.
    
    Args:
        original_model: Original model
        quantized_model: Quantized model
        input_shape: Input shape
        num_iterations: Number of iterations
        device: Device to benchmark on
    
    Returns:
        Dictionary with benchmark results
    """
    import time
    
    original_model = original_model.to(device).eval()
    quantized_model = quantized_model.to(device).eval()
    
    test_input = torch.randn(*input_shape, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = original_model(test_input)
            _ = quantized_model(test_input)
    
    # Benchmark original
    start = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = original_model(test_input)
    original_time = time.time() - start
    
    # Benchmark quantized
    start = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = quantized_model(test_input)
    quantized_time = time.time() - start
    
    speedup = original_time / quantized_time if quantized_time > 0 else 0
    
    results = {
        'original_time': original_time,
        'quantized_time': quantized_time,
        'speedup': speedup,
        'original_avg_ms': (original_time / num_iterations) * 1000,
        'quantized_avg_ms': (quantized_time / num_iterations) * 1000,
    }
    
    print(f"\nQuantization Speedup:")
    print(f"  Original: {results['original_avg_ms']:.2f} ms/iter")
    print(f"  Quantized: {results['quantized_avg_ms']:.2f} ms/iter")
    print(f"  Speedup: {speedup:.2f}x")
    
    return results


class QuantizationConfig:
    """
    Configuration for model quantization.
    """
    
    def __init__(
        self,
        method: str = 'dynamic',
        dtype: torch.dtype = torch.qint8,
        qconfig: str = 'fbgemm',
        calibration_batches: int = 100
    ):
        """
        Args:
            method: Quantization method ('dynamic', 'static', 'qat')
            dtype: Quantization dtype
            qconfig: Quantization backend config
            calibration_batches: Number of batches for calibration
        """
        self.method = method
        self.dtype = dtype
        self.qconfig = qconfig
        self.calibration_batches = calibration_batches
    
    def __repr__(self) -> str:
        return (
            f"QuantizationConfig(method={self.method}, "
            f"dtype={self.dtype}, qconfig={self.qconfig})"
        )
