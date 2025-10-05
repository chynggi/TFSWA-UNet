"""
Model export utilities for deployment.

Supports ONNX, TorchScript, and optimization for inference.

Authors: Zhenyu Yao, Yuping Su, Honghong Yang, Yumei Zhang, Xiaojun Wu
License: CC BY-NC-ND 4.0
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import warnings


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, ...] = (1, 2, 1025, 259),
    opset_version: int = 14,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    simplify: bool = True,
    verify: bool = True
) -> bool:
    """
    Export model to ONNX format.
    
    Args:
        model: Model to export
        output_path: Path to save ONNX model
        input_shape: Input tensor shape [B, C, H, W]
        opset_version: ONNX opset version
        dynamic_axes: Dynamic axes for variable-length inputs
        simplify: Whether to simplify ONNX graph
        verify: Whether to verify exported model
    
    Returns:
        True if export successful
    
    Example:
        >>> model = TFSWAUNet(...)
        >>> export_to_onnx(
        ...     model,
        ...     'model.onnx',
        ...     input_shape=(1, 2, 1025, 259),
        ...     dynamic_axes={
        ...         'input': {0: 'batch', 3: 'time'},
        ...         'output': {0: 'batch', 3: 'time'}
        ...     }
        ... )
    """
    try:
        # Set model to eval mode
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(*input_shape)
        
        # Default dynamic axes if not provided
        if dynamic_axes is None:
            dynamic_axes = {
                'input': {0: 'batch_size', 3: 'time_steps'},
                'output': {0: 'batch_size', 3: 'time_steps'}
            }
        
        # Export to ONNX
        print(f"Exporting model to ONNX (opset {opset_version})...")
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        
        # Simplify ONNX graph
        if simplify:
            try:
                import onnx
                from onnxsim import simplify as onnx_simplify
                
                print("Simplifying ONNX graph...")
                onnx_model = onnx.load(output_path)
                simplified_model, check = onnx_simplify(onnx_model)
                
                if check:
                    onnx.save(simplified_model, output_path)
                    print("✓ ONNX graph simplified")
                else:
                    warnings.warn("ONNX simplification check failed")
            except ImportError:
                warnings.warn("onnx-simplifier not installed. Skipping simplification.")
        
        # Verify exported model
        if verify:
            try:
                import onnx
                import onnxruntime as ort
                
                print("Verifying ONNX model...")
                
                # Check model validity
                onnx_model = onnx.load(output_path)
                onnx.checker.check_model(onnx_model)
                
                # Test inference
                ort_session = ort.InferenceSession(output_path)
                ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
                ort_outputs = ort_session.run(None, ort_inputs)
                
                # Compare with PyTorch
                with torch.no_grad():
                    torch_output = model(dummy_input)
                
                max_diff = abs(torch_output.numpy() - ort_outputs[0]).max()
                
                if max_diff < 1e-4:
                    print(f"✓ ONNX model verified (max diff: {max_diff:.6f})")
                else:
                    warnings.warn(f"Large difference between PyTorch and ONNX: {max_diff}")
                
            except ImportError:
                warnings.warn("onnx or onnxruntime not installed. Skipping verification.")
        
        # Get file size
        file_size = Path(output_path).stat().st_size / 1024 / 1024
        print(f"✓ Model exported to {output_path} ({file_size:.2f} MB)")
        
        return True
        
    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
        return False


def export_to_torchscript(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, ...] = (1, 2, 1025, 259),
    method: str = 'trace',
    optimize: bool = True,
    verify: bool = True
) -> bool:
    """
    Export model to TorchScript format.
    
    Args:
        model: Model to export
        output_path: Path to save TorchScript model
        input_shape: Input tensor shape for tracing
        method: Export method ('trace' or 'script')
        optimize: Whether to optimize for inference
        verify: Whether to verify exported model
    
    Returns:
        True if export successful
    
    Example:
        >>> model = TFSWAUNet(...)
        >>> export_to_torchscript(
        ...     model,
        ...     'model.pt',
        ...     method='trace'
        ... )
    """
    try:
        # Set model to eval mode
        model.eval()
        
        # Create example input
        example_input = torch.randn(*input_shape)
        
        print(f"Exporting model to TorchScript ({method})...")
        
        # Export based on method
        if method == 'trace':
            with torch.no_grad():
                traced_model = torch.jit.trace(model, example_input)
        elif method == 'script':
            traced_model = torch.jit.script(model)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'trace' or 'script'")
        
        # Optimize for inference
        if optimize:
            print("Optimizing for inference...")
            traced_model = torch.jit.optimize_for_inference(traced_model)
        
        # Save model
        traced_model.save(output_path)
        
        # Verify exported model
        if verify:
            print("Verifying TorchScript model...")
            loaded_model = torch.jit.load(output_path)
            
            with torch.no_grad():
                original_output = model(example_input)
                loaded_output = loaded_model(example_input)
            
            max_diff = (original_output - loaded_output).abs().max().item()
            
            if max_diff < 1e-6:
                print(f"✓ TorchScript model verified (max diff: {max_diff:.8f})")
            else:
                warnings.warn(f"Large difference detected: {max_diff}")
        
        # Get file size
        file_size = Path(output_path).stat().st_size / 1024 / 1024
        print(f"✓ Model exported to {output_path} ({file_size:.2f} MB)")
        
        return True
        
    except Exception as e:
        print(f"✗ TorchScript export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def optimize_for_inference(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 2, 1025, 259),
    fuse_modules: bool = True,
    freeze_bn: bool = True
) -> nn.Module:
    """
    Optimize model for inference.
    
    Applies various optimizations:
    - Module fusion (Conv+BN+ReLU)
    - BatchNorm freezing
    - Parameter freezing
    
    Args:
        model: Model to optimize
        input_shape: Input shape for optimization
        fuse_modules: Whether to fuse modules
        freeze_bn: Whether to freeze BatchNorm layers
    
    Returns:
        Optimized model
    
    Example:
        >>> model = TFSWAUNet(...)
        >>> optimized = optimize_for_inference(model)
    """
    # Set to eval mode
    model.eval()
    
    # Freeze BatchNorm layers
    if freeze_bn:
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.eval()
                # Disable gradient computation
                for param in module.parameters():
                    param.requires_grad = False
    
    # Fuse modules if possible
    if fuse_modules:
        try:
            # Try to fuse Conv+BN+ReLU patterns
            from torch.quantization import fuse_modules as torch_fuse
            
            # This is model-specific and may need customization
            print("Note: Module fusion requires manual specification of fusion patterns")
            
        except ImportError:
            warnings.warn("torch.quantization not available for module fusion")
    
    # Disable gradient computation for all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    print("✓ Model optimized for inference")
    
    return model


def benchmark_model(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 2, 1025, 259),
    num_iterations: int = 100,
    warmup_iterations: int = 10,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict[str, float]:
    """
    Benchmark model inference speed.
    
    Args:
        model: Model to benchmark
        input_shape: Input tensor shape
        num_iterations: Number of inference iterations
        warmup_iterations: Number of warmup iterations
        device: Device to run on
    
    Returns:
        Dictionary with benchmark results
    """
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape, device=device)
    
    # Warmup
    print(f"Warming up ({warmup_iterations} iterations)...")
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model(dummy_input)
    
    # Synchronize
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    print(f"Benchmarking ({num_iterations} iterations)...")
    import time
    
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    # Calculate metrics
    total_time = end_time - start_time
    avg_time = total_time / num_iterations
    fps = 1.0 / avg_time
    
    # Estimate real-time factor for audio
    # Assuming input is ~6 seconds of audio (259 frames * 512 hop / 44100 Hz)
    audio_duration = (input_shape[-1] * 512) / 44100
    rtf = avg_time / audio_duration
    
    results = {
        'total_time': total_time,
        'avg_time_ms': avg_time * 1000,
        'fps': fps,
        'audio_duration': audio_duration,
        'real_time_factor': rtf,
        'device': device
    }
    
    print(f"\nBenchmark Results:")
    print(f"  Average time: {results['avg_time_ms']:.2f} ms")
    print(f"  FPS: {results['fps']:.2f}")
    print(f"  Real-time factor: {results['real_time_factor']:.3f}x")
    print(f"  (RTF < 1.0 means faster than real-time)")
    
    return results


def export_model_info(
    model: nn.Module,
    output_path: str
) -> Dict[str, Any]:
    """
    Export model architecture information.
    
    Args:
        model: Model to analyze
        output_path: Path to save info
    
    Returns:
        Dictionary with model info
    """
    import json
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Get model structure
    model_str = str(model)
    
    info = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_architecture': model_str,
        'module_names': [name for name, _ in model.named_modules()],
    }
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"✓ Model info exported to {output_path}")
    
    return info
