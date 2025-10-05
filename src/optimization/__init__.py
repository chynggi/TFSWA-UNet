"""
Optimization utilities for TFSWA-UNet.

Authors: Zhenyu Yao, Yuping Su, Honghong Yang, Yumei Zhang, Xiaojun Wu
License: CC BY-NC-ND 4.0
"""

from .gradient_checkpoint import (
    enable_gradient_checkpointing,
    checkpoint_sequential,
    GradientCheckpointWrapper,
    get_memory_stats,
    estimate_memory_savings,
)
from .export import (
    export_to_onnx,
    export_to_torchscript,
    optimize_for_inference,
    benchmark_model,
    export_model_info,
)
from .quantization import (
    quantize_dynamic,
    quantize_static,
    prepare_qat,
    compare_models,
    benchmark_quantized_model,
    QuantizableModel,
    QuantizationConfig,
)
from .quantization import (
    quantize_dynamic,
    quantize_static,
    prepare_qat,
)

__all__ = [
    # Gradient checkpointing
    'enable_gradient_checkpointing',
    'checkpoint_sequential',
    # Export
    'export_to_onnx',
    'export_to_torchscript',
    'optimize_for_inference',
    # Quantization
    'quantize_dynamic',
    'quantize_static',
    'prepare_qat',
]
