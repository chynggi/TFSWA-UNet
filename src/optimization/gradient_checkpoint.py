"""
Gradient checkpointing utilities for memory optimization.

Implements gradient checkpointing to reduce memory usage during training
by trading compute for memory.

Authors: Zhenyu Yao, Yuping Su, Honghong Yang, Yumei Zhang, Xiaojun Wu
License: CC BY-NC-ND 4.0
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import List, Callable, Any
import warnings


def enable_gradient_checkpointing(
    model: nn.Module,
    checkpoint_blocks: List[str] = None
) -> nn.Module:
    """
    Enable gradient checkpointing for specified modules.
    
    Args:
        model: Model to enable checkpointing on
        checkpoint_blocks: List of module names to checkpoint
                          If None, checkpoints all TFSWABlock modules
    
    Returns:
        Model with gradient checkpointing enabled
    
    Example:
        >>> model = TFSWAUNet(...)
        >>> model = enable_gradient_checkpointing(model)
        >>> # During training, memory usage reduced by ~40%
    """
    if checkpoint_blocks is None:
        # Default: checkpoint all TFSWA blocks
        checkpoint_blocks = ['TFSWABlock']
    
    checkpointed_count = 0
    
    def _enable_checkpoint(module: nn.Module, name: str = ''):
        nonlocal checkpointed_count
        
        # Check if this module should be checkpointed
        module_type = type(module).__name__
        if any(block_name in module_type for block_name in checkpoint_blocks):
            # Store original forward
            original_forward = module.forward
            
            # Create checkpointed forward
            def checkpointed_forward(*args, **kwargs):
                # Only use checkpointing during training
                if module.training:
                    return checkpoint(original_forward, *args, **kwargs, use_reentrant=False)
                else:
                    return original_forward(*args, **kwargs)
            
            # Replace forward method
            module.forward = checkpointed_forward
            checkpointed_count += 1
        
        # Recursively apply to children
        for child_name, child in module.named_children():
            _enable_checkpoint(child, f"{name}.{child_name}" if name else child_name)
    
    _enable_checkpoint(model)
    
    print(f"Gradient checkpointing enabled for {checkpointed_count} modules")
    
    return model


def checkpoint_sequential(
    functions: List[Callable],
    segments: int,
    *inputs,
    **kwargs
) -> Any:
    """
    Checkpoint a sequential series of functions.
    
    Divides the functions into segments and checkpoints each segment
    to reduce memory usage.
    
    Args:
        functions: List of functions to apply sequentially
        segments: Number of checkpoint segments
        *inputs: Inputs to first function
        **kwargs: Additional arguments
    
    Returns:
        Output of final function
    
    Example:
        >>> layers = [layer1, layer2, layer3, layer4]
        >>> output = checkpoint_sequential(layers, segments=2, x)
    """
    if segments <= 0:
        raise ValueError("segments must be positive")
    
    if segments > len(functions):
        segments = len(functions)
    
    # Calculate segment size
    segment_size = (len(functions) + segments - 1) // segments
    
    def run_segment(start_idx: int, end_idx: int, *segment_inputs):
        """Run a segment of functions."""
        output = segment_inputs
        for idx in range(start_idx, min(end_idx, len(functions))):
            if isinstance(output, tuple):
                output = functions[idx](*output)
            else:
                output = functions[idx](output)
        return output
    
    # Process segments
    output = inputs
    for i in range(0, len(functions), segment_size):
        segment_end = min(i + segment_size, len(functions))
        
        # Checkpoint this segment
        if isinstance(output, tuple):
            output = checkpoint(
                run_segment,
                i,
                segment_end,
                *output,
                use_reentrant=False
            )
        else:
            output = checkpoint(
                run_segment,
                i,
                segment_end,
                output,
                use_reentrant=False
            )
    
    return output


class GradientCheckpointWrapper(nn.Module):
    """
    Wrapper module that applies gradient checkpointing to forward pass.
    
    Example:
        >>> block = TFSWABlock(...)
        >>> checkpointed_block = GradientCheckpointWrapper(block)
    """
    
    def __init__(self, module: nn.Module):
        """
        Args:
            module: Module to wrap with gradient checkpointing
        """
        super().__init__()
        self.module = module
    
    def forward(self, *args, **kwargs):
        """Forward pass with optional gradient checkpointing."""
        if self.training:
            return checkpoint(
                self.module,
                *args,
                **kwargs,
                use_reentrant=False
            )
        else:
            return self.module(*args, **kwargs)


def get_memory_stats() -> dict:
    """
    Get current GPU memory statistics.
    
    Returns:
        Dictionary with memory statistics
    """
    if not torch.cuda.is_available():
        return {
            'allocated': 0,
            'reserved': 0,
            'max_allocated': 0,
        }
    
    return {
        'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
        'reserved': torch.cuda.memory_reserved() / 1024**3,    # GB
        'max_allocated': torch.cuda.max_memory_allocated() / 1024**3,  # GB
    }


def estimate_memory_savings(
    model: nn.Module,
    input_shape: tuple,
    batch_size: int = 1,
    device: str = 'cuda'
) -> dict:
    """
    Estimate memory savings from gradient checkpointing.
    
    Args:
        model: Model to analyze
        input_shape: Input tensor shape (C, H, W)
        batch_size: Batch size
        device: Device to test on
    
    Returns:
        Dictionary with memory statistics
    """
    if not torch.cuda.is_available():
        warnings.warn("CUDA not available, skipping memory estimation")
        return {}
    
    # Move model to device
    model = model.to(device)
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, *input_shape, device=device)
    
    # Test without checkpointing
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    model.train()
    output = model(dummy_input)
    loss = output.mean()
    loss.backward()
    
    memory_without = get_memory_stats()
    
    # Test with checkpointing
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    model_checkpointed = enable_gradient_checkpointing(model)
    model_checkpointed.train()
    
    output = model_checkpointed(dummy_input)
    loss = output.mean()
    loss.backward()
    
    memory_with = get_memory_stats()
    
    # Calculate savings
    savings = {
        'without_checkpoint': memory_without,
        'with_checkpoint': memory_with,
        'savings_gb': memory_without['max_allocated'] - memory_with['max_allocated'],
        'savings_percent': (
            (memory_without['max_allocated'] - memory_with['max_allocated']) /
            memory_without['max_allocated'] * 100
        ) if memory_without['max_allocated'] > 0 else 0
    }
    
    return savings


class CheckpointConfig:
    """
    Configuration for gradient checkpointing.
    """
    
    def __init__(
        self,
        enabled: bool = False,
        checkpoint_blocks: List[str] = None,
        num_segments: int = 1
    ):
        """
        Args:
            enabled: Whether checkpointing is enabled
            checkpoint_blocks: Module types to checkpoint
            num_segments: Number of checkpoint segments
        """
        self.enabled = enabled
        self.checkpoint_blocks = checkpoint_blocks or ['TFSWABlock']
        self.num_segments = num_segments
    
    def apply(self, model: nn.Module) -> nn.Module:
        """
        Apply checkpointing configuration to model.
        
        Args:
            model: Model to configure
        
        Returns:
            Configured model
        """
        if self.enabled:
            return enable_gradient_checkpointing(
                model,
                checkpoint_blocks=self.checkpoint_blocks
            )
        return model
    
    def __repr__(self) -> str:
        return (
            f"CheckpointConfig(enabled={self.enabled}, "
            f"blocks={self.checkpoint_blocks}, "
            f"segments={self.num_segments})"
        )
