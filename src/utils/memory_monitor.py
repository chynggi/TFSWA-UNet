"""Memory monitoring utilities for debugging VRAM usage."""
import torch
from typing import Optional


def print_gpu_memory_usage(msg: Optional[str] = None):
    """Print current GPU memory usage."""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
    
    if msg:
        print(f"\n{'='*60}")
        print(f"GPU Memory - {msg}")
        print(f"{'='*60}")
    else:
        print(f"\nGPU Memory Usage:")
    
    print(f"  Allocated:     {allocated:.2f} GB")
    print(f"  Reserved:      {reserved:.2f} GB")
    print(f"  Max Allocated: {max_allocated:.2f} GB")
    print(f"  Free:          {reserved - allocated:.2f} GB (in reserved)")


def reset_peak_memory_stats():
    """Reset peak memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def get_memory_summary() -> str:
    """Get detailed memory summary."""
    if not torch.cuda.is_available():
        return "CUDA not available"
    
    return torch.cuda.memory_summary()


def check_memory_leak(threshold_gb: float = 0.5):
    """Check for potential memory leaks."""
    if not torch.cuda.is_available():
        return
    
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    
    unreleased = reserved - allocated
    
    if unreleased > threshold_gb:
        print(f"\n⚠️  Potential memory leak detected!")
        print(f"   Reserved but unallocated: {unreleased:.2f} GB")
        print(f"   Consider calling torch.cuda.empty_cache()")


def optimize_memory_allocation():
    """Apply memory optimization settings."""
    if torch.cuda.is_available():
        # Enable TF32 for faster training on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable cudnn benchmarking for better performance
        torch.backends.cudnn.benchmark = True
        
        # Set memory allocator to reduce fragmentation
        import os
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        print("✓ Memory optimizations enabled")


if __name__ == "__main__":
    # Demo
    optimize_memory_allocation()
    print_gpu_memory_usage("Initial")
    
    if torch.cuda.is_available():
        # Create some tensors
        x = torch.randn(1000, 1000, device='cuda')
        print_gpu_memory_usage("After creating tensor")
        
        # Delete tensor
        del x
        torch.cuda.empty_cache()
        print_gpu_memory_usage("After cleanup")
        
        check_memory_leak()
