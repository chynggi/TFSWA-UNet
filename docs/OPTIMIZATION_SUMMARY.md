# URGENT: Memory Optimization Summary

## Problems Fixed

### 1. ✅ Infinite Recursion Error
**Issue**: `_load_audio_segment_efficient` was recursively calling itself when filtering silent chunks.

**Fix**: Removed recursive retry logic. Silent chunk filtering should be done at dataset level if needed.

### 2. ⚠️ CUDA Out of Memory
**Issue**: Model uses 94+ GB VRAM - far exceeding available GPU memory.

**Root Causes**:
1. Large attention matrices in TSA/FSA operations
2. Long audio segments (originally 6 seconds)
3. Large FFT size (2048 → lots of frequency bins)
4. Batch size too large for available VRAM

## Changes Applied

### Configuration Files Updated

1. **configs/data/musdb.yaml**
   - ✅ `segment_seconds: 6.0 → 3.0` (50% reduction)
   - ✅ `window_size: 2048 → 1024` (50% reduction)
   - ✅ `hop_length: 512 → 256` (proportional reduction)

2. **configs/model/tfswa_unet.yaml**
   - ✅ Added `tsa_chunk_size: 32`
   - ✅ Added `fsa_chunk_size: 32`

3. **scripts/train.py defaults**
   - ✅ `segment_seconds: 6.0 → 3.0`
   - ✅ `n_fft: 2048 → 1024`
   - ✅ `hop_length: 512 → 256`

### Code Changes

1. **src/data/musdb_dataset.py**
   - ✅ Fixed infinite recursion in `_load_audio_segment_efficient`
   - ✅ Added chunk-based audio loading with `soundfile`
   - ✅ Added memory-efficient loading option

2. **scripts/train.py**
   - ✅ Added `--persistent_workers` flag
   - ✅ Added `--prefetch_factor` option
   - ✅ Added `--use_efficient_loading` flag
   - ✅ Added `--min_mean_abs` for silent filtering

### New Files Created

1. **docs/MEMORY_OPTIMIZATION.md** - Comprehensive optimization guide
2. **docs/OOM_EMERGENCY_FIX.md** - Emergency troubleshooting guide
3. **scripts/train_low_vram.sh** - Bash script for low VRAM training
4. **scripts/train_low_vram.ps1** - PowerShell script for Windows
5. **src/utils/memory_monitor.py** - Memory monitoring utilities

## Recommended Next Steps

### Immediate Action (Try This Now)

```bash
# 1. Set environment variable
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 2. Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# 3. Run with minimal settings
python scripts/train.py \
    --batch_size 1 \
    --segment_seconds 3.0 \
    --n_fft 1024 \
    --hop_length 256 \
    --use_efficient_loading \
    --use_amp \
    --use_checkpointing \
    --num_workers 2 \
    --persistent_workers \
    --prefetch_factor 2
```

Or use the pre-configured script:

```bash
bash scripts/train_low_vram.sh  # Linux/Mac
# OR
powershell scripts/train_low_vram.ps1  # Windows
```

### If Still OOM

Try progressively smaller settings:

```bash
# Level 1: Even smaller segments
--segment_seconds 2.0

# Level 2: Smaller FFT
--n_fft 512 --hop_length 128

# Level 3: Reduce model size
# Edit configs/model/tfswa_unet.yaml:
depths: [1, 1, 3, 1]
dims: [16, 32, 64, 128]
num_heads: 4
```

## Memory Usage Estimation

### Before Optimization
- Audio segment: 6s × 44100 Hz × 2 channels = 529,200 samples
- STFT: 2048 FFT → 1025 frequency bins
- Attention matrices: Can be enormous (T×T and F×F)
- **Peak VRAM: 94+ GB** ❌

### After Optimization
- Audio segment: 3s × 44100 Hz × 2 channels = 264,600 samples (50% ↓)
- STFT: 1024 FFT → 513 frequency bins (50% ↓)
- Attention chunking: Process in smaller pieces
- **Expected VRAM: 2-4 GB** ✅

### Memory Reduction
- Spectrogram size: 75% reduction (time × frequency)
- Attention matrices: 87.5% reduction (time² and frequency²)
- Total estimated: **95%+ reduction**

## Testing the Fix

### 1. Check GPU Availability
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
```

### 2. Monitor Memory Usage
```bash
# In separate terminal
watch -n 1 nvidia-smi

# Or
python -c "from src.utils.memory_monitor import print_gpu_memory_usage; print_gpu_memory_usage()"
```

### 3. Test Minimal Configuration
```bash
python scripts/train.py \
    --batch_size 1 \
    --segment_seconds 2.0 \
    --n_fft 512 \
    --hop_length 128 \
    --max_epochs 1 \
    --log_every_n_steps 1
```

## Understanding the Trade-offs

| Setting | VRAM Usage | Quality | Speed |
|---------|-----------|---------|-------|
| segment_seconds | ↓↓↓ | ↓ | ↑ |
| n_fft | ↓↓↓ | ↓↓ | ↑↑ |
| batch_size | ↓↓↓ | = | ↓ |
| model dims | ↓↓ | ↓↓ | ↑ |
| gradient checkpointing | ↓↓ | = | ↓ |
| mixed precision | ↓↓ | ≈ | ↑↑ |

**Key Insight**: 
- Reducing `segment_seconds` has minimal quality impact (more samples per epoch)
- Reducing `n_fft` below 1024 may hurt quality significantly
- Reducing `batch_size` to 1 is fine (can compensate with gradient accumulation)

## Common Pitfalls

### ❌ Don't Do This
```bash
# Too long segments
--segment_seconds 10.0  # Uses ~4x memory

# Too large FFT
--n_fft 4096  # Uses 4x frequency bins

# Large batch without checking VRAM
--batch_size 8  # May OOM

# No memory optimizations
# (not using --use_amp, --use_checkpointing)
```

### ✅ Do This Instead
```bash
# Start small
--segment_seconds 3.0
--n_fft 1024
--batch_size 1

# Enable all optimizations
--use_amp
--use_checkpointing
--use_efficient_loading
--persistent_workers
```

## Support Resources

1. **Emergency Guide**: `docs/OOM_EMERGENCY_FIX.md`
2. **Full Guide**: `docs/MEMORY_OPTIMIZATION.md`
3. **Memory Monitor**: `src/utils/memory_monitor.py`
4. **Low VRAM Scripts**: `scripts/train_low_vram.*`

## Expected Timeline

1. **Immediate** (5 min): Apply environment variables and restart
2. **Short-term** (30 min): Test minimal configuration
3. **Medium-term** (2-4 hours): Find optimal settings for your GPU
4. **Long-term** (ongoing): Train with validated configuration

## Success Criteria

You'll know it's working when:
- ✅ No "CUDA out of memory" errors
- ✅ `nvidia-smi` shows stable VRAM usage (not growing)
- ✅ Training progresses beyond first batch
- ✅ Loss decreases over iterations
- ✅ No infinite recursion errors

## Final Notes

The optimizations make training feasible on consumer GPUs (8-16GB VRAM) while maintaining reasonable quality. The model will still achieve good separation performance, just with:

- More epochs to converge (smaller batch size)
- Slightly lower frequency resolution (smaller FFT)
- More segments per track (shorter segments)

**This is normal and acceptable for research/development!**
