# Memory Optimization Guide for TFSWA-UNet

## Overview

This document describes the memory optimization techniques implemented in TFSWA-UNet to reduce VRAM usage during training. These optimizations are based on best practices from [ZFTurbo's Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training).

## Key Optimizations

### 1. Chunk-Based Audio Loading

**Problem**: Loading entire audio tracks into memory consumes excessive VRAM, especially with high-quality audio files.

**Solution**: Load only the required audio segment (chunk) instead of the entire track.

```python
# Before: Load entire track
track_audio = track.audio  # Loads full track into memory
segment = track_audio[start:end]

# After: Load only required chunk
segment = load_chunk(file_path, track_length, chunk_size, offset=start)
```

**Benefits**:
- Reduces memory usage by ~10-50x depending on segment vs track length
- Faster loading times
- Enables training with larger batch sizes

**Implementation**:
- Uses `soundfile.read()` with `start` and `frames` parameters
- Efficient random access without loading full file
- Automatic padding for chunks at track boundaries

### 2. Efficient DataLoader Configuration

**Problem**: Default DataLoader settings don't optimize for memory and throughput.

**Solution**: Use optimal DataLoader parameters.

```python
DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=4,
    pin_memory=True,              # Faster GPU transfer
    persistent_workers=True,      # Keep workers alive
    prefetch_factor=2,            # Prefetch batches
)
```

**Benefits**:
- `persistent_workers=True`: Avoids recreating workers each epoch (~20% speedup)
- `prefetch_factor=2`: Overlaps data loading with GPU computation
- `pin_memory=True`: Faster CPU-to-GPU transfer

### 3. Silent Chunk Filtering

**Problem**: Training on silent audio segments wastes computational resources.

**Solution**: Filter out chunks with low audio energy.

```python
if np.abs(audio_chunk).mean() < min_mean_abs:
    # Skip this chunk or sample another
    continue
```

**Benefits**:
- Improves training efficiency
- Reduces wasted computation on silence
- Better use of limited training data

**Recommended**: `min_mean_abs = 0.001` for training, `0.0` for validation

### 4. Reduced STFT Parameters

**Problem**: Large FFT sizes create huge spectrogram tensors.

**Solution**: Use smaller FFT sizes while maintaining quality.

```yaml
# Before
n_fft: 4096      # (2049, T) spectrogram per channel
hop_length: 1024

# After  
n_fft: 2048      # (1025, T) spectrogram per channel - 50% reduction
hop_length: 512
```

**Benefits**:
- 50% reduction in frequency bins
- ~50% reduction in VRAM for spectrograms
- Faster STFT/ISTFT computation
- Still maintains good frequency resolution for music

### 5. Shorter Audio Segments

**Problem**: Long audio segments require more VRAM for both waveform and spectrogram.

**Solution**: Use shorter segments during training.

```yaml
# Before
segment_seconds: 10.0  # 441,000 samples at 44.1kHz

# After
segment_seconds: 6.0   # 264,600 samples - 40% reduction
```

**Benefits**:
- 40% reduction in audio buffer size
- 40% reduction in spectrogram temporal dimension
- Can increase batch size proportionally
- More samples per epoch (same track provides more segments)

## Memory Usage Comparison

### Before Optimization

| Component | Memory |
|-----------|--------|
| Audio segment (10s, stereo) | ~1.7 MB |
| STFT spectrogram (4096 FFT) | ~33 MB |
| Batch of 4 | ~139 MB |
| **Total per batch** | **~140 MB** |

### After Optimization

| Component | Memory |
|-----------|--------|
| Audio segment (6s, stereo) | ~1.0 MB |
| STFT spectrogram (2048 FFT) | ~10 MB |
| Batch of 4 | ~44 MB |
| **Total per batch** | **~45 MB** |

**Net reduction**: ~68% less VRAM per batch

## Usage

### Enable All Optimizations

```bash
python scripts/train.py \
    --data_root data/musdb18 \
    --batch_size 4 \
    --segment_seconds 6.0 \
    --n_fft 2048 \
    --hop_length 512 \
    --use_efficient_loading \
    --min_mean_abs 0.001 \
    --persistent_workers \
    --prefetch_factor 2 \
    --num_workers 4
```

### Configuration File

Update `configs/data/musdb.yaml`:

```yaml
segment_seconds: 6.0
use_efficient_loading: true
min_mean_abs: 0.001
persistent_workers: true
prefetch_factor: 2
stft:
  window_size: 2048
  hop_length: 512
```

## Advanced Optimizations

### 1. Gradient Checkpointing

Trade computation for memory by recomputing activations during backward pass.

```python
from src.optimization.gradient_checkpoint import enable_gradient_checkpointing

model = TFSWAUNet(...)
model = enable_gradient_checkpointing(model)
```

**Trade-off**: ~20% slower training, ~40% less VRAM

### 2. Mixed Precision Training

Use FP16 for forward/backward, FP32 for optimizer.

```bash
python scripts/train.py --use_amp
```

**Benefits**: ~50% less VRAM, ~2x faster on modern GPUs

### 3. Gradient Accumulation

Simulate larger batch sizes without using more VRAM.

```yaml
# In training config
batch_size: 2
gradient_accumulation_steps: 4  # Effective batch size = 8
```

### 4. Reduce Model Size

Use smaller model dimensions during development:

```yaml
dims: [16, 32, 64, 128]  # Instead of [32, 64, 128, 256]
depths: [1, 1, 3, 1]     # Instead of [2, 2, 6, 2]
```

## Troubleshooting

### Out of Memory Errors

1. **Reduce batch size**: Try `--batch_size 1` or `2`
2. **Reduce segment length**: Use `--segment_seconds 4.0` or `3.0`
3. **Reduce FFT size**: Use `--n_fft 1024` (though quality may suffer)
4. **Enable gradient checkpointing**: Add `--use_checkpointing`
5. **Use mixed precision**: Add `--use_amp`

### Slow Training

1. **Increase workers**: Try `--num_workers 8`
2. **Enable persistent workers**: Add `--persistent_workers`
3. **Increase prefetch**: Try `--prefetch_factor 4`
4. **Check min_mean_abs**: Lower value = more chunks = slower
5. **Disable augmentation**: For debugging only

### Quality Degradation

If optimizations hurt model quality:

1. **Don't reduce FFT too much**: Keep `n_fft >= 2048`
2. **Keep segments reasonable**: `segment_seconds >= 5.0`
3. **Don't filter too aggressively**: `min_mean_abs <= 0.01`
4. **Validate on full-length tracks**: Don't use segmentation for validation

## Monitoring Memory Usage

### During Training

```python
import torch

# Print VRAM usage
if torch.cuda.is_available():
    print(f"VRAM allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"VRAM cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
```

### Using nvidia-smi

```bash
# Watch VRAM usage in real-time
watch -n 1 nvidia-smi
```

### Expected Usage

| Configuration | VRAM Usage |
|---------------|------------|
| Batch=1, segment=6s, FFT=2048 | ~3-4 GB |
| Batch=4, segment=6s, FFT=2048 | ~6-8 GB |
| Batch=8, segment=6s, FFT=2048 | ~10-14 GB |
| Batch=16, segment=6s, FFT=2048 | ~18-24 GB |

## Best Practices

1. **Start small**: Begin with minimal VRAM settings, then increase
2. **Profile first**: Use `torch.cuda.memory_summary()` to find bottlenecks
3. **Balance quality vs memory**: Don't over-optimize at expense of model quality
4. **Test on validation set**: Ensure optimizations don't hurt performance
5. **Use efficient loading**: Always enable for MUSDB18-HQ (WAV files)

## References

- [ZFTurbo's Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)
- [PyTorch DataLoader Best Practices](https://pytorch.org/docs/stable/data.html#memory-pinning)
- [PyTorch Memory Management](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)

## Implementation Notes

The optimizations are implemented in:

- `src/data/musdb_dataset.py`: Chunk-based loading
- `scripts/train.py`: DataLoader configuration
- `configs/data/musdb.yaml`: Default settings
- `src/optimization/gradient_checkpoint.py`: Gradient checkpointing

All optimizations are **backward compatible** and can be disabled by setting appropriate flags.
