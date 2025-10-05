# EMERGENCY: CUDA Out of Memory Fix

## 🚨 Immediate Actions

If you're getting "CUDA out of memory" errors, follow these steps **in order**:

### 1. Quick Fix (Try First)

```bash
# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# Set memory allocator environment variable
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Linux/Mac
# OR
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"  # Windows PowerShell
```

### 2. Minimal VRAM Configuration

Use the absolute minimum settings:

```bash
python scripts/train.py \
    --batch_size 1 \
    --segment_seconds 2.0 \
    --n_fft 512 \
    --hop_length 128 \
    --use_efficient_loading \
    --use_amp \
    --use_checkpointing \
    --num_workers 0
```

Or use the provided script:

```bash
# Linux/Mac
bash scripts/train_low_vram.sh

# Windows
powershell scripts/train_low_vram.ps1
```

### 3. Reduce Model Size

Edit `configs/model/tfswa_unet.yaml`:

```yaml
# Smaller model configuration
depths: [1, 1, 3, 1]      # Instead of [2, 2, 6, 2]
dims: [16, 32, 64, 128]   # Instead of [32, 64, 128, 256]
num_heads: 4              # Instead of 8
window_size: 4            # Instead of 8
tsa_chunk_size: 16        # Added for memory
fsa_chunk_size: 16        # Added for memory
```

### 4. Progressive Reduction Strategy

Try these settings in order until training works:

#### Level 1: Moderate (8-12GB VRAM)
```bash
--batch_size 2 \
--segment_seconds 4.0 \
--n_fft 1024 \
--hop_length 256
```

#### Level 2: Conservative (6-8GB VRAM)
```bash
--batch_size 1 \
--segment_seconds 3.0 \
--n_fft 1024 \
--hop_length 256
```

#### Level 3: Extreme (4-6GB VRAM)
```bash
--batch_size 1 \
--segment_seconds 2.0 \
--n_fft 512 \
--hop_length 128
```

#### Level 4: Last Resort (< 4GB VRAM)
```bash
--batch_size 1 \
--segment_seconds 1.0 \
--n_fft 512 \
--hop_length 128 \
--dims 8 16 32 64 \
--depths 1 1 2 1
```

## 🔍 Debugging Steps

### Check Current Memory Usage

```python
import torch
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
```

### Find Memory Bottleneck

Add to your training script:

```python
from src.utils.memory_monitor import print_gpu_memory_usage

# After model creation
print_gpu_memory_usage("After model creation")

# After loading batch
print_gpu_memory_usage("After loading batch")

# After forward pass
print_gpu_memory_usage("After forward pass")

# After backward pass
print_gpu_memory_usage("After backward pass")
```

### Monitor During Training

```bash
# In a separate terminal, watch GPU usage
watch -n 1 nvidia-smi

# Or on Windows
while ($true) { nvidia-smi; sleep 1; clear }
```

## 🛠️ Advanced Fixes

### 1. Gradient Accumulation

Simulate larger batch sizes without using more memory:

```python
# In train.py or config
gradient_accumulation_steps: 4  # Effective batch_size = 1 * 4 = 4
```

### 2. CPU Offloading

Move some tensors to CPU during training:

```python
# Not recommended for this model but possible
torch.cuda.set_per_process_memory_fraction(0.9)
```

### 3. Reduce Precision Further

Use bfloat16 instead of float16:

```python
# In trainer.py
scaler = torch.cuda.amp.GradScaler(enabled=True, dtype=torch.bfloat16)
```

### 4. Disable Validation During Training

```bash
--val_every_n_epochs 999999  # Essentially disable
```

### 5. Use Smaller Validation Set

```bash
--val_segments_per_track 1  # Only 1 segment per track
```

## 📊 Expected Memory Usage

| Configuration | Peak VRAM | Quality |
|--------------|-----------|---------|
| Full (batch=8, seg=6s, FFT=2048) | ~18-24 GB | Best |
| Standard (batch=4, seg=6s, FFT=2048) | ~8-12 GB | Good |
| Conservative (batch=2, seg=4s, FFT=1024) | ~4-6 GB | Acceptable |
| Minimal (batch=1, seg=3s, FFT=1024) | ~2-4 GB | Limited |
| Emergency (batch=1, seg=2s, FFT=512) | ~1-2 GB | Poor |

## 🐛 Common Issues

### Issue: "Efficient loading failed, using fallback"

**Solution**: This is a warning, not an error. The code automatically falls back to standard loading.

To disable efficient loading:
```bash
--use_efficient_loading False
```

### Issue: "maximum recursion depth exceeded"

**Solution**: Already fixed in latest code. Update your code or set:
```bash
--min_mean_abs 0.0  # Disable silent chunk filtering
```

### Issue: Memory keeps growing between epochs

**Solution**: Add to training loop:
```python
torch.cuda.empty_cache()  # After each epoch
```

### Issue: Works for 1st epoch, fails on 2nd

**Cause**: Memory fragmentation

**Solution**:
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
```

## 📝 Checklist

Before asking for help, verify:

- [ ] Used `--batch_size 1`
- [ ] Used `--segment_seconds 3.0` or less
- [ ] Used `--n_fft 1024` or less
- [ ] Enabled `--use_amp`
- [ ] Enabled `--use_checkpointing`
- [ ] Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- [ ] Tried reducing model size in config
- [ ] Cleared GPU memory before training
- [ ] Checked `nvidia-smi` for other processes
- [ ] Verified enough system RAM (32GB+ recommended)

## 🆘 Still Having Issues?

1. Check your GPU specs:
   ```bash
   nvidia-smi --query-gpu=name,memory.total --format=csv
   ```

2. Check PyTorch CUDA version:
   ```python
   import torch
   print(torch.version.cuda)
   print(torch.cuda.get_device_properties(0))
   ```

3. Try training on CPU (very slow):
   ```bash
   --device cpu
   ```

4. Consider using a cloud GPU service:
   - Google Colab (free T4)
   - Kaggle (free P100)
   - Lambda Labs
   - RunPod

## 📞 Getting Help

When reporting OOM issues, include:

1. GPU model and VRAM amount
2. Full error traceback
3. Command you ran
4. Output of `nvidia-smi`
5. PyTorch version: `python -c "import torch; print(torch.__version__)"`
6. Settings that worked/didn't work
