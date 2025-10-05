# Training script optimized for LOW VRAM (4-8GB) - Windows PowerShell
# This configuration minimizes memory usage at the cost of some quality

python scripts/train.py `
    --data_root data/musdb18 `
    --batch_size 1 `
    --segment_seconds 3.0 `
    --n_fft 1024 `
    --hop_length 256 `
    --num_workers 2 `
    --use_efficient_loading `
    --persistent_workers `
    --prefetch_factor 2 `
    --use_amp `
    --use_checkpointing `
    --gradient_clip_val 1.0 `
    --learning_rate 1e-4 `
    --max_epochs 300 `
    --val_every_n_epochs 10 `
    --output_dir outputs/tfswa_unet_low_vram `
    $args
