# Validation with SDR Metrics

## Overview

TFSWA-UNet 학습 과정에서 실제 음원 분리를 수행하고 SDR(Signal-to-Distortion Ratio) 지표를 계산하여 모델 성능을 평가합니다.

## Features

### 🎯 Real Source Separation Evaluation
- **실제 음원 분리**: 손실 함수 기반 평가가 아닌, 실제로 음원을 분리하여 평가
- **SDR 지표**: 업계 표준 SDR(Signal-to-Distortion Ratio) 및 SI-SDR 계산
- **Validation Set**: MUSDB18 validation subset을 사용하여 평가
- **효율적인 평가**: 전체 트랙 대신 일부 트랙만 평가하여 시간 절약

### 📊 Metrics Computed

1. **SDR (Signal-to-Distortion Ratio)**
   - 전체 분리 품질 측정
   - 높을수록 좋음 (단위: dB)

2. **SI-SDR (Scale-Invariant SDR)**
   - 스케일 불변 SDR
   - 볼륨 차이에 영향받지 않음

3. **Per-Stem Metrics**
   - `vocals_sdr`: Vocals 분리 SDR
   - `vocals_si_sdr`: Vocals 분리 SI-SDR
   - `other_sdr`: Other (accompaniment) 분리 SDR
   - `other_si_sdr`: Other 분리 SI-SDR

4. **Overall Average**
   - `avg_sdr`: 모든 stem의 평균 SDR

## Usage

### Basic Training with SDR Evaluation (기본값)

SDR 평가는 기본적으로 활성화되어 있습니다:

```bash
python scripts/train.py \
    --data_root /path/to/musdb18 \
    --eval_num_tracks 5 \
    --val_every_n_epochs 5
```

**SDR 평가 시점**: 
- 첫 번째 validation (epoch 5)
- 이후 5번째 validation마다 (epoch 25, 50, 75, 100...)

### Disable SDR Evaluation (Faster)

손실 함수 기반 평가만 수행:

```bash
python scripts/train.py \
    --data_root /path/to/musdb18 \
    --no_eval_sdr
```

### Adjust Evaluation Frequency

SDR 평가는 계산 비용이 높으므로, 첫 번째 validation과 이후 5번째 validation마다 수행됩니다:

```bash
python scripts/train.py \
    --val_every_n_epochs 5
    # Validation: epochs 5, 10, 15, 20, 25, 30...
    # SDR 평가: epochs 5 (val #1), 25 (val #5), 50 (val #10), 75 (val #15)...
```

첫 validation에서도 SDR 평가를 수행하여 초기 성능을 확인할 수 있습니다.

### Low VRAM Settings

메모리가 부족한 경우:

```bash
python scripts/train.py \
    --eval_num_tracks 3 \
    --segment_seconds 3.0 \
    --batch_size 1
```

## Configuration Options

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--eval_sdr` | `True` | SDR 평가 활성화 |
| `--no_eval_sdr` | - | SDR 평가 비활성화 |
| `--eval_num_tracks` | `5` | 평가할 validation 트랙 수 |
| `--val_every_n_epochs` | `5` | Validation 주기 (epoch) |

### Programmatic Configuration

```python
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    # ... other args ...
    eval_sdr=True,              # Enable SDR evaluation
    eval_num_tracks=5,          # Evaluate on 5 tracks
)
```

## How It Works

### 1. Overlap-Add Inference

긴 오디오 트랙을 처리하기 위해 overlap-add 방식 사용:

```python
# 10초 세그먼트로 분할
segment_length = 10.0 seconds
overlap = 50%  # 5초 overlap

# Hann window로 smooth blending
window = torch.hann_window(segment_samples)
```

### 2. Full Track Separation

```python
def _separate_track(mixture):
    # 1. Segment audio
    # 2. Process each segment with model
    # 3. Apply masks to mixture spectrogram
    # 4. ISTFT to reconstruct audio
    # 5. Overlap-add with windowing
    return separated_stems
```

### 3. SDR Computation

```python
# Ground truth stems
references = {
    'vocals': vocals_audio,
    'other': drums + bass + other
}

# Compute SDR for each stem
for stem_name in ['vocals', 'other']:
    sdr_value = sdr(estimate, reference)
    si_sdr_value = si_sdr(estimate, reference)
```

## Output Examples

### During Training

```
Epoch 25/300 - Train losses:
  l1_vocals: 0.0547
  l1_other: 0.0621
  total_loss: 0.0584

Performing SDR evaluation...
SDR Evaluation: 100%|████████████| 5/5 [01:23<00:00, 16.7s/it, avg_SDR=4.23dB]

SDR Metrics:
  vocals_sdr: 5.124 dB
  vocals_si_sdr: 4.987 dB
  other_sdr: 3.341 dB
  other_si_sdr: 3.198 dB
  avg_sdr: 4.233 dB

New best model with SDR: 4.233 dB
```

### TensorBoard Logs

SDR 지표가 자동으로 TensorBoard에 로깅됩니다:

```
sdr/vocals_sdr
sdr/vocals_si_sdr
sdr/other_sdr
sdr/other_si_sdr
sdr/avg_sdr
```

## Performance Considerations

### Evaluation Time

| Tracks | Segment | Time (GPU) |
|--------|---------|------------|
| 3 tracks | 10s | ~1 min |
| 5 tracks | 10s | ~1.5 min |
| 10 tracks | 10s | ~3 min |

### Memory Usage

- **Peak Memory**: ~4-6GB VRAM (depends on track length)
- **Gradient-free**: `@torch.no_grad()` 사용
- **Mixed Precision**: AMP 지원으로 메모리 절약

### Best Practices

1. **적절한 트랙 수**: 5개 트랙이 속도와 정확도의 균형점
2. **평가 주기**: 초기에는 자주, 후반에는 덜 자주 평가
3. **Best Model**: SDR 기반으로 best model 저장

## Model Selection Strategy

### Loss-based vs SDR-based

```python
# Every 5 epochs: Loss-based validation (fast)
if (epoch + 1) % 5 == 0:
    val_loss = validate()  # ~10 seconds
    
    # First validation and every 5th validation: SDR evaluation
    val_count = (epoch + 1) // 5  # validation number
    if val_count == 1 or val_count % 5 == 0:
        sdr_metrics = evaluate_sdr()  # ~90 seconds
        # Use avg_sdr for best model selection
```

**실제 실행**:
- Epoch 5 (val #1): Loss + SDR ✓
- Epoch 10 (val #2): Loss only
- Epoch 15 (val #3): Loss only
- Epoch 20 (val #4): Loss only
- Epoch 25 (val #5): Loss + SDR ✓
- Epoch 30 (val #6): Loss only
- ...

### Best Model Criteria

- **Primary**: Average SDR across all stems
- **Fallback**: Validation loss if SDR not available
- **Direction**: Higher SDR = Better (stored as negative for consistency)

## Troubleshooting

### Out of Memory

```bash
# Reduce evaluation tracks
--eval_num_tracks 3

# Use shorter segments
--segment_seconds 3.0

# Disable SDR evaluation temporarily
--no_eval_sdr
```

### Slow Evaluation

```bash
# Reduce number of tracks
--eval_num_tracks 3

# Increase validation interval
--val_every_n_epochs 10
# SDR will run every 50 epochs
```

### Inconsistent Results

- **Solution 1**: Increase `eval_num_tracks` for more stable metrics
- **Solution 2**: Use SI-SDR which is scale-invariant
- **Solution 3**: Evaluate on full test set after training

## Advanced Usage

### Custom Evaluation Tracks

validation dataset의 특정 트랙만 평가하려면:

```python
# Modify in trainer.py
eval_tracks = val_dataset.tracks[indices]  # Custom track selection
```

### Frame-wise SDR

긴 트랙의 프레임별 SDR을 계산하려면:

```python
from src.evaluation.metrics import MetricsCalculator

calculator = MetricsCalculator(
    sample_rate=44100,
    segment_length=44100 * 10  # 10-second frames
)

metrics = calculator.compute(estimate, reference, compute_all=True)
```

## Expected Performance

### Target Metrics (MUSDB18)

| Metric | Target | SOTA |
|--------|--------|------|
| Vocals SDR | >8.0 dB | 9.16 dB |
| Other SDR | >12.0 dB | 14.0 dB |
| Average SDR | >10.0 dB | 11.5 dB |

### Training Progress

Typical SDR progression during training:

```
Epoch 0-25:   avg_sdr ~ 2-4 dB   (초기 학습)
Epoch 25-100: avg_sdr ~ 4-7 dB   (빠른 향상)
Epoch 100-200: avg_sdr ~ 7-9 dB  (안정화)
Epoch 200-300: avg_sdr ~ 9-10 dB (미세 조정)
```

## References

1. **BSS Eval**: Vincent, E., et al. "Performance measurement in blind audio source separation." IEEE TASLP, 2006.
2. **SI-SDR**: Le Roux, J., et al. "SDR – half-baked or well done?" ICASSP, 2019.
3. **MUSDB18**: Rafii, Z., et al. "MUSDB18 - a corpus for music separation." 2017.

## See Also

- [IMPLEMENTATION_CHECKLIST.md](IMPLEMENTATION_CHECKLIST.md) - Implementation status
- [src/evaluation/metrics.py](src/evaluation/metrics.py) - Metrics implementation
- [src/evaluation/evaluator.py](src/evaluation/evaluator.py) - Full evaluation system
