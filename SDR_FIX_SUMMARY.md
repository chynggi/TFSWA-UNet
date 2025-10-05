# SDR Evaluation Fix Summary

## 문제
- `--eval_sdr` 옵션이 자동으로 작동하지 않음
- `action='store_true'`와 `default=True`의 충돌로 인해 실제로는 False가 기본값이 됨

## 해결 방법

### 1. 명령행 인자 수정 (train.py)
```python
# Before (문제)
parser.add_argument('--eval_sdr', action='store_true', default=True)

# After (수정)
parser.add_argument('--eval_sdr', type=lambda x: x.lower() == 'true', default=True)
parser.add_argument('--no_eval_sdr', dest='eval_sdr', action='store_false')
```

### 2. SDR 평가 시점 수정 (trainer.py)
```python
# 첫 번째 validation과 이후 5번째 validation마다 실행
val_count = (epoch + 1) // self.val_every_n_epochs
should_eval_sdr = self.eval_sdr and (val_count == 1 or val_count % 5 == 0)
```

### 3. 디버그 출력 추가
- 학습 시작 시 SDR 평가 설정 명확히 표시
- SDR 평가 실행 시 구분선과 validation 번호 표시

## 사용 방법

### 기본 사용 (SDR 평가 활성화)
```bash
python scripts/train.py --data_root /workspace/dataset
# SDR 평가가 자동으로 실행됨
```

### SDR 평가 비활성화
```bash
python scripts/train.py --data_root /workspace/dataset --no_eval_sdr
```

### 평가 트랙 수 조정
```bash
python scripts/train.py --data_root /workspace/dataset --eval_num_tracks 3
```

## 평가 일정

`--val_every_n_epochs 5` 기준:

| Epoch | Validation # | Loss Eval | SDR Eval |
|-------|--------------|-----------|----------|
| 5     | 1            | ✓         | ✓        |
| 10    | 2            | ✓         | ✗        |
| 15    | 3            | ✓         | ✗        |
| 20    | 4            | ✓         | ✗        |
| 25    | 5            | ✓         | ✓        |
| 30    | 6            | ✓         | ✗        |
| ...   | ...          | ...       | ...      |
| 50    | 10           | ✓         | ✓        |

## 확인 방법

### 1. 테스트 스크립트 실행
```bash
python test_sdr_eval.py --data_root /workspace/dataset
```

### 2. 학습 시작 시 출력 확인
```
Trainer initialized:
  Device: cuda
  Output dir: outputs/tfswa_unet_low_vram
  Max epochs: 300
  Mixed precision: True
  Target stems: ['vocals', 'other']

  SDR Evaluation: ENABLED  ← 이것을 확인!
    - Eval tracks: 5
    - Validation every: 5 epochs
    - SDR eval at validations: #1, #5, #10, #15, ...
    - SDR eval epochs: 5, 25, 50, 75, ...
```

### 3. Validation 시 출력 확인
```
Epoch 5/300 - Train losses:
  ...

Validation losses:
  ...

============================================================
Performing SDR evaluation (validation #1)...  ← 이것을 확인!
============================================================
SDR Evaluation: 100%|████| 5/5 [01:23<00:00, avg_SDR=4.23dB]

SDR Metrics:
  vocals_sdr: 5.124 dB
  vocals_si_sdr: 4.987 dB
  other_sdr: 3.341 dB
  other_si_sdr: 3.198 dB
  avg_sdr: 4.233 dB
```

## 트러블슈팅

### SDR 평가가 여전히 실행되지 않으면

1. **명시적으로 활성화**:
   ```bash
   python scripts/train.py --eval_num_tracks 5
   ```

2. **디버그 출력 확인**:
   ```python
   print(f"args.eval_sdr = {args.eval_sdr}")  # train.py에 추가
   print(f"self.eval_sdr = {self.eval_sdr}")  # trainer.py에 추가
   ```

3. **조건문 확인**:
   ```python
   val_count = (epoch + 1) // self.val_every_n_epochs
   should_eval = self.eval_sdr and (val_count == 1 or val_count % 5 == 0)
   print(f"Epoch {epoch}, val_count={val_count}, should_eval_sdr={should_eval}")
   ```

## 수정된 파일

1. `scripts/train.py` - 명령행 인자 수정
2. `src/training/trainer.py` - SDR 평가 로직 및 출력 개선
3. `VALIDATION_WITH_SDR.md` - 문서 업데이트
4. `test_sdr_eval.py` - 테스트 스크립트 추가
5. `SDR_FIX_SUMMARY.md` - 이 문서
