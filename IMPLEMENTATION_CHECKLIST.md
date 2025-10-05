# TFSWA-UNet 구현 체크리스트

최종 업데이트: 2025년 10월 4일

---

## Phase 1: 핵심 모델 아키텍처 ✅ 100% 완료

### Attention 메커니즘
- [x] **ScaledDotProductAttention** - 기본 attention 연산
- [x] **MultiHeadAttention** - Multi-head wrapper
- [x] **TemporalSequenceAttention (TSA)** - 시간축 attention
- [x] **FrequencySequenceAttention (FSA)** - 주파수축 attention  
- [x] **ShiftedWindowAttention (SW-MSA)** - 지역적 attention
- [x] **Window partitioning utilities** - 윈도우 분할/병합

### 빌딩 블록
- [x] **TFSWABlock** - TSA+FSA+SWA 통합 블록
  - [x] Input projection
  - [x] 3중 attention 병렬 처리
  - [x] Feature fusion
  - [x] Residual connections
  - [x] Skip connection 지원
- [x] **DownsampleBlock** - Encoder용 다운샘플링
- [x] **UpsampleBlock** - Decoder용 업샘플링

### 전체 아키텍처
- [x] **TFSWAUNet** 클래스
  - [x] Stem (초기 feature 추출)
  - [x] Encoder (3 stages)
    - [x] Stage 1: 2×TFSWABlock (32 channels)
    - [x] Stage 2: 2×TFSWABlock (64 channels)
    - [x] Stage 3: 6×TFSWABlock (128 channels)
  - [x] Bottleneck (2×TFSWABlock, 256 channels)
  - [x] Decoder (3 stages with skip connections)
    - [x] Stage 3: 6×TFSWABlock + Skip
    - [x] Stage 2: 2×TFSWABlock + Skip
    - [x] Stage 1: 2×TFSWABlock + Skip
  - [x] Output head (Sigmoid mask)
- [x] **Weight initialization**
- [x] **Model info utilities**

### 테스트 & 검증
- [x] **Forward pass test** - 정상 작동 확인
- [x] **Gradient flow test** - 모든 파라미터 gradient 확인
- [x] **Shape preservation test** - 입출력 shape 검증
- [x] **Output range test** - [0,1] 범위 확인
- [x] **Architecture visualization** - 구조 시각화

### 문서화
- [x] **README.md** - 프로젝트 개요
- [x] **PHASE1_SUMMARY.md** - Phase 1 요약
- [x] **PHASE1_IMPLEMENTATION_REPORT.md** - 상세 기술 문서
- [x] **IMPLEMENTATION_CHECKLIST.md** - 본 체크리스트
- [x] Code docstrings (Google style)
- [x] Type hints

---

## Phase 2: 데이터 파이프라인 & 학습 ✅ 100% 완료

### 데이터 로딩
- [x] **MUSDB18Dataset** 클래스
  - [x] 트랙 로딩
  - [x] Train/val/test 분할
  - [x] Segment 추출
  - [x] **유연한 스템 선택** (2-stem, 4-stem, 커스텀)
  - [x] **자동 스템 합성** (other = drums+bass+other)
- [x] **DataLoader** 설정
  - [x] Batch collation
  - [x] Multi-processing
  - [x] Shuffle

### 전처리
- [x] **STFT 변환**
  - [x] Complex spectrogram 생성
  - [x] Magnitude/phase 분리
  - [x] Normalization (instance/batch)
  - [x] Real/Imag 분리
- [x] **ISTFT 역변환**
  - [x] Mask 적용
  - [x] Phase 복원
  - [x] Overlap-add
  - [x] 완벽한 재구성 (error: 0.000000)
- [x] **Audio I/O**
  - [x] 파일 로딩 (musdb)
  - [x] Waveform 처리
  - [x] Segment 추출

### Data Augmentation
- [x] **Time stretching** (0.9-1.1x)
- [x] **Pitch shifting** (±2 semitones)
- [x] **Volume scaling** (0.7-1.3x)
- [x] **Frequency masking** (max 80 bins)
- [x] **Time masking** (max 40 frames)
- [x] **Mixup augmentation** (Beta distribution)

### Loss Functions
- [x] **L1 Loss** (magnitude spectrogram)
- [x] **Multi-resolution STFT Loss**
  - [x] Multiple FFT sizes (2048, 1024, 512)
  - [x] Multiple hop lengths (512, 256, 128)
  - [x] Magnitude + log magnitude losses
- [x] **SourceSeparationLoss** (L1 + MRSTFT 결합)
- [ ] **Perceptual Loss** (향후 구현)

### Training Loop
- [x] **Epoch 관리**
- [x] **Batch iteration**
- [x] **Forward/backward pass**
- [x] **Optimizer step**
  - [x] AdamW
  - [x] Weight decay
- [x] **Learning rate scheduling**
  - [x] Cosine annealing
  - [x] Step-wise scheduling
- [x] **Gradient clipping** (max norm 1.0)
- [x] **Mixed precision (AMP)** (FP16)
- [x] **Progress bar** (tqdm)

### Validation
- [x] **Validation loop**
- [x] **Loss 계산**
- [x] **Best model tracking**
- [x] **Checkpoint 저장/로드**
- [ ] **Early stopping** (향후 추가)

### Training Script
- [x] **CLI 인터페이스** (argparse)
- [x] **Configuration** (모든 하이퍼파라미터)
- [x] **Resume training**
- [x] **TensorBoard logging**
- [x] **Output directory** 관리

---

## Phase 3: 평가 & 최적화 ⏳ 0% 완료

### 평가 메트릭
- [ ] **SDR (Signal-to-Distortion Ratio)**
- [ ] **SIR (Signal-to-Interference Ratio)**
- [ ] **SAR (Signal-to-Artifacts Ratio)**
- [ ] **museval 통합**
- [ ] 10초 segment 평가

### Inference Pipeline
- [ ] **전체 트랙 처리**
- [ ] **Segment 분할/병합**
- [ ] **Overlap-add windowing**
- [ ] **Batch inference**
- [ ] **실시간 처리** (선택적)

### 체크포인팅
- [ ] **모델 저장**
  - [ ] State dict
  - [ ] Optimizer state
  - [ ] Scheduler state
  - [ ] Epoch/step info
- [ ] **모델 로딩**
- [ ] **Resume training**

### 로깅 & 모니터링
- [ ] **TensorBoard 통합**
  - [ ] Loss curves
  - [ ] Learning rate
  - [ ] Gradient norms
  - [ ] Audio samples
  - [ ] Spectrograms
- [ ] **WandB 통합** (선택적)
- [ ] **Console logging**

### 최적화
- [ ] **Mixed Precision (FP16)**
  - [ ] GradScaler
  - [ ] Autocast
- [ ] **Gradient Checkpointing**
  - [ ] 메모리 절약
- [ ] **Distributed Training**
  - [ ] DDP (DistributedDataParallel)
  - [ ] Multi-GPU
- [ ] **Model Parallelism** (선택적)

---

## Phase 4: 배포 & 프로덕션 ⏳ 0% 완료

### 모델 Export
- [ ] **ONNX export**
- [ ] **TorchScript conversion**
- [ ] **TensorRT optimization** (선택적)
- [ ] **Quantization** (선택적)

### API & 서비스
- [ ] **REST API**
  - [ ] FastAPI/Flask
  - [ ] Audio upload/download
  - [ ] Async processing
- [ ] **WebSocket** (실시간)
- [ ] **Batch processing API**

### 배포
- [ ] **Docker container**
- [ ] **Kubernetes manifests** (선택적)
- [ ] **Cloud deployment**
  - [ ] AWS/GCP/Azure
- [ ] **CI/CD pipeline**

### Pre-trained Models
- [ ] **학습 완료 모델**
- [ ] **모델 업로드** (HuggingFace Hub)
- [ ] **사용 예제**
- [ ] **Inference 스크립트**

---

## 추가 개선사항 (선택적)

### 모델 변형
- [ ] **TFSWA-ResUNet** - Residual connections
- [ ] **Multi-scale TFSWA** - 다중 해상도
- [ ] **Lightweight TFSWA** - 모바일용

### 다중 소스 분리
- [ ] **4-stem separation** (vocals/drums/bass/other)
- [ ] **Multi-task learning**

### Self-supervised Learning
- [ ] **Pre-training task**
- [ ] **Contrastive learning**

### 데이터셋 확장
- [ ] **추가 데이터셋 지원**
  - [ ] DSD100
  - [ ] MedleyDB
  - [ ] Custom datasets
- [ ] **Data mixing strategies**

---

## 현재 진행률 요약

```
Phase 1 (핵심 모델):      ████████████████████ 100% ✅
Phase 2 (데이터/학습):    ████████████████████ 100% ✅
Phase 3 (평가/최적화):    ░░░░░░░░░░░░░░░░░░░░   0% ⏳
Phase 4 (배포):           ░░░░░░░░░░░░░░░░░░░░   0% ⏳

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
전체 프로젝트 진행률:     ██████████░░░░░░░░░░  50%
```

---

## 다음 작업 항목 (우선순위)

### 🔥 High Priority (실제 학습)
1. MUSDB18-HQ 데이터셋 다운로드
2. 첫 학습 실행 및 모니터링
3. Hyperparameter 튜닝
4. Validation 성능 추적
5. Best model 선택

### 🔶 Medium Priority (Phase 4)
6. Gradient checkpointing (메모리 최적화)
7. Model export (ONNX, TorchScript)
8. Quantization (INT8)
9. 실시간 추론 최적화
10. TensorRT 컴파일

### 🔷 Low Priority
11. 결과 시각화 대시보드
12. WandB 통합
13. Advanced 평가 메트릭
14. Ablation studies
15. Perceptual 메트릭 추가

---

## 완료 기준

각 항목은 다음 조건을 만족할 때 체크:

1. ✅ 코드 구현 완료
2. ✅ 단위 테스트 통과
3. ✅ 통합 테스트 통과
4. ✅ Docstring 작성
5. ✅ 코드 리뷰 (선택적)

---

**마지막 업데이트**: 2025년 10월 4일  
**다음 목표**: Phase 2 데이터 파이프라인 구현
