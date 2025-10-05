# TFSWA-UNet: Music Source Separation Project

## Project Overview
This repository implements TFSWA-UNet (Temporal-Frequency and Shifted Window Attention Based U-Net) for music source separation, specifically focused on vocals/instrumental separation. The model achieves state-of-the-art performance with 9.16dB SDR on MUSDB18 dataset.

## Architecture Details

### Core Components
- **Base Architecture**: U-Net encoder-decoder with skip connections
- **TFSWA Block**: Custom attention mechanism combining temporal-frequency and shifted window attention
- **TSA (Temporal Sequence Attention)**: Models temporal dependencies in spectrogram sequences
- **FSA (Frequency Sequence Attention)**: Captures frequency-domain relationships
- **Swin Transformer Integration**: Uses shifted window attention for local correlation capture

### Model Hierarchy
```
TFSWA-UNet/
├── Encoder (Downsampling Path)
│   ├── TFSWA Block 1 (32 channels)
│   ├── TFSWA Block 2 (64 channels)
│   ├── TFSWA Block 3 (128 channels)
│   └── Bottleneck (256 channels)
├── Decoder (Upsampling Path)
│   ├── TFSWA Block 4 (128 channels)
│   ├── TFSWA Block 3 (64 channels)
│   └── TFSWA Block 2 (32 channels)
└── Output Layer (2 channels for vocals/accompaniment)
```

## Technology Stack

### Primary Framework
- **PyTorch**: 2.0+ for deep learning implementation
- **torchaudio**: For audio processing and STFT operations
- **torch.nn**: For neural network layers and attention mechanisms

### Audio Processing
- **librosa**: Audio loading, preprocessing, and feature extraction
- **soundfile**: Audio I/O operations
- **numpy**: Numerical computations and array operations

### Training & Evaluation
- **musdb**: MUSDB18 dataset handling
- **museval**: SDR/SIR/SAR metric calculations
- **tensorboard**: Training visualization and logging
- **wandb**: Experiment tracking (optional)

### Utilities
- **hydra-core**: Configuration management
- **omegaconf**: YAML configuration parsing
- **tqdm**: Progress bars for training loops

## Code Structure & Conventions

### Directory Layout
```
tfswa-unet/
├── configs/             # Hydra configuration files
│   ├── model/          # Model architecture configs
│   ├── data/           # Dataset configs
│   └── training/       # Training hyperparameters
├── src/
│   ├── models/         # Model implementations
│   │   ├── tfswa_unet.py
│   │   ├── attention.py
│   │   └── blocks.py
│   ├── data/           # Data loading and preprocessing
│   ├── training/       # Training logic
│   ├── evaluation/     # Evaluation metrics
│   └── utils/          # Utility functions
├── scripts/            # Training and evaluation scripts
├── notebooks/          # Jupyter notebooks for experiments
└── tests/              # Unit tests
```

### Naming Conventions
- **Classes**: PascalCase (e.g., `TFSWAUNet`, `AttentionBlock`)
- **Functions/Variables**: snake_case (e.g., `forward_pass`, `attention_weights`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `DEFAULT_WINDOW_SIZE`, `MAX_FREQ_BINS`)
- **Files**: lowercase with underscores (e.g., `tfswa_unet.py`, `data_loader.py`)

### Code Style Guidelines
- Use type hints for all function signatures
- Include comprehensive docstrings following Google style
- Implement proper error handling with custom exceptions
- Use dataclasses for configuration structures
- Follow PyTorch best practices for model implementation

## Model Implementation Guidelines

### TFSWA Block Structure
```python
class TFSWABlock(nn.Module):
    """
    Temporal-Frequency and Shifted Window Attention Block
    
    Args:
        in_channels: Input feature channels
        window_size: Window size for shifted window attention
        shift_size: Shift size for window attention
        num_heads: Number of attention heads
    """
```

### Attention Mechanisms
- **TSA**: Use multi-head self-attention along temporal dimension
- **FSA**: Apply attention across frequency bins
- **Swin Attention**: Implement shifted window mechanism for local attention
- **Cross Attention**: Combine TSA and FSA outputs

### Skip Connections
- Implement feature fusion between encoder and decoder
- Use 1x1 convolutions for channel dimension matching
- Apply batch normalization before skip connection addition

## Data Processing Pipeline

### Input Preprocessing
- **STFT Parameters**: 
  - Window size: 4096 samples (92ms at 44.1kHz)
  - Hop length: 1024 samples (23ms)
  - Window function: Hann window
- **Spectrogram**: Complex-valued with magnitude and phase
- **Normalization**: Instance normalization per frequency bin

### Data Augmentation
- **Time stretching**: Random tempo changes (0.9-1.1x)
- **Pitch shifting**: ±2 semitones
- **Volume scaling**: Random gain (0.7-1.3x)
- **Frequency masking**: Mask random frequency bands
- **Time masking**: Mask random time segments

### Output Postprocessing
- **Mask Application**: Apply learned masks to input mixture
- **Phase Recovery**: Use mixture phase for reconstruction
- **Overlap-Add**: Reconstruct time-domain signals using ISTFT

## Training Configuration

### Loss Functions
- **Primary**: L1 loss on magnitude spectrograms
- **Secondary**: Multi-resolution STFT loss
- **Perceptual**: Optional perceptual loss using pre-trained features

### Optimization
- **Optimizer**: AdamW with weight decay 1e-4
- **Learning Rate**: 1e-3 with cosine annealing
- **Batch Size**: 8-16 depending on GPU memory
- **Gradient Clipping**: Max norm 1.0

### Training Schedule
- **Warmup**: 1000 iterations linear warmup
- **Epochs**: 200-300 epochs
- **Validation**: Every 5 epochs
- **Checkpoint**: Save best model based on SDR

## Evaluation Metrics

### Primary Metrics
- **SDR (Signal-to-Distortion Ratio)**: Overall separation quality
- **SIR (Signal-to-Interference Ratio)**: Interference suppression
- **SAR (Signal-to-Artifacts Ratio)**: Artifact level measurement

### Implementation
- Use `museval` library for official metric calculation
- Report metrics for both vocals and accompaniment
- Calculate metrics on 10-second segments

## Memory Optimization

### Model Efficiency
- Use gradient checkpointing for large models
- Implement mixed precision training (FP16)
- Apply model parallelism for multi-GPU training

### Data Loading
- Use multi-processing data loading
- Implement efficient data caching
- Apply on-the-fly augmentation

## Debugging & Development

### Logging
- Log training metrics every 100 iterations
- Save model checkpoints every epoch
- Track attention weights visualization
- Monitor gradient norms and parameter updates

### Testing
- Unit tests for each model component
- Integration tests for full pipeline
- Sanity checks for data loading
- Gradient flow verification

## Model Variations

### Architecture Variants
- **TFSWA-ResUNet**: Add residual connections to TFSWA blocks
- **Multi-Scale TFSWA**: Multiple resolution processing
- **Lightweight TFSWA**: Reduced parameter version for mobile deployment

### Training Variants
- **Progressive Training**: Start with low-resolution, increase gradually
- **Multi-Task Learning**: Joint vocals/drums/bass/other separation
- **Self-Supervised Pre-training**: Pre-train on reconstruction task

## Deployment Considerations

### Model Export
- Support ONNX export for cross-platform deployment
- Implement TorchScript conversion for production
- Optimize for real-time inference with TensorRT

### API Interface
- RESTful API for audio separation service
- Batch processing support
- WebSocket for real-time streaming

## Configuration Examples

### Model Config (configs/model/tfswa_unet.yaml)
```yaml
_target_: src.models.tfswa_unet.TFSWAUNet
in_channels: 2  # Real and imaginary parts
out_channels: 2  # Vocals and accompaniment masks
depths: [2, 2, 6, 2]  # Number of TFSWA blocks per stage
dims: [32, 64, 128, 256]  # Feature dimensions
window_size: 8
shift_size: 4
num_heads: 8
```

### Training Config (configs/training/default.yaml)
```yaml
batch_size: 12
learning_rate: 1e-3
weight_decay: 1e-4
max_epochs: 300
gradient_clip_val: 1.0
precision: 16
```

## Performance Targets

### Quality Metrics
- **Vocals SDR**: >9.0 dB (target: 9.16 dB)
- **Accompaniment SDR**: >14.0 dB
- **Training Time**: <48 hours on single A100
- **Inference Speed**: <0.1x real-time on GPU

### Model Size
- **Parameters**: ~15M (target for efficiency)
- **Model Size**: <60MB
- **Memory Usage**: <8GB during training

## Common Implementation Patterns

### Forward Pass Structure
```python
def forward(self, x):
    # Encoder with skip connections
    skip_connections = []
    for block in self.encoder:
        x = block(x)
        skip_connections.append(x)
    
    # Decoder with attention fusion
    for i, block in enumerate(self.decoder):
        skip = skip_connections[-(i+2)]  # Reverse order
        x = block(x, skip)  # Attention-based fusion
    
    return self.output_head(x)
```

### Attention Weight Computation
```python
def compute_attention(self, q, k, v, mask=None):
    # Scaled dot-product attention
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dim)
    if mask is not None:
        scores.masked_fill_(mask == 0, -1e9)
    weights = F.softmax(scores, dim=-1)
    output = torch.matmul(weights, v)
    return output, weights
```

This project aims to advance the state-of-the-art in music source separation through innovative attention mechanisms and efficient neural architectures. Focus on clean, maintainable code that can serve as a foundation for future research and development.