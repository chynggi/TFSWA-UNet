"""Entry point for training the TFSWA-UNet model."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.tfswa_unet import TFSWAUNet
from src.data.musdb_dataset import MUSDB18Dataset, collate_fn
from src.data.stft_processor import STFTProcessor
from src.training.losses import SourceSeparationLoss
from src.training.trainer import Trainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train TFSWA-UNet for music source separation')
    
    # Data arguments
    parser.add_argument('--data_root', type=str, default='data/musdb18',
                        help='Path to MUSDB18 dataset')
    parser.add_argument('--target_stems', type=str, nargs='+', default=['vocals', 'other'],
                        help='Target stems to separate')
    parser.add_argument('--sample_rate', type=int, default=44100,
                        help='Audio sample rate')
    parser.add_argument('--segment_seconds', type=float, default=6.0,
                        help='Segment length in seconds')
    
    # Model arguments
    parser.add_argument('--in_channels', type=int, default=2,
                        help='Number of input channels (real, imag)')
    parser.add_argument('--out_channels', type=int, default=2,
                        help='Number of output channels per stem')
    parser.add_argument('--depths', type=int, nargs='+', default=[2, 2, 6, 2],
                        help='Number of blocks at each stage')
    parser.add_argument('--dims', type=int, nargs='+', default=[32, 64, 128, 256],
                        help='Channel dimensions at each stage')
    parser.add_argument('--window_size', type=int, default=8,
                        help='Window size for shifted window attention')
    parser.add_argument('--shift_size', type=int, default=4,
                        help='Shift size for shifted window attention')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    
    # STFT arguments
    parser.add_argument('--n_fft', type=int, default=2048,
                        help='FFT size')
    parser.add_argument('--hop_length', type=int, default=512,
                        help='Hop length')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=300,
                        help='Maximum number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--gradient_clip_val', type=float, default=1.0,
                        help='Gradient clipping value')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--val_segments_per_track', type=int, default=1,
                        help='Number of sequential validation segments per track (None for all)')
    
    # Loss arguments
    parser.add_argument('--l1_weight', type=float, default=1.0,
                        help='Weight for L1 loss')
    parser.add_argument('--mrstft_weight', type=float, default=0.5,
                        help='Weight for multi-resolution STFT loss')
    
    # Logging arguments
    parser.add_argument('--output_dir', type=str, default='outputs/tfswa_unet',
                        help='Output directory')
    parser.add_argument('--log_every_n_steps', type=int, default=50,
                        help='Logging frequency')
    parser.add_argument('--val_every_n_epochs', type=int, default=5,
                        help='Validation frequency')
    parser.add_argument('--save_every_n_epochs', type=int, default=10,
                        help='Checkpoint saving frequency')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use automatic mixed precision')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Print configuration
    print("=" * 80)
    print("TFSWA-UNet Training Configuration")
    print("=" * 80)
    print(f"Data root: {args.data_root}")
    print(f"Target stems: {args.target_stems}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Device: {args.device}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 80)
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = MUSDB18Dataset(
        root=args.data_root,
        split='train',
        target_stems=args.target_stems,
        sample_rate=args.sample_rate,
        segment_seconds=args.segment_seconds,
        random_segments=True,
    )
    
    val_dataset = MUSDB18Dataset(
        root=args.data_root,
        split='valid',
        target_stems=args.target_stems,
        sample_rate=args.sample_rate,
        segment_seconds=args.segment_seconds,
        random_segments=False,
        max_segments_per_track=(
            None if args.val_segments_per_track is None or args.val_segments_per_track <= 0
            else args.val_segments_per_track
        ),
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")
    
    # Create model
    print("\nCreating model...")
    
    # Adjust output channels based on number of stems
    model_out_channels = args.out_channels * len(args.target_stems)
    
    model = TFSWAUNet(
        in_channels=args.in_channels * 2,  # real + imag
        out_channels=model_out_channels,  # 2 channels per stem
        depths=args.depths,
        dims=args.dims,
        window_size=args.window_size,
        shift_size=args.shift_size,
        num_heads=args.num_heads,
    )
    
    print(f"Model created with {model.get_num_parameters():,} parameters")
    
    # Create STFT processor
    stft_processor = STFTProcessor(
        n_fft=args.n_fft,
        hop_length=args.hop_length,
    )
    
    # Create loss function
    loss_fn = SourceSeparationLoss(
        l1_weight=args.l1_weight,
        mrstft_weight=args.mrstft_weight,
        use_l1=True,
        use_mrstft=False,  # Disable for now (computationally expensive)
    )
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    # Create learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.max_epochs * len(train_loader),
        eta_min=1e-6,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        stft_processor=stft_processor,
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device(args.device),
        output_dir=args.output_dir,
        max_epochs=args.max_epochs,
        gradient_clip_val=args.gradient_clip_val,
        log_every_n_steps=args.log_every_n_steps,
        val_every_n_epochs=args.val_every_n_epochs,
        save_every_n_epochs=args.save_every_n_epochs,
        use_amp=args.use_amp,
        target_stems=args.target_stems,
    )
    
    # Resume from checkpoint if specified
    if args.resume is not None:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    print("\nStarting training...\n")
    trainer.train()
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()
