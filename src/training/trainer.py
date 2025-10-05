"""Training loop and trainer class."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..data.stft_processor import STFTProcessor
from .losses import SourceSeparationLoss


class Trainer:
    """
    Trainer for TFSWA-UNet music source separation.
    
    Args:
        model: TFSWA-UNet model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        loss_fn: Loss function
        stft_processor: STFT processor
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        output_dir: Directory to save checkpoints and logs
        max_epochs: Maximum number of epochs
        gradient_clip_val: Gradient clipping value (default: 1.0)
        log_every_n_steps: Log frequency (default: 100)
        val_every_n_epochs: Validation frequency (default: 1)
        save_every_n_epochs: Checkpoint saving frequency (default: 5)
        use_amp: Whether to use automatic mixed precision (default: False)
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        loss_fn: SourceSeparationLoss,
        stft_processor: STFTProcessor,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        device: torch.device,
        output_dir: str,
        max_epochs: int = 300,
        gradient_clip_val: float = 1.0,
        log_every_n_steps: int = 100,
        val_every_n_epochs: int = 1,
        save_every_n_epochs: int = 5,
        use_amp: bool = False,
        target_stems: Optional[list] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.stft_processor = stft_processor
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.max_epochs = max_epochs
        self.gradient_clip_val = gradient_clip_val
        self.log_every_n_steps = log_every_n_steps
        self.val_every_n_epochs = val_every_n_epochs
        self.save_every_n_epochs = save_every_n_epochs
        self.use_amp = use_amp
        self.target_stems = target_stems or ['vocals', 'other']
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Initialize TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.output_dir / 'logs'))
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Mixed precision
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # Move model to device
        self.model = self.model.to(device)
        self.stft_processor = self.stft_processor.to(device)
        
        print(f"Trainer initialized:")
        print(f"  Device: {device}")
        print(f"  Output dir: {output_dir}")
        print(f"  Max epochs: {max_epochs}")
        print(f"  Mixed precision: {use_amp}")
        print(f"  Target stems: {self.target_stems}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_losses = {}
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}/{self.max_epochs}')
        
        for batch_idx, (mixtures, targets) in enumerate(pbar):
            # Move to device
            mixtures = mixtures.to(self.device)  # (B, 2, samples)
            targets = {k: v.to(self.device) for k, v in targets.items()}
            
            # Compute STFT (outside autocast to avoid ComplexHalf issues)
            mixture_spec = self.stft_processor.stft(mixtures)  # (B, 2, F, T) complex64
            
            target_specs = {}
            for stem_name, stem_audio in targets.items():
                target_specs[stem_name] = self.stft_processor.stft(stem_audio)
            
            # Get mixture magnitude and phase (complex operations in float32)
            mixture_spec_mono = mixture_spec.mean(dim=1)  # (B, F, T) complex
            mixture_mag = torch.abs(mixture_spec_mono)  # (B, F, T) real
            
            # Get target magnitudes for loss
            target_mags = {}
            for k, v in target_specs.items():
                target_mono = v.mean(dim=1)  # (B, F, T) complex
                target_mags[k] = torch.abs(target_mono)  # (B, F, T) real
            
            # Debug: Check data statistics on first batch
            if batch_idx == 0 and self.current_epoch == 0:
                print(f"\n=== Data Statistics ===")
                print(f"Mixture audio - shape: {mixtures.shape}, range: [{mixtures.min():.4f}, {mixtures.max():.4f}], mean: {mixtures.mean():.4f}")
                print(f"Mixture magnitude - shape: {mixture_mag.shape}, range: [{mixture_mag.min():.4f}, {mixture_mag.max():.4f}], mean: {mixture_mag.mean():.4f}")
                for k, v in targets.items():
                    print(f"Target {k} audio - shape: {v.shape}, range: [{v.min():.4f}, {v.max():.4f}], mean: {v.mean():.4f}")
                for k, v in target_mags.items():
                    print(f"Target {k} magnitude - shape: {v.shape}, range: [{v.min():.4f}, {v.max():.4f}], mean: {v.mean():.4f}")
            
            # Forward pass with mixed precision (only for model inference)
            with torch.amp.autocast(device_type="cuda", enabled=self.use_amp, dtype=torch.float16):
                # Convert to model input (real, imag)
                model_input = self.stft_processor.to_model_input(mixture_spec)  # (B, 4, F, T)
                
                # Model prediction
                model_output = self.model(model_input)  # (B, n_stems*2, F, T)
                
                # Debug: Check model output on first batch
                if batch_idx == 0 and self.current_epoch == 0:
                    print(f"\n=== Model Output Statistics ===")
                    print(f"Model input - shape: {model_input.shape}, range: [{model_input.min():.4f}, {model_input.max():.4f}], mean: {model_input.mean():.4f}")
                    print(f"Model output - shape: {model_output.shape}, range: [{model_output.min():.4f}, {model_output.max():.4f}], mean: {model_output.mean():.4f}")
                
                # Convert model output to magnitude masks
                pred_mags = {}
                for idx, stem_name in enumerate(self.target_stems):
                    # Extract mask for this stem
                    stem_mask = model_output[:, idx*2:(idx+1)*2, :, :]  # (B, 2, F, T)
                    
                    # Compute mask magnitude and apply sigmoid
                    mask_mag = torch.sqrt(stem_mask[:, 0, :, :]**2 + stem_mask[:, 1, :, :]**2 + 1e-8)  # (B, F, T)
                    mask_mag = torch.sigmoid(mask_mag)  # Constrain to [0, 1]
                    
                    # Apply mask to mixture magnitude (in-place to save memory)
                    pred_mags[stem_name] = mixture_mag * mask_mag
                    
                    # Debug: Check predictions on first batch
                    if batch_idx == 0 and self.current_epoch == 0:
                        print(f"\n{stem_name} mask - range: [{mask_mag.min():.4f}, {mask_mag.max():.4f}], mean: {mask_mag.mean():.4f}")
                        print(f"{stem_name} pred_mag - range: [{pred_mags[stem_name].min():.4f}, {pred_mags[stem_name].max():.4f}], mean: {pred_mags[stem_name].mean():.4f}")
                
                # Compute loss
                loss_dict = self.loss_fn(
                    pred_specs=pred_mags,
                    target_specs=target_mags,
                )
                
                loss = loss_dict['total_loss']
                
                # Debug: Check if loss computation is working
                if batch_idx == 0 and self.current_epoch == 0:
                    print(f"\n=== Loss Computation ===")
                    for k in pred_mags.keys():
                        diff = torch.abs(pred_mags[k] - target_mags[k])
                        print(f"{k} L1 diff - range: [{diff.min():.6f}, {diff.max():.6f}], mean: {diff.mean():.6f}")
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                self.optimizer.step()
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Accumulate losses
            for key, value in loss_dict.items():
                if key not in epoch_losses:
                    epoch_losses[key] = 0.0
                loss_val = value.item() if torch.is_tensor(value) else value
                epoch_losses[key] += loss_val
                
                # Debug: print first batch losses
                if batch_idx == 0 and self.current_epoch == 0:
                    print(f"  Debug - {key}: {loss_val:.6f}")
            
            # Logging
            if self.global_step % self.log_every_n_steps == 0:
                lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('train/learning_rate', lr, self.global_step)
                
                for key, value in loss_dict.items():
                    self.writer.add_scalar(f'train/{key}', value.item(), self.global_step)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}',
            })
            
            self.global_step += 1
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        val_losses = {}
        num_batches = len(self.val_loader)
        
        pbar = tqdm(self.val_loader, desc='Validation')
        
        for mixtures, targets in pbar:
            # Move to device
            mixtures = mixtures.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}
            
            # Compute STFT
            mixture_spec = self.stft_processor.stft(mixtures)
            
            target_specs = {}
            for stem_name, stem_audio in targets.items():
                target_specs[stem_name] = self.stft_processor.stft(stem_audio)
            
            # Get mixture magnitude
            mixture_spec_mono = mixture_spec.mean(dim=1)  # (B, F, T) complex
            mixture_mag = torch.abs(mixture_spec_mono)  # (B, F, T) real
            
            # Get target magnitudes for loss
            target_mags = {}
            for k, v in target_specs.items():
                target_mono = v.mean(dim=1)  # (B, F, T) complex
                target_mags[k] = torch.abs(target_mono)  # (B, F, T) real
            
            # Convert to model input
            model_input = self.stft_processor.to_model_input(mixture_spec)
            
            # Model prediction
            model_output = self.model(model_input)
            
            # Convert model output to magnitude predictions
            pred_mags = {}
            for idx, stem_name in enumerate(self.target_stems):
                # Extract mask for this stem
                stem_mask = model_output[:, idx*2:(idx+1)*2, :, :]  # (B, 2, F, T)
                
                # Compute mask magnitude and apply sigmoid
                mask_mag = torch.sqrt(stem_mask[:, 0, :, :]**2 + stem_mask[:, 1, :, :]**2 + 1e-8)  # (B, F, T)
                mask_mag = torch.sigmoid(mask_mag)  # Constrain to [0, 1]
                
                # Apply mask to mixture magnitude
                pred_mags[stem_name] = mixture_mag * mask_mag
            
            # Compute loss
            loss_dict = self.loss_fn(
                pred_specs=pred_mags,
                target_specs=target_mags,
            )
            
            # Accumulate losses
            for key, value in loss_dict.items():
                if key not in val_losses:
                    val_losses[key] = 0.0
                val_losses[key] += value.item()
        
        # Average losses
        for key in val_losses:
            val_losses[key] /= num_batches
        
        return val_losses
    
    def save_checkpoint(self, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{self.current_epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest_model.pt'
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'scaler_state_dict' in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self) -> None:
        """Main training loop."""
        print(f"\nStarting training for {self.max_epochs} epochs...\n")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.max_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_losses = self.train_epoch()
            
            print(f"\nEpoch {epoch}/{self.max_epochs} - Train losses:")
            for key, value in train_losses.items():
                print(f"  {key}: {value:.4f}")
                self.writer.add_scalar(f'epoch_train/{key}', value, epoch)
            
            # Validation
            if (epoch + 1) % self.val_every_n_epochs == 0:
                val_losses = self.validate()
                
                if val_losses:
                    print(f"\nValidation losses:")
                    for key, value in val_losses.items():
                        print(f"  {key}: {value:.4f}")
                        self.writer.add_scalar(f'epoch_val/{key}', value, epoch)
                    
                    # Check if best model
                    val_loss = val_losses['total_loss']
                    is_best = val_loss < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_loss
                        print(f"  New best validation loss: {val_loss:.4f}")
                else:
                    is_best = False
            else:
                is_best = False
            
            # Save checkpoint
            if (epoch + 1) % self.save_every_n_epochs == 0 or is_best:
                self.save_checkpoint(is_best=is_best)
            
            print()
        
        elapsed_time = time.time() - start_time
        print(f"\nTraining completed in {elapsed_time / 3600:.2f} hours")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        self.writer.close()
