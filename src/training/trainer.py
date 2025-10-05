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
import numpy as np

from ..data.stft_processor import STFTProcessor
from .losses import SourceSeparationLoss
from ..evaluation.metrics import sdr, si_sdr


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
        eval_sdr: bool = True,
        eval_num_tracks: int = 5,
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
        self.eval_sdr = eval_sdr
        self.eval_num_tracks = eval_num_tracks
        
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
        
        print(f"\nTrainer initialized:")
        print(f"  Device: {device}")
        print(f"  Output dir: {output_dir}")
        print(f"  Max epochs: {max_epochs}")
        print(f"  Mixed precision: {use_amp}")
        print(f"  Target stems: {self.target_stems}")
        print(f"\n  SDR Evaluation: {'ENABLED' if eval_sdr else 'DISABLED'}")
        if eval_sdr:
            print(f"    - Eval tracks: {eval_num_tracks}")
            print(f"    - Validation every: {val_every_n_epochs} epochs")
            print(f"    - SDR eval at validations: #1, #5, #10, #15, ...")
            print(f"    - SDR eval epochs: {val_every_n_epochs}, {val_every_n_epochs*5}, {val_every_n_epochs*10}, ...")
    
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
    def evaluate_sdr(self) -> Dict[str, float]:
        """
        Evaluate model with real source separation and SDR metrics.
        
        Performs actual audio reconstruction and computes SDR on validation tracks.
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        # Get validation dataset
        val_dataset = self.val_loader.dataset
        
        # Select tracks to evaluate (avoid processing all to save time)
        num_tracks = min(self.eval_num_tracks, len(val_dataset.tracks))
        eval_tracks = val_dataset.tracks[:num_tracks]
        
        all_sdrs = {stem: [] for stem in self.target_stems}
        all_si_sdrs = {stem: [] for stem in self.target_stems}
        
        pbar = tqdm(eval_tracks, desc='SDR Evaluation')
        
        for track in pbar:
            try:
                # Get full track audio (not segmented)
                mixture = torch.from_numpy(track.audio.T).float()  # (2, samples)
                mixture = mixture.mean(dim=0, keepdim=True)  # (1, samples) - convert to mono
                mixture = mixture.to(self.device)
                
                # Get ground truth stems
                references = {}
                for stem_name in self.target_stems:
                    if stem_name == 'other' and len(self.target_stems) == 2 and 'vocals' in self.target_stems:
                        # Combine other stems for binary separation
                        other_audio = np.zeros_like(track.audio)
                        for other_stem in ['drums', 'bass', 'other']:
                            if other_stem in track.targets:
                                other_audio += track.targets[other_stem].audio
                        references[stem_name] = torch.from_numpy(other_audio.T).float().mean(dim=0, keepdim=True)
                    else:
                        references[stem_name] = torch.from_numpy(
                            track.targets[stem_name].audio.T
                        ).float().mean(dim=0, keepdim=True)  # (1, samples)
                    references[stem_name] = references[stem_name].to(self.device)
                
                # Separate sources using model
                separated = self._separate_track(mixture)
                
                # Compute SDR for each stem
                for stem_name in self.target_stems:
                    if stem_name in separated and stem_name in references:
                        # Trim to same length
                        min_len = min(separated[stem_name].shape[1], references[stem_name].shape[1])
                        est = separated[stem_name][:, :min_len].squeeze(0)  # (samples,)
                        ref = references[stem_name][:, :min_len].squeeze(0)  # (samples,)
                        
                        # Compute SDR metrics
                        sdr_val = sdr(est, ref).item()
                        si_sdr_val = si_sdr(est, ref).item()
                        
                        all_sdrs[stem_name].append(sdr_val)
                        all_si_sdrs[stem_name].append(si_sdr_val)
                
                # Update progress with current SDRs
                avg_sdr = np.mean([np.mean(sdrs) for sdrs in all_sdrs.values() if len(sdrs) > 0])
                pbar.set_postfix({'avg_SDR': f'{avg_sdr:.2f}dB'})
                
            except Exception as e:
                print(f"Warning: Failed to evaluate track {track.name}: {e}")
                continue
        
        # Compute average metrics
        metrics = {}
        for stem_name in self.target_stems:
            if len(all_sdrs[stem_name]) > 0:
                metrics[f'{stem_name}_sdr'] = np.mean(all_sdrs[stem_name])
                metrics[f'{stem_name}_si_sdr'] = np.mean(all_si_sdrs[stem_name])
        
        # Overall average
        if len(all_sdrs['vocals']) > 0 or len(all_sdrs.get('other', [])) > 0:
            all_stem_sdrs = []
            for stem in self.target_stems:
                if len(all_sdrs[stem]) > 0:
                    all_stem_sdrs.extend(all_sdrs[stem])
            metrics['avg_sdr'] = np.mean(all_stem_sdrs)
        
        return metrics
    
    def _separate_track(self, mixture: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Separate a full track using the model with overlap-add.
        
        Args:
            mixture: Input mixture (1, samples)
            
        Returns:
            Dictionary {stem_name: separated_audio (1, samples)}
        """
        segment_samples = int(10 * self.stft_processor.sample_rate)  # 10 second segments
        hop_samples = segment_samples // 2  # 50% overlap
        total_length = mixture.shape[1]
        
        # Initialize output buffers
        separated = {
            name: torch.zeros(1, total_length, device=self.device)
            for name in self.target_stems
        }
        normalization = torch.zeros(total_length, device=self.device)
        
        # Create Hann window for smooth overlap-add
        window = torch.hann_window(segment_samples, device=self.device)
        
        # Process in segments
        num_segments = (total_length - segment_samples) // hop_samples + 1
        
        for i in range(max(1, num_segments)):
            start = i * hop_samples
            end = start + segment_samples
            
            # Handle last segment
            if end > total_length:
                end = total_length
                start = max(0, end - segment_samples)
            
            # Extract segment
            segment = mixture[:, start:end]
            
            # Pad if needed
            if segment.shape[1] < segment_samples:
                pad_length = segment_samples - segment.shape[1]
                segment = torch.nn.functional.pad(segment, (0, pad_length))
            
            # Convert to stereo for STFT (duplicate mono channel)
            segment_stereo = segment.repeat(2, 1).unsqueeze(0)  # (1, 2, samples)
            
            # Compute STFT
            complex_spec = self.stft_processor.stft(segment_stereo)  # (1, 2, F, T)
            
            # Get mixture magnitude and phase
            mixture_spec_mono = complex_spec.mean(dim=1)  # (1, F, T)
            mixture_mag = torch.abs(mixture_spec_mono)  # (1, F, T)
            mixture_phase = torch.angle(mixture_spec_mono)  # (1, F, T)
            
            # Convert to model input
            model_input = self.stft_processor.to_model_input(complex_spec)  # (1, 4, F, T)
            
            # Model prediction with AMP
            with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
                model_output = self.model(model_input)  # (1, n_stems*2, F, T)
            
            # Extract masks and reconstruct each stem
            actual_length = min(end - start, segment.shape[1])
            segment_window = window[:actual_length]
            
            for idx, stem_name in enumerate(self.target_stems):
                # Extract mask for this stem
                stem_mask = model_output[:, idx*2:(idx+1)*2, :, :]  # (1, 2, F, T)
                
                # Compute mask magnitude
                mask_mag = torch.sqrt(stem_mask[:, 0, :, :]**2 + stem_mask[:, 1, :, :]**2 + 1e-8)
                mask_mag = torch.sigmoid(mask_mag)  # (1, F, T)
                
                # Apply mask to mixture magnitude
                masked_mag = mixture_mag * mask_mag  # (1, F, T)
                
                # Reconstruct complex spectrogram with original phase
                masked_spec = masked_mag * torch.exp(1j * mixture_phase)  # (1, F, T)
                
                # Add channel dimension and duplicate for stereo ISTFT
                masked_spec = masked_spec.unsqueeze(1).repeat(1, 2, 1, 1)  # (1, 2, F, T)
                
                # ISTFT to audio
                separated_audio = self.stft_processor.istft(masked_spec)  # (1, 2, samples)
                
                # Convert back to mono
                separated_audio = separated_audio.mean(dim=1)  # (1, samples)
                
                # Apply window and accumulate
                separated[stem_name][:, start:start+actual_length] += (
                    separated_audio[:, :actual_length] * segment_window
                )
            
            # Accumulate normalization
            normalization[start:start+actual_length] += segment_window
        
        # Normalize by window overlap
        normalization = torch.clamp(normalization, min=1e-8)
        for stem_name in self.target_stems:
            separated[stem_name] = separated[stem_name] / normalization.unsqueeze(0)
        
        return separated
    
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
                # Loss-based validation
                val_losses = self.validate()
                
                if val_losses:
                    print(f"\nValidation losses:")
                    for key, value in val_losses.items():
                        print(f"  {key}: {value:.4f}")
                        self.writer.add_scalar(f'epoch_val/{key}', value, epoch)
                    
                    val_loss = val_losses['total_loss']
                else:
                    val_loss = float('inf')
                
                # SDR-based evaluation (first validation and every 5 validations thereafter)
                val_count = (epoch + 1) // self.val_every_n_epochs
                should_eval_sdr = self.eval_sdr and (val_count == 1 or val_count % 5 == 0)
                
                if should_eval_sdr:
                    print(f"\n{'='*60}")
                    print(f"Performing SDR evaluation (validation #{val_count})...")
                    print(f"{'='*60}")
                    sdr_metrics = self.evaluate_sdr()
                    
                    if sdr_metrics:
                        print(f"\nSDR Metrics:")
                        for key, value in sdr_metrics.items():
                            print(f"  {key}: {value:.3f} dB")
                            self.writer.add_scalar(f'sdr/{key}', value, epoch)
                        
                        # Use average SDR for best model selection (higher is better)
                        if 'avg_sdr' in sdr_metrics:
                            avg_sdr = sdr_metrics['avg_sdr']
                            # Convert to loss (negative SDR so lower is better)
                            sdr_based_loss = -avg_sdr
                            is_best = sdr_based_loss < self.best_val_loss
                            if is_best:
                                self.best_val_loss = sdr_based_loss
                                print(f"  New best model with SDR: {avg_sdr:.3f} dB")
                        else:
                            is_best = val_loss < abs(self.best_val_loss)
                    else:
                        is_best = val_loss < abs(self.best_val_loss)
                else:
                    # Use loss for best model when not doing SDR eval
                    is_best = val_loss < abs(self.best_val_loss)
                    if is_best:
                        self.best_val_loss = val_loss
                        print(f"  New best validation loss: {val_loss:.4f}")
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
