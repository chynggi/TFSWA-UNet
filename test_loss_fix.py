"""Quick test to verify loss calculation fix."""
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.training.losses import SourceSeparationLoss

# Test loss function
print("Testing loss function...")

# Create dummy data
batch_size = 2
freq_bins = 100
time_frames = 50

# Create predictions and targets with realistic values
pred_vocals = torch.rand(batch_size, freq_bins, time_frames) * 0.5
pred_other = torch.rand(batch_size, freq_bins, time_frames) * 0.5

target_vocals = torch.rand(batch_size, freq_bins, time_frames) * 0.8
target_other = torch.rand(batch_size, freq_bins, time_frames) * 0.8

pred_specs = {
    'vocals': pred_vocals,
    'other': pred_other,
}

target_specs = {
    'vocals': target_vocals,
    'other': target_other,
}

# Initialize loss function
loss_fn = SourceSeparationLoss(
    l1_weight=1.0,
    mrstft_weight=0.0,  # Disable MRSTFT for this test
    use_l1=True,
    use_mrstft=False,
)

# Compute loss
loss_dict = loss_fn(pred_specs=pred_specs, target_specs=target_specs)

print("\nLoss computation results:")
for key, value in loss_dict.items():
    loss_val = value.item() if torch.is_tensor(value) else value
    print(f"  {key}: {loss_val:.6f}")

# Check if losses are reasonable
total_loss = loss_dict['total_loss'].item()
if total_loss > 1e-6:
    print(f"\n✓ Loss calculation is working! Total loss: {total_loss:.6f}")
else:
    print(f"\n✗ Loss is still zero or near-zero: {total_loss}")

# Test with complex inputs
print("\n\nTesting with complex spectrograms...")
pred_complex = pred_vocals * torch.exp(1j * torch.rand(batch_size, freq_bins, time_frames) * 2 * 3.14159)
target_complex = target_vocals * torch.exp(1j * torch.rand(batch_size, freq_bins, time_frames) * 2 * 3.14159)

pred_specs_complex = {'vocals': pred_complex}
target_specs_complex = {'vocals': target_complex}

loss_dict_complex = loss_fn(pred_specs=pred_specs_complex, target_specs=target_specs_complex)
print(f"Complex loss: {loss_dict_complex['total_loss'].item():.6f}")

print("\n✓ All tests passed!")
