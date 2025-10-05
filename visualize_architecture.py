"""Visualize TFSWA-UNet architecture structure."""
import torch
from src.models.tfswa_unet import TFSWAUNet


def print_architecture():
    """Print detailed architecture structure."""
    
    print("=" * 80)
    print(" " * 25 + "TFSWA-UNet Architecture")
    print("=" * 80)
    
    config = {
        "in_channels": 2,
        "out_channels": 2,
        "depths": [2, 2, 6, 2],
        "dims": [32, 64, 128, 256],
        "window_size": 8,
        "shift_size": 4,
        "num_heads": 8,
    }
    
    model = TFSWAUNet(**config)
    
    print("\n📊 Model Configuration:")
    print("-" * 80)
    for key, value in config.items():
        print(f"  {key:20s}: {value}")
    
    info = model.get_model_info()
    print(f"\n  {'num_parameters':20s}: {info['num_parameters']:,}")
    print(f"  {'model_size_fp32':20s}: ~{info['num_parameters'] * 4 / 1024 / 1024:.2f} MB")
    print(f"  {'model_size_fp16':20s}: ~{info['num_parameters'] * 2 / 1024 / 1024:.2f} MB")
    
    print("\n" + "=" * 80)
    print(" " * 30 + "Architecture Flow")
    print("=" * 80)
    
    # Trace through architecture
    B, C_in, T, F = 2, 2, 256, 512
    print(f"\n🔹 INPUT: ({B}, {C_in}, {T}, {F})")
    print("   ↓")
    
    # Stem
    print(f"┌─ STEM (Conv 7x7 + BatchNorm + GELU)")
    print(f"└─ → ({B}, {config['dims'][0]}, {T}, {F})")
    print("   ↓")
    
    # Encoder
    print("┌" + "─" * 78 + "┐")
    print("│" + " " * 32 + "ENCODER" + " " * 40 + "│")
    print("├" + "─" * 78 + "┤")
    
    current_T, current_F = T, F
    for i, (depth, dim) in enumerate(zip(config['depths'][:-1], config['dims'][:-1])):
        print(f"│ Stage {i+1}:")
        print(f"│   [{depth}x TFSWABlock] (dim={dim}, heads={config['num_heads']})")
        print(f"│   - TSA: Temporal attention along {current_T} time frames")
        print(f"│   - FSA: Frequency attention along {current_F} frequency bins")
        print(f"│   - SWA: Shifted window attention (window={config['window_size']})")
        print(f"│   Shape: ({B}, {dim}, {current_T}, {current_F})")
        print(f"│   → Skip Connection {i+1}")
        
        if i < len(config['depths']) - 2:
            current_T //= 2
            current_F //= 2
            print(f"│   ↓ DOWNSAMPLE (stride=2)")
            print(f"│   → ({B}, {config['dims'][i+1]}, {current_T}, {current_F})")
        
        if i < len(config['depths']) - 2:
            print("│" + " " * 78 + "│")
    
    print("└" + "─" * 78 + "┘")
    print("   ↓")
    
    # Bottleneck
    print("┌" + "─" * 78 + "┐")
    print("│" + " " * 31 + "BOTTLENECK" + " " * 37 + "│")
    print("├" + "─" * 78 + "┤")
    print(f"│ [{config['depths'][-1]}x TFSWABlock] (dim={config['dims'][-1]}, heads={config['num_heads']})")
    print(f"│   Shape: ({B}, {config['dims'][-1]}, {current_T}, {current_F})")
    print("└" + "─" * 78 + "┘")
    print("   ↓")
    
    # Decoder
    print("┌" + "─" * 78 + "┐")
    print("│" + " " * 32 + "DECODER" + " " * 41 + "│")
    print("├" + "─" * 78 + "┤")
    
    for i in range(len(config['depths']) - 2, -1, -1):
        current_T *= 2
        current_F *= 2
        depth = config['depths'][i]
        dim = config['dims'][i]
        
        print(f"│   ↑ UPSAMPLE (stride=2)")
        print(f"│   → ({B}, {dim}, {current_T}, {current_F})")
        print(f"│ Stage {i+1}:")
        print(f"│   [{depth}x TFSWABlock] (dim={dim}, heads={config['num_heads']})")
        print(f"│   + Skip Connection {i+1} from Encoder")
        print(f"│   Shape: ({B}, {dim}, {current_T}, {current_F})")
        
        if i > 0:
            print("│" + " " * 78 + "│")
    
    print("└" + "─" * 78 + "┘")
    print("   ↓")
    
    # Output
    print(f"┌─ OUTPUT HEAD (Conv + Sigmoid)")
    print(f"└─ → ({B}, {config['out_channels']}, {T}, {F})")
    print()
    print(f"🔹 OUTPUT: Separation masks in [0, 1] range")
    
    print("\n" + "=" * 80)
    print(" " * 28 + "Module Breakdown")
    print("=" * 80)
    
    # Count parameters by module
    def count_parameters(module):
        return sum(p.numel() for p in module.parameters())
    
    print(f"\n{'Module':<30s} {'Parameters':>15s} {'% of Total':>15s}")
    print("-" * 80)
    
    total_params = model.get_num_parameters()
    
    stem_params = count_parameters(model.stem)
    print(f"{'Stem':<30s} {stem_params:>15,d} {stem_params/total_params*100:>14.2f}%")
    
    encoder_params = sum(count_parameters(stage) for stage in model.encoder_stages)
    print(f"{'Encoder (all stages)':<30s} {encoder_params:>15,d} {encoder_params/total_params*100:>14.2f}%")
    
    downsample_params = sum(count_parameters(layer) for layer in model.downsample_layers)
    print(f"{'Downsample layers':<30s} {downsample_params:>15,d} {downsample_params/total_params*100:>14.2f}%")
    
    bottleneck_params = count_parameters(model.bottleneck)
    print(f"{'Bottleneck':<30s} {bottleneck_params:>15,d} {bottleneck_params/total_params*100:>14.2f}%")
    
    upsample_params = sum(count_parameters(layer) for layer in model.upsample_layers)
    print(f"{'Upsample layers':<30s} {upsample_params:>15,d} {upsample_params/total_params*100:>14.2f}%")
    
    decoder_params = sum(count_parameters(stage) for stage in model.decoder_stages)
    print(f"{'Decoder (all stages)':<30s} {decoder_params:>15,d} {decoder_params/total_params*100:>14.2f}%")
    
    output_params = count_parameters(model.output_head)
    print(f"{'Output head':<30s} {output_params:>15,d} {output_params/total_params*100:>14.2f}%")
    
    print("-" * 80)
    print(f"{'TOTAL':<30s} {total_params:>15,d} {'100.00%':>15s}")
    
    print("\n" + "=" * 80)
    print(" " * 25 + "Attention Mechanisms Detail")
    print("=" * 80)
    
    print("\n1️⃣  TSA (Temporal Sequence Attention)")
    print("   • Processes temporal dimension (time frames)")
    print("   • Each frequency bin independently")
    print("   • Captures long-range temporal dependencies")
    print("   • Transform: (B, C, T, F) → (B*F, T, C) → attention → (B, C, T, F)")
    
    print("\n2️⃣  FSA (Frequency Sequence Attention)")
    print("   • Processes frequency dimension (frequency bins)")
    print("   • Each time frame independently")
    print("   • Captures frequency-domain relationships")
    print("   • Transform: (B, C, T, F) → (B*T, F, C) → attention → (B, C, T, F)")
    
    print("\n3️⃣  SWA (Shifted Window Attention)")
    print("   • Processes 2D spatial structure")
    print(f"   • Window size: {config['window_size']}x{config['window_size']}")
    print(f"   • Shift size: {config['shift_size']} (for SW-MSA)")
    print("   • Alternates between W-MSA (shift=0) and SW-MSA (shift>0)")
    print("   • Captures local spatial correlations efficiently")
    
    print("\n💡 Feature Fusion Strategy:")
    print("   TSA_out ─┐")
    print("   FSA_out ─┼─→ Concat → 1x1 Conv → Fused Features")
    print("   SWA_out ─┘")
    
    print("\n" + "=" * 80)
    print("✅ Phase 1 Implementation Complete!")
    print("=" * 80)


if __name__ == "__main__":
    print_architecture()
