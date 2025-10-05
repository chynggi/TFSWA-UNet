"""
Evaluation script for TFSWA-UNet on MUSDB18.

Usage:
    python scripts/evaluate.py --checkpoint outputs/best_model.pth --data_root data/musdb18
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.evaluation.inference import load_separator_from_checkpoint
from src.evaluation.evaluator import MUSDB18Evaluator, CustomDatasetEvaluator


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate TFSWA-UNet on MUSDB18'
    )
    
    # Model
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    
    # Data
    parser.add_argument(
        '--data_root',
        type=str,
        required=True,
        help='Path to MUSDB18 dataset root'
    )
    parser.add_argument(
        '--subset',
        type=str,
        default='test',
        choices=['train', 'test'],
        help='Dataset subset to evaluate'
    )
    
    # Stems
    parser.add_argument(
        '--target_stems',
        type=str,
        nargs='+',
        default=['vocals', 'other'],
        choices=['vocals', 'drums', 'bass', 'other'],
        help='Target stems to separate and evaluate'
    )
    
    # Output
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/evaluation',
        help='Output directory for results'
    )
    parser.add_argument(
        '--save_estimates',
        action='store_true',
        help='Save separated audio estimates'
    )
    
    # Inference settings
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run inference on'
    )
    parser.add_argument(
        '--use_amp',
        action='store_true',
        default=True,
        help='Use automatic mixed precision'
    )
    parser.add_argument(
        '--segment_length',
        type=float,
        default=10.0,
        help='Segment length in seconds for processing'
    )
    parser.add_argument(
        '--overlap',
        type=float,
        default=0.25,
        help='Overlap ratio between segments'
    )
    
    # Evaluation
    parser.add_argument(
        '--use_museval',
        action='store_true',
        default=True,
        help='Use official museval metrics'
    )
    parser.add_argument(
        '--num_tracks',
        type=int,
        default=None,
        help='Number of tracks to evaluate (None = all)'
    )
    
    # Custom dataset
    parser.add_argument(
        '--custom_dataset',
        action='store_true',
        help='Evaluate on custom dataset instead of MUSDB18'
    )
    parser.add_argument(
        '--mixture_dir',
        type=str,
        default=None,
        help='Directory containing mixture audio files (for custom dataset)'
    )
    parser.add_argument(
        '--reference_dirs',
        type=str,
        nargs='+',
        default=None,
        help='Directories containing reference stems (for custom dataset)'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*60)
    print("TFSWA-UNet Evaluation")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data root: {args.data_root}")
    print(f"Subset: {args.subset}")
    print(f"Target stems: {', '.join(args.target_stems)}")
    print(f"Device: {args.device}")
    print(f"Output directory: {args.output_dir}")
    print("="*60 + "\n")
    
    # Load separator
    print("Loading model from checkpoint...")
    separator = load_separator_from_checkpoint(
        checkpoint_path=args.checkpoint,
        device=args.device,
        use_amp=args.use_amp,
        segment_length=args.segment_length,
        overlap=args.overlap
    )
    print("Model loaded successfully!\n")
    
    # Evaluate
    if args.custom_dataset:
        # Custom dataset evaluation
        print("Evaluating on custom dataset...")
        
        if not args.mixture_dir or not args.reference_dirs:
            raise ValueError(
                "Custom dataset requires --mixture_dir and --reference_dirs"
            )
        
        # Collect file paths
        mixture_dir = Path(args.mixture_dir)
        mixture_paths = sorted(mixture_dir.glob('*.wav'))
        
        reference_paths = {}
        for i, stem_name in enumerate(args.target_stems):
            ref_dir = Path(args.reference_dirs[i])
            reference_paths[stem_name] = sorted(ref_dir.glob('*.wav'))
        
        # Create evaluator
        evaluator = CustomDatasetEvaluator(
            separator=separator,
            output_dir=args.output_dir
        )
        
        # Evaluate
        results = evaluator.evaluate_pairs(
            mixture_paths=[str(p) for p in mixture_paths],
            reference_paths={
                k: [str(p) for p in v]
                for k, v in reference_paths.items()
            },
            stem_names=args.target_stems,
            verbose=True
        )
        
    else:
        # MUSDB18 evaluation
        print("Evaluating on MUSDB18...")
        
        # Create evaluator
        evaluator = MUSDB18Evaluator(
            separator=separator,
            data_root=args.data_root,
            subset=args.subset,
            output_dir=args.output_dir,
            save_estimates=args.save_estimates,
            use_museval=args.use_museval
        )
        
        # Evaluate
        results = evaluator.evaluate(
            stem_names=args.target_stems,
            num_tracks=args.num_tracks,
            verbose=True
        )
    
    print("\n" + "="*60)
    print("Evaluation complete!")
    print(f"Results saved to: {args.output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
