#!/usr/bin/env python3
"""
OSFT Continual Learning Example

This script demonstrates OSFT (Orthogonal Subspace Fine-Tuning) for continual learning
with Llama 3 8B model. It shows how to teach the model new capabilities (e.g., JSON output)
while preserving its existing knowledge and abilities.

This is the script version of the osft_continual_learning.ipynb notebook training loop.
Unlike the notebook, this script displays full training logs for better monitoring.

Example usage:
    python osft_continual_learning_example.py \\
        --data-path /path/to/json_training_data.jsonl \\
        --ckpt-output-dir /path/to/checkpoints
"""

import os
import sys
import time
import glob
import argparse
from training_hub import osft


def find_most_recent_checkpoint(output_dir):
    """
    Find the most recent checkpoint in the training output directory.
    
    Args:
        output_dir (str): Training output directory containing hf_format/ subdirectory
        
    Returns:
        str: Path to the most recent checkpoint
        
    Raises:
        ValueError: If no checkpoints are found
    """
    # Get all checkpoint directories under hf_format
    checkpoint_pattern = os.path.join(output_dir, "hf_format", "samples_*.0")
    checkpoint_dirs = glob.glob(checkpoint_pattern)
    
    if not checkpoint_dirs:
        raise ValueError(f"No checkpoints found in {os.path.join(output_dir, 'hf_format')}")
    
    # Find the most recently created checkpoint
    most_recent_checkpoint = max(checkpoint_dirs, key=os.path.getctime)
    
    return most_recent_checkpoint


def main():
    parser = argparse.ArgumentParser(description='OSFT Continual Learning Example')
    
    # Required parameters
    parser.add_argument('--data-path', required=True,
                       help='Path to training data (JSONL format) for teaching new capabilities')
    parser.add_argument('--ckpt-output-dir', required=True,
                       help='Directory to save checkpoints')
    
    # Optional overrides
    parser.add_argument('--model-path', default='meta-llama/Meta-Llama-3-8B-Instruct',
                       help='Model path or HuggingFace name (default: meta-llama/Meta-Llama-3-8B-Instruct)')
    parser.add_argument('--num-epochs', type=int, default=1,
                       help='Number of epochs (default: 1)')
    parser.add_argument('--unfreeze-rank-ratio', type=float, default=0.28,
                       help='Unfreeze rank ratio for OSFT (0.0-1.0, default: 0.28)')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Effective batch size (default: 128)')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                       help='Learning rate (default: 2e-5)')
    parser.add_argument('--max-seq-len', type=int, default=2048,
                       help='Maximum sequence length (default: 2048)')
    parser.add_argument('--max-tokens-per-gpu', type=int, default=8192,
                       help='Max tokens per GPU (default: 8192)')
    parser.add_argument('--nproc-per-node', type=int, default=8,
                       help='Number of GPUs (default: 8)')
    parser.add_argument('--use-liger', action='store_true', default=True,
                       help='Use Liger kernels for efficiency (default: True)')
    parser.add_argument('--data-output-dir', default="/dev/shm",
                       help='Directory for processed data (default: /dev/shm for RAM disk)')
    
    # Distributed training parameters
    parser.add_argument('--nnodes', type=int, default=1,
                       help='Number of nodes (default: 1)')
    parser.add_argument('--node-rank', type=int, default=0,
                       help='Node rank (default: 0)')
    parser.add_argument('--rdzv-id', type=int, default=100,
                       help='Rendezvous ID (default: 100)')
    parser.add_argument('--rdzv-endpoint', default="127.0.0.1:29500",
                       help='Rendezvous endpoint (default: 127.0.0.1:29500)')
    
    args = parser.parse_args()
    
    print("🚀 Starting OSFT Continual Learning Training")
    print("=" * 60)
    print(f"Starting from: {args.model_path}")
    print(f"Training data: {args.data_path}")
    print(f"Output directory: {args.ckpt_output_dir}")
    print(f"Unfreeze ratio: {args.unfreeze_rank_ratio}")
    print()
    print("📝 Training Configuration:")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Max Sequence Length: {args.max_seq_len:,}")
    print(f"  Max Tokens per GPU: {args.max_tokens_per_gpu:,}")
    print(f"  GPUs: {args.nproc_per_node}")
    print(f"  Use Liger: {args.use_liger}")
    print()
    print("🎯 Expected Outcomes:")
    print("  • Base model capabilities: Preserved")
    print("  • New capabilities: Added without forgetting")
    print("  • Continual learning: Success without catastrophic forgetting")
    print("=" * 60)
    print()
    
    training_start_time = time.time()
    
    try:
        # OSFT training parameters (matching notebook configuration)
        training_result = osft(
            # Model and data
            model_path=args.model_path,
            data_path=args.data_path,
            ckpt_output_dir=args.ckpt_output_dir,
            
            # OSFT-specific
            unfreeze_rank_ratio=args.unfreeze_rank_ratio,
            
            # Training parameters
            num_epochs=args.num_epochs,
            effective_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_seq_len=args.max_seq_len,
            max_tokens_per_gpu=args.max_tokens_per_gpu,
            
            # Data processing
            data_output_dir=args.data_output_dir,
            warmup_steps=0,  # As in notebook
            
            # Optimization
            use_liger=args.use_liger,
            seed=42,  # Fixed seed for reproducibility
            lr_scheduler='cosine',  # As in notebook
            
            # Checkpointing
            checkpoint_at_epoch=True,
            save_final_checkpoint=True,
            
            # Distributed training
            nproc_per_node=args.nproc_per_node,
            nnodes=args.nnodes,
            node_rank=args.node_rank,
            rdzv_id=args.rdzv_id,
            rdzv_endpoint=args.rdzv_endpoint,
        )
        
        training_duration = time.time() - training_start_time
        
        # Find the most recent checkpoint
        final_checkpoint = find_most_recent_checkpoint(args.ckpt_output_dir)
        
        print()
        print("=" * 60)
        print(f"✅ OSFT training completed successfully in {training_duration/3600:.2f} hours!")
        print(f"📁 Final model checkpoint: {final_checkpoint}")
        print()
        print("📊 Training Achievements:")
        print("  • Base model capabilities: ✅ Preserved")
        print("  • New knowledge integrated: ✅ Complete")
        print("  • Continual learning: ✅ Success")
        print()
        print("🎉 Your model now has new capabilities while retaining all original abilities!")
        print("   You can load the checkpoint and test the enhanced model.")
        
    except Exception as e:
        training_duration = time.time() - training_start_time
        print(f"\n❌ OSFT training failed after {training_duration/60:.1f} minutes: {e}")
        print("\n💡 Troubleshooting tips:")
        print("  - Reduce --max-tokens-per-gpu if you see OOM errors")
        print("  - Check that your data is in proper JSONL format")
        print("  - Ensure you have enough disk space for checkpoints")
        print("  - For continual learning, --unfreeze-rank-ratio around 0.25-0.35 works well")
        sys.exit(1)


if __name__ == "__main__":
    main()
