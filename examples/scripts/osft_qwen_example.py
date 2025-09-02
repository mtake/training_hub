#!/usr/bin/env python3
"""
OSFT Training Example: Qwen 2.5 7B Instruct

This script demonstrates OSFT (Orthogonal Subspace Fine-Tuning) training with Qwen 2.5 7B model
using a single-node, multi-GPU setup with training_hub.

OSFT allows continual training without catastrophic forgetting, making it ideal for:
- Adapting Qwen's multilingual capabilities to specific languages or domains
- Adding specialized knowledge while preserving general reasoning
- Efficient domain adaptation without replay mechanisms

Example usage:
    python osft_qwen_example.py \\
        --data-path /path/to/data.jsonl \\
        --ckpt-output-dir /path/to/checkpoints
"""

import os
import sys
import time
from datetime import datetime
import argparse
import glob

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
    parser = argparse.ArgumentParser(description='OSFT Training Example: Qwen 2.5 7B Instruct')
    
    # Required parameters
    parser.add_argument('--data-path', required=True,
                       help='Path to training data (JSONL format)')
    parser.add_argument('--ckpt-output-dir', required=True,
                       help='Directory to save checkpoints')
    
    # Optional overrides
    parser.add_argument('--model-path', default='Qwen/Qwen2.5-7B-Instruct',
                       help='Model path or HuggingFace name (default: Qwen/Qwen2.5-7B-Instruct)')
    parser.add_argument('--num-epochs', type=int, default=3,
                       help='Number of epochs (default: 3)')
    parser.add_argument('--unfreeze-rank-ratio', type=float, default=0.25,
                       help='Unfreeze rank ratio for OSFT (0.0-1.0, default: 0.25)')
    parser.add_argument('--max-tokens-per-gpu', type=int, default=16384,
                       help='Max tokens per GPU (default: 16384)')
    parser.add_argument('--nproc-per-node', type=int, default=8,
                       help='Number of GPUs (default: 8)')
    parser.add_argument('--use-processed-dataset', action='store_true',
                       help='Use pre-processed dataset with input_ids and labels')
    parser.add_argument('--unmask-messages', action='store_true',
                       help='Unmask all messages for pretraining-style learning')
    parser.add_argument('--learning-rate', type=float, default=5e-6,
                       help='Learning rate for training (default: 5e-6)')
    
    args = parser.parse_args()
    
    # Qwen 2.5 7B OSFT configuration
    print("🚀 OSFT Training: Qwen 2.5 7B Instruct")
    print("=" * 50)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_path}")
    print(f"Output: {args.ckpt_output_dir}")
    print(f"GPUs: {args.nproc_per_node}")
    print(f"Unfreeze Rank Ratio: {args.unfreeze_rank_ratio}")
    print(f"Max tokens per GPU: {args.max_tokens_per_gpu:,}")
    if args.use_processed_dataset:
        print("Using pre-processed dataset")
    if args.unmask_messages:
        print("Unmasking all messages (pretraining mode)")
    print()
    print("📝 OSFT Benefits for Qwen:")
    print("   • Preserve Qwen's multilingual and reasoning capabilities")
    print("   • Efficiently adapt to domain-specific or language-specific tasks")
    print("   • No need for complex data mixing strategies")
    print()
    
    # Training configuration optimized for Qwen 2.5 7B with OSFT
    start_time = time.time()
    
    try:
        result = osft(
            # Model and data
            model_path=args.model_path,
            data_path=args.data_path,
            ckpt_output_dir=args.ckpt_output_dir,
            
            # OSFT-specific parameters
            unfreeze_rank_ratio=args.unfreeze_rank_ratio,  # Conservative for preservation
            
            # Training parameters optimized for Qwen 2.5 7B
            num_epochs=args.num_epochs,
            effective_batch_size=128,           # Standard for 7B models
            learning_rate=args.learning_rate,   # Standard LR for Qwen fine-tuning
            max_seq_len=8192,                  # Qwen 2.5 supports long context
            max_tokens_per_gpu=args.max_tokens_per_gpu,
            
            # Data processing options
            data_output_dir="/dev/shm",         # Use RAM disk for speed
            use_processed_dataset=args.use_processed_dataset,
            unmask_messages=args.unmask_messages,
            warmup_steps=0,
            
            # Optimization
            use_liger=True,                     # Enable Liger kernels
            seed=42,
            
            # Learning rate scheduler
            lr_scheduler="cosine",              # Cosine scheduler works well with OSFT
            
            # Checkpointing
            checkpoint_at_epoch=True,
            save_final_checkpoint=True,
            
            # Single-node multi-GPU setup
            nproc_per_node=args.nproc_per_node,
            nnodes=1,
            node_rank=0,
            rdzv_id=104,
            rdzv_endpoint="127.0.0.1:29500",
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        most_recent_checkpoint = find_most_recent_checkpoint(args.ckpt_output_dir)
        
        print("=" * 50)
        print("✅ OSFT Training completed successfully!")
        print(f"⏱️  Duration: {duration/3600:.2f} hours")
        print(f"📁 Checkpoints: {args.ckpt_output_dir}/hf_format")
        print(f"   Most recent checkpoint: {most_recent_checkpoint}")
        print()
        print("🎯 Your Qwen model has been successfully adapted!")
        print("   The model now incorporates your specialized knowledge")
        print("   while maintaining its multilingual and reasoning abilities.")
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        print("=" * 50)
        print(f"❌ Training failed after {duration/60:.1f} minutes")
        print(f"Error: {e}")
        print()
        print("💡 Troubleshooting tips:")
        print("   - Reduce --max-tokens-per-gpu if you see OOM errors")
        print("   - For domain adaptation, try --unfreeze-rank-ratio between 0.2-0.4")
        sys.exit(1)


if __name__ == "__main__":
    main()

