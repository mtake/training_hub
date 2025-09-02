#!/usr/bin/env python3
"""
LAB Multi-Phase Training Script with OSFT

This script demonstrates how to perform LAB (Large-scale Alignment for chatBots) multi-phase 
training using the OSFT algorithm. With OSFT, we can execute the two-phase LAB training process
WITHOUT needing replay buffers:

1. Phase 1 - Knowledge Tuning (Phase07): Training on knowledge-heavy data to build foundational understanding
2. Phase 2 - Skills Training (Phase10): Training on skills data with a reduced unfreeze_rank_ratio
   to preserve the knowledge from Phase 1 while adding new skills

The key innovation with OSFT is that we don't need replay mechanisms - the algorithm naturally
preserves prior capabilities. We reduce the unfreeze_rank_ratio in Phase 2 to ensure better
preservation of Phase 1 knowledge while still allowing skill acquisition.

Example: If Phase 1 uses unfreeze_rank_ratio=0.3, Phase 2 might use 0.25 or 0.2
"""

import os
import sys
import time
from datetime import datetime
import argparse
import glob
import json

# Import training_hub for OSFT training
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
    """Main function to execute LAB multi-phase training with OSFT."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='LAB Multi-Phase Training Script with OSFT')
    parser.add_argument('--base-model-path', required=True, 
                       help='Path to the base model (e.g., Llama-3.1-8B-Instruct)')
    parser.add_argument('--phase07-data-path', required=True,
                       help='Path to knowledge data for Phase07 (JSONL format)')
    parser.add_argument('--phase10-data-path', required=True,
                       help='Path to skills data for Phase10 (JSONL format, no replay needed!)')
    parser.add_argument('--ckpt-output-base-dir', required=True,
                       help='Base directory for checkpoint outputs')
    parser.add_argument('--data-output-dir', default='/dev/shm',
                       help='Directory for data processing (default: /dev/shm)')
    parser.add_argument('--experiment-prefix', default='lab_osft_multiphase',
                       help='Prefix for experiment names (default: lab_osft_multiphase)')
    
    # OSFT-specific parameters
    parser.add_argument('--phase07-unfreeze-ratio', type=float, default=0.3,
                       help='Unfreeze rank ratio for Phase07 knowledge training (default: 0.3)')
    parser.add_argument('--phase10-unfreeze-reduction', type=float, default=0.1,
                       help='How much to reduce unfreeze ratio for Phase10 (default: 0.1, so 0.3->0.2)')
    
    # Training hyperparameters
    parser.add_argument('--num-epochs', type=int, default=2,
                       help='Number of training epochs per phase (default: 3)')
    parser.add_argument('--learning-rate', type=float, default=5e-6,
                       help='Learning rate for training (default: 5e-6)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--lr-scheduler', default='cosine',
                       help='Learning rate scheduler (default: cosine)')
    parser.add_argument('--lr-scheduler-kwargs', default='{}',
                       help='Additional keyword arguments for the learning rate scheduler (default: {})')
    parser.add_argument('--phase07-learning-rate', type=float, default=5e-6,
                       help='Learning rate for Phase07 (default: 5e-6)')
    parser.add_argument('--phase10-learning-rate', type=float, default=5e-6,
                       help='Learning rate for Phase10 (default: 5e-6)')
    parser.add_argument('--phase07-batch-size', type=int, default=128,
                       help='Effective batch size for Phase07 (default: 128)')
    parser.add_argument('--phase10-batch-size', type=int, default=128,
                       help='Effective batch size for Phase10 (default: 128)')
    parser.add_argument('--phase07-warmup-steps', type=int, default=0,
                       help='Warmup steps for Phase07 (default: 0)')
    parser.add_argument('--phase10-warmup-steps', type=int, default=0,
                       help='Warmup steps for Phase10 (default: 0)')

    # Message masking control
    parser.add_argument('--phase07-mask-messages', action='store_true', default=False,
                       help='Mask messages during Phase07 training (default: False, messages are unmasked)')

    # Performance Hyperparameters
    parser.add_argument('--max-tokens-per-gpu', type=int, default=10000,
                       help='Memory limit per GPU (reduce if hitting OOM errors, default: 10000)')
    parser.add_argument('--max-seq-len', type=int, default=8192,
                       help='Maximum sequence length (default: 8192)')
    parser.add_argument('--use-liger', action='store_true', default=False,
                       help='Use Liger kernels for efficiency (default: False)')
    
    # Distributed training parameters
    parser.add_argument('--nproc-per-node', type=int, default=8,
                       help='Number of GPUs per node (default: 8)')
    parser.add_argument('--nnodes', type=int, default=1,
                       help='Number of nodes (default: 1)')
    parser.add_argument('--node-rank', type=int, default=0,
                       help='Rank of this node (default: 0)')
    parser.add_argument('--rdzv-id', type=int, default=47,
                       help='Rendezvous ID (default: 47)')
    parser.add_argument('--rdzv-endpoint', default='0.0.0.0:12345',
                       help='Master node endpoint (default: 0.0.0.0:12345)')
    
    # Control options
    parser.add_argument('--skip-phase07', action='store_true',
                       help='Skip Phase07 and use existing checkpoint for Phase10')
    parser.add_argument('--phase07-checkpoint', 
                       help='Path to Phase07 checkpoint (required if --skip-phase07)')
    parser.add_argument('--checkpoint-at-epoch', action='store_true', default=False,
                       help='Checkpoint at epoch (default: False)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.skip_phase07 and not args.phase07_checkpoint:
        parser.error("--phase07-checkpoint is required when --skip-phase07 is used")

    # Get lr scheduler
    try:
        args.lr_scheduler_kwargs = json.loads(args.lr_scheduler_kwargs)
    except json.JSONDecodeError as e:
        print(f"Error parsing lr_scheduler_kwargs: {e}")
        sys.exit(1)
    finally:
        # this will be a default dict but we want it to specifically be None
        if not args.lr_scheduler_kwargs:
            args.lr_scheduler_kwargs = None
    
    
    # Calculate Phase10 unfreeze ratio
    phase10_unfreeze_ratio = max(0.1, args.phase07_unfreeze_ratio - args.phase10_unfreeze_reduction)
    
    print("🚀 LAB Multi-Phase Training with OSFT")
    print("=" * 50)
    print(f"Experiment prefix: {args.experiment_prefix}")
    print(f"Base model: {args.base_model_path}")
    print(f"Output directory: {args.ckpt_output_base_dir}")
    print(f"GPUs per node: {args.nproc_per_node}")
    print(f"Max tokens per GPU: {args.max_tokens_per_gpu:,}")
    print(f"Max sequence length: {args.max_seq_len:,}")
    print(f"Use Liger: {args.use_liger}")
    print(f"Learning rate scheduler: {args.lr_scheduler}")
    print(f"Learning rate scheduler kwargs: {args.lr_scheduler_kwargs}")
    print(f"Phase07 learning rate: {args.phase07_learning_rate}")
    print(f"Phase10 learning rate: {args.phase10_learning_rate}")
    print(f"Seed: {args.seed}")
    print(f"Phase07 warmup steps: {args.phase07_warmup_steps}")
    print(f"Phase10 warmup steps: {args.phase10_warmup_steps}")
    print(f"Checkpoint at epoch: {args.checkpoint_at_epoch}")
    print()
    print("✨ OSFT Configuration:")
    print(f"  Phase07 unfreeze_rank_ratio: {args.phase07_unfreeze_ratio}")
    print(f"  Phase10 unfreeze_rank_ratio: {phase10_unfreeze_ratio} (reduced by {args.phase10_unfreeze_reduction})")
    print(f"  Data output directory: {args.data_output_dir}")
    print()
    print("📊 Data composition:")
    print(f"  Phase07: Knowledge data only ({args.phase07_data_path})")
    if args.phase07_mask_messages:
        print(f"    !!Messages are being masked during Phase07 training!!")
    print(f"  Phase10: Skills data only ({args.phase10_data_path})")
    print()
    print("🎯 Key Advantage: No replay buffers needed with OSFT!")
    print("   The algorithm naturally preserves prior knowledge.")
    print()
    
    # Phase07: Knowledge Tuning
    most_recent_checkpoint = None
    
    if not args.skip_phase07:
        print("=" * 50)
        print("📚 Phase 1 (Phase07): Knowledge Tuning with OSFT")
        print("=" * 50)
        
        phase07_output_dir = os.path.join(
            args.ckpt_output_base_dir, 
            f"{args.experiment_prefix}_phase07_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        print(f"Training on knowledge data: {args.phase07_data_path}")
        print(f"Output directory: {phase07_output_dir}")
        print(f"Unfreeze rank ratio: {args.phase07_unfreeze_ratio}")
        print()
        
        phase07_start = time.time()
        
        try:
            # Phase07 OSFT training
            osft(
                # Model and data
                model_path=args.base_model_path,
                data_path=args.phase07_data_path,
                ckpt_output_dir=phase07_output_dir,
                
                # OSFT-specific
                unfreeze_rank_ratio=args.phase07_unfreeze_ratio,
                
                # Training configuration
                num_epochs=args.num_epochs,
                effective_batch_size=args.phase07_batch_size,
                learning_rate=args.phase07_learning_rate,
                max_seq_len=args.max_seq_len,
                max_tokens_per_gpu=args.max_tokens_per_gpu,
                
                # Data processing
                data_output_dir=args.data_output_dir,
                warmup_steps=args.phase07_warmup_steps,
                unmask_messages=not args.phase07_mask_messages,
                
                # Optimization
                use_liger=args.use_liger,
                lr_scheduler=args.lr_scheduler,
                lr_scheduler_kwargs=args.lr_scheduler_kwargs,
                seed=args.seed,
                
                # Checkpointing
                checkpoint_at_epoch=args.checkpoint_at_epoch,
                save_final_checkpoint=True,
                
                # Distributed training
                nproc_per_node=args.nproc_per_node,
                nnodes=args.nnodes,
                node_rank=args.node_rank,
                rdzv_id=args.rdzv_id,
                rdzv_endpoint=args.rdzv_endpoint,
            )
            
            phase07_duration = time.time() - phase07_start
            print(f"✅ Phase07 completed in {phase07_duration/3600:.2f} hours")
            
            # Get the most recent checkpoint for Phase10
            most_recent_checkpoint = find_most_recent_checkpoint(phase07_output_dir)
            
        except Exception as e:
            print(f"❌ Phase07 training failed: {e}")
            sys.exit(1)
    else:
        print("⏩ Skipping Phase07, using checkpoint: {args.phase07_checkpoint}")
        most_recent_checkpoint = args.phase07_checkpoint
    
    # Phase10: Skills Training
    print()
    print("=" * 50)
    print("🎯 Phase 2 (Phase10): Skills Training with OSFT")
    print("=" * 50)
    
    phase10_output_dir = os.path.join(
        args.ckpt_output_base_dir, 
        f"{args.experiment_prefix}_phase10_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    print(f"Starting from Phase07 checkpoint: {most_recent_checkpoint}")
    print(f"Training on skills data: {args.phase10_data_path}")
    print(f"Output directory: {phase10_output_dir}")
    print(f"Unfreeze rank ratio: {phase10_unfreeze_ratio} (reduced for better preservation)")
    print()
    print("💡 Note: With OSFT, we don't need replay data!")
    print("   The reduced unfreeze_rank_ratio preserves Phase07 knowledge")
    print("   while allowing the model to acquire new skills.")
    print()
    
    phase10_start = time.time()
    
    try:
        # Phase10 OSFT training (starting from Phase07 checkpoint)
        osft(
            # Start from Phase07 checkpoint
            model_path=most_recent_checkpoint,
            data_path=args.phase10_data_path,
            ckpt_output_dir=phase10_output_dir,
            
            # OSFT-specific: Reduced ratio for better preservation
            unfreeze_rank_ratio=phase10_unfreeze_ratio,
            
            # Training configuration (potentially larger batch for skills)
            num_epochs=args.num_epochs,
            effective_batch_size=args.phase10_batch_size,
            learning_rate=args.phase10_learning_rate,
            max_seq_len=args.max_seq_len,
            max_tokens_per_gpu=args.max_tokens_per_gpu,
            
            # Data processing
            data_output_dir=args.data_output_dir,
            warmup_steps=args.phase10_warmup_steps,  # Shorter warmup for continuation
            
            # Optimization
            use_liger=args.use_liger,
            lr_scheduler=args.lr_scheduler,
            lr_scheduler_kwargs=args.lr_scheduler_kwargs,
            seed=args.seed,
            
            # Checkpointing
            checkpoint_at_epoch=args.checkpoint_at_epoch,
            save_final_checkpoint=True,
            
            # Distributed training
            nproc_per_node=args.nproc_per_node,
            nnodes=args.nnodes,
            node_rank=args.node_rank,
            rdzv_id=args.rdzv_id,  # Different ID for Phase10
            rdzv_endpoint=args.rdzv_endpoint,
        )
        
        phase10_duration = time.time() - phase10_start
        phase10_checkpoint = find_most_recent_checkpoint(phase10_output_dir)
        print(f"✅ Phase10 completed in {phase10_duration/3600:.2f} hours")
        
    except Exception as e:
        print(f"❌ Phase10 training failed: {e}")
        sys.exit(1)
    
    # Final summary
    total_duration = (time.time() - phase07_start if not args.skip_phase07 
                     else time.time() - phase10_start)
    
    print()
    print("=" * 50)
    print("🎉 LAB Multi-Phase OSFT Training Complete!")
    print("=" * 50)
    print(f"Total training time: {total_duration/3600:.2f} hours")
    print(f"Final model checkpoint: {phase10_checkpoint}")
    print()
    print("📊 Training Summary:")
    print(f"  • Phase07: Knowledge training with ratio {args.phase07_unfreeze_ratio}")
    print(f"  • Phase10: Skills training with ratio {phase10_unfreeze_ratio}")
    print(f"  • No replay buffers needed!")
    print(f"  • Data processing directory: {args.data_output_dir}")
    print(f"  • Model preserves all capabilities through OSFT")
    print()
    print("🚀 Your model now has:")
    print("  1. Original base model capabilities (preserved)")
    print("  2. New knowledge from Phase07 (integrated)")
    print("  3. Task-specific skills from Phase10 (acquired)")
    print("  All without catastrophic forgetting!")
    print()
    print("Next steps:")
    print("  1. Test the model on evaluation sets")
    print("  2. Compare with base model to verify improvements")
    print("  3. Deploy with confidence - no regression expected!")


if __name__ == "__main__":
    main()



