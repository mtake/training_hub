#!/usr/bin/env python3
"""
LAB Multi-Phase Training Script

This script demonstrates how to perform LAB (Large-scale Alignment for chatBots) multi-phase 
training using the training_hub library. It executes the two-phase LAB training process:

1. Phase 1 - Knowledge Tuning (Phase07): Training on knowledge-heavy data to build foundational understanding
2. Phase 2 - Skills + Replay Training (Phase10): Training on skills data with replay of both Phase07 
   knowledge data AND the base model's original instruction tuning data to maintain all capabilities

This LAB multi-phase approach is specifically designed for instruction tuning where you first 
establish additional knowledge foundations, then add task-specific skills while preventing 
knowledge forgetting and preserving the base model's original instruction-following capabilities 
through comprehensive replay mechanisms.
"""

import os
import sys
import time
from datetime import datetime
import argparse

# Import training_hub for SFT training
from training_hub import sft


def main():
    """Main function to execute LAB multi-phase training."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='LAB Multi-Phase Training Script')
    parser.add_argument('--base-model-path', required=True, 
                       help='Path to the base model (e.g., granite-3.1-8b-starter-v2.1)')
    parser.add_argument('--phase07-data-path', required=True,
                       help='Path to knowledge data for Phase07 (JSONL format)')
    parser.add_argument('--phase10-data-path', required=True,
                       help='Path to skills + replay data for Phase10 (JSONL format)')
    parser.add_argument('--ckpt-output-base-dir', required=True,
                       help='Base directory for checkpoint outputs')
    parser.add_argument('--experiment-prefix', default='lab_multiphase_training',
                       help='Prefix for experiment names (default: lab_multiphase_training)')
    
    # Training hyperparameters
    parser.add_argument('--max-tokens-per-gpu', type=int, default=25000,
                       help='Memory limit per GPU (reduce if hitting OOM errors, default: 25000)')
    parser.add_argument('--max-seq-len', type=int, default=20000,
                       help='Maximum sequence length (default: 20000)')
    parser.add_argument('--num-epochs', type=int, default=7,
                       help='Number of training epochs per phase (default: 7)')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                       help='Learning rate for training (default: 2e-5)')
    parser.add_argument('--phase07-batch-size', type=int, default=128,
                       help='Effective batch size for Phase07 (default: 128)')
    parser.add_argument('--phase10-batch-size', type=int, default=3840,
                       help='Effective batch size for Phase10 (default: 3840)')
    
    # Distributed training parameters
    parser.add_argument('--nproc-per-node', type=int, default=8,
                       help='Number of GPUs per node (default: 8)')
    parser.add_argument('--nnodes', type=int, default=1,
                       help='Number of nodes (default: 1)')
    parser.add_argument('--node-rank', type=int, default=0,
                       help='Rank of this node (default: 0)')
    parser.add_argument('--rdzv-id', type=int, default=420,
                       help='Rendezvous ID (default: 420)')
    parser.add_argument('--rdzv-endpoint', default='0.0.0.0:12345',
                       help='Master node endpoint (default: 0.0.0.0:12345)')
    
    # Control options
    parser.add_argument('--skip-phase07', action='store_true',
                       help='Skip Phase07 and use existing checkpoint for Phase10')
    parser.add_argument('--phase07-checkpoint', 
                       help='Path to Phase07 checkpoint (required if --skip-phase07)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.skip_phase07 and not args.phase07_checkpoint:
        parser.error("--phase07-checkpoint is required when --skip-phase07 is used")
    
    print("🚀 LAB Multi-Phase Training Script")
    print("=" * 50)
    print(f"Experiment prefix: {args.experiment_prefix}")
    print(f"Base model: {args.base_model_path}")
    print(f"Output directory: {args.ckpt_output_base_dir}")
    print(f"GPUs per node: {args.nproc_per_node}")
    print(f"Max tokens per GPU: {args.max_tokens_per_gpu:,}")
    print(f"Max sequence length: {args.max_seq_len:,}")
    print("\nData composition:")
    print(f"  Phase07: Knowledge data only ({args.phase07_data_path})")
    print(f"  Phase10: Skills + Phase07 replay + Base model instruction replay ({args.phase10_data_path})")
    if args.max_tokens_per_gpu < 25000:
        print(f"\n💡 Note: Using reduced max_tokens_per_gpu for memory conservation")
    print()
    
    # Phase07: Knowledge Tuning
    most_recent_checkpoint = None
    
    if not args.skip_phase07:
        print("📚 Starting Phase07 (Knowledge Tuning)")
        print("-" * 40)
        
        experiment_prefix_phase07 = args.experiment_prefix + "_phase07"
        experiment_name_phase07 = experiment_prefix_phase07 + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        phase07_ckpt_output_dir = os.path.join(args.ckpt_output_base_dir, experiment_prefix_phase07)
        
        print(f"Phase07 Configuration:")
        print(f"  Experiment name: {experiment_name_phase07}")
        print(f"  Input model: {args.base_model_path}")
        print(f"  Data path: {args.phase07_data_path}")
        print(f"  Output directory: {phase07_ckpt_output_dir}")
        print()
        
        start_time = time.time()
        
        try:
            print("🔄 Executing Phase07 training...")
            result = sft(
                # Required parameters
                model_path=args.base_model_path,
                data_path=args.phase07_data_path,
                ckpt_output_dir=phase07_ckpt_output_dir,
                
                # Core training parameters
                num_epochs=args.num_epochs,
                effective_batch_size=args.phase07_batch_size,
                learning_rate=args.learning_rate,
                max_seq_len=args.max_seq_len,
                max_tokens_per_gpu=args.max_tokens_per_gpu,
                
                # Data and checkpointing parameters
                data_output_dir="/dev/shm",
                warmup_steps=0,
                save_samples=0,
                checkpoint_at_epoch=True,
                accelerate_full_state_at_epoch=False,  # Save space for intermediate checkpoints
                
                # Distributed training parameters
                nproc_per_node=args.nproc_per_node,
                nnodes=args.nnodes,
                node_rank=args.node_rank,
                rdzv_id=args.rdzv_id,
                rdzv_endpoint=args.rdzv_endpoint,
            )
            
            end_time = time.time()
            duration = end_time - start_time
            print(f"✅ Phase07 training completed successfully!")
            print(f"⏱️  Duration: {duration/3600:.2f} hours")
            
            # Find the most recent checkpoint
            phase07_checkpoint_location = f"{phase07_ckpt_output_dir}/hf_format"
            most_recent_checkpoint = find_most_recent_checkpoint(phase07_checkpoint_location)
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            print(f"❌ Phase07 training failed after {duration/60:.1f} minutes")
            print(f"Error: {e}")
            print("\n🔍 Common troubleshooting steps:")
            print("   - Check data path exists and is valid JSONL")
            print("   - Verify model path is correct")
            print("   - Ensure sufficient GPU memory (reduce --max-tokens-per-gpu)")
            print("   - Check disk space in output directory")
            sys.exit(1)
    else:
        print("⏭️  Skipping Phase07, using provided checkpoint")
        most_recent_checkpoint = args.phase07_checkpoint
        print(f"Using checkpoint: {most_recent_checkpoint}")
    
    # Phase10: Skills + Replay Training
    print("\n🎯 Starting Phase10 (Skills + Replay Training)")
    print("-" * 40)
    
    if not most_recent_checkpoint:
        print("❌ Cannot proceed with Phase10: No checkpoint available")
        print("   Either Phase07 failed or --phase07-checkpoint not provided")
        sys.exit(1)
    
    experiment_prefix_phase10 = args.experiment_prefix + "_phase10"
    experiment_name_phase10 = experiment_prefix_phase10 + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    phase10_ckpt_output_dir = os.path.join(args.ckpt_output_base_dir, experiment_prefix_phase10)
    
    print(f"Phase10 Configuration:")
    print(f"  Experiment name: {experiment_name_phase10}")
    print(f"  Input model (from Phase07): {most_recent_checkpoint}")
    print(f"  Data path: {args.phase10_data_path}")
    print(f"  Output directory: {phase10_ckpt_output_dir}")
    print(f"  Training on skills + comprehensive replay data...")
    print(f"  ↳ Skills data + Phase07 knowledge replay + Base model instruction replay")
    print()
    
    start_time = time.time()
    
    try:
        print("🔄 Executing Phase10 training...")
        result = sft(
            # Required parameters
            model_path=most_recent_checkpoint,
            data_path=args.phase10_data_path,
            ckpt_output_dir=phase10_ckpt_output_dir,
            
            # Core training parameters
            num_epochs=args.num_epochs,
            effective_batch_size=args.phase10_batch_size,
            learning_rate=args.learning_rate,
            max_seq_len=args.max_seq_len,
            max_tokens_per_gpu=args.max_tokens_per_gpu,
            
            # Data and checkpointing parameters
            data_output_dir="/dev/shm",
            warmup_steps=0,
            save_samples=0,
            checkpoint_at_epoch=True,
            accelerate_full_state_at_epoch=True,  # Enable for final model
            
            # Distributed training parameters
            nproc_per_node=args.nproc_per_node,
            nnodes=args.nnodes,
            node_rank=args.node_rank,
            rdzv_id=args.rdzv_id,
            rdzv_endpoint=args.rdzv_endpoint,
        )
        
        end_time = time.time()
        duration = end_time - start_time
        print(f"✅ Phase10 training completed successfully!")
        print(f"⏱️  Duration: {duration/3600:.2f} hours")
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"❌ Phase10 training failed after {duration/60:.1f} minutes")
        print(f"Error: {e}")
        print("\n🔍 Common troubleshooting steps:")
        print("   - Check Phase10 data path exists and is valid JSONL")
        print("   - Verify Phase07 checkpoint is accessible")
        print("   - Ensure sufficient GPU memory (reduce --max-tokens-per-gpu)")
        print("   - Check disk space in output directory")
        sys.exit(1)
    
    # Training Summary
    print("\n🎉 LAB Multi-Phase Training Summary")
    print("=" * 50)
    
    if not args.skip_phase07:
        print(f"📁 Phase07 (Knowledge Tuning) Output: {phase07_ckpt_output_dir}")
    
    print(f"📁 Phase10 (Skills + Replay) Output: {phase10_ckpt_output_dir}")
    print(f"\n🎯 Final trained model location:")
    print(f"   {phase10_ckpt_output_dir}/hf_format/[latest_checkpoint]")
    
    # List final checkpoints if available
    final_ckpt_dir = f"{phase10_ckpt_output_dir}/hf_format"
    if os.path.exists(final_ckpt_dir):
        final_checkpoints = [d for d in os.listdir(final_ckpt_dir) 
                           if os.path.isdir(os.path.join(final_ckpt_dir, d))]
        if final_checkpoints:
            print(f"\n📋 Available final checkpoints:")
            for ckpt in sorted(final_checkpoints):
                print(f"   - {ckpt}")
    
    print(f"\n🔧 LAB Training Configuration Used:")
    print(f"   - Max tokens per GPU: {args.max_tokens_per_gpu:,}")
    print(f"   - Max sequence length: {args.max_seq_len:,}")
    print(f"   - GPUs per node: {args.nproc_per_node}")
    print(f"   - Phase07 batch size: {args.phase07_batch_size}")
    print(f"   - Phase10 batch size: {args.phase10_batch_size}")
    print(f"   - Learning rate: {args.learning_rate}")
    print(f"   - Epochs per phase: {args.num_epochs}")
    
    print(f"\n📊 Data Composition:")
    print(f"   - Phase07: Knowledge data only")
    print(f"   - Phase10: Skills + Phase07 replay + Base model instruction replay")
    
    print(f"\n💡 Next Steps:")
    print(f"   1. Evaluate your model on relevant benchmarks")
    print(f"   2. Test with sample prompts to verify training quality")
    print(f"   3. Check knowledge retention from Phase07")
    print(f"   4. Verify new skills acquisition from Phase10")
    print(f"   5. Confirm base model instruction-following capabilities are preserved")
    print(f"   6. Deploy for inference using your preferred serving framework")
    
    print(f"\n✨ LAB Multi-Phase Training completed successfully!")


def find_most_recent_checkpoint(checkpoint_dir):
    """Find the most recent checkpoint in the specified directory."""
    
    print(f"Looking for checkpoints in: {checkpoint_dir}")
    
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return None
    
    checkpoints = os.listdir(checkpoint_dir)
    
    if not checkpoints:
        print(f"❌ No checkpoints found in {checkpoint_dir}")
        return None
    
    print(f"Found {len(checkpoints)} checkpoint(s):")
    for ckpt in checkpoints:
        print(f"  - {ckpt}")
    
    # Find the most recent checkpoint
    most_recent_checkpoint, most_recent_time = None, 0
    
    for checkpoint in checkpoints:
        full_ckpt_path = f"{checkpoint_dir}/{checkpoint}"
        if os.path.isdir(full_ckpt_path):
            ckpt_time = os.stat(full_ckpt_path).st_ctime
            if ckpt_time > most_recent_time:
                most_recent_checkpoint = full_ckpt_path
                most_recent_time = ckpt_time
    
    if most_recent_checkpoint:
        print(f"\n✅ Most recent checkpoint: {most_recent_checkpoint}")
        print(f"   Created: {datetime.fromtimestamp(most_recent_time)}")
        return most_recent_checkpoint
    else:
        print("❌ No valid checkpoint directories found")
        return None


if __name__ == "__main__":
    main()