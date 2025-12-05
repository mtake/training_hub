#!/usr/bin/env python3
"""
SFT Training Example: GPT-OSS 20B

This script demonstrates SFT training with GPT-OSS 20B model from OpenAI
using a single-node, multi-GPU setup with training_hub.

GPT-OSS 20B is a high-quality open source model that provides excellent
performance for supervised fine-tuning tasks.

Example usage:
    python sft_gpt_oss_example.py \
        --data-path /path/to/data.jsonl \
        --ckpt-output-dir /path/to/checkpoints
"""

import os
import sys
import time
import glob
from datetime import datetime
import argparse
import torch

from training_hub import sft


# =============================================================================
# MODEL CONFIGURATION EXAMPLE
# =============================================================================

model_name = 'GPT-OSS 20B'
default_model_path = 'openai/gpt-oss-20b'
example_min_nproc_per_node = 8
example_max_tokens_per_gpu = 12000
example_max_seq_len = 8192
example_batch_size = 32
example_learning_rate = 2e-5
default_num_epochs = 3
default_nproc_per_node = torch.cuda.device_count() if torch.cuda.is_available() else 0
default_model_weight = 0.5

# =============================================================================
# DATA CONFIGURATION EXAMPLE
# =============================================================================

# data_name = "nemotron"
# data_name = "teigaku-genzei"  # 14187 samples
# data_name = "teigaku-genzei-ibm_generic_tmpl"  # 14187 samples
# data_name = "teigaku-genzei-v0.2"  # 18347 samples
# data_name = "teigaku-genzei-ibm-v0"
# data_name = "teigaku-genzei-ibm-v2"
# data_name = "teigaku-genzei-ibm-v3"
# data_name = "teigaku-genzei-ibm-v4-d5"
# data_name = "teigaku-genzei-ibm-v5_d5"
# data_name = "teigaku-genzei-ibm-v6_d5"
data_name = "teigaku-genzei-ibm-v6"  # 16751 samples
# data_name = "ibm-newsroom-d5"
# data_name = "ibm-newsroom-d5-x100"
# data_name = "ibm-newsroom-en_d5"  # 699 samples
# data_name = "jfe-technical-report_r5"  # 53030 samples

_data_name = f"_{data_name}" if data_name is not None and len(data_name) > 0 else ""

# =============================================================================
# COMPLETE SFT PARAMETER CONFIGURATION
# =============================================================================

# Experiment identification
experiment_name = "sft_gpt_oss_example"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
full_experiment_name = f"{experiment_name}{_data_name}_{timestamp}"

default_data_path = f"messages_data{_data_name}.jsonl"  # Path to training data in JSONL format
default_ckpt_output_dir = f"experiments/{full_experiment_name}"  # Where to save checkpoints

data_output_dir=f"data/{full_experiment_name}"  # Directory for processed data
# data_output_dir=f"/dev/shm/data/{full_experiment_name}"  # Directory for processed data (RAM disk for speed)

# Copied from examples/scripts/osft_continual_learning_example.py
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
    checkpoint_pattern = os.path.join(output_dir, "hf_format", "samples_*")
    checkpoint_dirs = glob.glob(checkpoint_pattern)
    
    if not checkpoint_dirs:
        raise ValueError(f"No checkpoints found in {os.path.join(output_dir, 'hf_format')}")
    
    # Find the most recently created checkpoint
    most_recent_checkpoint = max(checkpoint_dirs, key=os.path.getctime)
    
    return most_recent_checkpoint


def main():
    parser = argparse.ArgumentParser(description=f'SFT Training Example: {model_name}')
    
    # Optional overrides
    parser.add_argument('--data-path', default=default_data_path,
                       help=f'Path to training data (JSONL format) (default: {default_data_path})')
    parser.add_argument('--ckpt-output-dir', default=default_ckpt_output_dir,
                       help=f'Directory to save checkpoints (default: {default_ckpt_output_dir})')
    parser.add_argument('--model-path', default=default_model_path,
                       help=f'Model path or HuggingFace name (default: {default_model_path})')
    parser.add_argument('--num-epochs', type=int, default=default_num_epochs,
                       help=f'Number of training epochs (default: {default_num_epochs})')
    parser.add_argument('--nproc-per-node', type=int, default=default_nproc_per_node,
                       help=f'Number of GPUs (default: {default_nproc_per_node})')
    parser.add_argument('--max-tokens-per-gpu', type=int, default=example_max_tokens_per_gpu,
                       help=f'Max tokens per GPU (default: {example_max_tokens_per_gpu})')
    parser.add_argument('--batch-size', type=int, default=example_batch_size,
                       help=f'Effective batch size for training (default: {example_batch_size})')
    parser.add_argument('--learning-rate', type=float, default=example_learning_rate,
                       help=f'Learning rate for training (default: {example_learning_rate})')
    parser.add_argument('--max-seq-len', type=int, default=example_max_seq_len,
                       help=f'Max sequence length (default: {example_max_seq_len})')
    parser.add_argument('--model-weight', type=float, default=default_model_weight,
                       help=f'Weight for trained model for interpolation (0.0-1.0, default: {default_model_weight})')
    
    args = parser.parse_args()

    if args.nproc_per_node < example_min_nproc_per_node:
        print(f"💡 Try --nproc-per-node {example_min_nproc_per_node} or larger if you see OOM errors")
    
    # GPT-OSS 20B configuration
    print(f"🚀 SFT Training: {model_name}")
    print("=" * 50)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_path}")
    print(f"Output: {args.ckpt_output_dir}")
    print(f"GPUs: {args.nproc_per_node}")
    print(f"Max tokens per GPU: {args.max_tokens_per_gpu:,}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Max sequence length: {args.max_seq_len:,}")
    print(f"Model weight: {args.model_weight}")
    print()

    # Training configuration optimized for GPT-OSS 20B
    start_time = time.time()
    
    try:
        result = sft(
            # Model and data
            model_path=args.model_path,
            data_path=args.data_path,
            ckpt_output_dir=args.ckpt_output_dir,

            # Training parameters optimized for GPT-OSS 20B
            num_epochs=args.num_epochs,
            effective_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_seq_len=args.max_seq_len,
            max_tokens_per_gpu=args.max_tokens_per_gpu,

            # Data processing
            data_output_dir=data_output_dir,
            warmup_steps=100,
            save_samples=0,                    # 0 disables sample-based checkpointing, use epoch-based only

            # Checkpointing
            checkpoint_at_epoch=True,
            accelerate_full_state_at_epoch=False, # Disable for smaller checkpoints (no auto-resumption)

            # Single-node multi-GPU setup
            nproc_per_node=args.nproc_per_node,
            nnodes=1,
            # node_rank=0,
            # rdzv_id=104,
            # rdzv_endpoint="127.0.0.1:29500",
        )

        end_time = time.time()
        duration = end_time - start_time

        print("=" * 50)
        print("✅ Training completed successfully!")
        print(f"⏱️  Duration: {duration/3600:.2f} hours")
        print(f"📁 Checkpoints: {args.ckpt_output_dir}/hf_format/")

        most_recent_checkpoint = find_most_recent_checkpoint(args.ckpt_output_dir)
        print(f"   Most recent checkpoint: {most_recent_checkpoint}")

        trained_model_weight = args.model_weight
        if 0.0 < trained_model_weight and trained_model_weight < 1.0:
            from interpolator import interpolate_models

            interp_model_path = interpolate_models(args.model_path, most_recent_checkpoint, trained_model_weight=trained_model_weight)

            print("=" * 50)
            print("✅ Interpolation completed successfully!")
            print(f"   Interpolated model checkpoint: {interp_model_path}")

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time

        print("=" * 50)
        print(f"❌ Training failed after {duration/60:.1f} minutes")
        print(f"Error: {e}")
        print()
        print("💡 Troubleshooting tips:")
        print("   - Reduce --max-tokens-per-gpu if you see OOM errors")
        sys.exit(1)


if __name__ == "__main__":
    main()