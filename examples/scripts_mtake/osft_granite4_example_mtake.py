#!/usr/bin/env python3
"""
OSFT Training Example: Granite-4.0-H-Small

This script demonstrates OSFT (Orthogonal Subspace Fine-Tuning) training with Granite-4.0-H-Small model
using a single-node, multi-GPU setup with training_hub.

OSFT allows continual training without catastrophic forgetting, making it ideal for:
- Adapting instruction-tuned models to new domains
- Adding new knowledge without losing existing capabilities
- Fine-tuning without replay buffers or supplementary datasets

Example usage:
    python osft_granite4_example.py \\
        --data-path /path/to/data.jsonl \\
        --ckpt-output-dir /path/to/checkpoints
"""

import os
import sys
import time
from datetime import datetime
import argparse
import glob
import json
import torch

from training_hub import osft


# Detect GPUs
assert torch.cuda.is_available()
default_nproc_per_node = torch.cuda.device_count()

## Model Configuration Examples

# Derived from generic_7b_example in examples/notebooks/osft_comprehensive_tutorial.ipynb
granite4_example = {
    "model_name": "Granite-4.0-H-Small",
    "model_path": "ibm-granite/granite-4.0-h-small",  # HuggingFace model name or local path
    "example_unfreeze_rank_ratio": 0.3,  # Balanced preservation vs adaptation
    "example_max_tokens_per_gpu": 10000,
    "example_max_seq_len": 4096,
    "example_batch_size": 128,
    "example_learning_rate": 5e-6,
    "notes": "Good baseline for most 7B instruction-tuned models",
}

selected_example = granite4_example  # Change this to your preferred example

model_name = selected_example['model_name']
default_model_path = selected_example['model_path']
default_unfreeze_rank_ratio = selected_example["example_unfreeze_rank_ratio"]
default_max_tokens_per_gpu = selected_example['example_max_tokens_per_gpu']
default_max_seq_len = selected_example['example_max_seq_len']
default_batch_size = selected_example['example_batch_size']
default_learning_rate = selected_example['example_learning_rate']
default_num_epochs = 3

## Data Configuration Examples

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
# data_name = "jfe-technical-report_r5"

_data_name = f"_{data_name}" if data_name is not None and len(data_name) > 0 else ""

# =============================================================================
# COMPLETE OSFT PARAMETER CONFIGURATION
# =============================================================================

# Experiment identification
experiment_name = "osft_granite4_example"
default_model_basename = os.path.basename(default_model_path)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# @@@ahoaho XXX
# full_experiment_name = f"{experiment_name}_{timestamp}"
full_experiment_name = f"{experiment_name}_{default_model_basename}{_data_name}_{timestamp}"

default_data_path = f"messages_data{_data_name}.jsonl"  # Path to training data in JSONL format
default_ckpt_output_dir = f"experiments/{full_experiment_name}"  # Where to save checkpoints

# data_output_dir=f"data/{full_experiment_name}"  # Directory for processed data
data_output_dir=f"/dev/shm/data/{full_experiment_name}"  # Directory for processed data (RAM disk for speed)

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
    parser = argparse.ArgumentParser(description=f'OSFT Training Example: {model_name}')
    
    # Optional overrides
    parser.add_argument('--data-path', default=default_data_path,
                       help=f'Path to training data (JSONL format) (default: {default_data_path})')
    parser.add_argument('--ckpt-output-dir', default=default_ckpt_output_dir,
                       help=f'Directory to save checkpoints (default: {default_ckpt_output_dir})')
    parser.add_argument('--model-path', default=default_model_path,
                       help=f'Model path or HuggingFace name (default: {default_model_path})')
    parser.add_argument('--num-epochs', type=int, default=default_num_epochs,
                       help=f'Number of training epochs (default: {default_num_epochs})')
    parser.add_argument('--unfreeze-rank-ratio', type=float, default=default_unfreeze_rank_ratio,
                       help=f'Unfreeze rank ratio for OSFT (0.0-1.0, default: {default_unfreeze_rank_ratio})')
    parser.add_argument('--max-tokens-per-gpu', type=int, default=default_max_tokens_per_gpu,
                       help=f'Max tokens per GPU (default: {default_max_tokens_per_gpu})')
    parser.add_argument('--max-seq-len', type=int, default=default_max_seq_len,
                       help=f'Max sequence length (default: {default_max_seq_len})')
    parser.add_argument('--nproc-per-node', type=int, default=default_nproc_per_node,
                       help=f'Number of GPUs (default: {default_nproc_per_node})')
    parser.add_argument('--batch-size', type=int, default=default_batch_size,
                       help=f'Effective batch size for training (default: {default_batch_size})')
    parser.add_argument('--learning-rate', type=float, default=default_learning_rate,
                       help=f'Learning rate for training (default: {default_learning_rate})')
    parser.add_argument('--unmask-messages', action='store_true', default=False,
                       help='Unmask messages during training (default: False)')
    
    args = parser.parse_args()
    
    assert args.nproc_per_node <= default_nproc_per_node, f"NPROC_PER_NODE must be smaller than or equal to {default_nproc_per_node}"
    assert args.nproc_per_node >= 8, "NPROC_PER_NODE must be larger than or equal to 8"

    # Granite-4.0-H-Small OSFT configuration
    print(f"🚀 OSFT Training: {model_name}")
    print("=" * 50)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_path}")
    print(f"Output: {args.ckpt_output_dir}")
    print(f"Epochs: {args.num_epochs}")
    print(f"GPUs: {args.nproc_per_node}")
    print(f"Unfreeze Rank Ratio: {args.unfreeze_rank_ratio}")
    print(f"Max tokens per GPU: {args.max_tokens_per_gpu:,}")
    print(f"Max sequence length: {args.max_seq_len:,}")
    print()
    print("📝 Note: OSFT enables continual learning without replay buffers")
    print("    The model will adapt to new data while preserving existing capabilities")
    print()
    
    # Training configuration optimized for Granite-4.0-H-Small with OSFT
    start_time = time.time()
    
    try:
        result = osft(
            # Model and data
            model_path=args.model_path,
            data_path=args.data_path,
            ckpt_output_dir=args.ckpt_output_dir,
            
            # OSFT-specific parameters
            unfreeze_rank_ratio=args.unfreeze_rank_ratio,  # Controls preservation vs adaptation
            
            # Training parameters optimized for Granite-4.0-H-Small
            num_epochs=args.num_epochs,
            effective_batch_size=args.batch_size,            # Smaller batch for efficient model
            learning_rate=args.learning_rate,                # Very low LR for smaller but dense model
            max_seq_len=args.max_seq_len,
            max_tokens_per_gpu=args.max_tokens_per_gpu,
            
            # Data processing
            data_output_dir=data_output_dir,
            warmup_steps=0,
            unmask_messages=args.unmask_messages,
            
            # Optimization
            # @@@ahoaho XXX
            # use_liger=True,                     # Enable Liger kernels for efficiency
            use_liger=False,                     # Enable Liger kernels for efficiency. NOTE: liger-kernel 0.6.2 doesn't support granitemoehybrid
            seed=42,
            lr_scheduler='cosine',              # Cosine scheduler works well with OSFT
            
            # Checkpointing
            checkpoint_at_epoch=True,
            save_final_checkpoint=True,
            
            # Single-node multi-GPU setup
            nproc_per_node=args.nproc_per_node,
            nnodes=1,
            node_rank=0,
            rdzv_id=102,
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
        print("💡 Your model has been adapted to the new domain while preserving")
        print("   its original instruction-following capabilities!")
        
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

