#!/usr/bin/env python3
"""
SFT Training Example: Granite 3.3 8B Instruct

This script demonstrates SFT training with Granite 3.3 8B Instruct model
using a single-node, multi-GPU setup with training_hub.

Example usage:
    python sft_granite_example.py \\
        --data-path /path/to/data.jsonl \\
        --ckpt-output-dir /path/to/checkpoints
"""

import os
import sys
import time
from datetime import datetime
import argparse
import torch

from training_hub import sft


# Detect GPUs
assert torch.cuda.is_available()
default_nproc_per_node = torch.cuda.device_count()

## Model Configuration Examples

# @@@ahoaho XXX
# granite_example = {
#     "model_name": "Granite 3.3 8B Instruct",
#     "model_path": "ibm-granite/granite-3.3-8b-instruct",  # HuggingFace model name or local path
#     # The following values are taken from https://github.com/instructlab/training/blob/bfd0d73b42e4b150543eda22b5497718122cd771/examples/01_building_a_reasoning_model.ipynb
#     "example_max_tokens_per_gpu": 30000,
#     "example_max_seq_len": 20000,
#     "example_batch_size": 256,
#     "example_learning_rate": 2e-5,
#     "notes": "Excellent for domain adaptation while preserving multilingual capabilities",
# }
granite_example = {
    "model_name": "Granite 3.3 8B Instruct",
    "model_path": "ibm-granite/granite-3.3-8b-instruct",  # HuggingFace model name or local path
    "example_max_tokens_per_gpu": 25000,
    "example_max_seq_len": 20000,
    "example_batch_size": 256,
    "example_learning_rate": 2e-5,
    "notes": "Good baseline for most 7B instruction-tuned models",
}

selected_example = granite_example  # Change this to your preferred example

default_model_name = selected_example['model_name']
default_model_path = selected_example['model_path']
default_max_tokens_per_gpu = selected_example['example_max_tokens_per_gpu']
default_max_seq_len = selected_example['example_max_seq_len']
default_batch_size = selected_example['example_batch_size']
default_learning_rate = selected_example['example_learning_rate']

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
# COMPLETE SFT PARAMETER CONFIGURATION
# =============================================================================

# Experiment identification
experiment_name = "sft_comprehensive_example"
default_model_basename = os.path.basename(default_model_path)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# @@@ahoaho XXX
# full_experiment_name = f"{experiment_name}_{timestamp}"
full_experiment_name = f"{experiment_name}_{default_model_basename}{_data_name}_{timestamp}"

default_data_path = f"messages_data{_data_name}.jsonl"  # Path to training data in JSONL format
default_ckpt_output_dir = f"experiments/{full_experiment_name}"  # Where to save checkpoints

# @@@ahoaho XXX
# data_output_dir=f"data/{full_experiment_name}"  # Directory for processed data
data_output_dir=f"/dev/shm/data/{full_experiment_name}"  # Directory for processed data (RAM disk for speed)

def main():
    parser = argparse.ArgumentParser(description=f'SFT Training Example: {default_model_name}')
    
    # Optional overrides
    parser.add_argument('--data-path', default=default_data_path,
                       help=f'Path to training data (JSONL format) (default: {default_data_path})')
    parser.add_argument('--ckpt-output-dir', default=default_ckpt_output_dir,
                       help=f'Directory to save checkpoints (default: {default_ckpt_output_dir})')
    parser.add_argument('--model-path', default=default_model_path,
                       help=f'Model path or HuggingFace name (default: {default_model_path})')
    parser.add_argument('--num-epochs', type=int, default=3,
                       help='Number of epochs (default: 3)')
    parser.add_argument('--max-tokens-per-gpu', type=int, default=default_max_tokens_per_gpu,
                       help=f'Max tokens per GPU (default: {default_max_tokens_per_gpu})')
    parser.add_argument('--nproc-per-node', type=int, default=default_nproc_per_node,
                       help=f'Number of GPUs (default: {default_nproc_per_node})')
    
    args = parser.parse_args()
    
    assert args.nproc_per_node <= default_nproc_per_node, f"NPROC_PER_NODE must be smaller than or equal to {default_nproc_per_node}"

    # Granite 3.3 8B Instruct configuration
    print(f"🚀 SFT Training: {default_model_name}")
    print("=" * 50)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_path}")
    print(f"Output: {args.ckpt_output_dir}")
    print(f"GPUs: {args.nproc_per_node}")
    print(f"Max tokens per GPU: {args.max_tokens_per_gpu:,}")
    print()

    # Training configuration optimized for Granite 3.3 8B Instruct
    start_time = time.time()
    
    try:
        result = sft(
            # Model and data
            model_path=args.model_path,
            data_path=args.data_path,
            ckpt_output_dir=args.ckpt_output_dir,
            
            # Training parameters optimized for Granite 3.3 8B Instruct
            num_epochs=args.num_epochs,
            effective_batch_size=default_batch_size,
            learning_rate=default_learning_rate,
            max_seq_len=default_max_seq_len,
            max_tokens_per_gpu=args.max_tokens_per_gpu,
            
            # Data processing
            data_output_dir=data_output_dir,
            warmup_steps=100,
            save_samples=0,                    # 0 disables sample-based checkpointing, use epoch-based only
            
            # Checkpointing
            checkpoint_at_epoch=True,
            accelerate_full_state_at_epoch=True,  # Enable for auto-resumption (larger checkpoints)
            
            # Single-node multi-GPU setup
            nproc_per_node=args.nproc_per_node,
            nnodes=1,
            node_rank=0,
            rdzv_id=102,
            rdzv_endpoint="127.0.0.1:29500",
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print("=" * 50)
        print("✅ Training completed successfully!")
        print(f"⏱️  Duration: {duration/3600:.2f} hours")
        print(f"📁 Checkpoints: {args.ckpt_output_dir}/hf_format/")
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        print("=" * 50)
        print(f"❌ Training failed after {duration/60:.1f} minutes")
        print(f"Error: {e}")
        print()
        print("💡 Try reducing --max-tokens-per-gpu if you see OOM errors")
        sys.exit(1)


if __name__ == "__main__":
    main()