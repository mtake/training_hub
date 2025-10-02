# %% [markdown]
# # Comprehensive SFT Training Tutorial
# 
# This notebook provides a comprehensive guide to Supervised Fine-Tuning (SFT) using the training_hub library. We'll cover:
# 
# - **All available parameters** and their detailed explanations
# - **Single-node and multi-node training** configurations
# - **Popular model examples** (Qwen 2.5 7B Instruct, Llama 3.1 8B Instruct, Phi 4 Mini, etc.)
# - **Best practices and troubleshooting**
# 
# This tutorial serves as both a learning resource and a template you can adapt for your specific fine-tuning needs.
# 
# **Note:** For production workflows, we also provide focused example scripts for popular models: `scripts/sft_qwen_example.py`, `scripts/sft_llama_example.py`, and `scripts/sft_phi_example.py` with better logging consistency.

# %% [markdown]
# ## Setup and Imports
# 
# Let's start by importing the necessary libraries and setting up our environment.

# %%
# Import training_hub for SFT training
from training_hub import sft

# Standard library imports
import os
import time
from datetime import datetime
from pathlib import Path

# %% [markdown]
# ## Data Format Requirements
# 
# Before configuring your training, ensure your data is in the correct format. Training Hub uses the instructlab-training backend, which expects data in a specific **messages format**.
# 
# ### Required Format: JSONL with Messages
# 
# Your training data must be a **JSON Lines (.jsonl)** file where each line contains a conversation sample:
# 
# ```json
# {"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello, how are you?"}, {"role": "assistant", "content": "I'm doing well, thank you! How can I help you today?"}]}
# {"messages": [{"role": "user", "content": "What is machine learning?"}, {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence..."}]}
# ```
# 
# ### Message Structure
# 
# Each conversation contains a `messages` array with message objects having:
# - **`role`**: One of `"system"`, `"user"`, `"assistant"`, or `"pretraining"`
# - **`content`**: The text content of the message
# - **`reasoning_content`** (optional): Additional reasoning traces
# 
# ### Masking Behavior with `unmask` Field
# 
# You can control which parts of the conversation are used for training loss by adding an `unmask` metadata field:
# 
# #### Standard Instruction Tuning (default)
# ```json
# {"messages": [...]}
# ```
# or
# ```json
# {"messages": [...], "unmask": false}
# ```
# - **Trains only on assistant responses** (standard instruction-following)
# - System messages are always masked (ignored for loss)
# - User messages are masked
# - Assistant messages are unmasked (used for loss calculation)
# 
# #### Pretraining Mode
# ```json
# {"messages": [...], "unmask": true}
# ```
# - **Trains on all content except system messages**
# - System messages are always masked
# - User and assistant messages are both unmasked
# - Useful for pretraining-style data where the model should learn from all text
# 
# ### Example Data Formats
# 
# **Standard SFT (instruction-following):**
# ```json
# {"messages": [{"role": "system", "content": "You are a coding assistant."}, {"role": "user", "content": "Write a Python function to calculate factorial"}, {"role": "assistant", "content": "Here's a Python function to calculate factorial:\n\n```python\ndef factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    return n * factorial(n - 1)\n```"}]}
# ```
# 
# **Pretraining-style (learn from all content):**
# ```json
# {"messages": [{"role": "user", "content": "The capital of France is"}, {"role": "assistant", "content": "Paris."}], "unmask": true}
# ```
# 
# ### Data Path Configuration
# 
# When configuring your training, point to your JSONL file:
# 
# ```python
# data_path = "/path/to/your/training_data.jsonl"  # Your messages-format JSONL file
# ```
# 
# The training pipeline will automatically:
# 1. Load and validate your JSONL data
# 2. Apply chat templates based on your model
# 3. Handle masking according to the `unmask` setting
# 4. Process the data for efficient training

# %% [markdown]
# ## Model Configuration Examples
# 
# Here are configuration examples for popular models. These serve as starting points - adjust based on your specific hardware and requirements.

# %%
# =============================================================================
# MODEL CONFIGURATION EXAMPLES
# These are example configurations - adjust based on your hardware and requirements
# =============================================================================

# Example 1: Qwen 2.5 7B Instruct
qwen_example = {
    "model_name": "Qwen 2.5 7B Instruct",
    "model_path": "Qwen/Qwen2.5-7B-Instruct",  # HuggingFace model name or local path
    "example_max_tokens_per_gpu": 20000,
    "example_max_seq_len": 16384,
    "example_batch_size": 128,
    "example_learning_rate": 1e-5,
    "notes": "Excellent for domain adaptation while preserving multilingual capabilities",
}

# Example 2: Llama 3.1 8B Instruct
llama_example = {
    "model_name": "Llama 3.1 8B Instruct",
    "model_path": "meta-llama/Meta-Llama-3.1-8B-Instruct",  # HuggingFace model name or local path
    "example_max_tokens_per_gpu": 18000,
    "example_max_seq_len": 16384,
    "example_batch_size": 128,
    "example_learning_rate": 1e-5,
    "notes": "Ideal for adding specialized knowledge without losing general capabilities",
}

# Example 3: Phi 4 Mini
phi_example = {
    "model_name": "Phi 4 Mini",
    "model_path": "microsoft/Phi-4-mini-instruct",  # HuggingFace model name or local path
    "example_max_tokens_per_gpu": 25000,
    "example_max_seq_len": 8192,
    "example_batch_size": 64,
    "example_learning_rate": 5e-6,
    "notes": "Efficient for edge deployment with continual adaptation",
}

# Example 4: Generic 7B Base Model
generic_7b_example = {
    "model_name": "Generic 7B Base",
    "model_path": "/path/to/your-7b-model",  # Local path to model directory
    "example_max_tokens_per_gpu": 25000,
    "example_max_seq_len": 20000,
    "example_batch_size": 256,
    "example_learning_rate": 2e-5,
    "notes": "Good baseline for most 7B instruction-tuned models",
}

# Example 5: Smaller Model (1B-3B)
small_model_example = {
    "model_name": "Small Model (1B-3B)",
    "model_path": "/path/to/small-model",  # Local path or HuggingFace name
    "example_max_tokens_per_gpu": 40000,
    "example_max_seq_len": 32768,
    "example_batch_size": 512,
    "example_learning_rate": 3e-5,
    "notes": "Smaller models can handle more aggressive adaptation",
}

# @@@ahoaho XXX
# Example 6: Granite 3.3 8B Instruct
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

# =============================================================================
# SELECT YOUR CONFIGURATION
# =============================================================================

# Choose one of the examples above as a starting point
# @@@ahoaho XXX
# selected_example = qwen_example  # Change this to your preferred example
selected_example = granite_example  # Change this to your preferred example

print(f"Selected Example: {selected_example['model_name']}")
print(f"Model Path: {selected_example['model_path']}")
print(f"Example Max Tokens per GPU: {selected_example['example_max_tokens_per_gpu']:,}")
print(f"Example Max Sequence Length: {selected_example['example_max_seq_len']:,}")
print(f"Example Batch Size: {selected_example['example_batch_size']:,}")
print(f"Example Learning Rate: {selected_example['example_learning_rate']}")
print(f"Notes: {selected_example['notes']}")
print("\n💡 Remember: These are example configurations. Adjust based on your hardware and requirements.")

# %% [markdown]
# ## Data Configuration Examples

# %%
model_path = selected_example["model_path"]  # HuggingFace model name or local path

model_basename = os.path.basename(model_path)

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

# %% [markdown]
# ## Complete Parameter Reference
# 
# Let's configure all available SFT parameters with detailed explanations.

# %%
# =============================================================================
# COMPLETE SFT PARAMETER CONFIGURATION
# =============================================================================

# Experiment identification
experiment_name = "sft_comprehensive_example"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# @@@ahoaho XXX
# full_experiment_name = f"{experiment_name}_{timestamp}"
full_experiment_name = f"{experiment_name}_{model_basename}{_data_name}_{timestamp}"

# =============================================================================
# REQUIRED PARAMETERS
# =============================================================================

model_path = selected_example["model_path"]  # HuggingFace model name or local path
# @@@ahoaho XXX
# data_path = "/path/to/your/training_data.jsonl"  # Path to training data in JSONL format
# ckpt_output_dir = f"/path/to/checkpoints/{full_experiment_name}"  # Where to save checkpoints
data_path = f"messages_data{_data_name}.jsonl"  # Path to training data in JSONL format
ckpt_output_dir = f"experiments/{full_experiment_name}"  # Where to save checkpoints

print("📋 Required Parameters:")
print(f"  model_path: Path to the model to fine-tune (HuggingFace name or local path)")
print(f"  data_path: Path to the training data (JSONL format)")
print(f"  ckpt_output_dir: Directory to save checkpoints")
print()

# =============================================================================
# CORE TRAINING PARAMETERS
# =============================================================================

num_epochs = 3  # Number of training epochs
effective_batch_size = selected_example["example_batch_size"]  # Effective batch size for training
learning_rate = selected_example["example_learning_rate"]  # Learning rate for training
max_seq_len = selected_example["example_max_seq_len"]  # Maximum sequence length
max_tokens_per_gpu = selected_example["example_max_tokens_per_gpu"]  # Maximum tokens per GPU in a mini-batch (hard-cap for memory to avoid OOMs)

print("🎯 Core Training Parameters:")
print(f"  num_epochs: {num_epochs} - Number of training epochs")
print(f"  effective_batch_size: {effective_batch_size} - Effective batch size for training")
print(f"  learning_rate: {learning_rate} - Learning rate for training")
print(f"  max_seq_len: {max_seq_len:,} - Maximum sequence length")
print(f"  max_tokens_per_gpu: {max_tokens_per_gpu:,} - Maximum tokens per GPU in a mini-batch (hard-cap for memory to avoid OOMs). Used to automatically calculate mini-batch size and gradient accumulation to maintain the desired effective_batch_size while staying within memory limits.")
print()

# =============================================================================
# DATA AND PROCESSING PARAMETERS
# =============================================================================

# @@@ahoaho XXX
# data_output_dir = f"data/{full_experiment_name}"  # Directory for processed data
data_output_dir = f"/dev/shm/data/{full_experiment_name}"  # Directory for processed data (RAM disk for speed)
warmup_steps = 100  # Number of warmup steps

print("💾 Data Processing Parameters:")
print(f"  data_output_dir: '{data_output_dir}' - Directory to save processed data")
print(f"  warmup_steps: {warmup_steps} - Number of warmup steps")
print()

# =============================================================================
# CHECKPOINTING PARAMETERS
# =============================================================================

save_samples = 0  # Number of samples to save after training (0 disables saving based on sample count)
checkpoint_at_epoch = True  # Whether to checkpoint at each epoch
accelerate_full_state_at_epoch = True  # Whether to save full state at epoch for automatic checkpoint resumption

print("💾 Checkpointing Parameters:")
print(f"  save_samples: {save_samples} - Number of samples to save after training (0 disables saving based on sample count)")
print(f"  checkpoint_at_epoch: {checkpoint_at_epoch} - Whether to checkpoint at each epoch")
print(f"  accelerate_full_state_at_epoch: {accelerate_full_state_at_epoch} - Whether to save full state at epoch for automatic checkpoint resumption")
print()

# %% [markdown]
# ## Distributed Training Configuration
# 
# Configure distributed training for both single-node and multi-node setups.

# %%
import torch

# @@@ahoaho XXX
# Detect GPUs
assert torch.cuda.is_available()
nproc_per_node = torch.cuda.device_count()
print(f"Number of GPUs detected on this node: {nproc_per_node}")

# %%
# =============================================================================
# DISTRIBUTED TRAINING PARAMETERS
# =============================================================================

# Configuration options for different setups
distributed_configs = {
    "single_gpu_dev": {
        "nproc_per_node": 1,
        "nnodes": 1,
        "node_rank": 0,
        "rdzv_id": 1,
        "rdzv_endpoint": "127.0.0.1:29500",
        "description": "Development setup with single GPU"
    },
    "single_node_8gpu": {
        "nproc_per_node": 8,
        "nnodes": 1,
        "node_rank": 0,
        "rdzv_id": 100,
        "rdzv_endpoint": "127.0.0.1:29500",
        "description": "Single node with 8 GPUs"
    },
    # @@@ahoaho XX
    "single_node_all_gpu": {
        "nproc_per_node": nproc_per_node,
        "nnodes": 1,
        "node_rank": 0,
        "rdzv_id": 100,
        "rdzv_endpoint": "127.0.0.1:29500",
        "description": "Single node with all GPUs"
    },
    "multi_node_master": {
        "nproc_per_node": 8,
        "nnodes": 4,
        "node_rank": 0,
        "rdzv_id": 42,
        "rdzv_endpoint": "10.0.0.1:29500",  # Replace with actual master IP
        "description": "Multi-node master (rank 0) - 4 nodes total"
    },
    "multi_node_worker": {
        "nproc_per_node": 8,
        "nnodes": 4,
        "node_rank": 1,  # Change this for each worker node (1, 2, 3, ...)
        "rdzv_id": 42,
        "rdzv_endpoint": "10.0.0.1:29500",  # Same as master
        "description": "Multi-node worker (rank 1) - change rank for each worker"
    }
}

# Select your distributed configuration
# @@@ahoaho XXX
# selected_distributed = "single_node_8gpu"  # Change this to match your setup
selected_distributed = "single_node_all_gpu"  # Change this to match your setup
dist_config = distributed_configs[selected_distributed]

# Extract distributed training parameters
nproc_per_node = dist_config["nproc_per_node"]  # Number of processes (GPUs) per node
nnodes = dist_config["nnodes"]  # Total number of nodes
node_rank = dist_config["node_rank"]  # Rank of this node (0 to nnodes-1)
rdzv_id = dist_config["rdzv_id"]  # Unique job ID for rendezvous
rdzv_endpoint = dist_config["rdzv_endpoint"]  # Master node endpoint for multi-node training

# Calculate total resources
total_gpus = nproc_per_node * nnodes
per_gpu_batch_size = effective_batch_size // total_gpus

print("🖥️  Distributed Training Parameters:")
print(f"  Configuration: {dist_config['description']}")
print(f"  nproc_per_node: {nproc_per_node} - Number of processes (GPUs) per node")
print(f"  nnodes: {nnodes} - Total number of nodes")
print(f"  node_rank: {node_rank} - Rank of this node (0 to nnodes-1)")
print(f"  rdzv_id: {rdzv_id} - Unique job ID for rendezvous")
print(f"  rdzv_endpoint: '{rdzv_endpoint}' - Master node endpoint for multi-node training")
print()
print(f"📊 Resource Calculation:")
print(f"  Total GPUs: {total_gpus} ({nproc_per_node} × {nnodes})")
print(f"  Effective batch size: {effective_batch_size}")
print(f"  Approximate per-GPU batch size: {per_gpu_batch_size}")
print(f"  (Actual micro-batch size determined automatically by gradient accumulation)")
print()

# Multi-node setup instructions
if nnodes > 1:
    print("🔧 Multi-Node Setup Instructions:")
    print(f"  1. Ensure all nodes can reach the master at {rdzv_endpoint}")
    print(f"  2. Use the same rdzv_id ({rdzv_id}) on all nodes")
    print(f"  3. Set node_rank to 0 for master, 1,2,3... for workers")
    print(f"  4. Start training on ALL nodes simultaneously")
    print()

# %% [markdown]
# ## Execute Training
# 
# Now let's run the actual SFT training with all our configured parameters.

# %%
# =============================================================================
# TRAINING EXECUTION
# =============================================================================

print("🚀 Starting SFT Training")
print("=" * 60)
print(f"Experiment: {full_experiment_name}")
print(f"Model: {selected_example['model_name']}")
print(f"Total GPUs: {total_gpus} ({nproc_per_node} per node × {nnodes} nodes)")
print(f"Configuration: {dist_config['description']}")
print()

# Prepare all training parameters
training_params = {
    # Required parameters
    'model_path': model_path,
    'data_path': data_path,
    'ckpt_output_dir': ckpt_output_dir,
    
    # Core training parameters
    'num_epochs': num_epochs,
    'effective_batch_size': effective_batch_size,
    'learning_rate': learning_rate,
    'max_seq_len': max_seq_len,
    'max_tokens_per_gpu': max_tokens_per_gpu,
    
    # Data and processing parameters
    'data_output_dir': data_output_dir,
    'warmup_steps': warmup_steps,
    'save_samples': save_samples,
    
    # Checkpointing parameters
    'checkpoint_at_epoch': checkpoint_at_epoch,
    'accelerate_full_state_at_epoch': accelerate_full_state_at_epoch,
    
    # Distributed training parameters
    'nproc_per_node': nproc_per_node,
    'nnodes': nnodes,
    'node_rank': node_rank,
    'rdzv_id': rdzv_id,
    'rdzv_endpoint': rdzv_endpoint,
}

# Display final configuration summary
print("📋 Final Training Configuration:")
for key, value in training_params.items():
    print(f"  {key}: {value}")

print("\n" + "="*60)
print("⏳ Training starting...")
print("="*60)

# Execute training
start_time = time.time()

try:
    result = sft(**training_params)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "="*60)
    print("✅ Training completed successfully!")
    print(f"⏱️  Total duration: {duration/3600:.2f} hours ({duration/60:.1f} minutes)")
    print(f"📁 Checkpoints saved to: {ckpt_output_dir}")
    print("="*60)
    
except Exception as e:
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "="*60)
    print(f"❌ Training failed after {duration/60:.1f} minutes")
    print(f"Error: {e}")
    print("="*60)
    
    print("\n🔍 Quick Troubleshooting Checklist:")
    print("  □ Check that model_path exists or is a valid HuggingFace model name")
    print("  □ Verify data_path points to valid JSONL file")
    print("  □ Ensure ckpt_output_dir parent directory exists and is writable")
    print("  □ Try reducing max_tokens_per_gpu if you see OOM errors")
    print("  □ For multi-node: verify network connectivity and endpoints")
    print("  □ Check that all file paths are accessible from the training process")
    
    raise

# %% [markdown]
# ## Post-Training Analysis
# 
# After training completes, let's analyze the results and provide guidance for next steps.

# %%
# =============================================================================
# POST-TRAINING ANALYSIS AND NEXT STEPS
# =============================================================================

print("📊 Post-Training Analysis")
print("=" * 50)

# Check for saved checkpoints
checkpoint_dir = f"{ckpt_output_dir}/hf_format"

if os.path.exists(checkpoint_dir):
    checkpoints = [d for d in os.listdir(checkpoint_dir) 
                  if os.path.isdir(os.path.join(checkpoint_dir, d))]
    
    if checkpoints:
        print(f"✅ Found {len(checkpoints)} checkpoint(s):")
        for ckpt in sorted(checkpoints):
            ckpt_path = os.path.join(checkpoint_dir, ckpt)
            print(f"  📁 {ckpt}")
        
        # Identify the final checkpoint
        final_checkpoint = sorted(checkpoints)[-1]
        final_checkpoint_path = os.path.join(checkpoint_dir, final_checkpoint)
        
        print(f"\n🎯 Final model checkpoint: {final_checkpoint_path}")
        
        # Provide model loading example
        print(f"\n💻 Model Loading Example:")
        print(f"```python")
        print(f"from transformers import AutoModelForCausalLM, AutoTokenizer")
        print(f"")
        print(f"# Load your fine-tuned model")
        print(f"model = AutoModelForCausalLM.from_pretrained('{final_checkpoint_path}')")
        print(f"tokenizer = AutoTokenizer.from_pretrained('{final_checkpoint_path}')")
        print(f"")
        print(f"# Generate text")
        print(f"inputs = tokenizer('Your prompt here:', return_tensors='pt')")
        print(f"outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True)")
        print(f"response = tokenizer.decode(outputs[0], skip_special_tokens=True)")
        print(f"print(response)")
        print(f"```")
    else:
        print(f"❌ No checkpoints found in {checkpoint_dir}")
else:
    print(f"❌ Checkpoint directory not found: {checkpoint_dir}")

# Training summary
print(f"\n📈 Training Summary:")
print(f"  Model: {selected_example['model_name']}")
print(f"  Epochs: {num_epochs}")
print(f"  Global Batch Size: {effective_batch_size}")
print(f"  Learning Rate: {learning_rate}")
print(f"  Max Tokens per GPU: {max_tokens_per_gpu:,}")
print(f"  Max Sequence Length: {max_seq_len:,}")
print(f"  Total GPUs: {total_gpus}")
print(f"  Distributed Config: {dist_config['description']}")

# Next steps recommendations
print(f"\n🚀 Recommended Next Steps:")
print(f"  1. 🧪 Test your model with sample inputs to verify it's working")
print(f"  2. 📊 Evaluate performance on your validation/test datasets")
print(f"  3. 🔄 Compare outputs with the original base model")
print(f"  4. 🎯 Fine-tune hyperparameters if needed (learning rate, batch size)")
print(f"  5. 📝 Document your configuration and results for reproducibility")
print(f"  6. 🚢 Deploy for inference using your preferred serving framework")

# Performance optimization tips
print(f"\n⚡ Performance Optimization Tips:")
print(f"  • If training was slow: increase max_tokens_per_gpu or effective_batch_size")
print(f"  • If you hit OOM errors: reduce max_tokens_per_gpu or effective_batch_size")
print(f"  • For better convergence: try different learning rates or warmup_steps")
print(f"  • For production training: consider using the script version for better logging")

print(f"\n✨ SFT Training Complete!")

# %% [markdown]
# ## Parameter Reference Summary
# 
# Quick reference for all SFT parameters and their purposes.

# %% [markdown]
# ### Core Parameters
# 
# | Parameter | Required | Description | Example Values |
# |-----------|----------|-------------|----------------|
# | `model_path` | ✅ | Path to the model to fine-tune | `"Qwen/Qwen2.5-7B"`, `"/path/to/model"` |
# | `data_path` | ✅ | Path to the training data | `"/path/to/train.jsonl"` |
# | `ckpt_output_dir` | ✅ | Directory to save checkpoints | `"/path/to/checkpoints"` |
# | `num_epochs` | ❌ | Number of training epochs | `1`, `3`, `5` |
# | `effective_batch_size` | ❌ | Effective batch size for training | `64`, `128`, `256` |
# | `learning_rate` | ❌ | Learning rate for training | `1e-5`, `2e-5`, `5e-6` |
# | `max_seq_len` | ❌ | Maximum sequence length | `2048`, `8192`, `16384` |
# | `max_tokens_per_gpu` | ❌ | Maximum tokens per GPU in a mini-batch (hard-cap for memory) | `15000`, `25000`, `40000` |
# 
# ### Data Processing Parameters
# 
# | Parameter | Description | Default/Example |
# |-----------|-------------|------------------|
# | `data_output_dir` | Directory to save processed data | `"/dev/shm"` (RAM disk) |
# | `warmup_steps` | Number of warmup steps | `100`, `500` |
# 
# ### Checkpointing Parameters
# 
# | Parameter | Description | Recommended |
# |-----------|-------------|-------------|
# | `checkpoint_at_epoch` | Whether to checkpoint at each epoch | `True` |
# | `accelerate_full_state_at_epoch` | Whether to save full state at epoch for automatic checkpoint resumption | `True` |
# | `save_samples` | Number of samples to save after training (0 disables) | `1000`, `0` (disabled) |
# 
# ### Distributed Training Parameters
# 
# | Parameter | Description | Example Values |
# |-----------|-------------|----------------|
# | `nproc_per_node` | Number of processes (GPUs) per node | `1`, `4`, `8` |
# | `nnodes` | Total number of nodes | `1`, `2`, `4` |
# | `node_rank` | Rank of this node (0 to nnodes-1) | `0` (master), `1`, `2`... |
# | `rdzv_id` | Unique job ID for rendezvous | `42`, `100` |
# | `rdzv_endpoint` | Master node endpoint for multi-node training | `"127.0.0.1:29500"` |
# 
# ### Memory Optimization Guidelines
# 
# - **Start conservative**: Begin with lower `max_tokens_per_gpu` values and increase gradually
# - **Monitor usage**: Watch GPU memory during training and adjust accordingly
# - **Balance batch size**: Larger `effective_batch_size` can improve training stability
# - **Use RAM disk**: Set `data_output_dir="/dev/shm"` for faster data loading
# 
# ### Multi-Node Setup Checklist
# 
# 1. ✅ Ensure network connectivity between all nodes
# 2. ✅ Use the same `rdzv_id` and `rdzv_endpoint` on all nodes
# 3. ✅ Set unique `node_rank` for each node (0, 1, 2, ...)
# 4. ✅ Verify all nodes can access model and data paths
# 5. ✅ Start training simultaneously on all nodes
# 
# ### Popular Model Examples
# 
# | Model | HuggingFace Path | Example Config |
# |-------|------------------|----------------|
# | Qwen 2.5 7B | `Qwen/Qwen2.5-7B-Instruct` | `max_tokens_per_gpu=20000` |
# | Llama 3.1 8B | `meta-llama/Meta-Llama-3.1-8B-Instruct` | `max_tokens_per_gpu=18000` |
# | Phi 4 Mini | `microsoft/Phi-4-mini-instruct` | `max_tokens_per_gpu=25000` |
# 
# ### Script Alternative
# 
# For production workloads or long-running training, use the script version:
# 
# ```bash
# python scripts/sft_qwen_example.py \
#   --data-path /path/to/data.jsonl \
#   --ckpt-output-dir /path/to/checkpoints
# 
# python scripts/sft_llama_example.py \
#   --data-path /path/to/data.jsonl \
#   --ckpt-output-dir /path/to/checkpoints
# 
# python scripts/sft_phi_example.py \
#   --data-path /path/to/data.jsonl \
#   --ckpt-output-dir /path/to/checkpoints
# ```


