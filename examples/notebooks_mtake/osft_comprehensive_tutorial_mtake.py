# %% [markdown]
# # Comprehensive OSFT Training Tutorial
# 
# This notebook provides a comprehensive guide to Orthogonal Subspace Fine-Tuning (OSFT) using the training_hub library. We'll cover:
# 
# - **All available parameters** and their detailed explanations
# - **Single-node and multi-node training** configurations
# - **Popular model examples** (Qwen 2.5 7B Instruct, Llama 3.1 8B Instruct, Phi 4 Mini, etc.)
# - **Best practices and troubleshooting**
# 
# OSFT (Orthogonal Subspace Fine-Tuning) is an algorithm based on [Nayak et al. (2025), arXiv:2504.07097](https://arxiv.org/abs/2504.07097) that enables continual training of pre-trained or instruction-tuned models **without** catastrophic forgetting and **without** needing replay buffers or supplementary datasets.
# 
# This tutorial serves as both a learning resource and a template you can adapt for your specific continual learning needs.
# 
# **Note:** For production workflows, we also provide focused example scripts for popular models: `scripts/osft_qwen_example.py`, `scripts/osft_llama_example.py`, and `scripts/osft_phi_example.py` with better logging consistency.
# 

# %% [markdown]
# ## What is OSFT?
# 
# OSFT (Orthogonal Subspace Fine-Tuning) is a continual learning algorithm that allows you to adapt pre-trained or instruction-tuned models to new domains **without catastrophic forgetting**. Based on [Nayak et al. (2025), arXiv:2504.07097](https://arxiv.org/abs/2504.07097), OSFT fundamentally changes how we approach model adaptation.
# 
# ### Key Innovation
# 
# Traditional fine-tuning updates all model parameters, which can overwrite previously learned knowledge. OSFT instead:
# 1. **Identifies orthogonal subspaces** in the model's weight matrices
# 2. **Restricts updates to these subspaces**, preserving existing knowledge
# 3. **Eliminates the need for replay buffers** or supplementary datasets
# 
# ### OSFT vs Traditional Fine-Tuning
# 
# | Aspect | Traditional SFT | OSFT |
# |--------|----------------|------|
# | **Catastrophic Forgetting** | Common problem | Prevented by design |
# | **Data Requirements** | Needs replay/mixed data | Only new domain data |
# | **Preservation Method** | Data mixing ratios | Algorithm (math guarantees) |
# | **Memory Usage** | Similar | Similar |
# | **Complexity** | Complex data pipelines | Simple, direct |
# 
# ### When to Use OSFT
# 
# **Perfect for:**
# - Adding domain-specific knowledge (medical, legal, technical)
# - Adapting to new languages or dialects
# - Customizing instruction formats
# - Continual learning across multiple domains
# - Any scenario where you need to preserve existing capabilities
# 
# **Not needed for:**
# - Training from scratch
# - Base model pre-training
# - When you want to completely replace model behavior
# 

# %% [markdown]
# ## Understanding the Key Parameter: `unfreeze_rank_ratio`
# 
# The `unfreeze_rank_ratio` is the most important OSFT-specific parameter. It controls the balance between preservation and adaptation.
# 
# ### What Does It Do?
# 
# - Controls **how much of each weight matrix** can be updated during training
# - Range: `0.0` to `1.0`
# - Lower values = more preservation, slower adaptation
# - Higher values = more adaptation, slightly less preservation
# 
# ### Visual Intuition
# 
# Think of a weight matrix as a building:
# - `unfreeze_rank_ratio = 0.1`: You can only renovate 10% of the rooms
# - `unfreeze_rank_ratio = 0.3`: You can renovate 30% of the rooms
# - `unfreeze_rank_ratio = 1.0`: You can renovate the entire building (standard fine-tuning)
# 
# The "rooms" you renovate are carefully chosen to be orthogonal to existing knowledge, preventing damage to what's already there.
# 
# ### Recommended Settings by Use Case
# 
# | Use Case | Recommended Ratio | Why? |
# |----------|-------------------|------|
# | **Minor format adjustments** | 0.1-0.15 | Minimal changes needed |
# | **Domain vocabulary addition** | 0.15-0.25 | Add terms without losing general knowledge |
# | **Domain specialization** | 0.25-0.35 | Balance preservation and new expertise |
# | **Major capability expansion** | 0.35-0.5 | Significant new learning required |
# | **Complete repurposing** | >0.5 | Rarely needed, approaching standard fine-tuning |
# 
# ### Practical Guidelines
# 
# ```python
# # Conservative: Maximum preservation
# unfreeze_rank_ratio = 0.2  # Great for adding specialized knowledge
# 
# # Balanced: Good for most use cases  
# unfreeze_rank_ratio = 0.3  # Ideal default for domain adaptation
# 
# # Aggressive: When you need significant changes
# unfreeze_rank_ratio = 0.4  # Use when preservation is less critical
# ```
# 
# **Pro tip:** Start conservative (0.2-0.3) and increase only if needed. It's easier to train again with higher ratio than to recover lost capabilities!
# 

# %% [markdown]
# ## The `target_patterns` Parameter (Advanced Users Only)
# 
# There's an optional `target_patterns` parameter that allows targeting specific model layers for OSFT:
# 
# ```python
# target_patterns = None  # Default: applies OSFT to all appropriate layers (RECOMMENDED)
# ```
# 
# **⚠️ Important:** This is an expert-level parameter. Unless you have deep knowledge of model architecture and a specific reason to limit OSFT to certain layers, **leave it as `None`**.
# 
# If you do need to use it, it performs simple substring matching on module names:
# - `target_patterns = ["attention"]` → Targets modules with "attention" in the name
# - `target_patterns = ["mlp"]` → Targets modules with "mlp" in the name
# 
# **For 99% of users:** Just use the default (`None`) and let OSFT handle layer selection automatically. The algorithm knows what it's doing.
# 

# %% [markdown]
# ## Setup and Imports
# 
# Let's start by importing the necessary libraries and setting up our environment.
# 

# %%
# Import training_hub for OSFT training
from training_hub import osft

# Standard library imports
import os
import time
from datetime import datetime
from pathlib import Path


# %% [markdown]
# ## Data Format Requirements
# 
# Before configuring your training, ensure your data is in the correct format. OSFT uses the mini-trainer backend, which supports both standard messages format and pre-processed datasets.
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
# ### Masking Control with `unmask_messages` Parameter
# 
# Control which parts of the conversation are used for training loss:
# 
# #### Standard Instruction Tuning (default)
# ```python
# osft(..., unmask_messages=False)  # Only assistant responses used for loss
# ```
# - **Trains only on assistant responses** (standard instruction-following)
# - System messages are always masked (ignored for loss)
# - User messages are masked
# - Assistant messages are unmasked (used for loss calculation)
# 
# #### Pretraining Mode
# ```python
# osft(..., unmask_messages=True)   # All content except system messages used for loss
# ```
# - **Trains on all content except system messages**
# - System messages are always masked
# - User and assistant messages are both unmasked
# - Useful for pretraining-style data where the model should learn from all text
# 
# ### Pre-processed Dataset Option
# 
# If you have pre-processed data with `input_ids` and `labels` fields:
# 
# ```json
# {"input_ids": [1, 2, 3, ...], "labels": [1, 2, 3, ...]}
# ```
# 
# Use with:
# ```python
# osft(..., use_processed_dataset=True)
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
# 3. Handle masking according to the `unmask_messages` setting
# 4. Process the data for efficient training
# 

# %% [markdown]
# ## Model Configuration Examples
# 
# Here are configuration examples for popular models. These serve as starting points - adjust based on your specific hardware and continual learning requirements.
# 

# %%
# =============================================================================
# MODEL CONFIGURATION EXAMPLES FOR OSFT
# These are example configurations - adjust based on your hardware and requirements
# =============================================================================

# Example 1: Qwen 2.5 7B Instruct
qwen_example = {
    "model_name": "Qwen 2.5 7B Instruct",
    "model_path": "Qwen/Qwen2.5-7B-Instruct",  # HuggingFace model name or local path
    "example_unfreeze_rank_ratio": 0.25,  # Conservative for preserving multilingual capabilities
    "example_max_tokens_per_gpu": 10000,
    "example_max_seq_len": 8196,  # Qwen 2.5 supports long context
    "example_batch_size": 128,
    "example_learning_rate": 5e-6, 
    "notes": "Excellent for domain adaptation while preserving multilingual capabilities"
}

# Example 2: Llama 3.1 8B Instruct
llama_example = {
    "model_name": "Llama 3.1 8B Instruct",
    "model_path": "meta-llama/Meta-Llama-3.1-8B-Instruct",  # HuggingFace model name or local path
    "example_unfreeze_rank_ratio": 0.3,  # Slightly higher for more adaptation freedom
    "example_max_tokens_per_gpu": 10000,
    "example_max_seq_len": 8192,  # Supports up to 128K but 8K is common
    "example_batch_size": 128,
    "example_learning_rate": 5e-6,
    "notes": "Ideal for adding specialized knowledge without losing general capabilities"
}

# Example 3: Phi 4 Mini
phi_example = {
    "model_name": "Phi 4 Mini",
    "model_path": "microsoft/Phi-4-mini-instruct",  # HuggingFace model name or local path
    "example_unfreeze_rank_ratio": 0.25,  # Conservative for smaller model
    "example_max_tokens_per_gpu": 8192,
    "example_max_seq_len": 4096,
    "example_batch_size": 64,
    "example_learning_rate": 5e-6,
    "notes": "Efficient for edge deployment with continual adaptation"
}

# Example 4: Generic 7B Base Model
generic_7b_example = {
    "model_name": "Generic 7B Base",
    "model_path": "/path/to/your-7b-model",  # Local path to model directory
    "example_unfreeze_rank_ratio": 0.3,  # Balanced preservation vs adaptation
    "example_max_tokens_per_gpu": 10000,
    "example_max_seq_len": 4096,
    "example_batch_size": 128,
    "example_learning_rate": 5e-6,
    "notes": "Good baseline for most 7B instruction-tuned models"
}

# Example 5: Smaller Model (1B-3B)
small_model_example = {
    "model_name": "Small Model (1B-3B)",
    "model_path": "/path/to/small-model",  # Local path or HuggingFace name
    "example_unfreeze_rank_ratio": 0.4,  # Higher ratio for smaller models
    "example_max_tokens_per_gpu": 16_000,
    "example_max_seq_len": 4096,
    "example_batch_size": 128,
    "example_learning_rate": 3e-5,
    "notes": "Smaller models can handle more aggressive adaptation"
}

# @@@ahoaho XXX
# Example 6: Granite 3.3 8B Instruct
# granite_example = {
#     "model_name": "Granite 3.3 8B Instruct",
#     "model_path": "ibm-granite/granite-3.3-8b-instruct",  # HuggingFace model name or local path
#     "example_unfreeze_rank_ratio": 0.3,  # Balanced preservation vs adaptation
#     # The following values are taken from https://github.com/instructlab/training/blob/bfd0d73b42e4b150543eda22b5497718122cd771/examples/01_building_a_reasoning_model.ipynb
#     "example_max_tokens_per_gpu": 30000,
#     "example_max_seq_len": 20000,
#     "example_batch_size": 256,
#     "example_learning_rate": 2e-5,
#     "notes": "Excellent for domain adaptation while preserving multilingual capabilities",
# }
# Derived from generic_7b_example
granite_example = {
    "model_name": "Granite 3.3 8B Instruct",
    "model_path": "ibm-granite/granite-3.3-8b-instruct",  # HuggingFace model name or local path
    "example_unfreeze_rank_ratio": 0.3,  # Balanced preservation vs adaptation
    "example_max_tokens_per_gpu": 10000,
    "example_max_seq_len": 4096,
    "example_batch_size": 128,
    "example_learning_rate": 5e-6,
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
print(f"OSFT Unfreeze Rank Ratio: {selected_example['example_unfreeze_rank_ratio']}")
print(f"Example Max Tokens per GPU: {selected_example['example_max_tokens_per_gpu']:,}")
print(f"Example Max Sequence Length: {selected_example['example_max_seq_len']:,}")
print(f"Example Batch Size: {selected_example['example_batch_size']:,}")
print(f"Example Learning Rate: {selected_example['example_learning_rate']}")
print(f"Notes: {selected_example['notes']}")
print("\n💡 Remember: OSFT preserves original capabilities without needing replay buffers!")
print("   Adjust unfreeze_rank_ratio based on preservation vs adaptation needs.")


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
# Let's configure all available OSFT parameters with detailed explanations.
# 

# %%
# =============================================================================
# COMPLETE OSFT PARAMETER CONFIGURATION
# =============================================================================

# Experiment identification
experiment_name = "osft_comprehensive_example"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# @@@ahoaho XXX
# full_experiment_name = f"{experiment_name}_{timestamp}"
full_experiment_name = f"{experiment_name}_{model_basename}{_data_name}_{timestamp}"

# =============================================================================
# REQUIRED PARAMETERS
# =============================================================================

# TODO: revert these overrides after we've concluded training
model_path = selected_example["model_path"]  # HuggingFace model name or local path
# @@@ahoaho XXX
# data_path = "/path/to/your/training_data.jsonl"  # Path to training data in JSONL format
# ckpt_output_dir = f"/path/to/checkpoints/{full_experiment_name}"  # Where to save checkpoints
data_path = f"messages_data{_data_name}.jsonl"  # Path to training data in JSONL format
ckpt_output_dir = f"experiments/{full_experiment_name}"  # Where to save checkpoints
unfreeze_rank_ratio = selected_example["example_unfreeze_rank_ratio"]  # OSFT-specific parameter
effective_batch_size = selected_example["example_batch_size"]  # Effective batch size for training
max_tokens_per_gpu = selected_example["example_max_tokens_per_gpu"]  # Maximum tokens per GPU (memory limit)
max_seq_len = selected_example["example_max_seq_len"]  # Maximum sequence length
learning_rate = selected_example["example_learning_rate"]  # Learning rate for training

print("📋 Required Parameters (all must be specified):")
print(f"  • model_path: {model_path}")
print(f"  • data_path: {data_path}")
print(f"  • ckpt_output_dir: {ckpt_output_dir}")
print(f"  • unfreeze_rank_ratio: {unfreeze_rank_ratio}")
print(f"  • effective_batch_size: {effective_batch_size}")
print(f"  • max_tokens_per_gpu: {max_tokens_per_gpu:,}")
print(f"  • max_seq_len: {max_seq_len:,}")
print(f"  • learning_rate: {learning_rate}")
print()

# =============================================================================
# OSFT-SPECIFIC PARAMETERS
# =============================================================================

target_patterns = None  # Optional: Patterns to match specific modules for OSFT
# Example: ["*attention*", "*mlp*"] to target attention and MLP layers

print("🔧 OSFT-Specific Parameters:")
print(f"  unfreeze_rank_ratio: {unfreeze_rank_ratio} - Controls how much of each matrix is unfrozen")
print(f"    • 0.1-0.3: Conservative, maximum preservation")
print(f"    • 0.3-0.5: Balanced adaptation")
print(f"    • >0.5: Rarely needed for typical use cases")
print(f"  target_patterns: {target_patterns} - Optional patterns for selecting specific modules")
print()

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================

num_epochs = 3  # Number of training epochs
# num_epochs = 1  # Number of training epochs
seed = 42  # Random seed for reproducibility
lr_scheduler = "cosine"  # Learning rate scheduler
lr_scheduler_kwargs = {}  # Scheduler parameters
warmup_steps = 0  # Number of warmup steps

print("🎯 Training Hyperparameters:")
print(f"  effective_batch_size: {effective_batch_size} - Effective batch size for training")
print(f"  learning_rate: {learning_rate} - Learning rate for model updates")
print(f"  num_epochs: {num_epochs} - Number of training epochs")
print(f"  lr_scheduler: '{lr_scheduler}' - Learning rate scheduler type")
print(f"  lr_scheduler_kwargs: {lr_scheduler_kwargs} - Scheduler parameters")
print(f"  warmup_steps: {warmup_steps} - Number of warmup steps")
print(f"  seed: {seed} - Random seed for reproducibility")
print()

# =============================================================================
# MEMORY AND PERFORMANCE PARAMETERS
# =============================================================================

use_liger = True  # Use Liger kernels for efficiency

print("⚡ Memory and Performance Parameters:")
print(f"  max_tokens_per_gpu: {max_tokens_per_gpu:,} - Maximum tokens per GPU (hard-cap for memory)")
print(f"  max_seq_len: {max_seq_len:,} - Maximum sequence length")
print(f"  use_liger: {use_liger} - Use Liger kernels for efficiency")
print()

# =============================================================================
# DATA PROCESSING PARAMETERS
# =============================================================================

# data_output_dir = f"data/{full_experiment_name}"  # Directory for processed data
data_output_dir = f"/dev/shm/data/{full_experiment_name}"  # Directory for processed data (RAM disk for speed)
use_processed_dataset = False  # Whether data is pre-processed
unmask_messages = False  # Whether to unmask all messages for pretraining-style learning

print("💾 Data Processing Parameters:")
print(f"  data_path: '{data_path}' - Path to training data (JSONL format)")
print(f"  data_output_dir: '{data_output_dir}' - Directory to save processed data")
print(f"  use_processed_dataset: {use_processed_dataset} - Whether to use pre-processed data")
print(f"  unmask_messages: {unmask_messages} - Whether to unmask all messages")
print()

# =============================================================================
# CHECKPOINTING PARAMETERS
# =============================================================================

checkpoint_at_epoch = True  # Whether to checkpoint at each epoch
save_final_checkpoint = True  # Whether to save final checkpoint

print("💾 Checkpointing Parameters:")
print(f"  ckpt_output_dir: '{ckpt_output_dir}' - Directory to save checkpoints")
print(f"  checkpoint_at_epoch: {checkpoint_at_epoch} - Whether to checkpoint at each epoch")
print(f"  save_final_checkpoint: {save_final_checkpoint} - Whether to save final checkpoint")
print()


# %% [markdown]
# ## Distributed Training Configuration
# 
# Configure distributed training for both single-node and multi-node setups.
# 

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
        "nnodes": 2,  # 2 nodes
        "node_rank": 0,
        "rdzv_id": 42,
        # master node IP
        "rdzv_endpoint": "10.241.128.23:1738",  # Replace with actual master IP
        "description": "Multi-node master (rank 0) - 4 nodes total"
    },
    "multi_node_worker": {
        "nproc_per_node": 8,
        "nnodes": 2,  # 2 nodes
        "node_rank": 1,  # Change this for each worker node (1, 2, 3, ...)
        "rdzv_id": 42,
        "rdzv_endpoint": "10.241.128.23:1738",  # Same as master
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

# OSFT-specific multi-node considerations
print("📝 OSFT Multi-Node Considerations:")
print("  • OSFT works seamlessly across multiple nodes")
print("  • No special replay buffer coordination needed (unlike SFT)")
print("  • Each node processes its data portion with the same unfreeze_rank_ratio")
print("  • Gradients are synchronized automatically across all nodes")
print()


# %% [markdown]
# ## Execute Training
# 
# Now let's run the actual OSFT training with all our configured parameters.
# 

# %%
# =============================================================================
# TRAINING EXECUTION
# =============================================================================

print("🚀 Starting OSFT Training")
print("=" * 60)
print(f"Experiment: {full_experiment_name}")
print(f"Model: {selected_example['model_name']}")
print(f"Total GPUs: {total_gpus} ({nproc_per_node} per node × {nnodes} nodes)")
print(f"Configuration: {dist_config['description']}")
print(f"Unfreeze Rank Ratio: {unfreeze_rank_ratio}")
print()
print("✨ OSFT Advantages:")
print("  • No catastrophic forgetting")
print("  • No replay buffer needed")
print("  • Preserves original model capabilities")
print()

# Prepare all training parameters
training_params = {
    # Required parameters
    'model_path': model_path,
    'data_path': data_path,
    'ckpt_output_dir': ckpt_output_dir,
    'unfreeze_rank_ratio': unfreeze_rank_ratio,
    'effective_batch_size': effective_batch_size,
    'max_tokens_per_gpu': max_tokens_per_gpu,
    'max_seq_len': max_seq_len,
    'learning_rate': learning_rate,
    
    # Optional OSFT-specific parameters
    'target_patterns': target_patterns,
    
    # Training duration
    'num_epochs': num_epochs,
    
    # Data processing parameters
    'data_output_dir': data_output_dir,
    'use_processed_dataset': use_processed_dataset,
    'unmask_messages': unmask_messages,
    'warmup_steps': warmup_steps,
    
    # Optimization parameters
    'use_liger': use_liger,
    'seed': seed,
    'lr_scheduler': lr_scheduler,
    'lr_scheduler_kwargs': lr_scheduler_kwargs,
    
    # Checkpointing parameters
    'checkpoint_at_epoch': checkpoint_at_epoch,
    'save_final_checkpoint': save_final_checkpoint,
    
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
    if value is not None:  # Only show non-None values
        print(f"  {key}: {value}")

print("\n" + "="*60)
print("⏳ Training starting...")
print("="*60)

# Execute training
start_time = time.time()

try:
    result = osft(**training_params)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "="*60)
    print("✅ OSFT Training completed successfully!")
    print(f"⏱️  Total duration: {duration/3600:.2f} hours ({duration/60:.1f} minutes)")
    print(f"📁 Checkpoints saved to: {ckpt_output_dir}")
    print("="*60)
    print()
    print("🎯 What you've achieved with OSFT:")
    print("  • Model adapted to new domain/task")
    print("  • Original capabilities preserved")
    print("  • No catastrophic forgetting occurred")
    print("  • Ready for deployment without regression testing!")
    
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
    print("  □ Try adjusting unfreeze_rank_ratio (lower = more preservation)")
    print("  □ For multi-node: verify network connectivity and endpoints")
    print("  □ Check that mini-trainer backend dependencies are installed")
    
    raise


# %% [markdown]
# ## Post-Training Analysis
# 
# After training completes, let's analyze the results and provide guidance for next steps.
# 

# %%
# =============================================================================
# POST-TRAINING ANALYSIS AND NEXT STEPS
# =============================================================================

print("📊 Post-Training Analysis")
print("=" * 50)

# Check for saved checkpoints
checkpoint_dir = ckpt_output_dir

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
        print(f"# Load your OSFT-adapted model")
        print(f"model = AutoModelForCausalLM.from_pretrained('{final_checkpoint_path}')")
        print(f"tokenizer = AutoTokenizer.from_pretrained('{final_checkpoint_path}')")
        print(f"")
        print(f"# Test the model - it should maintain original capabilities")
        print(f"# while excelling at your new domain/task")
        print(f"inputs = tokenizer('Your domain-specific prompt:', return_tensors='pt')")
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
print(f"  Algorithm: OSFT (Orthogonal Subspace Fine-Tuning)")
print(f"  Unfreeze Rank Ratio: {unfreeze_rank_ratio}")
print(f"  Epochs: {num_epochs}")
print(f"  Global Batch Size: {effective_batch_size}")
print(f"  Learning Rate: {learning_rate}")
print(f"  Max Tokens per GPU: {max_tokens_per_gpu:,}")
print(f"  Max Sequence Length: {max_seq_len:,}")
print(f"  Total GPUs: {total_gpus}")
print(f"  Distributed Config: {dist_config['description']}")

# OSFT-specific validation recommendations
print(f"\n🧪 OSFT-Specific Validation Steps:")
print(f"  1. **Test Original Capabilities**: Verify the model still performs well on")
print(f"     general tasks it was originally trained for")
print(f"  2. **Test New Domain**: Confirm improved performance on your target domain")
print(f"  3. **No Regression Testing Needed**: Unlike SFT, OSFT preserves capabilities")
print(f"     by design, reducing validation overhead")
print(f"  4. **Compare with Base Model**: Run side-by-side comparisons to see")
print(f"     improvements without degradation")

# Next steps recommendations
print(f"\n🚀 Recommended Next Steps:")
print(f"  1. 🎯 Test on domain-specific evaluation sets")
print(f"  2. 📊 Compare performance with base model on both general and domain tasks")
print(f"  3. 🔄 If more adaptation needed, slightly increase unfreeze_rank_ratio")
print(f"  4. 💡 If too much change occurred, reduce unfreeze_rank_ratio")
print(f"  5. 📝 Document the unfreeze_rank_ratio that works best for your use case")
print(f"  6. 🚢 Deploy with confidence - no catastrophic forgetting!")

# Performance optimization tips
print(f"\n⚡ OSFT-Specific Optimization Tips:")
print(f"  • Current unfreeze_rank_ratio ({unfreeze_rank_ratio}):")
if unfreeze_rank_ratio < 0.2:
    print(f"    Very conservative - great preservation, slower adaptation")
    print(f"    Consider increasing to 0.25-0.3 if need more adaptation")
elif unfreeze_rank_ratio < 0.35:
    print(f"    Balanced - good preservation with reasonable adaptation")
    print(f"    This is ideal for most use cases")
else:
    print(f"    Aggressive - faster adaptation, slightly less preservation")
    print(f"    Consider reducing if seeing any capability degradation")

print(f"  • Memory usage is similar to SFT - adjust max_tokens_per_gpu as needed")
print(f"  • For production: use the script version for better logging and resumption")

print(f"\n✨ OSFT Training Complete!")
print(f"Your model has been successfully adapted without forgetting!")


# %% [markdown]
# ## Parameter Reference Summary
# 
# Quick reference for all OSFT parameters and their purposes.
# 

# %% [markdown]
# ### Required Parameters
# 
# | Parameter | Description | Example Values |
# |-----------|-------------|----------------|
# | `model_path` | Path to the model to fine-tune | `"Qwen/Qwen2.5-7B"`, `"/path/to/model"` |
# | `data_path` | Path to the training data | `"/path/to/train.jsonl"` |
# | `ckpt_output_dir` | Directory to save checkpoints | `"/path/to/checkpoints"` |
# | `unfreeze_rank_ratio` | **OSFT-specific**: Controls preservation vs adaptation | `0.25`, `0.3`, `0.4` |
# | `effective_batch_size` | Effective batch size for training | `64`, `128`, `256` |
# | `max_tokens_per_gpu` | Maximum tokens per GPU (memory limit) | `16384`, `25000`, `40000` |
# | `max_seq_len` | Maximum sequence length | `2048`, `8192`, `32768` |
# | `learning_rate` | Learning rate for training | `1e-5`, `2e-5`, `5e-6` |
# 
# ### OSFT-Specific Parameters
# 
# | Parameter | Description | Recommended Values | Use Case |
# |-----------|-------------|-------------------|----------|
# | `unfreeze_rank_ratio` | Controls how much of each matrix is unfrozen | `0.1-0.3` | Conservative preservation |
# |           |             | `0.3-0.5` | Balanced adaptation |
# |           |             | `>0.5` | Rarely needed |
# | `target_patterns` | Optional patterns to match specific modules | `None` | Default (all modules) |
# 
# ### Training Configuration Parameters
# 
# | Parameter | Description | Default/Example |
# |-----------|-------------|-----------------|
# | `num_epochs` | Number of training epochs | `1` |
# | `seed` | Random seed for reproducibility | `42` |
# | `use_liger` | Enable Liger kernels for efficiency | `False` |
# | `warmup_steps` | Number of warmup steps | `0` |
# | `lr_scheduler` | Learning rate scheduler | `"cosine"` |
# | `lr_scheduler_kwargs` | Additional scheduler parameters | `{"eta_min": 1e-6}` |
# 
# ### Data Processing Parameters
# 
# | Parameter | Description | Default/Example |
# |-----------|-------------|-----------------|
# | `data_output_dir` | Directory to save processed data | Defaults to `f"{ckpt_output_dir}/_internal_data_processing"`, Recommended value is `"/dev/shm"` (shared memory) |
# | `use_processed_dataset` | Use pre-processed data with input_ids/labels | `False` |
# | `unmask_messages` | Unmask all messages for pretraining-style learning | `False` |
# 
# ### Checkpointing Parameters
# 
# | Parameter | Description | Recommended |
# |-----------|-------------|-------------|
# | `checkpoint_at_epoch` | Whether to checkpoint at each epoch | `True` |
# | `save_final_checkpoint` | Whether to save final checkpoint | `True` |
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
# ### Unfreeze Rank Ratio Guidelines
# 
# | Use Case | Recommended Ratio | Rationale |
# |----------|-------------------|-----------|
# | **Minor format changes** | 0.1-0.15 | Maximum preservation, minimal changes |
# | **Domain vocabulary addition** | 0.15-0.25 | Add specialized terms without losing general knowledge |
# | **Domain specialization** | 0.25-0.35 | Balance between preservation and adaptation |
# | **Major capability expansion** | 0.35-0.5 | More freedom for significant new capabilities |
# | **Complete repurposing** | >0.5 | Rarely needed, approaching standard fine-tuning |
# 
# ### OSFT vs SFT Key Differences
# 
# | Aspect | OSFT | SFT |
# |--------|------|-----|
# | **Catastrophic Forgetting** | Prevented by design | Requires replay buffers |
# | **Data Requirements** | Only new domain data | Needs mixed/replay data |
# | **Memory Usage** | Similar to SFT | Similar to OSFT |
# | **Key Parameter** | `unfreeze_rank_ratio` | N/A |
# | **Backend** | mini-trainer | instructlab-training |
# | **Best For** | Continual learning, domain adaptation | Initial fine-tuning |
# 
# ### Popular Model Examples for OSFT
# 
# | Model | HuggingFace Path | Recommended `unfreeze_rank_ratio` | `max_tokens_per_gpu` |
# |-------|------------------|-----------------------------------|----------------------|
# | Qwen 2.5 7B | `Qwen/Qwen2.5-7B-Instruct` | 0.25 | 10000 |
# | Llama 3.1 8B | `meta-llama/Meta-Llama-3.1-8B-Instruct` | 0.3 | 10000 |
# | Phi 4 Mini | `microsoft/Phi-4-mini-instruct` | 0.25 | 15000 |
# 
# ### Script Alternative
# 
# For production workloads or long-running training, use the script version:
# 
# ```bash
# # Qwen example
# python scripts/osft_qwen_example.py \
#   --data-path /path/to/data.jsonl \
#   --ckpt-output-dir /path/to/checkpoints \
#   --unfreeze-rank-ratio 0.25
# 
# # Llama example
# python scripts/osft_llama_example.py \
#   --data-path /path/to/data.jsonl \
#   --ckpt-output-dir /path/to/checkpoints \
#   --unfreeze-rank-ratio 0.3
# 
# # Phi example
# python scripts/osft_phi_example.py \
#   --data-path /path/to/data.jsonl \
#   --ckpt-output-dir /path/to/checkpoints \
#   --unfreeze-rank-ratio 0.25
# ```
# 
# ### When to Use OSFT vs SFT
# 
# **Use OSFT when:**
# - Adding domain-specific knowledge to an already-trained model
# - Need to preserve original capabilities without regression
# - Don't have access to original training data for replay
# - Want to avoid catastrophic forgetting
# - Performing continual learning across multiple domains
# 
# **Use SFT when:**
# - Training a model from scratch or base model
# - Have comprehensive training data covering all desired capabilities  
# - Don't need to preserve specific prior behaviors
# - Performing initial instruction tuning
# 


