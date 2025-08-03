# SFT Algorithm Usage Examples

This document shows how to use the SFT (Supervised Fine-Tuning) algorithm in training_hub.

## Simple Usage with Convenience Function

The easiest way to run SFT training is using the convenience function:

```python
from training_hub import sft

# Basic SFT training with default parameters
result = sft(
    model_path="/path/to/your/model",
    data_path="/path/to/your/training/data", 
    ckpt_output_dir="/path/to/save/checkpoints"
)

# SFT training with custom parameters
result = sft(
    model_path="/path/to/your/model",
    data_path="/path/to/your/training/data",
    ckpt_output_dir="/path/to/save/checkpoints",
    num_epochs=3,
    learning_rate=1e-5,
    effective_batch_size=2048,
    max_seq_len=2048
)

# Using a different backend (when available)
result = sft(
    model_path="/path/to/your/model",
    data_path="/path/to/your/training/data",
    ckpt_output_dir="/path/to/save/checkpoints",
    backend="instructlab-training"  # This is the default
)
```

## Using the Factory Pattern

For more control over the algorithm instance:

```python
from training_hub import create_algorithm

# Create an SFT algorithm instance
sft_algo = create_algorithm('sft', 'instructlab-training')

# Run training
result = sft_algo.train(
    model_path="/path/to/your/model",
    data_path="/path/to/your/training/data",
    ckpt_output_dir="/path/to/save/checkpoints",
    num_epochs=2,
    learning_rate=2e-6
)

# Check required parameters
required_params = sft_algo.get_required_params()
print("Required parameters:", list(required_params.keys()))
```

## Algorithm and Backend Discovery

Explore available algorithms and backends:

```python
from training_hub import AlgorithmRegistry

# List all available algorithms
algorithms = AlgorithmRegistry.list_algorithms()
print("Available algorithms:", algorithms)  # ['sft']

# List backends for SFT
sft_backends = AlgorithmRegistry.list_backends('sft')
print("SFT backends:", sft_backends)  # ['instructlab-training']

# Get algorithm class directly
SFTAlgorithm = AlgorithmRegistry.get_algorithm('sft')
```

## Parameter Reference

### Required Parameters

- `model_path` (str): Path to the model to fine-tune
- `data_path` (str): Path to the training data
- `ckpt_output_dir` (str): Directory to save checkpoints

### Optional Parameters

- `num_epochs` (int, default=1): Number of training epochs
- `effective_batch_size` (int, default=3840): Effective batch size for training
- `learning_rate` (float, default=2e-6): Learning rate
- `max_seq_len` (int, default=4096): Maximum sequence length
- `max_batch_len` (int, default=60000): Maximum batch length

### Backend Selection

- `backend` (str, default="instructlab-training"): Backend implementation to use

## Error Handling

```python
from training_hub import sft, AlgorithmRegistry

try:
    # This will work
    result = sft(
        model_path="/valid/model/path",
        data_path="/valid/data/path",
        ckpt_output_dir="/valid/output/path"
    )
except ValueError as e:
    print(f"Configuration error: {e}")
except Exception as e:
    print(f"Training error: {e}")

# Check if algorithm exists before using
if 'sft' in AlgorithmRegistry.list_algorithms():
    print("SFT algorithm is available")

# Check if backend exists
if 'instructlab-training' in AlgorithmRegistry.list_backends('sft'):
    print("InstructLab Training backend is available")
```

## Multi-Node Training

The SFT algorithm supports multi-node distributed training through torchrun parameters:

```python
from training_hub import sft

# Single-node, multi-GPU training (2 GPUs)
result = sft(
    model_path="/path/to/model",
    data_path="/path/to/data",
    ckpt_output_dir="/path/to/checkpoints",
    nproc_per_node=2,  # Number of GPUs per node
    nnodes=1,          # Single node
    node_rank=0,       # This node's rank
    rdzv_id=12345,     # Rendezvous ID
    rdzv_endpoint=""   # Empty for single node
)

# Multi-node training (2 nodes, 4 GPUs each)
# Run this on the first node (rank 0):
result = sft(
    model_path="/path/to/model",
    data_path="/path/to/data", 
    ckpt_output_dir="/path/to/checkpoints",
    nproc_per_node=4,           # 4 GPUs per node
    nnodes=2,                   # 2 total nodes
    node_rank=0,                # This is node 0
    rdzv_id=12345,              # Shared rendezvous ID
    rdzv_endpoint="node0:29500" # Master node endpoint
)

# Run this on the second node (rank 1):
result = sft(
    model_path="/path/to/model",
    data_path="/path/to/data",
    ckpt_output_dir="/path/to/checkpoints", 
    nproc_per_node=4,           # 4 GPUs per node
    nnodes=2,                   # 2 total nodes
    node_rank=1,                # This is node 1
    rdzv_id=12345,              # Same rendezvous ID
    rdzv_endpoint="node0:29500" # Same master endpoint
)
```

### Torchrun Parameters

- `nproc_per_node` (int): Number of processes (GPUs) per node
- `nnodes` (int): Total number of nodes in the cluster
- `node_rank` (int): Rank of this node (0 to nnodes-1)
- `rdzv_id` (int): Unique job ID for rendezvous
- `rdzv_endpoint` (str): Master node endpoint (format: "host:port")

If these parameters are not provided, single-node defaults will be used.

## Future Extensions

This architecture supports adding new algorithms and backends:

```python
# Future algorithms might include:
# - DPO (Direct Preference Optimization)
# - LoRA (Low-Rank Adaptation)
# - OSFT (Continual Learning via OSFT)

# Example of what future usage might look like:
# from training_hub import dpo, lora
# 
# dpo_result = dpo(model_path="...", data_path="...", ckpt_output_dir="...")
# lora_result = lora(model_path="...", data_path="...", rank=16)
```