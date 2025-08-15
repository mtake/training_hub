# Training Hub Examples

This directory contains documentation, tutorials, and examples for using training_hub algorithms.

## Directory Structure

- **`docs/`** - Usage documentation and guides for supported algorithms
- **`notebooks/`** - Interactive Jupyter notebooks with step-by-step tutorials
- **`scripts/`** - Standalone Python scripts for automation and examples

## Supported Algorithms

### Supervised Fine-Tuning (SFT)

The SFT algorithm supports training language models on supervised datasets with both single-node and multi-node distributed training capabilities.

**Documentation:**
- [SFT Usage Guide](docs/sft_usage.md) - Comprehensive usage documentation with parameter reference and examples

**Tutorials:**
- [LAB Multi-Phase Training Tutorial](notebooks/lab_multiphase_training_tutorial.ipynb) - Interactive notebook demonstrating LAB multi-phase training workflow
- [SFT Comprehensive Tutorial](notebooks/sft_comprehensive_tutorial.ipynb) - Interactive notebook covering all SFT parameters with popular model examples

**Scripts:**
- [LAB Multi-Phase Training Script](scripts/lab_multiphase_training.py) - Example script for LAB multi-phase training with full command-line interface
- [SFT with Qwen 2.5 7B](scripts/sft_qwen_example.py) - Single-node multi-GPU training example with Qwen 2.5 7B Instruct
- [SFT with Llama 3.1 8B](scripts/sft_llama_example.py) - Single-node multi-GPU training example with Llama 3.1 8B Instruct  
- [SFT with Phi 4 Mini](scripts/sft_phi_example.py) - Single-node multi-GPU training example with Phi 4 Mini Instruct

**Quick Example:**
```python
from training_hub import sft

result = sft(
    model_path="/path/to/model",
    data_path="/path/to/data",
    ckpt_output_dir="/path/to/checkpoints",
    num_epochs=3,
    learning_rate=2e-5,
    max_tokens_per_gpu=45000
)
```

## Getting Started

1. **For detailed parameter documentation**: Check the relevant guide in `docs/`
2. **For hands-on learning**: Open the interactive notebooks in `notebooks/`
3. **For automation scripts**: Refer to examples in `scripts/`