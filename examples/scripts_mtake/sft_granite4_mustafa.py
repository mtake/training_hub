import torch
from training_hub import sft
from instructlab.training import FSDPOptions

# SFT training with custom parameters
result = sft(
    model_path="ibm-granite/granite-4.0-h-small",
    # @@@ahoaho XXX
    # data_path="/home/lab/meyceoz/testing-train-hub/training_hub/examples/debug-dataset.jsonl",
    # ckpt_output_dir="granite-custom-ckpt",
    # data_output_dir="data_output",
    data_path="messages_data_teigaku-genzei-ibm-v6.jsonl",
    ckpt_output_dir="experiments-mustafa",
    data_output_dir="data-mustafa",
    save_samples=0,
    num_epochs=2,
    learning_rate=2e-5,
    effective_batch_size=128,
    max_seq_len=150,
    max_tokens_per_gpu=1000,
    warmup_steps=25,
    accelerate_full_state_at_epoch = False,
    checkpoint_at_epoch = True,
    # @@@ahoaho XXX
    # nproc_per_node=8,
    nproc_per_node=torch.cuda.device_count() if torch.cuda.is_available() else 0,
    fsdp_options = FSDPOptions(
        cpu_offload_params=True  # Boolean: True to enable CPU offloading, False to disable
    )
)
