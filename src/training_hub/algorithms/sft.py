from typing import Any, Dict, Type, Optional
from instructlab.training import run_training, TorchrunArgs, TrainingArgs

from . import Algorithm, Backend, AlgorithmRegistry


class InstructLabTrainingSFTBackend(Backend):
    """InstructLab Training backend for SFT algorithm."""
    
    def execute_training(self, algorithm_params: Dict[str, Any]) -> Any:
        """Execute SFT training using instructlab-training."""
        model_path = algorithm_params['model_path']
        data_path = algorithm_params['data_path'] 
        ckpt_output_dir = algorithm_params['ckpt_output_dir']
        
        # Extract optional parameters with defaults
        num_epochs = algorithm_params.get('num_epochs', 1)
        effective_batch_size = algorithm_params.get('effective_batch_size', 3840)
        learning_rate = algorithm_params.get('learning_rate', 2e-6)
        max_seq_len = algorithm_params.get('max_seq_len', 4096)
        max_batch_len = algorithm_params.get('max_batch_len', 60000)
        
        # Set up training arguments
        training_args = TrainingArgs(
            model_path=model_path,
            data_path=data_path,
            ckpt_output_dir=ckpt_output_dir,
            num_epochs=num_epochs,
            effective_batch_size=effective_batch_size,
            learning_rate=learning_rate,
            max_seq_len=max_seq_len,
            max_batch_len=max_batch_len,
            # Add other training args as needed
        )
        
        # Extract torchrun parameters for multi-node support
        torchrun_params = {}
        torchrun_keys = ['nproc_per_node', 'nnodes', 'node_rank', 'rdzv_id', 'rdzv_endpoint']
        
        for key in torchrun_keys:
            if key in algorithm_params:
                torchrun_params[key] = algorithm_params[key]
        
        # Set up torchrun arguments - use defaults if not specified
        if torchrun_params:
            torchrun_args = TorchrunArgs(**torchrun_params)
        else:
            # Use single-node defaults
            torchrun_args = TorchrunArgs(
                nproc_per_node=1,
                nnodes=1, 
                node_rank=0,
                rdzv_id=0,
                rdzv_endpoint=""
            )
        
        # Execute training
        return run_training(
            torchrun_args=torchrun_args,
            training_args=training_args
        )


class SFTAlgorithm(Algorithm):
    """Supervised Fine-Tuning algorithm."""
    
    def __init__(self, backend: Backend, **kwargs):
        self.backend = backend
        self.config = kwargs
    
    def train(self, 
              model_path: str,
              data_path: str, 
              ckpt_output_dir: str,
              num_epochs: int = 1,
              effective_batch_size: int = 3840,
              learning_rate: float = 2e-6,
              max_seq_len: int = 4096,
              max_batch_len: int = 60000,
              # Torchrun parameters for multi-node support
              nproc_per_node: int = None,
              nnodes: int = None,
              node_rank: int = None,
              rdzv_id: int = None,
              rdzv_endpoint: str = None,
              **kwargs) -> Any:
        """Execute SFT training."""
        params = {
            'model_path': model_path,
            'data_path': data_path,
            'ckpt_output_dir': ckpt_output_dir,
            'num_epochs': num_epochs,
            'effective_batch_size': effective_batch_size,
            'learning_rate': learning_rate,
            'max_seq_len': max_seq_len,
            'max_batch_len': max_batch_len,
            **kwargs
        }
        
        # Add torchrun parameters if provided
        torchrun_params = {
            'nproc_per_node': nproc_per_node,
            'nnodes': nnodes, 
            'node_rank': node_rank,
            'rdzv_id': rdzv_id,
            'rdzv_endpoint': rdzv_endpoint
        }
        
        # Only add non-None torchrun parameters
        for key, value in torchrun_params.items():
            if value is not None:
                params[key] = value
        
        return self.backend.execute_training(params)
    
    def get_required_params(self) -> Dict[str, Type]:
        """Return required parameters for SFT."""
        return {
            'model_path': str,
            'data_path': str,
            'ckpt_output_dir': str,
            'num_epochs': int,
            'effective_batch_size': int,
            'learning_rate': float,
            'max_seq_len': int,
            'max_batch_len': int,
        }


# Register the algorithm and backend
AlgorithmRegistry.register_algorithm('sft', SFTAlgorithm)
AlgorithmRegistry.register_backend('sft', 'instructlab-training', InstructLabTrainingSFTBackend)


# Convenience function for backwards compatibility
def sft(model_path: str, 
        data_path: str, 
        ckpt_output_dir: str,
        backend: str = "instructlab-training",
        # Multi-node support
        nproc_per_node: int = None,
        nnodes: int = None,
        node_rank: int = None,
        rdzv_id: int = None,
        rdzv_endpoint: str = None,
        **kwargs) -> Any:
    """Convenience function to run SFT training.
    
    Args:
        model_path: Path to the model to fine-tune
        data_path: Path to the training data
        ckpt_output_dir: Directory to save checkpoints
        backend: Backend implementation to use (default: "instructlab-training")
        **kwargs: Additional training parameters
    
    Returns:
        Training result from the backend
    """
    from . import create_algorithm
    
    algorithm = create_algorithm('sft', backend)
    return algorithm.train(
        model_path=model_path,
        data_path=data_path,
        ckpt_output_dir=ckpt_output_dir,
        nproc_per_node=nproc_per_node,
        nnodes=nnodes,
        node_rank=node_rank,
        rdzv_id=rdzv_id,
        rdzv_endpoint=rdzv_endpoint,
        **kwargs
    )

