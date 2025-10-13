import os
from curses.ascii import isdigit
from importlib.metadata import pass_none
from typing import get_origin, get_args

def format_type_name(tp):
    # Handle None
    if tp is type(None):
        return 'None'
    
    # Handle basic types
    if hasattr(tp, '__name__'):
        return tp.__name__
    
    # Handle typing generics
    origin = get_origin(tp)
    args = get_args(tp)
    
    if origin is not None:
        origin_name = getattr(origin, '__name__', str(origin))
        if args:
            arg_names = [format_type_name(arg) for arg in args]
            return f"{origin_name}[{', '.join(arg_names)}]"
        return origin_name
    
    # Fallback: clean up the string representation
    type_str = str(tp)
    if type_str.startswith("<class '") and type_str.endswith("'>"):
        return type_str[8:-2]
    
    return type_str


def get_torchrun_params(args: dict):
    """
    Parse and load PyTorch variables from dict with fallback to environment variables.

    Args:
        args (dict): Dictionary containing PyTorch configuration parameters

    Returns:
        dict: Dictionary with PyTorch parameters loaded from args or environment
    """
    pytorch_vars = ['nproc_per_node', 'nnodes', 'node_rank', 'rdzv_id', 'rdzv_endpoint', 'master_addr', 'master_port']
    torchrun_args = {}

    def validate_nproc_per_node(value):
        """Validate and convert nproc_per_node value."""
        if isinstance(value, str):
            if value.lower() == 'auto':
                return 'gpu'
            elif value.lower() == 'gpu':
                return 'gpu'
            else:
                try:
                    return int(value)
                except ValueError:
                    raise ValueError(f"nproc_per_node must be 'auto', 'gpu', or an integer, got: {value}")
        elif isinstance(value, int):
            return value
        else:
            raise ValueError(f"nproc_per_node must be 'auto', 'gpu', or an integer, got: {value}")

    def get_env_var_name(var_name):
        """Get environment variable name based on PyTorch convention."""
        return var_name.upper() if var_name in ['master_addr', 'master_port'] else f"PET_{var_name.upper()}"

    for var_name in pytorch_vars:
        # Try args dict first
        if var_name in args and args[var_name] is not None and args[var_name] != "":
            value = args[var_name]
            if var_name == 'nproc_per_node':
                torchrun_args[var_name] = validate_nproc_per_node(value)
            elif var_name in ['nnodes', 'node_rank', 'rdzv_id', 'master_port']:
                torchrun_args[var_name] = int(value) if isinstance(value, (str, int)) else value
            else:
                torchrun_args[var_name] = value
        else:
            # Fallback to environment variable
            env_value = os.getenv(get_env_var_name(var_name))
            if env_value is not None:
                if var_name == 'nproc_per_node':
                    torchrun_args[var_name] = validate_nproc_per_node(env_value)
                elif var_name in ['nnodes', 'node_rank', 'rdzv_id', 'master_port']:
                    try:
                        torchrun_args[var_name] = int(env_value)
                    except ValueError:
                        torchrun_args[var_name] = env_value
                else:
                    torchrun_args[var_name] = env_value
            else:
                # Set defaults
                defaults = {'nnodes': 1, 'rdzv_id': 0}
                torchrun_args[var_name] = defaults.get(var_name, "")

    # Validate mutually exclusive parameters
    if (torchrun_args.get('rdzv_endpoint', '') != "" and
        (torchrun_args.get('master_addr', '') != "" or torchrun_args.get('master_port', '') != "")):
        raise ValueError("Cannot specify both rdzv_endpoint and master_addr/master_port. These are mutually exclusive parameters.")

    return torchrun_args
