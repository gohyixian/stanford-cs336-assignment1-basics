"""Utils for Resource Accounting."""

import torch
from torch import nn, Tensor
from typing import Mapping, Union



def count_weight_params(
    model_or_state: Union[nn.Module, Mapping[str, Tensor]],
    deduplicate: bool = True,
) -> int:
    """
    Count the total number of parameters that end with '.weight' in a model/state_dict.
    
    Args:
        model_or_state: An nn.Module or a state_dict (mapping str -> Tensor).
        deduplicate: If True, avoid double-counting tied/shared weights by
                     hashing underlying storage pointers.
    
    Returns:
        int: total number of elements across all .weight tensors.
    """
    
    # Get a state_dict view
    state = model_or_state.state_dict() if isinstance(model_or_state, nn.Module) else model_or_state
    
    total = 0
    seen_ptrs = set()
    
    for name, tensor in state.items():
        if not name.endswith(".weight"):
            continue
        if not isinstance(tensor, torch.Tensor):
            continue
        
        if deduplicate:
            # Robust pointer fetch across PyTorch versions
            try:
                ptr = tensor.untyped_storage().data_ptr()
            except AttributeError:
                try:
                    ptr = tensor.storage().data_ptr()  # older PyTorch
                except Exception:
                    ptr = tensor.data_ptr()  # fallback
            
            if ptr in seen_ptrs:
                continue
            seen_ptrs.add(ptr)
        
        total += tensor.numel()
    
    return int(total)




if __name__ == "__main__":
    
    # python -m cs336_basics.model.utils.resource_accounting
    
    from cs336_basics.model.models.transformer_lm import TransformerLM
    
    # GPT-2 XL
    model_config = {
        "vocab_size"     : 50257,
        "context_length" : 1024,
        "num_layers"     : 48,
        "d_model"        : 1600,
        "num_heads"      : 25,
        "d_ff"           : 6400,
        "rope_theta"     : 10000,
        "rmsnorm_eps"    : 1e-5,
        "device"         : torch.device("cpu"),
        "dtype"          : torch.float32
    }
    
    net = TransformerLM(**model_config)
    
    num_params = count_weight_params(net, deduplicate=True)
    print(f"Number of Trainable Parameters (deduplicated): {num_params}")  # 2127057600
    
    num_params = count_weight_params(net, deduplicate=False)
    print(f"Number of Trainable Parameters               : {num_params}")  # 2127057600
    