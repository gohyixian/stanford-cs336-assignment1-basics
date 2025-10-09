"""Utils for Resource Accounting."""

import torch
from torch import nn, Tensor
from typing import Mapping, Union



def count_weight_params(
    model_or_state: Union[nn.Module, Mapping[str, Tensor]],
    deduplicate: bool = True,
    drop_list: list[str] = [],
    target_list: list[str] = []
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
        if any(i in name for i in drop_list):
            continue
        
        if len(target_list) > 0:
            if not any(i in name for i in target_list):
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



def bytes_from_params(n_params: int, dtype: torch.dtype = torch.float32) -> int:
    # element_size() returns bytes per element for the dtype (float32 -> 4)
    return n_params * torch.empty((), dtype=dtype).element_size()


def get_flops_per_batch(n_params: int, seq_len: int) -> int:
    # FLOPS
    # flops = 2 (# data points: B) (# params: D*K)
    # NOTE: this is for batch_size=1
    return 2 * seq_len * n_params


def human_bytes(n: int) -> str:
    gb = n / (1024**3)
    mb = n / (1024**2)
    kb = n / 1024
    if gb >= 1:
        return f"{gb:,.3f} GB"
    if mb >= 1:
        return f"{mb:,.3f} MB"
    if kb >= 1:
        return f"{kb:,.3f} KB"
    return f"{n} B"



def count_lm_matrix_multiplications(
    model_config: dict,
    batch_size: int = 1,
    sequence_length: int = 1024
) -> dict:
    """
    Count the number of matrix multiplications (A @ B) in a forward pass of the transformer model.
    
    Args:
        model_config: Dictionary containing model configuration parameters
        batch_size: Batch size for the forward pass
        sequence_length: Sequence length for the forward pass
    
    Returns:
        Dictionary with counts of different types of matrix multiplications
    """
    
    # Extract model parameters
    vocab_size = model_config["vocab_size"]
    d_model    = model_config["d_model"]
    num_layers = model_config["num_layers"]
    num_heads  = model_config["num_heads"]
    d_ff       = model_config["d_ff"]
    
    d_k = d_model // num_heads  # dimension per attention head
    
    counts = {}
    
    # 1. Token Embedding (lookup, not a matrix multiplication)
    counts["token_embedding"] = 0  # This is just indexing, not matmul
    
    # 2. Per Transformer Layer
    per_layer_counts = {}
    
    # 2a. Attention Layer
    # Q, K, V projections: 3 linear layers
    per_layer_counts["attention_qkv_projections"] = 3
    
    # Attention computation: Q @ K^T and attn @ V
    per_layer_counts["attention_qk_matmul"] = 1  # Q @ K^T
    per_layer_counts["attention_attn_v_matmul"] = 1  # attn @ V
    
    # Output projection
    per_layer_counts["attention_output_projection"] = 1
    
    # 2b. FFN Layer (SwiGLU)
    # W1, W3 projections: 2 linear layers
    per_layer_counts["ffn_w1_w3_projections"] = 2
    # W2 projection: 1 linear layer
    per_layer_counts["ffn_w2_projection"] = 1
    
    # Total per layer
    per_layer_counts["total_per_layer"] = sum(per_layer_counts.values())
    
    # 3. Final Layer Norm (no matrix multiplication)
    
    # 4. LM Head (output projection)
    counts["lm_head_projection"] = 1
    
    # Calculate totals
    counts["per_layer"] = per_layer_counts
    counts["total_attention_per_layer"] = (
        per_layer_counts["attention_qkv_projections"] + 
        per_layer_counts["attention_qk_matmul"] + 
        per_layer_counts["attention_attn_v_matmul"] + 
        per_layer_counts["attention_output_projection"]
    )
    counts["total_ffn_per_layer"] = (
        per_layer_counts["ffn_w1_w3_projections"] + 
        per_layer_counts["ffn_w2_projection"]
    )
    counts["total_per_layer"]    = per_layer_counts["total_per_layer"]
    counts["total_all_layers"]   = num_layers * per_layer_counts["total_per_layer"]
    counts["total_forward_pass"] = counts["total_all_layers"] + counts["lm_head_projection"]
    
    # Add detailed breakdown
    counts["breakdown"] = {
        "token_embedding": 0,  # Not a matrix multiplication
        "attention_layers": {
            "qkv_projections"    : num_layers * per_layer_counts["attention_qkv_projections"],
            "qk_matmul"          : num_layers * per_layer_counts["attention_qk_matmul"],
            "attn_v_matmul"      : num_layers * per_layer_counts["attention_attn_v_matmul"],
            "output_projections" : num_layers * per_layer_counts["attention_output_projection"],
            "total"              : num_layers * counts["total_attention_per_layer"]
        },
        "ffn_layers": {
            "w1_w3_projections" : num_layers * per_layer_counts["ffn_w1_w3_projections"],
            "w2_projections"    : num_layers * per_layer_counts["ffn_w2_projection"],
            "total"             : num_layers * counts["total_ffn_per_layer"]
        },
        "lm_head" : counts["lm_head_projection"]
    }
    
    return counts



if __name__ == "__main__":
    
    # python -m cs336_basics.lm.arch.utils.resource_accounting
    
    from cs336_basics.lm.arch.models.transformer_lm import TransformerLM
    
    # model_name = "gpt-2-small"
    # model_config = {
    #     "vocab_size"     : 50257,
    #     "context_length" : 1024,
    #     "num_layers"     : 12,
    #     "d_model"        : 768,
    #     "num_heads"      : 12,
    #     "d_ff"           : 6400,
    #     "rope_theta"     : 10000,
    #     "rmsnorm_eps"    : 1e-5,
    #     "device"         : torch.device("cpu"),
    #     "dtype"          : torch.float32
    # }
    # model_name = "gpt-2-medium"
    # model_config = {
    #     "vocab_size"     : 50257,
    #     "context_length" : 1024,
    #     "num_layers"     : 24,
    #     "d_model"        : 1024,
    #     "num_heads"      : 16,
    #     "d_ff"           : 6400,
    #     "rope_theta"     : 10000,
    #     "rmsnorm_eps"    : 1e-5,
    #     "device"         : torch.device("cpu"),
    #     "dtype"          : torch.float32
    # }
    # model_name = "gpt-2-large"
    # model_config = {
    #     "vocab_size"     : 50257,
    #     "context_length" : 1024,
    #     "num_layers"     : 36,
    #     "d_model"        : 1280,
    #     "num_heads"      : 20,
    #     "d_ff"           : 6400,
    #     "rope_theta"     : 10000,
    #     "rmsnorm_eps"    : 1e-5,
    #     "device"         : torch.device("cpu"),
    #     "dtype"          : torch.float32
    # }
    # model_name = "gpt-2-xl"
    # model_config = {
    #     "vocab_size"     : 50257,
    #     "context_length" : 1024,
    #     "num_layers"     : 48,
    #     "d_model"        : 1600,
    #     "num_heads"      : 25,
    #     "d_ff"           : 6400,
    #     "rope_theta"     : 10000,
    #     "rmsnorm_eps"    : 1e-5,
    #     "device"         : torch.device("cpu"),
    #     "dtype"          : torch.float32
    # }
    model_name = "gpt-2-xl"
    model_config = {
        "vocab_size"     : 50257,
        "context_length" : 16384,
        "num_layers"     : 48,
        "d_model"        : 1600,
        "num_heads"      : 25,
        "d_ff"           : 6400,
        "rope_theta"     : 10000,
        "rmsnorm_eps"    : 1e-5,
        "device"         : torch.device("cpu"),
        "dtype"          : torch.float32
    }
    
    tok_emb = ["token_embeddings"]
    attn    = ["attn"]
    ffn     = ["ffn", "lm_head"]
    
    
    net = TransformerLM(**model_config)
    # input_seq_len = 1024
    input_seq_len = 16384
    
    
    def report(name: str, n_params: int, dtype: torch.dtype = torch.float32) -> None:
        b = bytes_from_params(n_params, dtype)
        flops = get_flops_per_batch(n_params, input_seq_len)
        print(f"{name:<48} : {n_params:>13,} params  |  {b:>13,} bytes  ({human_bytes(b):>10})  |  {flops:>16,} FLOPs for seq_len={input_seq_len}")
    
    
    print(model_name)
    print("="*20)
    
    num_params = count_weight_params(net, deduplicate=True)
    report("Number of Trainable Parameters (deduplicated)", num_params, model_config["dtype"])
    
    num_params = count_weight_params(net, deduplicate=False)
    report("Number of Trainable Parameters", num_params, model_config["dtype"])
    
    num_params = count_weight_params(net, deduplicate=False, drop_list=tok_emb)
    report("Number of Trainable Parameters (drop tok emb)", num_params, model_config["dtype"])
    
    num_params = count_weight_params(net, deduplicate=False, target_list=attn)
    report("Number of Trainable Parameters (attn)", num_params, model_config["dtype"])
    
    num_params = count_weight_params(net, deduplicate=False, target_list=ffn)
    report("Number of Trainable Parameters (ffn)", num_params, model_config["dtype"])
    
    print("\n" + "="*60)
    print("MATRIX MULTIPLICATION COUNTS")
    print("="*60)
    
    # Count matrix multiplications
    matmul_counts = count_lm_matrix_multiplications(model_config)
    
    print(f"Per Transformer Layer:")
    print(f"  Attention QKV projections   : {matmul_counts['per_layer']['attention_qkv_projections']}")
    print(f"  Attention Q@K^T             : {matmul_counts['per_layer']['attention_qk_matmul']}")
    print(f"  Attention attn@V            : {matmul_counts['per_layer']['attention_attn_v_matmul']}")
    print(f"  Attention output proj       : {matmul_counts['per_layer']['attention_output_projection']}")
    print(f"  FFN W1,W3 projections       : {matmul_counts['per_layer']['ffn_w1_w3_projections']}")
    print(f"  FFN W2 projection           : {matmul_counts['per_layer']['ffn_w2_projection']}")
    print(f"  Total per layer             : {matmul_counts['per_layer']['total_per_layer']}")
    
    print(f"\nTotal for {model_config['num_layers']} layers:")
    print(f"  Total attention operations  : {matmul_counts['breakdown']['attention_layers']['total']}")
    print(f"  Total FFN operations        : {matmul_counts['breakdown']['ffn_layers']['total']}")
    print(f"  Total all layers            : {matmul_counts['total_all_layers']}")
    
    print(f"\nFinal output:")
    print(f"  LM head projection          : {matmul_counts['lm_head_projection']}")
    
    print(f"\nTOTAL MATRIX MULTIPLICATIONS  : {matmul_counts['total_forward_pass']}")
