"""Custom Attention implementation."""

import math
import torch
from jaxtyping import Bool, Float, Int
from einops import einsum, rearrange, repeat

from cs336_basics.lm.arch.layers.linear import Linear
from cs336_basics.lm.arch.layers.activations import softmax
from cs336_basics.lm.arch.layers.embedding import RotaryPositionalEmbedding



def scaled_dot_product_attention(
    Q:    Float[torch.Tensor, " ... q d_k"],
    K:    Float[torch.Tensor, " ... k d_k"],
    V:    Float[torch.Tensor, " ... k d_v"],
    mask: Bool[torch.Tensor,  " ... q k"] = None
) -> Float[torch.Tensor, " ... q d_v"]:
    """
    Scaled dot-product attention with optional boolean mask.
    Shapes:
        Q: (..., q, d_k)
        K: (..., k, d_k)
        V: (..., k, d_v)
        mask: (..., q, k) with True for valid positions / “information flow”
    
    Returns:
        (..., q, d_v)
    
    NOTE:
    - All `q,k,v` dimensions are `n` / seq_len
    """
    
    d_k = Q.shape[-1]
    
    # Scores: (Q.T @ K) / sqrt(d_k)
    scores = einsum(Q, K, "... q d_k, ... k d_k -> ... q k") / math.sqrt(d_k)
    
    # Apply mask if given
    if mask is not None:
        scores = scores.masked_fill(~mask.bool(), float("-inf"))
    
    attn = softmax(scores, dim=-1)
    
    # If a query row is fully masked (all False), softmax gives NaNs; force zeros
    # torch.where(cond, A, B) chooses elements from A where cond is True, else from B
    if mask is not None:
        has_any = mask.any(dim=-1, keepdim=True)  # (..., q, 1)
        attn = torch.where(has_any, attn, torch.zeros_like(attn))
    
    # softmax((Q.T @ K) / sqrt(d_k)) @ V
    out = einsum(attn, V, "... q k, ... k d_v -> ... q d_v")
    
    return out



class CausalMultiHeadSelfAttention(torch.nn.Module):
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionalEmbedding = None,
        device: torch.device = torch.device("cpu"), 
        dtype: torch.dtype = torch.float32
    ):
        """
        Causal multi-head self-attention layer.
        
        d_model: int Dimensionality of the Transformer block inputs.
        num_heads: int Number of heads to use in multi-head self-attention.
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        
        NOTE:
        - d_model must be divisible by num_heads
        - Each head has dimension d_k = d_model / num_heads
        - Uses scaled dot-product attention with causal masking
        - Uses a single linear layer W_{q,k,v} for each of q/k/v to project inputs to {Q,K,V} for all heads
        - Uses a single linear layer W_O to project concatenated heads back to d_model
        """
        
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        factory_kwargs = {"device": device, "dtype": dtype}
        
        # Linear layers to project input to Q, K, V for all heads
        self.q_proj = Linear(d_model, d_model, **factory_kwargs)
        self.k_proj = Linear(d_model, d_model, **factory_kwargs)
        self.v_proj = Linear(d_model, d_model, **factory_kwargs)
        
        # Output projection layer
        self.output_proj = Linear(d_model, d_model, **factory_kwargs)
        
        self.rope = rope
    
    
    def forward(
        self, 
        x: Float[torch.Tensor, " ... n in"], 
        token_positions: Int[torch.Tensor, " ... n"] | None = None
    ) -> Float[torch.Tensor, " ... n out"]:
        """
        Compute causal multi-head self-attention on `x` with optional RoPE on Q/K.
        
        Causal masking: prevent attention to future positions so token i can only
        attend to positions j where j <= i. Use a lower-triangular boolean mask.
        
        RoPE: if `self.rope` is not None, apply RoPE to the per-head queries and
        keys (but not values). Treat the head dimension as part of the batch so
        the same rotation is applied for every head at the same position. Token
        positions are inferred as arange(seq_len) and broadcast across leading
        dimensions.
        
        Args:
            x: Tensor of shape (..., n, d_model). Leading dims (e.g., batch) are
               supported. The last dim must equal `self.d_model`.
        
        Returns:
            Tensor of shape (..., n, d_model): attention output.
        """
        
        # Project to Q, K, V in a single batched operation for all heads
        # ------------------------------------------
        # Shapes after projection: (..., n, d_model)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        
        # Split Q, K, V into multiple heads
        # ------------------------------------------
        # NOTE: d_model = h * d_k
        # rearrange(tensor, "pattern", **axes_lengths)
        q = rearrange(q, "... n (h d_k) -> ... h n d_k", h=self.num_heads)
        k = rearrange(k, "... n (h d_k) -> ... h n d_k", h=self.num_heads)
        v = rearrange(v, "... n (h d_k) -> ... h n d_k", h=self.num_heads)
        
        # seq_len
        n = q.shape[-2]
        
        
        # Optionally apply RoPE to Q and K only
        # ------------------------------------------
        if self.rope is not None:
            
            if token_positions is None:
                # base positions: (n,): [0, n-1]
                pos = torch.arange(n, device=q.device, dtype=torch.long)
                
                # broadcast to (..., h, n):
                pos = rearrange(pos, 'n -> 1 1 n')                   # (1, 1, n)
                token_positions = pos.expand(*q.shape[:-2], n)       # (..., h, n)
            
            else:
                # use token positions if provided
                token_positions = repeat(token_positions, "... n -> ... h n", h=self.num_heads)   # (..., h, n)
            
            # rotate per head, same positions across heads at the same time index
            q = self.rope(q, token_positions)      # (..., h, n, d_k)
            k = self.rope(k, token_positions)      # (..., h, n, d_k)
        
        
        # Causal Mask
        # ------------------------------------------
        # Build causal mask of shape (..., head, n, n): True where allowed (j <= i)
        i = torch.arange(n, device=q.device)
        j = torch.arange(n, device=q.device)
        
        # base causal (n, n)
        causal_mask = rearrange(i, "n -> n 1") >= rearrange(j, "n -> 1 n")  # (n, n)
        
        # broadcast to (..., head, n, n)
        lead_shape = q.shape[:-2]  # (..., head)
        causal_mask = causal_mask.view(*((1,) * len(lead_shape)), n, n).expand(*lead_shape, n, n)
        
        
        # Scaled dot-product attention per head
        # ------------------------------------------
        context = scaled_dot_product_attention(q, k, v, mask=causal_mask)  # (..., head, n, d_k)
        
        
        # Merge heads back: (..., head, n, d_k) -> (..., n, d_model)
        # ------------------------------------------
        context = rearrange(context, "... h n d_k -> ... n (h d_k)")
        
        
        # Final output projection on merged heads
        # ------------------------------------------
        out = self.output_proj(context)
        
        return out
