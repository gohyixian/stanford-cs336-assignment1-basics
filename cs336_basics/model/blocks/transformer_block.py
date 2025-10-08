"""Custom Tranformer Block implementation."""

import torch
from jaxtyping import Float, Int

from cs336_basics.model.layers.ffn import SwiGLU
from cs336_basics.model.layers.rmsnorm import RMSNorm
from cs336_basics.model.layers.embedding import RotaryPositionalEmbedding
from cs336_basics.model.layers.attention import CausalMultiHeadSelfAttention



class TransformerBlock(torch.nn.Module):
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        rope_theta: float,
        rmsnorm_eps: float = 1e-5,
        device: torch.device = torch.device("cpu"), 
        dtype: torch.dtype = torch.float32
    ):
        """
        Custom implementation of a Transformer Block, that inherits from 
        torch.nn.Module and consists of a Causal Multi-Head Self-Attention 
        layer followed by a SwiGLU Feed-Forward Network (FFN), with RMSNorm 
        applied before each layer and residual connections after each layer.
        
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        rope_theta (float): RoPE parameter.
        rmsnorm_eps (float): Epsilon value for numerical stability
        """
        
        super().__init__()
        
        factory_kwargs = {"device": device, "dtype": dtype}
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        d_k = d_model // num_heads
        rope = RotaryPositionalEmbedding(rope_theta, d_k, max_seq_len, **factory_kwargs)
        
        # Causal Multi-Head Self-Attention with RoPE
        self.ln1 = RMSNorm(d_model, rmsnorm_eps, **factory_kwargs)
        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, rope, **factory_kwargs)
        
        # Position-Wise Feed-Forward
        self.ln2 = RMSNorm(d_model, rmsnorm_eps, **factory_kwargs)
        self.ffn = SwiGLU(d_model, d_ff, **factory_kwargs)
    
    
    def forward(
        self, 
        x: Float[torch.Tensor, " b n d_model"], 
        token_positions: Int[torch.Tensor, " ... n"] | None = None
    ) -> Float[torch.Tensor, " b n d_model"]:
        """
        Apply the Transformer block to the input.
        
        Args:
            x (Float[Tensor, b n d_model]): Input tensor of shape (batch_size, seq_len, d_model)
            token_positions (Int[Tensor, ... n] | None): Optional tensor of shape 
                (batch_size, seq_len) containing the position indices of each token in the sequence.
                If None, positions are assumed to be [0, 1, 2, ..., seq_len - 1] for each sequence in the batch.
        
        Returns:
            Float[Tensor, b n d_model]: Output tensor of shape (batch_size, seq_len, d_model)
        """
        
        # Pre-LN + Causal MHSA + Residual
        x = x + self.attn(self.ln1(x), token_positions=token_positions)
        
        # Pre-LN + SwiGLU FFN + Residual
        x = x + self.ffn(self.ln2(x))
        
        return x