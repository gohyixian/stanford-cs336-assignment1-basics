"""Custom implementation of a Transformer Language Model."""

import torch
from jaxtyping import Float, Int

from cs336_basics.lm.arch.layers.linear import Linear
from cs336_basics.lm.arch.layers.rmsnorm import RMSNorm
from cs336_basics.lm.arch.layers.embedding import Embedding
from cs336_basics.lm.arch.blocks.transformer_block import TransformerBlock



class TransformerLM(torch.nn.Module):
    
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        rmsnorm_eps: float = 1e-5,
        device: torch.device = torch.device("cpu"), 
        dtype: torch.dtype = torch.float32
    ):
        """
        Custom implementation of a Transformer Language Model, that inherits from 
        torch.nn.Module and consists of an embedding layer, followed by a stack of 
        Transformer blocks, and a final linear layer to project the output to the 
        vocabulary size.
        
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\Theta$ parameter.
        rmsnorm_eps (float): Epsilon value for numerical stability
        """
        
        super().__init__()
        
        factory_kwargs = {"device": device, "dtype": dtype}
        
        # Token Embedding layer
        self.token_embeddings = Embedding(vocab_size, d_model, **factory_kwargs)
        
        # Transformer Blocks
        self.layers = torch.nn.ModuleList([
            TransformerBlock(
                d_model, num_heads, d_ff, context_length, rope_theta, rmsnorm_eps, **factory_kwargs
            ) for _ in range(num_layers)
        ])
        
        # Norm before Output Unembedding
        self.ln_final = RMSNorm(d_model, rmsnorm_eps, **factory_kwargs)
        
        # Output Unembedding
        self.lm_head = Linear(d_model, vocab_size, **factory_kwargs)
    
    
    
    def forward(
        self, 
        token_ids: Int[torch.Tensor, " b n"], 
    ) -> Float[torch.Tensor, " b n vocab_size"]:
        """
        Forward pass for the Transformer Language Model.
        
        token_ids (Int[torch.Tensor, b n]): Input token IDs of shape (batch_size, seq_len).
        
        Returns:
            Float[torch.Tensor, b n vocab_size]: Logits over the vocabulary for each token position.
        """
        
        # Token embeddings (b, n, d_model)
        x = self.token_embeddings(token_ids)
        
        # Transformer layers (b, n, d_model)
        for layer in self.layers:
            x = layer(x)
        
        # Final RMSNorm and projection to vocabulary logits
        x = self.ln_final(x)
        logits = self.lm_head(x)  # (b, n, vocab_size)
        
        return logits