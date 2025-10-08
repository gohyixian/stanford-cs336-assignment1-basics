"""Custom `torch.nn.Embedding` and `RoPE` implementation."""

import torch
from jaxtyping import Int, Float
from einops import einsum, rearrange



class Embedding(torch.nn.Module):
    
    def __init__(
        self,
        num_embeddings: int, 
        embedding_dim: int, 
        device: torch.device = torch.device("cpu"), 
        dtype: torch.dtype = torch.float32
    ):
        """
        Custom implementation of an embedding layer, that inherits from 
        torch.nn.Module and performs an embedding lookup.
        
        num_embeddings: int Size of the vocabulary
        embedding_dim: int Dimension of the embedding vectors, i.e., d_model
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        
        NOTE:
        - Embedding weights E stored as E \in \ℝ^{num_emb x d_model}
        - Init weights using torch.nn.init.trunc_normal_ with mean=0, var=1, truncated at [-3, 3].
        """
        
        super().__init__()
        
        factory_kwargs = {"device": device, "dtype": dtype}
        
        # Store embeddings E as E \in \ℝ^{num_emb x model}
        self.weight = torch.nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), **factory_kwargs)
        )
        
        # Initialize weights with truncated normal
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)
    
    
    def forward(
        self, 
        token_ids: Int[torch.Tensor, " ..."]
    ) -> Float[torch.Tensor, " ... emb_dim"]:
        """
        Lookup the embedding vectors for the given token IDs.
        
        Args:
            token_ids (LongTensor): Indices of shape (batch_size, seq_len)
        Returns:
            Tensor: Embedding vectors of shape (batch_size, seq_len, embedding_dim)
        """
        
        return self.weight[token_ids]




class RotaryPositionalEmbedding(torch.nn.Module):
    
    def __init__(
        self, 
        theta: float, 
        d_k: int, 
        max_seq_len: int, 
        device: torch.device = torch.device("cpu"), 
        dtype: torch.dtype = torch.float32
    ):
        """
        Construct the RoPE module and create buffers if needed.
        
        theta: float Θ value for the RoPE
        d_k: int dimension of query and key vectors
        max_seq_len: int Maximum sequence length that will be inputted
        device: torch.device | None = None Device to store the buffer on
        
        NOTE: No learnable parameters
        """
        
        super().__init__()
        assert d_k % 2 == 0, "RoPE requires even d_k"
        
        factory_kwargs = {"device": device, "dtype": dtype}
        
        # Compute frequencies for each pair of dims
        # freq = \theta ^ (2k-2)/d
        k = torch.arange(1, d_k // 2 + 1, **factory_kwargs)  # k \in {1,..,d/2}
        freq = theta ** ((2*k - 2) / d_k)
        freq = 1.0 / freq  # (d_k // 2,)
        
        # All possible Token Positions [0...max_seq_len-1]
        i = torch.arange(max_seq_len, **factory_kwargs)  # (max_seq_len,)
        angles = einsum(i, freq, "n, d_half -> n d_half")    # (max_seq_len, d_k // 2)
        
        # Precompute cos/sin
        cos = torch.cos(angles)  # (max_seq_len, d_k // 2)
        sin = torch.sin(angles)  # (max_seq_len, d_k // 2)
        
        # Register as buffers
        # NOTE: persistent=False so they are not saved in the model's state_dict / checkpoints
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.d_k = d_k
    
    
    def forward(
        self, 
        x: Float[torch.Tensor, " ... n d_k"],
        token_positions: Int[torch.Tensor, " ... n"],
    ) -> Float[torch.Tensor, " ... n d_k"]:
        """
        Apply RoPE to input tensor.
        
        Apply rotation (a more computationally efficient way)
        >
        >  Each R in \big{R} = \
        >        [[cos m \theta, -sin m \theta],
        >         [sin m \theta,  cos m \theta]]
        
        When applying rotation:
        >
        >  [[cos m \theta, -sin m \theta],     [x1,     [x1 * cos m \theta - x2 * sin m \theta,     [x1d1,
        >   [sin m \theta,  cos m \theta]]  X   x2]  =   x1 * sin m \theta + x2 * cos m \theta]  =   x2d1]
        >
        
        Args:
            x: Tensor of shape (..., seq_len, d_k)
            token_positions: LongTensor of shape (..., seq_len)
        
        Returns:
            Tensor of shape (..., seq_len, d_k)
        """
        
        assert x.shape[-1] == self.d_k, f"Expected last dim {self.d_k}, got {x.shape[-1]}"
        if token_positions.dtype != torch.long:
            token_positions = token_positions.long()
        
        # Gather cos/sin for positions
        cos = self.cos[token_positions]  # (..., seq_len, d_k/2)
        sin = self.sin[token_positions]  # (..., seq_len, d_k/2)
        
        # Split last dim into pairs (x1, x2)
        x1 = x[..., ::2]   # (..., seq_len, d_k/2)
        x2 = x[..., 1::2]  # (..., seq_len, d_k/2)
        
        # Temporary stack: shape (..., seq_len, d_k/2, 2)
        # Last dim are pairs of [x1d1, x2d1]
        x_rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)  
        
        # Merge back last two dims -> (..., seq_len, d_k)
        # [[x1d1,x2d1],
        #  [x1d2,x2d2]]  ->  [x1d1,x2d1,x1d2,x2d2]
        out = rearrange(x_rotated, "... n d two -> ... n (d two)")
        
        return out