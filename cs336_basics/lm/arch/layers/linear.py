"""Custom `torch.nn.Linear` implementation."""

import torch
from jaxtyping import Float
from einops import einsum



class Linear(torch.nn.Module):
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        device: torch.device = torch.device("cpu"), 
        dtype: torch.dtype = torch.float32
    ):
        """
        Custom implementation of a linear (fully-connected) layer.
        
        in_features: int final dimension of the input
        out_features: int final dimension of the output
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        
        NOTE:
        - Weight matrix W is stored as: W \in \ℝ^{out x in}
        - Does not include bias term (i.e., y = Wx), as per modern LLMs
        - Init weights using torch.nn.init.trunc_normal_ with mean=0, var=2/(in + out), truncated at [-3std, +3std].
        """
        
        super().__init__()
        
        factory_kwargs = {"device": device, "dtype": dtype}
        
        # Store weights W as W \in \ℝ^{out x in}, not W^T
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        
        # Initialize weights with truncated normal
        std = (2.0 / (in_features + out_features)) ** 0.5
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
    
    
    def forward(
        self, 
        x: Float[torch.Tensor, " ... in"]
    ) -> Float[torch.Tensor, " ... out"]:
        """
        Apply the linear transformation to the input:
        
          y = Wx
        
        where:
        - x \in \ℝ^{..., in}
        - W \in \ℝ^{out x in}
        - y \in \ℝ^{..., out}
        """
        
        y = einsum(self.weight, x, "out in, ... in -> ... out")
        return y
