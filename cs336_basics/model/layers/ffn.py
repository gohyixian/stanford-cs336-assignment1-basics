"""Custom `SwiGLU` implementation."""

import torch
from jaxtyping import Float

from cs336_basics.model.layers.linear import Linear
from cs336_basics.model.layers.activations import SiLU



class SwiGLU(torch.nn.Module):
    
    def __init__(
        self, 
        d_model: int,
        d_ff: int,
        device: torch.device = torch.device("cpu"), 
        dtype: torch.dtype = torch.float32
    ):
        """
        Custom implementation of a SwiGLU layer.
        
        in_features: int final dimension of the input
        out_features: int final dimension of the output
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        
        FFN(x) = SwiGLU(x, W1, W2, W3) = W2(SiLU(W1x) ⊙ W3x)
        
        where:
        - x \in \ℝ^{..., d_model}
        - W1 \in \ℝ^{d_ff x d_model}
        - W2 \in \ℝ^{d_model x d_ff}
        - W3 \in \ℝ^{d_ff x d_model}
        
        and canonically, d_ff = 8/3 * d_model
        """
        
        super().__init__()
        
        factory_kwargs = {"device": device, "dtype": dtype}
        
        self.w1 = Linear(d_model, d_ff, **factory_kwargs)
        self.w2 = Linear(d_ff, d_model, **factory_kwargs)
        self.w3 = Linear(d_model, d_ff, **factory_kwargs)
    
    
    def forward(
        self, 
        x: Float[torch.Tensor, " ... d_model"]
    ) -> Float[torch.Tensor, " ... d_model"]:
        """
        Forward pass of SwiGLU.
        
        Args:
            x (Tensor): Input tensor of shape (..., d_model)
        
        Returns:
            Tensor: Output tensor of shape (..., d_model)
        """
        # Project input: ff model, ... model -> ... ff
        x1 = self.w1(x)
        x3 = self.w3(x)
        
        # SiLU activation: x * sigmoid(x)
        x1 = SiLU(x1)
        
        # Elem-wise product
        hidden = x1 * x3
        
        # Final projection back to d_model: model ff, ... ff -> ... model
        out = self.w2(hidden)
        
        return out
