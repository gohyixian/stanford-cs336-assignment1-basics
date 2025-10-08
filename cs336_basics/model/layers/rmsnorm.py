"""Custom `torch.nn.RMSNorm` implementation."""

import torch
from jaxtyping import Float



class RMSNorm(torch.nn.Module):
    
    def __init__(
        self, 
        d_model: int, 
        eps: float = 1e-5, 
        device: torch.device = torch.device("cpu"), 
        dtype: torch.dtype = torch.float32
    ):
        """
        Custom implementation of a Root-Mean-Square Layer Norm module, 
        that inherits from torch.nn.Module.
        
        d_model: int Hidden dimension of the model
        eps: float = 1e-5 Epsilon value for numerical stability
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        
        super().__init__()
        
        self.eps = eps
        factory_kwargs = {"device": device, "dtype": dtype}
        
        # Learnable gain parameter g \in \ℝ^{d_model}
        self.weight = torch.nn.Parameter(
            torch.ones(d_model, **factory_kwargs)
        )
    
    
    def forward(
        self, 
        x: Float[torch.Tensor, " ... d_model"],
    ) -> Float[torch.Tensor, " ... d_model"]:
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model) 
        and return a tensor of the same shape.
        """
        
        in_dtype = x.dtype
        x = x.to(torch.float32)
        
        # Compute RMS over the last dim (which is often d_model)
        rms = torch.sqrt(
            torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps
        )
        
        # Normalize and rescale with learnable gain
        out = (x / rms) * self.weight
        
        return out.to(in_dtype)
