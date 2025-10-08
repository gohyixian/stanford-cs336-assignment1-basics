"""Activation functions."""

import torch
from jaxtyping import Float



def SiLU(x: Float[torch.Tensor, " ..."]) -> Float[torch.Tensor, " ..."]:
    """
    SiLU activation function, also known as the swish function.
    
    SiLU(x) = x * sigmoid(x)
    
    Args:
        x (Tensor): Input tensor of any shape.
    Returns:
        Tensor: Output tensor of the same shape as input.
    """
    
    return x * torch.sigmoid(x)


def softmax(x: Float[torch.Tensor, " ..."], dim: int = -1) -> torch.Tensor:
    """
    Numerically stable softmax function.
    Subtract max for numerical stability: exp(big_values) = inf / NaN
    
    Args:
        x (Tensor): Input tensor of any shape.
        dim (int): Dimension along which to apply softmax. Default is -1 (last dimension).
    
    Returns:
        Tensor: Output tensor of the same shape as input, with softmax applied along specified dimension.
    """
    
    # same shape as x but with size 1 along `dim`
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    sum_exp = torch.sum(x_exp, dim=dim, keepdim=True)
    
    return x_exp / sum_exp