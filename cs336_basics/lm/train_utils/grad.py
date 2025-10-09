"""Custom implementation of Gradient utils."""

import torch
from typing import Iterable
from jaxtyping import Float



@torch.no_grad()
def clip_grad_norm(
    params: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6,
) -> Float[torch.Tensor, ""]:
    """
    Clip gradients in-place so the global L2 norm across all params is <= max_norm.
    
    Args:
        params: Iterable of parameters whose `.grad` will be clipped (in-place).
        max_l2_norm: Target maximum L2 norm M.
        eps: Small epsilon for numerical stability (default 1e-6).
    
    Returns:
        The pre-clipping global L2 norm (0-D tensor).
    """
    
    grads = [p.grad for p in params if (p is not None and p.grad is not None)]
    if not grads:
        return torch.tensor(0.0)
    
    # Accumulate norms in fp32 for stability
    device = grads[0].device
    total_sq = torch.zeros((), dtype=torch.float32, device=device)
    for g in grads:
        total_sq += g.detach().to(torch.float32).pow(2).sum()
    total_norm = total_sq.sqrt()
    
    # Scale in-place
    if max_l2_norm > 0:
        scale = (max_l2_norm / (total_norm + eps)).clamp(max=1.)
        if scale < 1.0:
            for g in grads:
                g.mul_(scale.to(g.dtype))
    
    return total_norm
