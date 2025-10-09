"""Custom Perplexity metric implementation."""

import torch
from jaxtyping import Float, Int

from cs336_basics.lm.criterion.cross_entropy import cross_entropy



def perplexity(
    inputs: Float[torch.Tensor, " batch_size vocab_size"],
    targets: Int[torch.Tensor, " batch_size"],
    reduction: str = "mean"
) -> Float[torch.Tensor, ""]:
    return torch.exp(cross_entropy(inputs, targets, reduction))