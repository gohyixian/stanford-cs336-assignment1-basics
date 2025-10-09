"""Custom Cross-Entropy implementation."""

import torch
from jaxtyping import Float, Int



def cross_entropy(
    inputs: Float[torch.Tensor, " batch_size vocab_size"], 
    targets: Int[torch.Tensor, " batch_size"],
    reduction: str = "mean"
) -> Float[torch.Tensor, ""]:
    """
    Compute the cross-entropy loss between predicted logits and target indices.
    
    This function implements a numerically stable cross-entropy loss computation that takes
    predicted logits and target class indices, computing the cross-entropy loss as:
    l_i = -log(softmax(o_i)[x_{i+1}])
    
    The implementation includes several optimizations for numerical stability:
    - Subtracts the maximum logit value to prevent overflow in exp() operations
    - Uses log-sum-exp trick to avoid computing softmax explicitly
    - Cancels out log and exp operations where possible for efficiency
    
    Args:
        inputs: Tensor of shape [batch_size, vocab_size] containing predicted logits.
                Each row represents the unnormalized log-probabilities for one example.
        targets: Tensor of shape [batch_size] containing target class indices.
                 Each value must be in range [0, vocab_size-1].
        reduction: One of: "mean" | "sum" | "none"
    
    Returns:
        Scalar tensor containing the average cross-entropy loss across the batch.
    """
    
    # Subtract the largest element for numerical stability
    x_max = torch.max(inputs, dim=-1, keepdim=True).values  # [b, 1]
    x_stable = inputs - x_max                               # [b, v]
    
    # Use log-sum-exp for numerical stability
    log_sum_exp = torch.log(torch.sum(torch.exp(x_stable), dim=-1))  # [b,]
    
    # Collect logits for the target classes
    index = targets.unsqueeze(-1)                                            # [b, 1]
    target_logits = torch.gather(x_stable, dim=-1, index=index).squeeze(-1)  # [b,]
    
    # Cross-entropy loss: -log(softmax(x)[target]) = -x[target] + log(sum(exp(x)))
    # Since we already subtracted max, this becomes: -x_stable[target] + log_sum_exp
    losses = -target_logits + log_sum_exp  # [b,]
    
    if reduction == "mean":
        return torch.mean(losses)
    elif reduction == "sum":
        return torch.sum(losses)
    elif reduction == "none":
        return losses
    else:
        raise ValueError(f"Invalid reduction: {reduction}")