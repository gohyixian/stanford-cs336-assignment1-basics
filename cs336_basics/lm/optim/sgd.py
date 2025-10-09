"""Custom implementation of Stochastic Gradient Descent (SGD)"""

import math
import torch
from collections.abc import Callable
from typing import Optional, Dict, Any, Iterable, Union



# substitute for torch.nn.optim.Optimizer.Params
ParamsLike = Union[Iterable[torch.nn.Parameter], Iterable[Dict[str, Any]]]


class SGD(torch.optim.Optimizer):
    
    def __init__(
        self, 
        params: ParamsLike, 
        lr: float = 1e-3
    ):
        """
        A slight variation of SGD where the learning rate decays over training,
        starting with an initial learning rate \alpha and taking successively smaller 
        steps over time:
        
        \theta_{t+1} = \theta_{t} - ( \alpha / \sqrt(t+1) ) * \nabla L(\theta_t; \B_t)
        
        In case `params` are just a single collection of torch.nn.Parameter objects, 
        the base constructor will create a single group and assign it the default 
        hyperparameters. Then, in step, we iterate over each parameter group, then 
        over each parameter in that group, and apply the above Equation.
        
        The API specifies that the user might pass in a callable closure to re-compute 
        the loss before the optimizer step. We won't need this for the optimizers we'll 
        use, but we add it to comply with the API.
        """
        
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        
        defaults = {"lr": lr}
        
        super().__init__(params, defaults)
    
    
    @torch.no_grad()
    def step(
        self, 
        closure: Optional[Callable] = None
    ):
        """A single SGD update step on model parameters."""
        
        loss = None if closure is None else closure()
        
        for group in self.param_groups:
            
            # Get the learning rate.
            lr = group["lr"]
            
            for p in group["params"]:
                
                # Skip if no gradients computed
                if p.grad is None:
                    continue
                
                # Get state associated with p (state used to keep track of items, 
                # i.e. rolling average EMA for stateful optims like AdamW)
                state = self.state[p]
                t = state.get("t", 0)  # Get iteration number from the state, or initial value.
                
                # Get the gradient of loss with respect to p.
                grad = p.grad.data
                
                # Update weight tensor in-place.
                p.data -= lr / math.sqrt(t + 1) * grad
                
                # Increment iteration number.
                state["t"] = t + 1
        
        return loss




if __name__ == "__main__":
    
    # python -m cs336_basics.lm.optim.sgd
    
    lr = 1e1  # slow monotonic convergence
    lr = 1e2  # converges very quickly, monotonically
    lr = 1e3  # diverges to very high values
    
    iter = 10
    
    # set seed
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=lr)
    
    for t in range(iter):
        
        opt.zero_grad()            # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean() # Compute a scalar loss value.
        
        print(loss.cpu().item())
        
        loss.backward()            # Run backward pass, which computes gradients.
        opt.step()                 # Run optimizer step.