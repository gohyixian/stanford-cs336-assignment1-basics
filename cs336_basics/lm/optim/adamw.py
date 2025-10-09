"""Custom implementation of AdamW."""

import math
import torch
from collections.abc import Callable
from typing import Optional, Dict, Any, Iterable, Union

from cs336_basics.lm.optim.sgd import ParamsLike



class AdamW(torch.optim.Optimizer):
    
    def __init__(
        self,
        params: ParamsLike,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        """
        AdamW optimizer:
        
        m_t : first moment vector, same shape as \theta, initialised as 0
        v_t : second moment vector, same shape as \theta, initialised as 0
        
        Each step() call does:
        -----------------------------------------------------------------
        1.  m_t        = \beta_1 * m_{t-1} + (1 - \beta1) * g_t              <-- update first moment estimate
            v_t        = \beta_2 * v_{t-1} + (1 - \beta2) * g_t^2            <-- update second moment estimate
        2.  \alpha_t   = \alpha * sqrt( 1 - \beta2^t ) / (1 - \beta1^t)      <-- compute adjusted \alpha for iteration \t
        3.  \theta     = \theta - \alpha_t * m_t / ( sqrt(v_t) + \epsilon )  <-- parameter update
        4.  \theta     = \theta - \alpha * \lambda * \theta                  <-- decoupled weight decay
        -----------------------------------------------------------------
        
        Args mirror PyTorch's AdamW for familiarity.
        
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
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        
        defaults = {
            "lr"           : lr, 
            "betas"        : betas, 
            "eps"          : eps, 
            "weight_decay" : weight_decay
        }
        
        super().__init__(params, defaults)
    
    
    @torch.no_grad()
    def step(
        self, 
        closure: Optional[Callable] = None
    ):
        """A single AdamW update step on model parameters."""
        
        loss = None if closure is None else closure()
        
        for group in self.param_groups:
            
            # Get the learning rate.
            lr:    float = group["lr"]
            beta1: float = group["betas"][0]
            beta2: float = group["betas"][1]
            eps:   float = group["eps"]
            lambd: float = group["weight_decay"]
            
            for p in group["params"]:
                
                # Skip if no gradients computed
                if p.grad is None:
                    continue
                
                # Skip if gradients are stored in a sparse format (only non-zero 
                # entries + their indices), instead of a full dense tensor.
                grad = p.grad.data
                if p.grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")
                
                # State initialization
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                m_t = state["exp_avg"]
                v_t = state["exp_avg_sq"]
                
                # Start with t=1
                state["step"] += 1
                t = state["step"]
                
                
                # 1. First and Second Moment Estimate update:
                # -------------------------------------------
                #    m_t = \beta_1 * m_{t-1} + (1 - \beta1) * g_t
                #    v_t = \beta_2 * v_{t-1} + (1 - \beta2) * g_t^2
                m_t = (beta1 * m_t) + ((1 - beta1) * grad)
                v_t = (beta2 * v_t) + ((1 - beta2) * grad * grad)
                
                
                # 2. Adjust \alpha for iteration \t:
                # -------------------------------------------
                #    \alpha_t = \alpha * sqrt( 1 - \beta2^t ) / (1 - \beta1^t)
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1
                
                
                # 3. Parameter update:
                # -------------------------------------------
                #    \theta = \theta - \alpha_t * m_t / ( sqrt(v_t) + \epsilon )
                denom = torch.sqrt(v_t) + eps
                p.data -= (step_size * (m_t / denom))
                
                
                # 4. Decoupled weight decay:
                # -------------------------------------------
                #    \theta = \theta - \alpha * \lambda * \theta
                if lambd != 0.0:
                    p.data -= (lr * lambd * p.data)
                
                
                # update state
                state["exp_avg"] = m_t
                state["exp_avg_sq"] = v_t
        
        return loss
