"""Custom implementation of Cosine Annealing LR Scheduler."""

import math



class CosineAnnealingScheduler:
    
    def __init__(
        self,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters: int,
    ):
        """
        Args:
            max_learning_rate (float): alpha_max, the maximum learning rate for
                cosine learning rate schedule (with warmup).
            min_learning_rate (float): alpha_min, the minimum / final learning rate for
                the cosine learning rate schedule (with warmup).
            warmup_iters (int): T_w, the number of iterations to linearly warm-up
                the learning rate.
            cosine_cycle_iters (int): T_c, the number of cosine annealing iterations,
                (a.k.a the TOTAL number of training iters, inc. warmup).
        """
        
        if max_learning_rate <= 0:
            raise ValueError("max_learning_rate must be > 0.")
        if min_learning_rate < 0:
            raise ValueError("min_learning_rate must be >= 0.")
        if min_learning_rate > max_learning_rate:
            raise ValueError("min_learning_rate cannot exceed max_learning_rate.")
        if warmup_iters < 0 or cosine_cycle_iters < 0:
            raise ValueError("warmup_iters and cosine_cycle_iters must be >= 0.")
        if cosine_cycle_iters < warmup_iters:
            raise ValueError("cosine_cycle_iters (T_c) must be >= warmup_iters (T_w).")
        
        self.alpha_max = float(max_learning_rate)
        self.alpha_min = float(min_learning_rate)
        self.Tw = int(warmup_iters)
        self.Tc = int(cosine_cycle_iters)
        
        
        # ---- precompute schedule for t in [0, Tc] ----
        self.lr_table = [0.0] * (self.Tc + 1)
        for t in range(self.Tc + 1):
            
            # (Warm-up)
            # -------------------------
            #    \alpha_t = (t / T_w) * \alpha_max
            if self.Tw > 0 and t < self.Tw:
                lr = (t / self.Tw) * self.alpha_max
            
            
            # (Cosine annealing)
            # -------------------------
            #    \alpha_t = \alpha_min + 0.5 * (1 + cos( (t - T_w)/(T_c - T_w) * \pi )) * (\alpha_max - \alpha_min)
            else:
                # Handle the degenerate case T_c == T_w
                denom = max(1, self.Tc - self.Tw)
                progress = (t - self.Tw) / denom
                lr = self.alpha_min + 0.5 * (1.0 + math.cos(progress * math.pi)) * (self.alpha_max - self.alpha_min)
            
            self.lr_table[t] = lr
    
    
    def get_lr(self, t: int) -> float:
        """
        Given an iteration number `t`, return the learning rate at the 
        given iteration.
        
        Args:
            it (int): Iteration number to get learning rate for.
        
        Returns:
            Learning rate at the given iteration.
        """
        
        if t < 0:
            raise ValueError("t must be >= 0")
        
        if t <= self.Tc:
            return self.lr_table[t]
        
        # Post Annealing
        return self.alpha_min
