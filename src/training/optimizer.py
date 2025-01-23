import torch
from torch.optim import AdamW
from typing import Dict, Any, List

class QuantumOptimizer:
    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_grad_norm: float = 1.0
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        
        # Initialize optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.current_step = 0
        
    def step(
        self,
        loss: torch.Tensor,
        scaler: Optional[torch.cuda.amp.GradScaler] = None
    ) -> Dict[str, Any]:
        """
        Perform optimization step with learning rate scheduling.
        """
        # Increment step counter
        self.current_step += 1
        
        # Calculate learning rate
        if self.current_step < self.warmup_steps:
            lr_scale = min(1., float(self.current_step) / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate * lr_scale
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm
            )
            scaler.step(self.optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm
            )
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        
        return {
            "loss": loss.item(),
            "learning_rate": self.optimizer.param_groups[0]['lr']
        } 