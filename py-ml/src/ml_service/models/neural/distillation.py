import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class DistillationLoss(nn.Module):
    """Knowledge distillation loss function."""
    
    def __init__(self, temperature: float = 3.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        
    def forward(self,
                student_logits: torch.Tensor,
                teacher_logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the knowledge distillation loss.
        Combines soft targets from teacher and hard targets from ground truth.
        """
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_prob = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        soft_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean')
        hard_loss = F.cross_entropy(student_logits, targets)
        
        return (self.alpha * soft_loss * (self.temperature ** 2) + 
                (1 - self.alpha) * hard_loss) 