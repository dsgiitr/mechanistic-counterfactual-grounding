import torch
import torch.nn as nn
import torch.nn.functional as F

class ERMLoss(nn.Module):
    def __init__(self,reduction: str='mean'):
        super().__init__()
        self.reduction=reduction

    def forward(
        self,
        logits: torch.Tensor,
        answer_scores:torch.Tensor,

    ) -> torch.Tensor:

        loss=F.binary_cross_entropy_with_logits(
            logits,
            answer_scores,
            reduction=self.reduction
        )

        return loss