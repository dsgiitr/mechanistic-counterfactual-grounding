import torch
import torch.nn as nn
importtorch.nn.functional as F 

class CRMLoss(nn.Module):
    def __init__(self,clip_M: float=10.0, lambda_cf: float=0.5):
        super().__init__()
        self.clip_M=clip_M
        self.lambda_cf=lambda_cf

    def forward(
        self,
        logits: torch.Tensor,
        logits_cf:torch.Tensor,
        omega:torch.Tensor,
        answer_scores:torch.Tensor,
        answer_idx:torch.Tnesor,

    ) -> torch.Tensor:

        loss_per_sample=F.binary_cross_entropy_with_logits(
            logits,
            answer_scores,
            reduction='none',

        ).mean(dim=-1)


        weighted_loss=(loss_per_sample*omega).mean()


        loss_cf=F.cross_entropy(logits_cf,answer_idx)

        total_loss=weighted_loss+self.lambda*loss_cf
        return total_loss

