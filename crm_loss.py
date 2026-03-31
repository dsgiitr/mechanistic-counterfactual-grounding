import torch
import torch.nn as nn
import torch.nn.functional as F 

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
        answer_idx:torch.Tensor,

    ) -> torch.Tensor:

        loss_per_sample=F.binary_cross_entropy_with_logits(
            logits,
            answer_scores,
            reduction='none',

        ).mean(dim=-1)
        

        return (loss_per_sample*omega.detach()).mean()


    def compute_omega(
        self,
        logits:torch.Tensor,
        logits_cf:torch.Tensor,
        answer_idx: torch.Tensor,

    ) -> torch.Tensor:

        with torch.no_grad():
            log_p_factual=F.log_softmax(logits,dim=-1)
            log_p_cf=F.log_softmax(logits_cf,dim=-1)

            log_p_a_factual=log_p_factual.gather(
                1,answer_idx.unsqueeze(1)
           ).squeeze(1)

            log_p_a_cf=log_p_cf.gather(
            1,answer_idx.unsqueeze(1)
           ).squeeze(1)


            log_omega=log_p_a_factual-log_p_a_cf
            omega=torch.exp(log_omega).clamp(max=self.clip_M)

            return omega