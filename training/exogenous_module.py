import torch
import torch.nn as nn
from torch.distributions import Beta
from typing import Tuple,Optional

class ExogenousVariableModule(nn.Module):
    def __init__(
        self,
        alpha0:float=0.1,
        beta0:float=0.1,
        epsilon: float=0.1,
    ):

        super().__init__()
        self.alpha0=alpha0
        self.beta0=beta0
        self.epsilon=epsilon


        self.register_buffer("alpha_v",torch.tensor(alpha0))
        self.register_buffer("beta_v",torch.tensor(beta0))
        self.register_buffer("alpha_q",torch.tensor(alpha0))
        self.register_buffer("beta_q",torch.tensor(beta0))

        self.register_buffer("n_updates",torch.tensor(0,dtype=torch.long))

    def update_posterior(
        self,
        logits:torch.Tensor,
        answer_idx:torch.Tensor,

    ) -> None:

        with torch.no_grad():
            preds=logits.argmax(dim=-1)
            correct=(preds==answer_idx).float()
            n_correct=correct.sum().item()
            n_incorrect=(1-correct).sum().item()

            self.alpha_v.add_(n_correct)
            self.beta_v.add_(n_incorrect)
            self.alpha_q.add_(n_correct)
            self.beta_q.add_(n_incorrect)
            self.n_updates.add_(1)

    def sample_u(
            self,
            batch_size: int,
            modality: str='vision',
            device: Optional[torch.device]=None,
            
        ) -> torch.Tensor:


            if modality=='vision':
                alpha,beta=self.alpha_v,self.beta_v
            else:
                alpha,beta=self.alpha_q,self.beta_q

            
            dist=Beta(alpha.clamp(min=1e-6),beta.clamp(min=1e-6))
            u=dist.sample((batch_size,))
            u=u.clamp(0.0,1.0-1e-6)

            if device is not None:
                u=u.to(device)

            return u

    def get_posterior_mean(
        self,
        modality:str='vision',
    ) -> float:

        if modality=='vision':
            alpha,beta=self.alpha_v,self.beta_v
        else:
            alpha,beta=self.alpha_q,self.beta_q
        return (alpha/(alpha+beta)).item()

    def reset_posterior(self) -> None:
 
        self.alpha_v.fill_(self.alpha0)
        self.beta_v.fill_(self.beta0)
        self.alpha_q.fill_(self.alpha0)
        self.beta_q.fill_(self.beta0)
        self.n_updates.fill_(0)
 
    def forward(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:

        u_v = self.sample_u(batch_size, modality="vision", device=device)
        u_q = self.sample_u(batch_size, modality="language", device=device)
        return u_v, u_q
 




