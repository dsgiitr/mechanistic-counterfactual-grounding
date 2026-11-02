import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple ,Dict,Optional 


class CounterFactualGenerator(nn.Module):
    def __init__(
        self,
        epsilon:float=0.1,
        cf_grad_steps:int=5,
        cf_lr:float=0.01,
        lambda_cf:float=0.5,
        clip_M: float=10.0,
        intervene_vision: bool=True,
        intervene_language: bool=True,
    ):
        super().__init__()
        self.epsilon=epsilon
        self.cf_grad_steps=cf_grad_steps
        self.cf_lr=cf_lr
        self.lambda_cf=lambda_cf
        self.clip_M=clip_M
        self.intervene_vision=intervene_vision
        self.intervene_language=intervene_language


    def interpolate_embeddings(
        self,
        e_orig:torch.Tensor,
        e_random:torch.Tensor,
        u:torch.Tensor,

    ) -> torch.Tensor:


        threshold=1-self.epsilon

        if e_orig.shape != e_random.shape:
            raise ValueError("Factual and donor embeddings must have the same shape")
        u_exp = u.to(device=e_orig.device, dtype=e_orig.dtype)
        while u_exp.ndim < e_orig.ndim:
            u_exp = u_exp.unsqueeze(-1)
        factual_mask = u_exp >= threshold

        interpolated=u_exp*e_orig + (1.0-u_exp)*e_random

        e_cf=torch.where(factual_mask, e_orig, interpolated)

        return e_cf

    
    def optimize_u_for_intervention(
        self,
        e_v:torch.Tensor,
        e_q:torch.Tensor,
        e_v_rand:torch.Tensor,
        e_q_rand:torch.Tensor,
        u_v_init: torch.Tensor,
        u_q_init:torch.Tensor,
        answer_idx:torch.Tensor,
        fusion_fn,
    )  -> Tuple[torch.Tensor,torch.Tensor]:
    
        e_v_d=e_v.detach()
        e_q_d=e_q.detach()
        e_v_rand_d=e_v_rand.detach()
        e_q_rand_d=e_q_rand.detach()


        u_v=u_v_init.clone().detach().requires_grad_(self.intervene_vision)
        u_q=u_q_init.detach().clone().requires_grad_(self.intervene_language)

        params=[]
        if self.intervene_vision and u_v.requires_grad:
            params.append(u_v)

        if self.intervene_language and u_q.requires_grad:
            params.append(u_q)


        if not params:
            return u_v_init, u_q_init


        optimizer=torch.optim.Adam(params,lr=self.cf_lr)


        for _ in range(self.cf_grad_steps):
            optimizer.zero_grad()


        
            # Use continuous interpolation during search. The hard factual gate
            # is applied by generate(), otherwise initial u above the threshold
            # would have no gradient from the prediction objective.
            def mix(original, donor, u):
                weight = u.clamp(0, 1-1e-6).to(original)
                while weight.ndim < original.ndim:
                    weight = weight.unsqueeze(-1)
                return weight * original + (1-weight) * donor
            e_v_cf = mix(e_v_d,e_v_rand_d,u_v) if self.intervene_vision else e_v_d
            e_q_cf = mix(e_q_d,e_q_rand_d,u_q) if self.intervene_language else e_q_d


            logits_cf=fusion_fn(e_v_cf,e_q_cf)
            log_probs=F.log_softmax(logits_cf,dim=-1)

            log_p_correct=log_probs.gather(
                1,answer_idx.unsqueeze(1)
               ).squeeze(1)



            u_norm_sq=(u_v**2).mean() + (u_q**2).mean()
            loss=-u_norm_sq + self.lambda_cf*log_p_correct.mean()

            # Inner optimization must not accumulate gradients on VLM weights.
            grads = torch.autograd.grad(loss, params)
            for param, grad in zip(params, grads):
                param.grad = grad
            optimizer.step()

            with torch.no_grad():
                if self.intervene_vision:
                    u_v.clamp_(0.0,1.0-1e-6)
                
                if self.intervene_language:
                    u_q.clamp_(0.0,1.0-1e-6)


        return u_v.detach(),u_q.detach()


    
    def generate(
        self,
        e_v: torch.Tensor,
        e_q: torch.Tensor,
        e_v_rand: torch.Tensor,
        e_q_rand: torch.Tensor,
        u_v: torch.Tensor,
        u_q: torch.Tensor,
    ) -> Tuple[torch.Tensor,torch.Tensor]:



        e_v_cf=e_v
        e_q_cf=e_q



        if self.intervene_vision:
            e_v_cf=self.interpolate_embeddings(e_v,e_v_rand,u_v)

        if self.intervene_language:
            e_q_cf=self.interpolate_embeddings(e_q,e_q_rand,u_q)

        
        return e_v_cf,e_q_cf
