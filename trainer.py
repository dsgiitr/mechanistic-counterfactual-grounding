"""
CVL Training Loop.

Implements the alternating ERM / CRM training strategy from the paper
(Section 4.2): "In practice, we alternate between the conventional ERM
(i.e. when ω(θ) = 1) and the counterfactual risk."

Training algorithm:
  For each batch:
    1. Forward pass (factual) → logits, image_features, text_embeds
    2. Update Beta posterior p(u|D) based on predictions (Eq. 1)
    3. If this is a CRM step:
       a. Sample random batch for interpolation (uniform from dataset)
       b. Optimize u to find minimum-edit counterfactual (Eq. 3)
       c. Generate counterfactual embeddings (Eq. 2)
       d. Forward pass (counterfactual) → logits_cf
       e. Compute ω importance weights (Eq. 4)
       f. Compute CRM loss
    4. Otherwise: compute standard ERM loss
    5. Backprop and update θ
"""

import os
import time
import random
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import GradScaler
from transformers import get_scheduler
from tqdm.auto import tqdm

from llava_cvl import LLaVACVL
from erm_loss import ERMLoss
from crm_loss import CRMLoss
from evaluator import VQAEvaluator

logger = logging.getLogger(__name__)


class CVLTrainer:
    """
    Trainer for CVL-LLaVA.

    Alternates between:
      - ERM steps: standard VQA cross-entropy on factual data
      - CRM steps: counterfactual risk minimization with importance weighting
    """

    def __init__(
        self,
        model: LLaVACVL,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        train_dataset,  # needed for random sampling during CF generation
        config: Dict,
        device: torch.device,
        output_dir: str,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.train_dataset = train_dataset
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training config
        tcfg = config["training"]
        cfcfg = config["counterfactual"]
        self.num_epochs = tcfg["num_epochs"]
        self.grad_accum_steps = tcfg["gradient_accumulation_steps"]
        self.max_grad_norm = tcfg["max_grad_norm"]
        self.logging_steps = tcfg["logging_steps"]
        self.eval_steps = tcfg["eval_steps"]
        self.save_steps = tcfg["save_steps"]
        self.save_total_limit = tcfg.get("save_total_limit", 3)
        self.mixed_precision = tcfg.get("mixed_precision", "bf16")

        # CVL config
        self.alternate_every = cfcfg["alternate_every"]
        self.cf_prob = cfcfg.get("cf_prob", 1.0)
        self.lambda_cf = cfcfg["lambda_cf"]

        # Loss functions
        self.erm_loss = ERMLoss()
        self.crm_loss = CRMLoss(clip_M=cfcfg["clip_M"])

        # Optimizer
        self.optimizer = self._build_optimizer(tcfg)

        # LR scheduler
        total_steps = (
            len(train_dataloader) // self.grad_accum_steps * self.num_epochs
        )
        warmup_steps = int(total_steps * tcfg["warmup_ratio"])
        self.scheduler = get_scheduler(
            tcfg["lr_scheduler"],
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Mixed precision scaler (fp16 only; bf16 doesn't need it)
        self.use_amp = self.mixed_precision == "fp16"
        self.scaler = GradScaler(enabled=self.use_amp)

        # Evaluator
        self.evaluator = VQAEvaluator()

        # State
        self.global_step = 0
        self.best_val_acc = 0.0
        self._checkpoint_history = []

    def _build_optimizer(self, tcfg: Dict) -> torch.optim.Optimizer:
        # Don't apply weight decay to biases and LayerNorm
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        param_groups = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": tcfg["weight_decay"],
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        return torch.optim.AdamW(param_groups, lr=tcfg["learning_rate"])

    def _get_random_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a random mini-batch from the training dataset.
        Used as "x'" (negatives) for counterfactual interpolation.
        """
        indices = random.sample(range(len(self.train_dataset)), batch_size)
        samples = [self.train_dataset[i] for i in indices]
        # Re-use the same collator (processor is stored in the dataset's loader)
        # We access it through the dataloader's collate_fn
        collator = self.train_dataloader.collate_fn
        batch = collator(samples)
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

    def _is_crm_step(self) -> bool:
        """Determine if this step should use CRM instead of ERM."""
        if self.alternate_every <= 0:
            return False
        return (self.global_step % (self.alternate_every * 2)) >= self.alternate_every

    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Single training step — either ERM or CRM."""
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        pixel_values = batch["pixel_values"].to(self.device)
        answer_scores = batch["answer_scores"].to(self.device)
        answer_idx = batch["answer_idx"].to(self.device)

        use_crm = self._is_crm_step() and (random.random() < self.cf_prob)

        dtype = torch.bfloat16 if self.mixed_precision == "bf16" else torch.float16
        use_autocast = self.mixed_precision in ("fp16", "bf16")

        with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_autocast):
            if use_crm:
                # ── CRM step ──────────────────────────────────────────────
                # Get random batch for counterfactual interpolation
                rand_batch = self._get_random_batch(input_ids.size(0))

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    answer_idx=answer_idx,
                    run_counterfactual=True,
                    rand_input_ids=rand_batch["input_ids"],
                    rand_attention_mask=rand_batch["attention_mask"],
                    rand_pixel_values=rand_batch["pixel_values"],
                )

                logits = outputs["logits"]
                logits_cf = outputs["logits_cf"]
                omega = outputs["omega"]

                loss = self.crm_loss(
                    logits=logits,
                    logits_cf=logits_cf,
                    omega=omega,
                    answer_scores=answer_scores,
                )

                # Update Beta posterior based on this batch's predictions
                self.model.exogenous_module.update_posterior(
                    logits.detach(), answer_idx
                )

                step_info = {
                    "loss": loss.item(),
                    "loss_type": "crm",
                    "omega_mean": omega.mean().item(),
                    "omega_max": omega.max().item(),
                }

            else:
                # ── ERM step ──────────────────────────────────────────────
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                )

                logits = outputs["logits"]
                loss = self.erm_loss(logits, answer_scores)

                # Update posterior even on ERM steps
                self.model.exogenous_module.update_posterior(
                    logits.detach(), answer_idx
                )

                step_info = {
                    "loss": loss.item(),
                    "loss_type": "erm",
                }

        # Scale and backprop
        scaled_loss = loss / self.grad_accum_steps
        if self.use_amp:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        return step_info

    def optimizer_step(self) -> float:
        """Apply accumulated gradients."""
        if self.use_amp:
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )
            self.optimizer.step()

        self.scheduler.step()
        self.optimizer.zero_grad()
        return grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

    def train(self) -> None:
        """Full training loop."""
        logger.info(f"Starting CVL training for {self.num_epochs} epochs")
        logger.info(f"Steps per epoch: {len(self.train_dataloader)}")
        logger.info(f"Total steps: {len(self.train_dataloader) * self.num_epochs}")

        has_device_map = (
            hasattr(self.model, "llava")
            and hasattr(self.model.llava, "hf_device_map")
            and self.model.llava.hf_device_map is not None
        )
        if has_device_map:
            logger.info("Detected device-mapped LLaVA model; skipping explicit model.to(device)")
        else:
            self.model.to(self.device)
        self.optimizer.zero_grad()

        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_start = time.time()
            running_losses = {"erm": [], "crm": []}

            progress = tqdm(
                enumerate(self.train_dataloader),
                total=len(self.train_dataloader),
                desc=f"Epoch {epoch + 1}/{self.num_epochs}",
                leave=True,
            )

            for step, batch in progress:
                step_info = self.train_step(batch)

                loss_type = step_info["loss_type"]
                running_losses[loss_type].append(step_info["loss"])

                # Optimizer step
                if (step + 1) % self.grad_accum_steps == 0:
                    grad_norm = self.optimizer_step()
                    self.global_step += 1

                    progress.set_postfix(
                        loss=f"{step_info['loss']:.4f}",
                        mode=step_info["loss_type"],
                        step=self.global_step,
                    )

                    # Logging
                    if self.global_step % self.logging_steps == 0:
                        lr = self.scheduler.get_last_lr()[0]
                        erm_avg = (
                            sum(running_losses["erm"]) / len(running_losses["erm"])
                            if running_losses["erm"] else 0
                        )
                        crm_avg = (
                            sum(running_losses["crm"]) / len(running_losses["crm"])
                            if running_losses["crm"] else 0
                        )

                        log_str = (
                            f"Step {self.global_step} | "
                            f"ERM loss: {erm_avg:.4f} | "
                            f"CRM loss: {crm_avg:.4f} | "
                            f"LR: {lr:.2e} | "
                            f"GradNorm: {grad_norm:.3f}"
                        )
                        if "omega_mean" in step_info:
                            log_str += f" | ω_mean: {step_info['omega_mean']:.3f}"

                        logger.info(log_str)
                        running_losses = {"erm": [], "crm": []}

                    # Evaluation
                    if self.global_step % self.eval_steps == 0:
                        val_metrics = self.evaluate()
                        val_acc = val_metrics["vqa_accuracy"]
                        logger.info(
                            f"Step {self.global_step} | Val VQA Acc: {val_acc:.4f}"
                        )

                        if val_acc > self.best_val_acc:
                            self.best_val_acc = val_acc
                            self.save_checkpoint("best_model")
                            logger.info(f"New best model! VQA Acc: {val_acc:.4f}")

                        # Log posterior state
                        u_v_mean = self.model.exogenous_module.get_posterior_mean("vision")
                        u_q_mean = self.model.exogenous_module.get_posterior_mean("language")
                        logger.info(
                            f"Beta posterior means — u_v: {u_v_mean:.4f}, u_q: {u_q_mean:.4f}"
                        )

                        self.model.train()

                    # Periodic save
                    if self.global_step % self.save_steps == 0:
                        self.save_checkpoint(f"checkpoint-{self.global_step}")

            epoch_time = time.time() - epoch_start
            logger.info(f"Epoch {epoch+1}/{self.num_epochs} done in {epoch_time:.1f}s")

        # Final save
        self.save_checkpoint("final_model")
        logger.info(f"Training complete. Best val acc: {self.best_val_acc:.4f}")

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation on the validation set."""
        self.model.eval()
        all_predictions = []
        all_targets = []

        eval_progress = tqdm(
            self.val_dataloader,
            total=len(self.val_dataloader),
            desc="Validation",
            leave=False,
        )

        for batch in eval_progress:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            pixel_values = batch["pixel_values"].to(self.device)
            answer_scores = batch["answer_scores"]

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
            )

            logits = outputs["logits"]
            preds = logits.argmax(dim=-1).cpu()

            all_predictions.extend(preds.tolist())
            all_targets.extend(answer_scores.tolist())

        metrics = self.evaluator.compute_accuracy(all_predictions, all_targets)
        return metrics

    def save_checkpoint(self, name: str) -> None:
        """Save model checkpoint."""
        ckpt_dir = self.output_dir / name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model state dict (LoRA weights + VQA head + CVL modules)
        state = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_val_acc": self.best_val_acc,
            "exogenous_alpha_v": self.model.exogenous_module.alpha_v,
            "exogenous_beta_v": self.model.exogenous_module.beta_v,
            "exogenous_alpha_q": self.model.exogenous_module.alpha_q,
            "exogenous_beta_q": self.model.exogenous_module.beta_q,
        }
        torch.save(state, ckpt_dir / "checkpoint.pt")

        # Manage checkpoint rotation
        self._checkpoint_history.append(str(ckpt_dir))
        while len(self._checkpoint_history) > self.save_total_limit:
            old = self._checkpoint_history.pop(0)
            if "best_model" not in old and "final_model" not in old:
                import shutil
                shutil.rmtree(old, ignore_errors=True)

        logger.info(f"Saved checkpoint to {ckpt_dir}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load a checkpoint to resume training."""
        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.scheduler.load_state_dict(state["scheduler_state_dict"])
        self.global_step = state["global_step"]
        self.best_val_acc = state["best_val_acc"]

        # Restore posterior state
        self.model.exogenous_module.alpha_v = state["exogenous_alpha_v"]
        self.model.exogenous_module.beta_v = state["exogenous_beta_v"]
        self.model.exogenous_module.alpha_q = state["exogenous_alpha_q"]
        self.model.exogenous_module.beta_q = state["exogenous_beta_q"]

        logger.info(f"Resumed from step {self.global_step}")