from __future__ import annotations

from copy import deepcopy
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import TASKS
from evaluation.metrics import aggregate_metrics
from .losses import blend_weight, mixup, normalized_mse


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    y_true: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    model.eval()
    preds = []
    for batch in loader:
        batch = [tensor.cuda(non_blocking=True) if torch.cuda.is_available() else tensor for tensor in batch]
        inputs = batch[:4] if len(batch) == 4 else batch[:4]
        outputs = model(*inputs[:3])
        preds.append(outputs.cpu().numpy())
    preds = np.vstack(preds)
    denorm = preds * target_std + target_mean
    return aggregate_metrics(y_true, denorm, TASKS)


class DistillationTrainer:
    def __init__(self, model: torch.nn.Module, config: Dict[str, float]) -> None:
        self.model = model
        self.config = config

    def _make_optimizer(self) -> torch.optim.Optimizer:
        params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            params,
            lr=self.config["lr"],
            weight_decay=self.config["wd"],
        )

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        target_mean: np.ndarray,
        target_std: np.ndarray,
        y_val: np.ndarray,
        max_epochs: int,
        patience: int,
    ) -> Tuple[float, int, Dict[str, Dict[str, float]]]:
        self.model.gate.logits.requires_grad_(False)
        if hasattr(self.model, "log_sigma2"):
            self.model.log_sigma2.requires_grad_(False)

        optimizer = self._make_optimizer()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=max(1, patience // 3),
            min_lr=1e-5,
        )

        best_metric = -1e9
        best_epoch = -1
        best_state: Optional[Dict[str, torch.Tensor]] = None
        stale = 0

        for epoch in range(1, max_epochs + 1):
            self.model.train()
            total_loss = []
            alpha = blend_weight(
                epoch,
                max_epochs,
                self.config["mix_alpha_start"],
                self.config["mix_alpha_end"],
                self.config["mix_turn"],
            )
            for batch in train_loader:
                batch = [tensor.cuda(non_blocking=True) if torch.cuda.is_available() else tensor for tensor in batch]
                mix_inputs, _ = mixup(alpha, *batch)
                xv, xx, xl, y = mix_inputs[:4]
                if len(mix_inputs) > 4:
                    y_teacher, y_v, y_x, y_l = mix_inputs[4:]
                    preds, aux = self.model(xv, xx, xl, return_aux=True)
                    kd_loss = torch.mean(normalized_mse(preds, y_teacher))
                    feat_loss = torch.mean(
                        (
                            normalized_mse(aux["V"], y_v)
                            + normalized_mse(aux["X"], y_x)
                            + normalized_mse(aux["L"], y_l)
                        )
                        / 3.0
                    )
                else:
                    preds = self.model(xv, xx, xl)
                    kd_loss = torch.tensor(0.0, device=preds.device)
                    feat_loss = torch.tensor(0.0, device=preds.device)
                gt_loss = torch.mean(normalized_mse(preds, y))
                loss = (
                    self.config["gt_w"] * gt_loss
                    + self.config["out_kd_w"] * kd_loss
                    + self.config["feat_kd_w"] * feat_loss
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.config["clip"],
                )
                optimizer.step()
                total_loss.append(float(loss.item()))

            metrics = evaluate_model(self.model, val_loader, target_mean, target_std, y_val)
            mean_r2 = metrics["mean"]["R2"]
            scheduler.step(mean_r2)
            if mean_r2 > best_metric + 1e-4:
                best_metric = mean_r2
                best_epoch = epoch
                best_state = deepcopy({k: v.cpu() for k, v in self.model.state_dict().items()})
                stale = 0
            else:
                stale += 1
            print(
                f"[KD] epoch {epoch:03d}/{max_epochs} "
                f"| alpha={alpha:.3f} | loss={np.mean(total_loss):.6f} "
                f"| mean R2={mean_r2:.4f} | best={best_metric:.4f} "
                f"| patience={stale}/{patience}"
            )
            if stale >= patience:
                print("Early stopping.")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state, strict=True)
        final_metrics = evaluate_model(self.model, val_loader, target_mean, target_std, y_val)
        return best_metric, best_epoch, final_metrics
