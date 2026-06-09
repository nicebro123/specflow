"""Training loop for static and dual-graph SpecFlow configurations."""

import math
from typing import Dict, Iterable, List, Mapping, Optional

import torch
from tqdm.auto import tqdm

from specflow.flow.flow_matching import ControlAnchoredFlowMatching
from specflow.flow.mmd_loss import MMDLoss
from specflow.flow.ode_solver import EulerSampler
from specflow.training.ema import EMA


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    total_steps: int,
    warmup_steps: int = 0,
    eta_min: float = 1e-6,
):
    if scheduler_name == "none":
        return None
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=eta_min
    )
    if warmup_steps <= 0:
        return cosine
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-2, end_factor=1.0, total_iters=warmup_steps
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
    )


class SpecFlowTrainer:
    """Optimize flow matching with static or perturbation-conditioned spectra."""

    def __init__(
        self,
        model: torch.nn.Module,
        spectral_embedding,
        sigma: float = 0.5,
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-5,
        grad_clip: float = 1.0,
        mmd_weight: float = 0.0,
        mmd_interval: int = 10,
        mmd_steps: int = 8,
        delta_corr_weight: float = 0.0,
        device: Optional[torch.device] = None,
        ema_decay: float = 0.0,
        scheduler: str = "none",
        max_epochs: int = 100,
        warmup_epochs: int = 0,
        eta_min: float = 1e-6,
        use_amp: bool = False,
        show_progress: bool = True,
        scheduler_interval: str = "epoch",
        ot_coupling: bool = False,
    ) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.spectral_embedding = self._move_spectral(spectral_embedding)
        self.flow = ControlAnchoredFlowMatching(sigma=sigma, ot_coupling=ot_coupling)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.grad_clip = grad_clip
        if mmd_weight < 0 or mmd_interval < 1:
            raise ValueError("mmd_weight must be non-negative and mmd_interval positive")
        self.mmd_weight = mmd_weight
        self.mmd_interval = mmd_interval
        self.mmd_loss = MMDLoss()
        self.mmd_sampler = EulerSampler(self.model, sigma=sigma, n_steps=mmd_steps)
        if delta_corr_weight < 0:
            raise ValueError("delta_corr_weight must be non-negative")
        self.delta_corr_weight = delta_corr_weight
        self._training_step = 0

        self.use_amp = use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.show_progress = show_progress
        if scheduler_interval not in {"epoch", "step"}:
            raise ValueError("scheduler_interval must be 'epoch' or 'step'")
        self.scheduler_interval = scheduler_interval

        self.ema = EMA(self.model, decay=ema_decay) if ema_decay > 0 else None
        self.scheduler = _build_scheduler(
            self.optimizer, scheduler, max_epochs, warmup_epochs, eta_min
        )

    def _move_spectral(self, spectral_source):
        if callable(spectral_source):
            return spectral_source
        if torch.is_tensor(spectral_source):
            return spectral_source.to(self.device)
        if isinstance(spectral_source, Mapping):
            return {
                name: values.to(self.device) for name, values in spectral_source.items()
            }
        raise ValueError("spectral_embedding must be a tensor, mapping, or callable")

    def _spectral_for_batch(self, batch: Dict[str, object]):
        source = self.spectral_embedding
        if callable(source):
            return source(batch["pert_mask"])
        return source

    def _move_batch(self, batch: Dict[str, object]) -> Dict[str, object]:
        return {
            name: value.to(self.device) if torch.is_tensor(value) else value
            for name, value in batch.items()
        }

    def current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    @staticmethod
    def _condition_groups(conditions, batch_size: int, device: torch.device):
        if conditions is None:
            return [torch.arange(batch_size, device=device)]
        if torch.is_tensor(conditions):
            keys = conditions.detach().cpu().tolist()
        else:
            keys = list(conditions)
        if len(keys) != batch_size:
            return [torch.arange(batch_size, device=device)]
        groups = {}
        for idx, key in enumerate(keys):
            groups.setdefault(str(key), []).append(idx)
        return [
            torch.tensor(indices, device=device, dtype=torch.long)
            for indices in groups.values()
        ]

    @staticmethod
    def _pearson_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        predicted_centered = predicted - predicted.mean()
        target_centered = target - target.mean()
        denom = (predicted_centered.norm() * target_centered.norm()).clamp_min(1e-8)
        corr = torch.dot(predicted_centered, target_centered) / denom
        return 1.0 - corr.clamp(min=-1.0, max=1.0)

    def _delta_correlation_loss(
        self,
        predicted_velocity: torch.Tensor,
        target_delta: torch.Tensor,
        conditions,
    ) -> torch.Tensor:
        groups = self._condition_groups(
            conditions, predicted_velocity.shape[0], predicted_velocity.device
        )
        losses = []
        for indices in groups:
            pred_mean = predicted_velocity.index_select(0, indices).mean(dim=0)
            target_mean = target_delta.index_select(0, indices).mean(dim=0)
            losses.append(self._pearson_loss(pred_mean, target_mean))
        return torch.stack(losses).mean()

    def _progress(self, iterable, description: str):
        if not self.show_progress:
            return iterable
        total = len(iterable) if hasattr(iterable, "__len__") else None
        return tqdm(iterable, total=total, desc=description, leave=False)

    def train_batch(self, raw_batch: Dict[str, object]) -> float:
        self.model.train()
        batch = self._move_batch(raw_batch)
        spectral = self._spectral_for_batch(batch)
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            output = self.flow.compute_loss(
                self.model, batch, spectral
            )
            loss = output.loss
            if self.delta_corr_weight > 0:
                # pearson_delta is computed from mean perturbation residuals:
                # pred_mean - ctrl_mean vs. true_mean - ctrl_mean. Align the
                # batch-level residual directly, using the noise-free target
                # delta rather than the flow target x_1 - (ctrl + sigma eps).
                with torch.amp.autocast(self.device.type, enabled=False):
                    delta_loss = self._delta_correlation_loss(
                        output.prediction.float(),
                        output.target_delta.float(),
                        batch.get("condition"),
                    )
                loss = loss + self.delta_corr_weight * delta_loss.to(loss.dtype)
            if self.mmd_weight > 0 and self._training_step % self.mmd_interval == 0:
                predicted = self.mmd_sampler.sample_with_grad(
                    batch["ctrl_expr"], batch["pert_mask"], spectral
                )
                loss = loss + self.mmd_weight * self.mmd_loss(
                    predicted, batch["pert_expr"]
                )
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.ema is not None:
            self.ema.update()
        self._training_step += 1
        if self.scheduler is not None and self.scheduler_interval == "step":
            self.scheduler.step()
        return float(loss.detach().cpu())

    def train_epoch(
        self,
        loader: Iterable[Dict[str, object]],
        epoch: Optional[int] = None,
        max_epochs: Optional[int] = None,
    ) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        if epoch is not None and max_epochs is not None:
            description = f"train {epoch}/{max_epochs}"
        else:
            description = "train"
        progress = self._progress(loader, description)
        for raw_batch in progress:
            loss = self.train_batch(raw_batch)
            total_loss += loss
            n_batches += 1
            if self.show_progress and hasattr(progress, "set_postfix"):
                progress.set_postfix(loss=f"{total_loss / n_batches:.4f}", lr=f"{self.current_lr():.2e}")
        if n_batches == 0:
            raise ValueError("training loader yielded no batches")
        if self.scheduler is not None and self.scheduler_interval == "epoch":
            self.scheduler.step()
        return total_loss / n_batches

    @torch.no_grad()
    def evaluate_loss(
        self,
        loader: Iterable[Dict[str, object]],
        epoch: Optional[int] = None,
        max_epochs: Optional[int] = None,
    ) -> float:
        self.model.eval()
        ctx = self.ema.shadow_context() if self.ema is not None else _nullcontext()
        with ctx:
            total_loss = 0.0
            n_batches = 0
            if epoch is not None and max_epochs is not None:
                description = f"val {epoch}/{max_epochs}"
            else:
                description = "val"
            progress = self._progress(loader, description)
            for raw_batch in progress:
                batch = self._move_batch(raw_batch)
                spectral = self._spectral_for_batch(batch)
                output = self.flow.compute_loss(
                    self.model, batch, spectral
                )
                total_loss += float(output.loss.cpu())
                n_batches += 1
                if self.show_progress and hasattr(progress, "set_postfix"):
                    progress.set_postfix(loss=f"{total_loss / n_batches:.4f}")
        if n_batches == 0:
            raise ValueError("evaluation loader yielded no batches")
        return total_loss / n_batches

    def fit(
        self,
        train_loader: Iterable[Dict[str, object]],
        n_epochs: int,
        validation_loader: Optional[Iterable[Dict[str, object]]] = None,
    ) -> List[Dict[str, float]]:
        history = []
        for epoch in range(n_epochs):
            metrics = {
                "epoch": float(epoch + 1),
                "train_loss": self.train_epoch(train_loader, epoch + 1, n_epochs),
                "lr": self.current_lr(),
            }
            if validation_loader is not None:
                metrics["validation_loss"] = self.evaluate_loss(
                    validation_loader, epoch + 1, n_epochs
                )
            history.append(metrics)
        return history


class _nullcontext:
    def __enter__(self):
        return None
    def __exit__(self, *args):
        pass
