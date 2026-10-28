from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from az.model import load_checkpoint, save_checkpoint
from az.replay_buffer import ReplayBuffer


class Trainer:
    """Loss, optimizer, cosine schedule and checkpoints for the AlphaZero net."""

    def __init__(self, model, cfg: dict, buffer: ReplayBuffer | None = None, device=None, seed: int = 0) -> None:
        self.cfg = cfg
        self.buffer = buffer
        self.seed = int(seed)
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        # Adam's weight_decay is the L2 term in the loss, so it is not added twice.
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(cfg["learning_rate"]),
            weight_decay=float(cfg["weight_decay"]),
        )
        self.rng = np.random.default_rng(self.seed)
        self.global_step = 0
        self.n_iter = 0
        self.eval_batch: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self.eval_indices: np.ndarray | None = None

    @property
    def total_steps(self) -> int:
        return int(self.cfg["n_iter"]) * int(self.cfg["train_steps_per_iter"])

    def _total_loss(self, logits: torch.Tensor, values: torch.Tensor, pi: torch.Tensor, z: torch.Tensor):
        policy_loss = -(pi * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        value_loss = F.mse_loss(values.reshape(-1), z.reshape(-1))
        loss = self.cfg["policy_loss_weight"] * policy_loss + self.cfg["value_loss_weight"] * value_loss
        return loss, policy_loss, value_loss

    def _set_lr(self) -> None:
        """Cosine schedule indexed by global_step, so resume continues the curve instead of restarting it."""
        base = float(self.cfg["learning_rate"])
        total = self.total_steps
        if total <= 0 or self.cfg.get("lr_schedule") != "cosine":
            lr = base
        else:
            progress = min(self.global_step, total) / total
            lr = base * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def train_on_batch(self, features, pi, z) -> dict[str, float]:
        """Run one optimizer step on the given batch."""
        self._set_lr()
        self.model.train()
        x = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        target_pi = torch.as_tensor(pi, dtype=torch.float32, device=self.device)
        target_z = torch.as_tensor(z, dtype=torch.float32, device=self.device)

        logits, values = self.model(x)
        loss, policy_loss, value_loss = self._total_loss(logits, values, target_pi, target_z)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.global_step += 1
        return {
            "loss": float(loss.detach()),
            "policy_loss": float(policy_loss.detach()),
            "value_loss": float(value_loss.detach()),
        }

    def train_steps(self, n_steps: int | None = None) -> dict[str, float]:
        """Draw `n_steps` batches from the buffer and train on each.

        `distinct_samples` counts the buffer slots actually drawn, so early iterations are not
        reported as if they used `n_steps * batch_size` different samples.
        """
        if self.buffer is None or len(self.buffer) == 0:
            raise ValueError("Cannot train: the buffer is empty.")
        n_steps = int(self.cfg["train_steps_per_iter"] if n_steps is None else n_steps)
        if n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {n_steps}")

        totals = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}
        used: set[int] = set()
        for _ in range(n_steps):
            features, pi, z, indices = self.buffer.sample(
                self.cfg["batch_size"],
                self.cfg["rule_set"],
                self.rng,
                return_indices=True,
                exclude_indices=self.eval_indices,
            )
            used.update(int(i) for i in indices)
            step = self.train_on_batch(features.numpy(), pi.numpy(), z.numpy())
            for key in totals:
                totals[key] += step[key]

        return {key: value / n_steps for key, value in totals.items()} | {"distinct_samples": len(used)}

    def freeze_eval_batch(self, size: int | None = None) -> None:
        """Freeze the metric batch once per run and keep it out of every training draw."""
        if self.buffer is None or len(self.buffer) == 0:
            raise ValueError("Cannot freeze the eval batch: the buffer is empty.")
        size = int(self.cfg["eval_fixed_batch_size"] if size is None else size)
        rng = np.random.default_rng(self.seed)
        indices = rng.integers(0, len(self.buffer), size=size)
        self.eval_indices = np.unique(indices)
        self.eval_batch = (
            self.buffer.features[indices].astype(np.float32),
            self.buffer.pi[indices].astype(np.float32),
            self.buffer.z[indices].astype(np.float32),
        )

    def loss_on_fixed_batch(self) -> float:
        """Loss on the frozen batch. The only number used to claim the loss decreases."""
        if self.eval_batch is None:
            raise ValueError("No frozen eval batch; call freeze_eval_batch() first.")
        features, pi, z = self.eval_batch
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                x = torch.as_tensor(features, dtype=torch.float32, device=self.device)
                logits, values = self.model(x)
                loss, _, _ = self._total_loss(
                    logits,
                    values,
                    torch.as_tensor(pi, dtype=torch.float32, device=self.device),
                    torch.as_tensor(z, dtype=torch.float32, device=self.device),
                )
                return float(loss)
        finally:
            self.model.train(was_training)

    def save(self, checkpoint_path: str | Path, buffer_path: str | Path | None = None) -> None:
        """Write `last.pt` and, when checkpoint_buffer is on, `buffer_last.npz` beside it."""
        meta = {
            "n_iter": self.n_iter,
            "global_step": self.global_step,
            "rule_set": self.cfg["rule_set"],
            "config": dict(self.cfg),
        }
        save_checkpoint(self.model, checkpoint_path, meta=meta, optimizer=self.optimizer)

        if buffer_path is None or self.buffer is None:
            return
        if not self.cfg.get("checkpoint_buffer", True):
            return
        extra = {}
        if self.eval_batch is not None:
            features, pi, z = self.eval_batch
            extra = {"eval_features": features, "eval_pi": pi, "eval_z": z}
        if self.eval_indices is not None:
            extra["eval_indices"] = np.asarray(self.eval_indices)
        self.buffer.save_npz(buffer_path, extra=extra)

    @classmethod
    def load(cls, checkpoint_path: str | Path, cfg: dict, buffer: ReplayBuffer | None = None, device=None, seed: int = 0) -> Trainer:
        """Restore a trainer: weights, optimizer state, global step and iteration counter."""
        model, meta = load_checkpoint(checkpoint_path, device=device)
        trainer = cls(model, cfg, buffer=buffer, device=device, seed=seed)

        state = torch.load(checkpoint_path, map_location=trainer.device, weights_only=False).get("optimizer_state_dict")
        if state is not None:
            trainer.optimizer.load_state_dict(state)
        trainer.global_step = int(meta.get("global_step", 0))
        trainer.n_iter = int(meta.get("n_iter", 0))
        if buffer is not None and buffer.eval_batch is not None:
            trainer.eval_batch = buffer.eval_batch
        if buffer is not None and buffer.eval_indices is not None:
            trainer.eval_indices = buffer.eval_indices
        return trainer
