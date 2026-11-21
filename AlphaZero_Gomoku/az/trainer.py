from __future__ import annotations

import math
import json
import os
import re
import tempfile
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from az.model import load_checkpoint, save_checkpoint
from az.replay_buffer import ReplayBuffer, _file_sha256

def _sync_file(path: Path) -> None:
    with path.open("r+b") as stream:
        os.fsync(stream.fileno())


def _sync_directory(path: Path) -> None:
    if os.name != "nt":
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _write_pointer(path: Path, generation: str) -> None:
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump({"generation": generation}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def resolve_generation(pointer: str | Path, generation: str | None = None) -> tuple[Path, Path]:
    """Validate an immutable generation selected by a pointer or explicit ID."""
    pointer = Path(pointer)
    if pointer.is_symlink():
        raise ValueError("Invalid generation pointer.")
    current = json.loads(pointer.read_text(encoding="utf-8"))["generation"]
    target = generation if generation is not None else current
    if not isinstance(target, str) or not re.fullmatch(r"[0-9a-f]{32}", target):
        raise ValueError("Invalid generation ID.")
    root = pointer.parent.resolve()
    seen = set()
    while True:
        if not isinstance(current, str) or not re.fullmatch(r"[0-9a-f]{32}", current) or current in seen:
            raise ValueError("Invalid generation ID or ancestry.")
        seen.add(current)
        directory = root / "generations" / current
        if directory.resolve() != directory or directory.is_symlink():
            raise ValueError("Invalid generation path.")
        manifest_path = directory / "manifest.json"
        checkpoint = directory / "checkpoint.pt"
        replay = directory / "replay.npz"
        if any(path.is_symlink() for path in (manifest_path, checkpoint, replay)):
            raise ValueError("Invalid generation file path.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("generation") != current:
            raise ValueError("Invalid generation manifest.")
        if current == target:
            break
        current = manifest.get("commit_previous", manifest.get("previous"))
        if current is None:
            raise ValueError("Generation is not in committed history.")
    if (manifest.get("generation") != current or
            manifest.get("checkpoint_sha256") != _file_sha256(checkpoint) or
            manifest.get("replay_sha256") != _file_sha256(replay)):
        raise ValueError("Generation hash mismatch.")
    return checkpoint, replay


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
        self.promotion_eval_games = 0
        self.eval_batch: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self.bootstrap = None
        self.run_id: str | None = None
        self.lineage = None
        self.eval_indices: np.ndarray | None = None
        self._generation_checkpoint: Path | None = None

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

    def train_steps(self, n_steps: int | None = None) -> dict:
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

        game_ids = {str(self.buffer.game_ids[i]) for i in used if self.buffer.game_ids[i]}
        return ({key: value / n_steps for key, value in totals.items()}
                | {"distinct_samples": len(used), "identified_games_used": len(game_ids),
                   "identified_game_ids_used": sorted(game_ids),
                   "unidentified_samples_used": sum(1 for i in used if not self.buffer.game_ids[i])})

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

    def save(self, pointer_path: str | Path) -> Path:
        """Commit a new immutable checkpoint/replay generation via an atomic pointer."""
        if self.buffer is None or not self.cfg.get("checkpoint_buffer", True):
            raise ValueError("A replay buffer is required for a generation.")
        pointer_path = Path(pointer_path)
        pointer_path.parent.mkdir(parents=True, exist_ok=True)
        if pointer_path.suffix != ".json" or pointer_path.is_symlink():
            raise ValueError("Generation pointer must be a JSON file, not a symlink.")
        generations = pointer_path.parent / "generations"
        generations.mkdir(exist_ok=True)
        if generations.is_symlink():
            raise ValueError("Generation directory must not be a symlink.")
        previous = None
        if pointer_path.exists():
            previous = json.loads(pointer_path.read_text(encoding="utf-8"))["generation"]
            if not isinstance(previous, str) or not re.fullmatch(r"[0-9a-f]{32}", previous):
                raise ValueError("Invalid previous generation ID.")
        commit_previous = previous
        if (previous is not None and self._generation_checkpoint is not None
                and self._generation_checkpoint.parent.parent == generations.resolve()):
            source_id = self._generation_checkpoint.parent.name
            source, _ = resolve_generation(pointer_path, source_id)
            if source.resolve() != self._generation_checkpoint:
                raise ValueError("Loaded checkpoint does not match the generation.")
            previous = source_id
        generation = uuid.uuid4().hex
        directory = generations / generation
        directory.mkdir()
        checkpoint_path = directory / "checkpoint.pt"
        buffer_path = directory / "replay.npz"
        meta = {
            "n_iter": self.n_iter,
            "global_step": self.global_step,
            "rule_set": self.cfg["rule_set"],
            "promotion_eval_games": self.promotion_eval_games,
            "rng_state": self.rng.bit_generator.state,
            "config": dict(self.cfg),
            "bootstrap": self.bootstrap,
            "run_id": self.run_id,
            "lineage": self.lineage,
            "recovery": getattr(self, "recovery", None),
        }
        extra = {}
        if self.eval_batch is not None:
            features, pi, z = self.eval_batch
            extra = {"eval_features": features, "eval_pi": pi, "eval_z": z}
        if self.eval_indices is not None:
            extra["eval_indices"] = np.asarray(self.eval_indices)
        self.buffer.save_npz(buffer_path, extra=extra)
        _sync_file(buffer_path)
        meta["replay_sha256"] = _file_sha256(buffer_path)
        save_checkpoint(self.model, checkpoint_path, meta=meta, optimizer=self.optimizer)
        _sync_file(checkpoint_path)
        manifest = {"generation": generation, "previous": previous,
                    "checkpoint_sha256": _file_sha256(checkpoint_path),
                    "parent_checkpoint_sha256": (_file_sha256(self._generation_checkpoint)
                                                 if self._generation_checkpoint is not None else None),
                    "replay_sha256": meta["replay_sha256"]}
        if commit_previous != previous:
            manifest["commit_previous"] = commit_previous
        with (directory / "manifest.json").open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream)
            stream.flush()
            os.fsync(stream.fileno())
        _sync_directory(directory)
        _sync_directory(generations)
        _write_pointer(pointer_path, generation)
        self._generation_checkpoint = checkpoint_path.resolve()
        return checkpoint_path

    @classmethod
    def from_pretrained(cls, checkpoint_path: str | Path, cfg: dict, buffer: ReplayBuffer, device=None, seed: int = 0) -> Trainer:
        """Fine-tune weights from another rule set with fresh optimizer and training state."""
        model, meta = load_checkpoint(checkpoint_path, device=device)
        if meta.get("rule_set") not in (None, "normal"):
            raise ValueError("fine-tuning source checkpoint must use Normal rules")
        if cfg.get("rule_set") != "renju":
            raise ValueError("fine-tuning target config must use Renju rules")
        if len(buffer):
            raise ValueError("fine-tuning requires an empty Renju replay buffer")
        if (model.n_blocks, model.n_filters) != (int(cfg["n_blocks"]), int(cfg["n_filters"])):
            raise ValueError("source checkpoint architecture does not match target config")
        return cls(model, cfg, buffer=buffer, device=device, seed=seed)

    @classmethod
    def bootstrap_normal(cls, checkpoint_path: str | Path, cfg: dict, buffer: ReplayBuffer,
                         optimizer: str, device=None, seed: int = 0) -> Trainer:
        """Start a Normal run from weights, without importing historical replay."""
        if optimizer not in ("reset", "restore"):
            raise ValueError("bootstrap optimizer must be reset or restore")
        model, meta = load_checkpoint(checkpoint_path, device=device)
        if meta.get("rule_set") != "normal" or cfg.get("rule_set") != "normal":
            raise ValueError("bootstrap requires Normal source and target rules")
        if (model.n_blocks, model.n_filters) != (int(cfg["n_blocks"]), int(cfg["n_filters"])):
            raise ValueError("source checkpoint architecture does not match target config")
        if len(buffer):
            raise ValueError("bootstrap requires an empty replay buffer")
        trainer = cls(model, cfg, buffer=buffer, device=device, seed=seed)
        if optimizer == "restore":
            source_cfg = meta.get("config", {})
            keys = ("learning_rate", "weight_decay", "lr_schedule", "n_iter", "train_steps_per_iter")
            if any(key not in source_cfg or source_cfg[key] != cfg.get(key) for key in keys):
                raise ValueError("restored optimizer requires matching learning rate schedule and horizon")
            step = meta.get("global_step")
            if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step < trainer.total_steps:
                raise ValueError("restored optimizer requires a valid global_step within the schedule horizon")
            state = torch.load(checkpoint_path, map_location=trainer.device, weights_only=False).get("optimizer_state_dict")
            if not isinstance(state, dict) or not state.get("state") or not state.get("param_groups"):
                raise ValueError("restored optimizer requires populated optimizer state")
            trainer.optimizer.load_state_dict(state)
            if any(group.get("weight_decay") != float(cfg["weight_decay"])
                   for group in trainer.optimizer.param_groups):
                raise ValueError("restored optimizer weight decay does not match the schedule config")
            states = trainer.optimizer.state
            parameters = [parameter for group in trainer.optimizer.param_groups for parameter in group["params"]]
            if len(states) != len(parameters):
                raise ValueError("restored optimizer state is incomplete")
            for parameter in parameters:
                entry = states[parameter]
                if (not all(key in entry for key in ("step", "exp_avg", "exp_avg_sq"))
                        or int(entry["step"]) != step
                        or entry["exp_avg"].shape != parameter.shape
                        or entry["exp_avg_sq"].shape != parameter.shape
                        or not torch.isfinite(entry["exp_avg"]).all()
                        or not torch.isfinite(entry["exp_avg_sq"]).all()):
                    raise ValueError("restored optimizer state is invalid or inconsistent with global_step")
            steps_per_iteration = int(cfg["train_steps_per_iter"])
            if step % steps_per_iteration:
                raise ValueError("restored optimizer global_step must be at an iteration boundary")
            trainer.n_iter = step // steps_per_iteration
            trainer.global_step = step
        trainer.bootstrap = {
            "parent_checkpoint_sha256": _file_sha256(Path(checkpoint_path)),
            "parent_config": meta.get("config", {}),
            "optimizer": optimizer,
            "parent_global_step": meta.get("global_step"),
        }
        return trainer

    @classmethod
    def load(cls, checkpoint_path: str | Path, cfg: dict, buffer: ReplayBuffer | None = None, device=None, seed: int = 0) -> Trainer:
        """Restore a trainer: weights, optimizer state, global step and iteration counter."""
        model, meta = load_checkpoint(checkpoint_path, device=device)
        if buffer is not None:
            if not buffer._cursor_known:
                raise ValueError("Cannot resume: replay buffer cursor is unknown.")
            if not meta.get("replay_sha256") or meta["replay_sha256"] != buffer._source_sha256:
                raise ValueError("Cannot resume: checkpoint and replay identity is missing or mismatched.")
        trainer = cls(model, cfg, buffer=buffer, device=device, seed=seed)
        source_cfg = meta.get("config", {})
        keys = ("rule_set", "n_blocks", "n_filters", "learning_rate", "weight_decay",
                "lr_schedule", "n_iter", "train_steps_per_iter")
        if any(key not in source_cfg or source_cfg[key] != cfg.get(key) for key in keys):
            raise ValueError("Cannot resume: checkpoint training configuration differs.")
        if any(type(meta.get(key)) is not int or meta[key] < 0 for key in ("n_iter", "global_step")):
            raise ValueError("Cannot resume: checkpoint iteration or global step is invalid.")
        trainer._generation_checkpoint = Path(checkpoint_path).resolve()

        state = torch.load(checkpoint_path, map_location=trainer.device, weights_only=False).get("optimizer_state_dict")
        if not isinstance(state, dict) or not state.get("param_groups"):
            raise ValueError("Cannot resume: optimizer state is missing.")
        if meta.get("global_step", 0) and not state.get("state"):
            raise ValueError("Cannot resume: optimizer state is incomplete.")
        if not meta.get("rng_state"):
            raise ValueError("Cannot resume: RNG state is missing.")
        trainer.optimizer.load_state_dict(state)
        trainer.n_iter = int(meta.get("n_iter", 0))
        trainer.global_step = int(meta.get("global_step", 0))
        trainer.bootstrap = meta.get("bootstrap")
        trainer.run_id = meta.get("run_id")
        trainer.lineage = meta.get("lineage")
        trainer.recovery = meta.get("recovery")
        trainer.promotion_eval_games = int(meta.get("promotion_eval_games", 0))
        if meta.get("rng_state") is not None:
            trainer.rng.bit_generator.state = meta["rng_state"]
        if buffer is not None and buffer.eval_batch is not None:
            trainer.eval_batch = buffer.eval_batch
        if buffer is not None and buffer.eval_indices is not None:
            trainer.eval_indices = buffer.eval_indices

        return trainer
