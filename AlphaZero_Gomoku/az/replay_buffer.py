from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
import warnings

import numpy as np
import torch

from gomoku.features import N_ACTIONS, N_PLANES, N_SYMMETRIES, SIZE, symmetry_transform

RULE_SET_IDS: dict[str, int] = {"normal": 0, "renju": 1}

def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()



class ReplayBuffer:
    """Fixed-capacity sample store. One rule set per buffer."

    Samples are stored before augmentation; `sample` applies one random symmetry per draw.
    """

    def __init__(self, capacity: int = 50000) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = int(capacity)
        self.features = np.zeros((self.capacity, N_PLANES, SIZE, SIZE), dtype=np.uint8)
        self.pi = np.zeros((self.capacity, N_ACTIONS), dtype=np.float16)
        self.z = np.zeros(self.capacity, dtype=np.float32)
        self.rule_set = np.zeros(self.capacity, dtype=np.uint8)
        self.game_ids = np.full(self.capacity, "", dtype="U64")
        self._size = 0
        self._cursor = 0
        self._label: int | None = None
        self._source_sha256: str | None = None
        self._cursor_known = True
        # Fixed eval batch (features, pi, z) persisted next to the buffer so the metric survives resume.
        self.eval_batch: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self.eval_indices: np.ndarray | None = None

    def __len__(self) -> int:
        """Number of stored samples, before augmentation."""
        return self._size

    def add(self, features: np.ndarray, pi: np.ndarray, z: float, rule_set: str, game_id: str = "") -> None:
        """Store one sample, overwriting the oldest when full."""
        if rule_set not in RULE_SET_IDS:
            raise ValueError(f"Unknown rule_set '{rule_set}'. Expected one of {sorted(RULE_SET_IDS)}")
        label = RULE_SET_IDS[rule_set]
        if self._label is not None and label != self._label:
            raise ValueError("Cannot mix rule sets in one buffer (hard rule 7).")

        features = np.asarray(features, dtype=np.uint8)
        if features.shape != (N_PLANES, SIZE, SIZE):
            raise ValueError(f"features must have shape {(N_PLANES, SIZE, SIZE)}, got {features.shape}")
        pi = np.asarray(pi, dtype=np.float32)
        if pi.shape != (N_ACTIONS,):
            raise ValueError(f"pi must have shape {(N_ACTIONS,)}, got {pi.shape}")
        if not isinstance(game_id, str) or len(game_id) > 64:
            raise ValueError("game_id must be a string of at most 64 characters.")

        self.features[self._cursor] = features
        self.pi[self._cursor] = pi
        self.z[self._cursor] = float(z)
        self.rule_set[self._cursor] = label
        self.game_ids[self._cursor] = game_id
        self._source_sha256 = None
        self._label = label
        self._cursor = (self._cursor + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        rule_set: str,
        rng: np.random.Generator | None = None,
        return_indices: bool = False,
        exclude_indices=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
        """Draw a batch with one random symmetry per sample, applied to features and pi together.

        The source indices are returned as a fourth element when `return_indices` is set.
        `exclude_indices` keeps the fixed eval samples out of training draws.
        """
        if self._size == 0:
            raise ValueError("Cannot sample from an empty buffer.")
        if rule_set not in RULE_SET_IDS:
            raise ValueError(f"Unknown rule_set '{rule_set}'.")
        if self._label != RULE_SET_IDS[rule_set]:
            raise ValueError(f"Buffer holds rule set id {self._label}, not '{rule_set}'.")

        rng = rng if rng is not None else np.random.default_rng()
        indices = rng.integers(0, self._size, size=batch_size)
        if exclude_indices is not None:
            blocked = {int(i) for i in np.asarray(exclude_indices).reshape(-1) if int(i) < self._size}
            if len(blocked) < self._size:
                for i in range(batch_size):
                    while int(indices[i]) in blocked:
                        indices[i] = rng.integers(0, self._size)
        symmetries = rng.integers(0, N_SYMMETRIES, size=batch_size)

        features = np.empty((batch_size, N_PLANES, SIZE, SIZE), dtype=np.float32)
        pi = np.empty((batch_size, N_ACTIONS), dtype=np.float32)
        for i, (index, k) in enumerate(zip(indices, symmetries)):
            features[i] = symmetry_transform(self.features[index], k)
            pi[i] = symmetry_transform(self.pi[index].astype(np.float32), k)

        batch = (
            torch.from_numpy(features),
            torch.from_numpy(pi),
            torch.from_numpy(self.z[indices]),
        )
        return (*batch, indices) if return_indices else batch

    def save_npz(self, path: str | Path, extra: dict | None = None) -> None:
        """Write the stored samples (and any extra arrays) to `path`. Local disk only (ADR-0002)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "features": self.features[: self._size],
            "pi": self.pi[: self._size],
            "z": self.z[: self._size],
            "rule_set": self.rule_set[: self._size],
            "game_ids": self.game_ids[: self._size],
            "capacity": np.array(self.capacity),
            "cursor": np.array(self._cursor),
        }
        arrays.update(extra or {})
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".npz", delete=False) as stream:
            temp_path = Path(stream.name)
        try:
            np.savez(temp_path, **arrays)
            os.replace(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)

    @classmethod
    def load_npz(cls, path: str | Path) -> ReplayBuffer:
        """Read a buffer written by `save_npz`, including the fixed eval batch when present."""
        with np.load(path) as data:
            buffer = cls(capacity=int(data["capacity"]))
            n = int(data["z"].shape[0])
            buffer.features[:n] = data["features"]
            buffer.pi[:n] = data["pi"]
            buffer.z[:n] = data["z"]
            buffer.rule_set[:n] = data["rule_set"]
            if "game_ids" in data.files:
                ids = data["game_ids"]
                if ids.shape != (n,) or ids.dtype.kind not in "US" or any(len(str(x)) > 64 for x in ids):
                    raise ValueError("Invalid replay game identities.")
                buffer.game_ids[:n] = ids
            buffer._size = n
            if "cursor" in data.files:
                buffer._cursor = int(data["cursor"])
                if not 0 <= buffer._cursor < buffer.capacity or (n < buffer.capacity and buffer._cursor != n):
                    raise ValueError("Invalid replay buffer cursor.")
            else:
                buffer._cursor = n % buffer.capacity
                if n == buffer.capacity:
                    buffer._cursor_known = False
                    warnings.warn("Legacy full replay buffer has unknown oldest sample; overwrite order may differ.", stacklevel=2)
            if n:
                buffer._label = int(data["rule_set"][0])
            if "eval_features" in data.files:
                buffer.eval_batch = (data["eval_features"], data["eval_pi"], data["eval_z"])
            if "eval_indices" in data.files:
                buffer.eval_indices = data["eval_indices"]
        buffer._source_sha256 = _file_sha256(path)
        return buffer
