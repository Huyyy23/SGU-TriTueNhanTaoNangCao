from __future__ import annotations

import numpy as np
import torch

from gomoku.features import N_ACTIONS, N_PLANES, N_SYMMETRIES, SIZE, symmetry_transform

RULE_SET_IDS: dict[str, int] = {"normal": 0, "renju": 1}


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
        self._size = 0
        self._cursor = 0
        self._label: int | None = None

    def __len__(self) -> int:
        """Number of stored samples, before augmentation."""
        return self._size

    def add(self, features: np.ndarray, pi: np.ndarray, z: float, rule_set: str) -> None:
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

        self.features[self._cursor] = features
        self.pi[self._cursor] = pi
        self.z[self._cursor] = float(z)
        self.rule_set[self._cursor] = label
        self._label = label
        self._cursor = (self._cursor + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        rule_set: str,
        rng: np.random.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Draw a batch with one random symmetry per sample, applied to features and pi together."""
        if self._size == 0:
            raise ValueError("Cannot sample from an empty buffer.")
        if rule_set not in RULE_SET_IDS:
            raise ValueError(f"Unknown rule_set '{rule_set}'.")
        if self._label != RULE_SET_IDS[rule_set]:
            raise ValueError(f"Buffer holds rule set id {self._label}, not '{rule_set}'.")

        rng = rng if rng is not None else np.random.default_rng()
        indices = rng.integers(0, self._size, size=batch_size)
        symmetries = rng.integers(0, N_SYMMETRIES, size=batch_size)

        features = np.empty((batch_size, N_PLANES, SIZE, SIZE), dtype=np.float32)
        pi = np.empty((batch_size, N_ACTIONS), dtype=np.float32)
        for i, (index, k) in enumerate(zip(indices, symmetries)):
            features[i] = symmetry_transform(self.features[index], k)
            pi[i] = symmetry_transform(self.pi[index].astype(np.float32), k)

        return (
            torch.from_numpy(features),
            torch.from_numpy(pi),
            torch.from_numpy(self.z[indices]),
        )
