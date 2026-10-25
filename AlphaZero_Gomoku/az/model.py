from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CHECKPOINT_FORMAT_VERSION = 1


class ResBlock(nn.Module):
    """Residual block with two 3x3 conv layers and batch norm."""

    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        return self.relu2(out)


class AZNet(nn.Module):
    """AlphaZero dual-head ResNet for 11x11 Gomoku."""

    def __init__(self, n_blocks: int = 6, n_filters: int = 64) -> None:
        super().__init__()
        self.n_blocks = n_blocks
        self.n_filters = n_filters

        # Input block: (B, 4, 11, 11) -> (B, n_filters, 11, 11)
        self.conv_in = nn.Conv2d(4, n_filters, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(n_filters)
        self.relu_in = nn.ReLU(inplace=True)

        # Residual tower
        self.blocks = nn.ModuleList([ResBlock(n_filters) for _ in range(n_blocks)])

        # Policy head: Conv 1x1 (2 filters) -> BN -> ReLU -> Flatten -> Linear(2*11*11, 121)
        self.policy_conv = nn.Conv2d(n_filters, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_relu = nn.ReLU(inplace=True)
        self.policy_fc = nn.Linear(2 * 11 * 11, 121)

        # Value head: Conv 1x1 (1 filter) -> BN -> ReLU -> Flatten -> Linear(1*11*11, 64) -> ReLU -> Linear(64, 1) -> Tanh
        self.value_conv = nn.Conv2d(n_filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_relu = nn.ReLU(inplace=True)
        self.value_fc1 = nn.Linear(1 * 11 * 11, 64)
        self.value_fc_relu = nn.ReLU(inplace=True)
        self.value_fc2 = nn.Linear(64, 1)
        self.value_tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.relu_in(self.bn_in(self.conv_in(x)))
        for block in self.blocks:
            out = block(out)

        # Policy logits (B, 121)
        p = self.policy_relu(self.policy_bn(self.policy_conv(out)))
        p = torch.flatten(p, start_dim=1)
        logits = self.policy_fc(p)

        # Value (B, 1) in [-1, 1]
        v = self.value_relu(self.value_bn(self.value_conv(out)))
        v = torch.flatten(v, start_dim=1)
        v = self.value_fc_relu(self.value_fc1(v))
        values = self.value_tanh(self.value_fc2(v))

        return logits, values

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(features, np.ndarray):
            raise TypeError(f"features must be a numpy ndarray, got {type(features)}")
        if features.ndim != 4 or features.shape[1:] != (4, 11, 11):
            raise ValueError(f"features must have shape (B, 4, 11, 11), got {features.shape}")

        device = next(self.parameters()).device
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                x = torch.from_numpy(features).to(device=device, dtype=torch.float32)
                logits, values = self.forward(x)
                probs = F.softmax(logits, dim=1)
                return (
                    probs.detach().cpu().numpy().astype(np.float32),
                    values.squeeze(-1).detach().cpu().numpy().astype(np.float32),
                )
        finally:
            self.train(was_training)


def save_checkpoint(
    model: AZNet,
    path: str | Path,
    meta: dict | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_data = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "meta": {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "n_iter": 0,
            "rule_set": None,
            "config": {},
            **(meta or {}),
            # Architecture always comes from the model, never from caller meta.
            "n_blocks": model.n_blocks,
            "n_filters": model.n_filters,
        },
    }
    torch.save(checkpoint_data, path)


def load_checkpoint(
    path: str | Path,
    device: str | torch.device | None = None,
) -> tuple[AZNet, dict]:
    path = Path(path)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    checkpoint_data = torch.load(path, map_location=device, weights_only=False)
    meta = checkpoint_data.get("meta", {})
    # Older checkpoints may lack these keys; keep the six-key contract on load.
    meta.setdefault("format_version", CHECKPOINT_FORMAT_VERSION)
    meta.setdefault("n_iter", 0)
    meta.setdefault("rule_set", None)
    meta.setdefault("config", {})
    n_blocks = meta.get("n_blocks", 6)
    n_filters = meta.get("n_filters", 64)

    model = AZNet(n_blocks=n_blocks, n_filters=n_filters)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.to(device)
    model.eval()
    return model, meta
