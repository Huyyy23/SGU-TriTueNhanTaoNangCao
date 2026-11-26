"""Value-head diagnostics: compare predictions with recorded game outcomes.

Training loss is dominated by the policy term and the value head can memorise a
degenerate dataset, so the stored `value_loss` is not evidence that the evaluator
generalises. These helpers score a checkpoint on real positions and report the
metrics that expose the colour shortcut: sign accuracy and the mean prediction per
moving side.
"""

from __future__ import annotations

import numpy as np


def value_metrics(
    predictions: np.ndarray,
    z: np.ndarray,
    mover_black: np.ndarray,
    *,
    confident_threshold: float = 0.99,
) -> dict[str, float | int]:
    """Summarise `predictions` against the side-to-move labels `z` in {-1, +1}."""
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    mover_black = np.asarray(mover_black, dtype=bool).reshape(-1)
    if not (predictions.shape == z.shape == mover_black.shape):
        raise ValueError("predictions, z and mover_black must have the same length")
    if predictions.size == 0:
        raise ValueError("no positions to score")

    decided = z != 0
    sign_accuracy = (
        float((np.sign(predictions[decided]) == np.sign(z[decided])).mean())
        if decided.any()
        else 0.0
    )
    black = mover_black
    white = ~mover_black
    return {
        "positions": int(predictions.size),
        "mse": float(((predictions - z) ** 2).mean()),
        "mean_abs_error": float(np.abs(predictions - z).mean()),
        "sign_accuracy": sign_accuracy,
        "confident_fraction": float((np.abs(predictions) > confident_threshold).mean()),
        "mean_prediction_black_to_move": float(predictions[black].mean()) if black.any() else 0.0,
        "mean_prediction_white_to_move": float(predictions[white].mean()) if white.any() else 0.0,
    }
