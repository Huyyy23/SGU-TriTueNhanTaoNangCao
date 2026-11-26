import numpy as np
import pytest

from az.value_probe import value_metrics


def test_perfect_predictions_have_zero_error():
    predictions = np.array([1.0, -1.0, 1.0, -1.0])
    z = np.array([1.0, -1.0, 1.0, -1.0])
    mover_black = np.array([True, False, True, False])
    metrics = value_metrics(predictions, z, mover_black)
    assert metrics["positions"] == 4
    assert metrics["mse"] == pytest.approx(0.0)
    assert metrics["sign_accuracy"] == pytest.approx(1.0)
    assert metrics["confident_fraction"] == pytest.approx(1.0)
    assert metrics["mean_prediction_black_to_move"] == pytest.approx(1.0)
    assert metrics["mean_prediction_white_to_move"] == pytest.approx(-1.0)


def test_colour_shortcut_is_visible_in_the_means_and_low_accuracy():
    # The model answers "Black wins" no matter who moves; half the games went to White.
    z = np.array([1.0, -1.0, 1.0, -1.0])
    mover_black = np.array([True, False, True, False])
    predictions = np.array([1.0, -1.0, 1.0, -1.0])
    wrong = value_metrics(predictions, np.array([-1.0, 1.0, -1.0, 1.0]), mover_black)
    assert wrong["sign_accuracy"] == pytest.approx(0.0)
    assert wrong["mse"] == pytest.approx(4.0)
    assert wrong["mean_prediction_black_to_move"] > 0.9
    assert wrong["mean_prediction_white_to_move"] < -0.9

    correct = value_metrics(predictions, z, mover_black)
    assert correct["mean_prediction_black_to_move"] > 0.9
    assert correct["mean_prediction_white_to_move"] < -0.9


def test_confident_fraction_uses_the_threshold():
    metrics = value_metrics(
        np.array([0.5, -0.5, 0.999]), np.array([1.0, -1.0, 1.0]),
        np.array([True, False, True]), confident_threshold=0.9,
    )
    assert metrics["confident_fraction"] == pytest.approx(1 / 3)


def test_draws_are_excluded_from_sign_accuracy():
    metrics = value_metrics(
        np.array([1.0, 1.0]), np.array([0.0, -1.0]), np.array([True, False])
    )
    assert metrics["sign_accuracy"] == pytest.approx(0.0)


def test_empty_or_mismatched_input_is_rejected():
    with pytest.raises(ValueError, match="no positions"):
        value_metrics(np.array([]), np.array([]), np.array([], dtype=bool))
    with pytest.raises(ValueError, match="same length"):
        value_metrics(np.array([1.0]), np.array([1.0, -1.0]), np.array([True]))
