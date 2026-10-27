from pathlib import Path

import pytest

from config import DEFAULT_CONFIG, load_config

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def test_defaults_when_no_path():
    config = load_config()
    assert config["n_iter"] == 20
    assert config["games_per_iter"] == 50
    assert config["n_simulations"] == 200
    assert config["buffer_size"] == 50000


def test_normal_yaml_matches_agents_defaults():
    config = load_config(CONFIGS / "normal.yaml")
    assert config["rule_set"] == "normal"
    assert config["n_iter"] * config["games_per_iter"] == 1000
    assert config["n_simulations"] == 200
    assert config["c_puct"] == 1.5
    assert config["dirichlet_alpha"] == 0.15
    assert config["dirichlet_eps"] == 0.25
    assert config["temp_moves"] == 15
    assert config["train_steps_per_iter"] == 200
    assert config["batch_size"] == 128
    assert config["arena_win_threshold"] == 0.55
    assert config["learning_rate"] == pytest.approx(1e-3)
    assert config["optimizer"] == "adam"
    assert config["weight_decay"] == pytest.approx(1e-4)
    assert config["value_loss_weight"] == 1.0
    assert config["policy_loss_weight"] == 1.0
    assert config["lr_schedule"] == "cosine"


def test_renju_inherits_normal_and_overrides_rule_set():
    normal = load_config(CONFIGS / "normal.yaml")
    renju = load_config(CONFIGS / "renju.yaml")
    assert renju["rule_set"] == "renju"
    assert renju["n_simulations"] == normal["n_simulations"]
    assert renju["games_per_iter"] == normal["games_per_iter"]


def test_missing_keys_fall_back_to_defaults(tmp_path):
    path = tmp_path / "tiny.yaml"
    path.write_text("n_iter: 2\n", encoding="utf-8")
    config = load_config(path)
    assert config["n_iter"] == 2
    assert config["n_simulations"] == DEFAULT_CONFIG["n_simulations"]
    assert config["rule_set"] == "normal"
