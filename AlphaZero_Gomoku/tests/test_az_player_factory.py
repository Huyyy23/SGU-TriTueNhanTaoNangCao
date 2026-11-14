from az.player_factory import wrap_az_player
from az.player_tactical import TacticalAZPlayer
from az.threat_space import ThreatSpaceAZPlayer


class BasePlayer:
    rules = object()


def test_wrap_az_player_returns_base_when_overrides_disabled():
    base = BasePlayer()

    assert wrap_az_player(base, {}) is base


def test_wrap_az_player_uses_tactical_wrapper_when_enabled():
    player = wrap_az_player(BasePlayer(), {"tactical": {"enabled": True}})

    assert isinstance(player, TacticalAZPlayer)
    assert not isinstance(player, ThreatSpaceAZPlayer)


def test_wrap_az_player_uses_threat_wrapper_and_keeps_tactical_enabled():
    player = wrap_az_player(
        BasePlayer(),
        {"tactical": {"enabled": True}, "threat_space": {"enabled": True}},
    )

    assert isinstance(player, ThreatSpaceAZPlayer)
    assert player.enabled
    assert player.threat_enabled


def test_three_variants_share_checkpoint_and_simulation_budget(monkeypatch):
    from players.factory import create_player
    from gomoku.rules import get_rules
    import players.az_player as module

    loaded = []
    monkeypatch.setattr(module, "load_checkpoint", lambda path, device=None: (loaded.append(path) or object(), {}))
    rules = get_rules("normal")
    bases = []
    for variant in ("az_v0", "az_v1_tactical", "az_v1_threat"):
        player = create_player(variant, rules, checkpoint="same.pt", n_simulations=7)
        bases.append(getattr(player, "base", player))
    assert loaded == ["same.pt"] * 3
    assert [base.cfg["n_simulations"] for base in bases] == [7, 7, 7]


def test_time_budget_runs_beyond_configured_simulation_ceiling(monkeypatch):
    import numpy as np
    import players.az_player as module
    from gomoku.board import Board, SIZE
    from gomoku.rules import get_rules

    class Model:
        def predict(self, planes):
            return np.full((1, SIZE * SIZE), 1.0 / (SIZE * SIZE)), np.zeros(1)

    monkeypatch.setattr(module, "load_checkpoint", lambda *args, **kwargs: (Model(), {}))
    ticks = iter(i / 10 for i in range(100))
    monkeypatch.setattr(module.time, "monotonic", lambda: next(ticks))
    player = module.AZPlayer("same.pt", get_rules("normal"), cfg={"n_simulations": 1}, time_budget_sec=0.5)
    board = Board()
    before = board.grid.copy()
    move = player.select_move(board)
    assert player.simulations_run > 1
    assert player.nn_calls == player.simulations_run
    assert player.nodes_visited is None
    assert move[0] * SIZE + move[1] in player.rules.legal_moves(board)
    assert np.array_equal(board.grid, before)
    assert not board.history
    player.time_budget_sec = 0
    player.select_move(board)
    assert player.simulations_run == player.nn_calls == player.nodes_visited == 0
