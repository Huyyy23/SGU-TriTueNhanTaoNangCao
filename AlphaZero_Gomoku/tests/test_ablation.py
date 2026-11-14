"""T32: ablation script on tiny inputs (fake agents for MCTS, crowded boards for Alpha-Beta)."""

import json

import pytest

from gomoku.board import SIZE
from scripts import ablation as A


class FirstLegal:
    def __init__(self, rules, seed=None):
        self.rules = rules

    def select_move(self, board):
        return divmod(min(self.rules.legal_moves(board)), SIZE)


class SeededRandom:
    def __init__(self, rules, seed=None):
        import random

        self.rules = rules
        self.rng = random.Random(seed)
        self.n_rollouts = 0
        self.simulations_run = 0

    def select_move(self, board):
        self.simulations_run = self.n_rollouts
        return divmod(self.rng.choice(self.rules.legal_moves(board)), SIZE)


def fake_builder(settings, rules, seed):
    agent = settings["factory"]["agent"]
    player = SeededRandom(rules, seed) if agent == "mcts-pure" else FirstLegal(rules, seed)
    if agent == "mcts-pure":
        player.n_rollouts = settings["factory"]["rollouts"]
    return player


def quiet(_msg):
    pass


def test_mcts_levels_come_from_parameters(tmp_path):
    path, payload = A.run_mcts(sims=[3, 9], games=4, opponent="heuristic", seed=1, run_id="m",
                               out_root=tmp_path, builder=fake_builder, log=quiet)
    assert path == tmp_path / "m.json"
    on_disk = json.loads(path.read_text())
    assert on_disk["config"]["sims"] == [3, 9]
    assert on_disk["config"]["games_per_level"] == 4
    assert [r["sims"] for r in on_disk["results"]] == [3, 9]
    for level in on_disk["results"]:
        assert level["games"] == 4
        assert level["games_black"] == level["games_white"] == 2
        assert level["wins"] + level["draws"] + level["losses"] == 4
        low, high = level["wilson95"]
        assert 0.0 <= low <= level["score"] <= high <= 1.0
        assert level["mcts"]["rollouts_per_move"] == level["sims"]
    assert len(on_disk["games"]) == 8
    assert on_disk["seed"] == 1



def test_mcts_real_player_reports_completed_search(tmp_path):
    _, payload = A.run_mcts(sims=[1], games=2, opponent="random", seed=7,
                            run_id="real", out_root=tmp_path, log=quiet)
    level = payload["results"][0]
    assert level["games_black"] == level["games_white"] == 1
    assert level["illegal_moves"] == 0
    assert 0 <= level["mcts"]["rollouts_per_move"] <= 1
    assert level["mcts"]["mean_nodes_per_move"] >= 0
    assert all("times" not in stats for game in payload["games"] for stats in game["stats"].values())


@pytest.mark.parametrize("ruleset", ["normal", "renju"])
def test_positions_repeat_and_remain_legal(ruleset):
    from gomoku.board import Board
    from gomoku.rules import get_rules

    positions = A.make_positions(ruleset, 2, 8, seed=11)
    assert positions == A.make_positions(ruleset, 2, 8, seed=11)
    rules = get_rules(ruleset)
    for moves in positions:
        board = Board()
        assert len(moves) == 8
        for move in moves:
            assert not rules.is_terminal(board)
            assert move in rules.legal_moves(board)
            board.play(*divmod(move, SIZE))
        assert not rules.is_terminal(board)

def test_mcts_rejects_odd_games_and_existing_file(tmp_path):
    with pytest.raises(ValueError):
        A.run_mcts(sims=[3], games=3, out_root=tmp_path, builder=fake_builder, log=quiet)
    A.run_mcts(sims=[3], games=2, run_id="same", out_root=tmp_path, builder=fake_builder, log=quiet)
    with pytest.raises(FileExistsError):
        A.run_mcts(sims=[3], games=2, run_id="same", out_root=tmp_path, builder=fake_builder, log=quiet)


def test_alphabeta_pruning_on_off_same_depth(tmp_path):
    positions = A.make_positions("normal", 1, 100, seed=0)
    path, payload = A.run_alphabeta(depths=[1, 2], positions=positions, seed=0, run_id="ab",
                                    out_root=tmp_path, position_plies=100, log=quiet)
    rows = {r["depth"]: r for r in payload["results"]}
    # Pruning never changes the chosen move, only the work done.
    assert all(r["same_move"] for r in payload["results"])
    # Depth 1: nothing to cut, both searches visit the same nodes.
    assert rows[1]["nodes_on"] == rows[1]["nodes_off"]
    # Depth 2: pruning visits fewer nodes.
    assert rows[2]["nodes_on"] < rows[2]["nodes_off"]
    assert json.loads(path.read_text())["summary"][1]["depth"] == 2


@pytest.mark.parametrize("mode", ["mcts", "alphabeta"])
def test_plot_writes_png(tmp_path, mode):
    pytest.importorskip("matplotlib")
    if mode == "mcts":
        path, _ = A.run_mcts(sims=[3, 9], games=2, run_id="p", out_root=tmp_path,
                             builder=fake_builder, log=quiet)
    else:
        path, _ = A.run_alphabeta(depths=[1], positions=A.make_positions("normal", 1, 100, seed=0),
                                  run_id="p", out_root=tmp_path, log=quiet)
    out = A.plot(path)
    assert out.suffix == ".png" and out.stat().st_size > 0


def test_cli_rejects_bad_lists():
    with pytest.raises(SystemExit):
        A.main(["mcts", "--sims", "50,abc"])
    with pytest.raises(SystemExit):
        A.main(["alphabeta", "--depths", "0"])
