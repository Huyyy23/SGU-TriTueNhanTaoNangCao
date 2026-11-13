"""T31: round-robin tournament with fake agents (fast, no network, no checkpoint)."""

import csv
import json

import pytest

from gomoku.board import SIZE
from scripts import tournament as T


class FirstLegal:
    """Plays the lowest legal move: deterministic, never illegal, no search."""

    def __init__(self, rules, seed=None):
        self.rules = rules
        self.name = "first"

    def select_move(self, board):
        return divmod(min(self.rules.legal_moves(board)), SIZE)


class SeededRandom:
    """Random legal move from its own seed; exposes a fake search counter."""

    def __init__(self, rules, seed=None):
        import random

        self.rules = rules
        self.rng = random.Random(seed)
        self.name = "seeded"
        self.nodes_visited = 0

    def select_move(self, board):
        self.nodes_visited = 7
        return divmod(self.rng.choice(self.rules.legal_moves(board)), SIZE)


class PlaysOccupied:
    """Always tries the centre, so its second move is illegal."""

    def __init__(self, rules, seed=None):
        self.rules = rules
        self.name = "cheater"

    def select_move(self, board):
        return (5, 5)


FAKES = {"first": FirstLegal, "seeded": SeededRandom, "cheater": PlaysOccupied}


def fake_builder(settings, rules, seed):
    return FAKES[settings["factory"]["agent"]](rules, seed)


def entries(*names):
    return {name: {"factory": {"agent": name}, "budget_note": "fake"} for name in names}


def run(tmp_path, names=("first", "seeded"), games=20, **kwargs):
    kwargs.setdefault("budget", T.Budget.parse("sims:10"))
    return T.run_tournament(
        entries(*names), games=games, out_root=tmp_path, builder=fake_builder, log=lambda _msg: None, **kwargs
    )


def test_twenty_games_split_ten_ten_by_colour(tmp_path):
    run_dir, summary = run(tmp_path, run_id="r1")

    games = [json.loads(line) for line in (run_dir / "games.jsonl").read_text().splitlines()]
    assert len(games) == 20 == summary["games_played"]
    assert sum(g["first_player"] == "first" for g in games) == 10
    assert sum(g["first_player"] == "seeded" for g in games) == 10

    pair = summary["pairs_summary"][0]
    assert pair["games"] == 20
    assert pair["games_a_black"] == pair["games_a_white"] == 10
    assert pair["games_b_black"] == pair["games_b_white"] == 10
    assert pair["wins_a"] + pair["draws"] + pair["wins_b"] == 20
    assert pair["score_a"] + pair["score_b"] == pytest.approx(1.0)
    for agent in summary["agents_summary"].values():
        assert agent["games_black"] == agent["games_white"] == 10


def test_matrix_is_full_and_wilson_ci_inside_unit_interval(tmp_path):
    names = ("first", "seeded", "cheater")
    run_dir, summary = run(tmp_path, names=names, games=4, run_id="r2")

    with open(run_dir / "matrix.csv", newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["agent", *names, "overall_score"]
    assert [row[0] for row in rows[1:]] == list(names)
    for i, row in enumerate(rows[1:]):
        for j, cell in enumerate(row[1:4]):
            if i == j:
                assert cell == ""
            else:
                assert 0.0 <= float(cell) <= 1.0
                assert float(cell) + float(rows[j + 1][i + 1]) == pytest.approx(1.0)

    for pair in summary["pairs_summary"]:
        for side in ("a", "b"):
            low, high = pair[f"wilson95_{side}"]
            assert 0.0 <= low <= pair[f"score_{side}"] <= high <= 1.0
    for agent in summary["agents_summary"].values():
        low, high = agent["wilson95"]
        assert 0.0 <= low <= agent["score"] <= high <= 1.0
    assert len(summary["pairs_summary"]) == 3


def test_summary_has_required_fields(tmp_path):
    run_dir, summary = run(tmp_path, games=2, run_id="r3", seed=42)
    for name in ("config.json", "games.jsonl", "matrix.csv", "pairs.csv", "summary.json"):
        assert (run_dir / name).is_file(), name
    on_disk = json.loads((run_dir / "summary.json").read_text())
    for key in ("seed", "budget", "ruleset", "agents", "agent_settings", "games_per_pair", "matrix",
                "agents_summary", "pairs_summary", "checkpoint_unchanged", "git", "versions"):
        assert key in on_disk, key
    assert on_disk["seed"] == 42
    assert on_disk["budget"] == {"mode": "sims", "value": 10, "text": "sims:10"}
    seeded = on_disk["agents_summary"]["seeded"]
    for key in ("mean_time_per_move_sec", "p95_time_per_move_sec", "mean_nodes_per_move", "score_black", "score_white"):
        assert key in seeded, key
    assert seeded["mean_nodes_per_move"] == 7


def test_same_seed_reproduces_the_same_games(tmp_path):
    first_dir, _ = run(tmp_path, games=4, run_id="a", seed=5, opening_plies=2)
    second_dir, _ = run(tmp_path, games=4, run_id="b", seed=5, opening_plies=2)
    other_dir, _ = run(tmp_path, games=4, run_id="c", seed=6, opening_plies=2)

    def moves(run_dir):
        return [json.loads(line)["moves"] for line in (run_dir / "games.jsonl").read_text().splitlines()]

    assert moves(first_dir) == moves(second_dir)
    assert moves(first_dir) != moves(other_dir)


def test_colour_swapped_games_share_opening(tmp_path):
    run_dir, _ = run(tmp_path, games=4, run_id="open", seed=1, opening_plies=3)
    games = [json.loads(line) for line in (run_dir / "games.jsonl").read_text().splitlines()]
    assert games[0]["moves"][:3] == games[1]["moves"][:3]
    assert games[0]["first_player"] != games[1]["first_player"]
    assert games[0]["seeds"]["opening"] == games[1]["seeds"]["opening"]


def test_existing_run_folder_is_never_overwritten(tmp_path):
    run(tmp_path, games=2, run_id="same")
    with pytest.raises(FileExistsError):
        run(tmp_path, games=2, run_id="same")


def test_illegal_move_loses_and_is_counted(tmp_path):
    _, summary = run(tmp_path, names=("first", "cheater"), games=2, run_id="illegal")
    pair = summary["pairs_summary"][0]
    assert pair["illegal_moves_b"] == 2
    assert pair["wins_a"] == 2
    assert summary["illegal_moves_total"] == 2
    assert any("illegal" in w for w in summary["warnings"])


# -- parsing and budget rules --------------------------------------------------


def test_budget_parse():
    assert T.Budget.parse("sims:200") == T.Budget("sims", 200)
    assert T.Budget.parse("time:1.5") == T.Budget("time", 1.5)
    for bad in ("200", "sims:", "sims:-1", "steps:5", "time:abc"):
        with pytest.raises(ValueError):
            T.Budget.parse(bad)


def test_agents_are_parsed_not_hardcoded():
    specs = T.parse_agents("random, mcts_pure@sims=50 ,alphabeta@depth=2@pruning=false,az_v1")
    assert [s.kind for s in specs] == ["random", "mcts-pure", "alphabeta", "az_v1_tactical"]
    assert specs[1].params == {"sims": 50}
    assert specs[2].params == {"depth": 2, "pruning": False}
    for bad in ("random", "random,random", "random,chess", "random,heuristic@depth=2", "random,alphabeta@depth=0"):
        with pytest.raises(ValueError):
            T.parse_agents(bad)


def test_budget_meaning_per_agent(tmp_path):
    sims, timed = T.Budget.parse("sims:300"), T.Budget.parse("time:0.5")
    defaults = T.Defaults(alphabeta_depth=1)
    spec = T.AgentSpec.parse

    mcts = T.agent_settings(spec("mcts_pure"), sims, defaults)
    assert mcts["factory"]["rollouts"] == 300 and mcts["budget_enforced"]
    assert T.agent_settings(spec("mcts_pure@sims=50"), sims, defaults)["factory"]["rollouts"] == 50
    assert T.agent_settings(spec("mcts_pure"), timed, defaults)["budget_enforced"] is False

    ab = T.agent_settings(spec("alphabeta"), sims, defaults)
    assert ab["factory"]["depth"] == 1 and ab["budget_enforced"] is False
    assert "does not apply" in ab["budget_note"]
    rnd = T.agent_settings(spec("random"), sims, defaults)
    assert rnd["budget_applies"] is False

    with pytest.raises(ValueError):
        T.agent_settings(spec("az_v0"), sims, defaults)  # no checkpoint
    ckpt = tmp_path / "fake.pt"
    ckpt.write_bytes(b"x")
    az = T.agent_settings(spec("az_v0"), sims, T.Defaults(checkpoint=str(ckpt)))
    assert az["factory"]["n_simulations"] == 300 and az["factory"]["time_budget_sec"] is None
    az_t = T.agent_settings(spec("az_v1_threat"), timed, T.Defaults(checkpoint=str(ckpt), time_sims_cap=900))
    assert az_t["factory"]["time_budget_sec"] == 0.5 and az_t["factory"]["n_simulations"] == 900


def test_odd_game_count_rejected(tmp_path):
    with pytest.raises(ValueError):
        run(tmp_path, games=3, run_id="odd")


def test_cli_dry_run(capsys):
    assert T.main(["--agents", "random,heuristic,alphabeta", "--games", "20", "--budget", "sims:200", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "3 pairs x 20 games = 60 games" in out
    with pytest.raises(SystemExit):
        T.main(["--agents", "random,heuristic", "--games", "5"])
