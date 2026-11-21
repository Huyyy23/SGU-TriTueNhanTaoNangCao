"""T31: round-robin tournament with fake agents (fast, no network, no checkpoint)."""

import sqlite3
import json
import hashlib

from scripts.tournament import main, parse_budget

import pytest

from gomoku.board import SIZE
from scripts import tournament as T
from az.model import AZNet, save_checkpoint


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
    kwargs.setdefault("db_path", tmp_path / "tournament.sqlite")
    return T.run_tournament(
        entries(*names), games=games, builder=fake_builder, log=lambda _msg: None, **kwargs
    )


def stored_games(db_path, experiment_id):
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            "SELECT payload_json FROM games WHERE experiment_id=? ORDER BY CAST(local_game_id AS INTEGER)",
            (experiment_id,),
        ).fetchall()
    return [json.loads(row[0]) for row in rows]



def test_tournament_records_completed_games_and_moves_in_sqlite(tmp_path):
    import sqlite3

    db_path = tmp_path / "explicit.sqlite"
    run(tmp_path, games=2, run_id="db", db_path=db_path)
    with sqlite3.connect(db_path) as connection:
        assert connection.execute("select count(*) from experiments").fetchone()[0] == 1
        assert connection.execute("select count(*) from agents").fetchone()[0] == 2
        assert connection.execute("select count(*) from matches").fetchone()[0] == 1
        assert connection.execute("select count(*) from games").fetchone()[0] == 2
        assert connection.execute("select sum(moves_count) from games").fetchone()[0] == connection.execute(
            "select count(*) from moves"
        ).fetchone()[0]
        payload, = connection.execute("select payload_json from games order by local_game_id limit 1").fetchone()
        assert json.loads(payload)["raw_moves"]

def test_tournament_none_skips_capture_and_storage_without_changing_result(tmp_path):
    _, standard = run(tmp_path, games=2, run_id="standard", seed=19)
    none_dir, none = run(tmp_path, games=2, run_id="none", seed=19, logging_level="none")

    assert none["matrix"] == standard["matrix"]
    assert none["games_played"] == standard["games_played"] == 2
    assert not (tmp_path / "none").exists()
    assert none["agents_summary"]["first"]["moves_timed"] == 0
    import sqlite3
    with sqlite3.connect(tmp_path / "tournament.sqlite") as connection:
        assert connection.execute("select count(*) from experiments").fetchone()[0] == 1

def test_tournament_none_cannot_resume(tmp_path):
    with pytest.raises(ValueError, match="resume requires STANDARD"):
        run(tmp_path, games=2, logging_level="none", resume=True)
    assert not (tmp_path / "tournament.sqlite").exists()


def test_twenty_games_split_ten_ten_by_colour(tmp_path):
    run_dir, summary = run(tmp_path, run_id="r1")

    games = stored_games(run_dir, "r1")
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


def test_pairs_summary_keeps_seed_of_every_game(tmp_path):
    run_dir, summary = run(tmp_path, games=4, run_id="r_seeds", seed=7)
    games = stored_games(run_dir, "r_seeds")
    pair = summary["pairs_summary"][0]
    assert [s["pair_game"] for s in pair["game_seeds"]] == [0, 1, 2, 3]
    for entry, game in zip(pair["game_seeds"], sorted(games, key=lambda g: g["pair_game"])):
        assert entry["black_agent"] == game["first_player"] and entry["white_agent"] == game["second_player"]
        assert entry["seeds"] == game["seeds"]


def test_broken_agent_fails_before_any_game_or_folder(tmp_path):
    def broken_builder(settings, rules, seed):
        if settings["factory"]["agent"] == "seeded":
            raise RuntimeError("size mismatch for conv.weight")
        return fake_builder(settings, rules, seed)

    with pytest.raises(ValueError, match="seeded"):
        T.run_tournament(entries("first", "seeded"), games=2, builder=broken_builder,
                         log=lambda _msg: None, budget=T.Budget.parse("sims:10"), run_id="broken",
                         db_path=tmp_path / "broken.sqlite")
    assert not (tmp_path / "broken.sqlite").exists()


def test_matrix_is_full_and_wilson_ci_inside_unit_interval(tmp_path):
    names = ("first", "seeded", "cheater")
    run_dir, summary = run(tmp_path, names=names, games=4, run_id="r2")

    matrix = summary["matrix"]
    for i, row in enumerate(names):
        for j, col in enumerate(names):
            if i == j:
                assert col not in matrix[row]
            else:
                assert 0.0 <= matrix[row][col] <= 1.0
                assert matrix[row][col] + matrix[col][row] == pytest.approx(1.0)

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
    with sqlite3.connect(run_dir) as connection:
        payload, = connection.execute(
            "select payload_json from events where experiment_id=? and event='tournament_summary'",
            ("r3",),
        ).fetchone()
    on_disk = json.loads(payload)
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

    def moves(db_path, experiment_id):
        return [game["moves"] for game in stored_games(db_path, experiment_id)]

    assert moves(first_dir, "a") == moves(second_dir, "b")
    assert moves(first_dir, "a") != moves(other_dir, "c")


def test_colour_swapped_games_share_opening(tmp_path):
    run_dir, _ = run(tmp_path, games=4, run_id="open", seed=1, opening_plies=3)
    games = stored_games(run_dir, "open")
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
    specs = T.parse_agents("random, mcts_pure@sims=50 ,alphabeta@depth=2@pruning=false,az_v1_threat")
    assert [s.kind for s in specs] == ["random", "mcts-pure", "alphabeta", "az_v1_threat"]
    assert specs[1].params == {"sims": 50}
    assert specs[2].params == {"depth": 2, "pruning": False}
    for bad in ("random", "random,random", "random,chess", "random,heuristic@depth=2", "random,alphabeta@depth=0", "random,az_v1"):
        with pytest.raises(ValueError):
            T.parse_agents(bad)


def test_budget_meaning_per_agent(tmp_path):
    sims, timed = T.Budget.parse("sims:300"), T.Budget.parse("time:0.5")
    defaults = T.Defaults(alphabeta_depth=1)
    spec = T.AgentSpec.parse

    mcts = T.agent_settings(spec("mcts_pure"), sims, defaults)
    assert mcts["factory"]["rollouts"] == 300 and mcts["budget_enforced"]
    assert T.agent_settings(spec("mcts_pure@sims=50"), sims, defaults)["factory"]["rollouts"] == 50
    assert T.agent_settings(spec("mcts_pure"), timed, defaults)["budget_enforced"] is True
    assert T.agent_settings(spec("mcts_pure"), timed, defaults)["factory"]["time_budget_sec"] == 0.5

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


def test_direct_tournament_run_uses_default_logger(tmp_path, capsys):
    run_dir, summary = T.run_tournament(
        entries("first", "seeded"), games=2, budget=T.Budget.parse("sims:1"),
        builder=fake_builder, db_path=tmp_path / "tournament.sqlite",
    )
    assert summary["games_played"] == 2
    assert "run" in capsys.readouterr().out
    with sqlite3.connect(run_dir) as connection:
        assert connection.execute(
            "select count(*) from events where event='tournament_summary'"
        ).fetchone()[0] == 1

@pytest.mark.parametrize("budget", ["sims:0", "sims:1.5", "time:0", "time:nan", "time:inf", "other:1"])
def test_invalid_budgets_rejected(budget):
    with pytest.raises(ValueError):
        T.Budget.parse(budget)
    with pytest.raises(ValueError):
        parse_budget(budget)


def test_tournament_dry_run_does_not_touch_existing_output_path(tmp_path):
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("user data")
    assert T.main(["--agents", "random,heuristic", "--games", "2", "--dry-run"]) == 0
    assert sentinel.read_text() == "user data"
    with pytest.raises(SystemExit):
        T.main(["--agents", "random,heuristic", "--games", "3", "--dry-run"])
    assert not (tmp_path / "odd").exists()


def test_checkpoint_hash_and_aliases(tmp_path):
    checkpoint = tmp_path / "renju.pt"
    checkpoint.write_bytes(b"checkpoint bytes")
    assert T.resolve_agent("mcts_pure") == "mcts-pure"
    assert T.resolve_agent("az") == "az_v1_threat"
    assert T.resolve_agent("az_renju") == "az_v0"
    with pytest.raises(ValueError):
        T.resolve_agent("az_v1")
    result = T.checkpoint_manifest({"az_renju": str(checkpoint)})
    assert result["az_renju"]["sha256"] == hashlib.sha256(b"checkpoint bytes").hexdigest()


def test_raw_telemetry_matches_integer_moves(tmp_path):
    run_dir, _ = run(tmp_path, games=2, run_id="raw")
    records = stored_games(run_dir, "raw")
    for record in records:
        assert len(record["raw_moves"]) == len(record["moves"]) == record["moves_count"]
        assert all(isinstance(move, int) for move in record["moves"])
        assert [move["row"] * SIZE + move["col"] for move in record["raw_moves"]] == record["moves"]
        assert all(move["latency_sec"] >= 0 for move in record["raw_moves"])


def test_probe_keeps_unavailable_nodes_null():
    from gomoku.board import Board
    player = FirstLegal(T.get_rules("normal"))
    player.nodes_visited = None
    probe = T.Probe(player, "az-like")
    assert probe.select_move(Board()) == (0, 0)
    assert probe.moves[0]["raw"]["nodes_visited"] is None
    assert "nodes" not in T._move_stats(probe.moves, None)


@pytest.mark.parametrize("scheme", ["paired-crc32", "legacy-v1"])
def test_filtered_pair_keeps_full_lineup_seeds(tmp_path, scheme):
    names = ("first", "seeded", "cheater")
    full_dir, _ = run(tmp_path, names=names, games=2, run_id="full", seed=7, seed_scheme=scheme)
    pair_dir, summary = run(tmp_path, names=names, games=2, run_id="pair", seed=7,
                            pair="seeded,cheater", seed_scheme=scheme)
    full = stored_games(full_dir, "full")
    pair = stored_games(pair_dir, "pair")
    expected = [record for record in full if record["pair"] == ["seeded", "cheater"]]
    assert [r["game_id"] for r in pair] == [0, 1]
    assert [(r["seeds"], r["moves"]) for r in pair] == [
        (r["seeds"], r["moves"]) for r in expected]
    assert {r["first_player"] for r in pair} == {"seeded", "cheater"}
    if scheme == "legacy-v1":
        assert pair[0]["seeds"]["black"] == 200007


def test_resume_preserves_prefix_and_skips_logged_games(tmp_path, monkeypatch):
    original = T.play_game_task
    def interrupted(task, builder=None):
        if task["game_id"] == 2:
            raise RuntimeError("interrupted")
        return original(task, builder)
    monkeypatch.setattr(T, "play_game_task", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        run(tmp_path, games=4, run_id="resume", seed=13, pair="first,seeded")
    db_path = tmp_path / "tournament.sqlite"
    prefix = stored_games(db_path, "resume")
    seen = []
    def resumed(task, builder=None):
        seen.append(task["game_id"])
        return original(task, builder)
    monkeypatch.setattr(T, "play_game_task", resumed)
    _, summary = run(tmp_path, games=4, run_id="resume", seed=13, pair="first,seeded", resume=True)
    assert seen == [2, 3]
    assert stored_games(db_path, "resume")[:len(prefix)] == prefix
    assert summary["games_played"] == 4
    with pytest.raises(ValueError, match="configuration"):
        run(tmp_path, games=4, run_id="resume", seed=14, pair="first,seeded", resume=True)


def test_resume_rejects_corrupted_identity(tmp_path):
    db_path, _ = run(tmp_path, games=2, run_id="corrupt", pair="first,seeded")
    with sqlite3.connect(db_path) as connection:
        payload, = connection.execute(
            "select payload_json from games where experiment_id='corrupt' and local_game_id='0'"
        ).fetchone()
        record = json.loads(payload)
        record["seeds"]["black"] += 1
        connection.execute(
            "update games set payload_json=? where experiment_id='corrupt' and local_game_id='0'",
            (json.dumps(record),),
        )
    before = stored_games(db_path, "corrupt")
    with pytest.raises(ValueError, match="identity|seed"):
        run(tmp_path, games=2, run_id="corrupt", pair="first,seeded", resume=True)
    assert stored_games(db_path, "corrupt") == before


def test_resume_requires_explicit_run_id(tmp_path):
    db_path = tmp_path / "tournament.sqlite"
    with pytest.raises(ValueError, match="explicit --run-id"):
        T.run_tournament(
            entries("first", "seeded"), games=2, budget=T.Budget.parse("sims:10"),
            resume=True, builder=fake_builder, db_path=db_path,
            backends={"mcts": "python", "patterns": "python"},
        )
    assert not db_path.exists()
def test_resume_uses_sqlite_after_legacy_files_are_unavailable(tmp_path, monkeypatch):
    db_path, _ = run(tmp_path, games=2, run_id="sqlite_resume", pair="first,seeded")
    with sqlite3.connect(db_path) as connection:
        assert connection.execute(
            "select count(*) from events where experiment_id='sqlite_resume' and event='tournament_summary'"
        ).fetchone()[0] == 1
    assert db_path.is_file()
    assert not (tmp_path / "sqlite_resume").exists()
    seen = []
    original = T.play_game_task

    def unexpected_replay(task, builder=None):
        seen.append(task["game_id"])
        return original(task, builder)

    monkeypatch.setattr(T, "play_game_task", unexpected_replay)
    _, summary = run(tmp_path, games=2, run_id="sqlite_resume", pair="first,seeded", resume=True)
    assert seen == []
    assert summary["games_played"] == 2
    with sqlite3.connect(db_path) as connection:
        assert connection.execute(
            "select count(*) from events where experiment_id='sqlite_resume' and event='tournament_summary'"
        ).fetchone()[0] == 1







def test_normal_az_variants_default_to_the_official_checkpoint():
    normal = T.checkpoint_mapping(
        [], ["az_v0", "az_v1_tactical", "az_v1_threat"], "normal"
    )
    assert set(normal.values()) == {"checkpoints/normal/latest.json"}
    assert T.checkpoint_mapping([], ["az_renju"], "renju") == {
        "az_renju": "checkpoints/renju_latest.json"
    }


def test_checkpoint_manifest_hashes_resolved_generation(tmp_path, monkeypatch):
    import hashlib
    import az.model

    pointer = tmp_path / "best.json"
    checkpoint = tmp_path / "checkpoint.pt"
    pointer.write_text("{}")
    checkpoint.write_bytes(b"model")
    monkeypatch.setattr(az.model, "resolve_checkpoint_path", lambda path: checkpoint)
    manifest = T.checkpoint_manifest({"az": str(pointer)})
    assert manifest["az"] == {
        "path": str(checkpoint.resolve()),
        "sha256": hashlib.sha256(b"model").hexdigest(),
    }
def _az_entries(checkpoint, config):
    return {
        "az_a": {"factory": {"agent": "az_v0", "checkpoint": str(checkpoint), "config": str(config)}},
        "az_b": {"factory": {"agent": "az_v0", "checkpoint": str(checkpoint), "config": str(config)}},
    }


def test_tournament_az_normal_cpp_records_components_used(tmp_path):
    import az.backend

    checkpoint = tmp_path / "model.pt"
    save_checkpoint(AZNet(n_blocks=1, n_filters=4), checkpoint)
    config = tmp_path / "normal.yaml"
    config.write_text("n_simulations: 2\ntemp_moves: 0\nrule_set: normal\n", encoding="utf-8")
    backends = az.backend.configure_backends("cpp", mcts_backend="cpp", patterns_backend="python",
                                             allow_build=False)
    _, summary = T.run_tournament(
        _az_entries(checkpoint, config), games=2, budget=T.Budget.parse("sims:2"),
        ruleset="normal", seed=1, builder=T.default_builder,
        backends=backends, db_path=tmp_path / "tournament.sqlite", logging_level="standard",
    )
    assert summary["components_used"]["mcts"] == "cpp"
    assert summary["backends"]["mcts"] == "cpp"


def test_tournament_az_renju_cpp_records_components_used(tmp_path):
    import az.backend

    checkpoint = tmp_path / "model.pt"
    save_checkpoint(AZNet(n_blocks=1, n_filters=4), checkpoint)
    config = tmp_path / "renju.yaml"
    config.write_text("n_simulations: 2\ntemp_moves: 0\nrule_set: renju\n", encoding="utf-8")
    backends = az.backend.configure_backends("cpp", mcts_backend="cpp", patterns_backend="python",
                                             allow_build=False)
    _, summary = T.run_tournament(
        _az_entries(checkpoint, config), games=2, budget=T.Budget.parse("sims:2"),
        ruleset="renju", seed=1, builder=T.default_builder,
        backends=backends, db_path=tmp_path / "tournament_renju.sqlite", logging_level="standard",
    )
    assert summary["components_used"]["mcts"] == "cpp"
    assert summary["backends"]["mcts"] == "cpp"
