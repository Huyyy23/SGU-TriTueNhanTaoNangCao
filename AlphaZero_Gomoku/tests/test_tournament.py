"""T31: round-robin tournament with fake agents (fast, no network, no checkpoint)."""

import csv
import json
import hashlib

from scripts.tournament import main, parse_budget

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


@pytest.mark.parametrize("budget", ["sims:0", "sims:1.5", "time:0", "time:nan", "time:inf", "other:1"])
def test_invalid_budgets_rejected(budget):
    with pytest.raises(ValueError):
        T.Budget.parse(budget)
    with pytest.raises(ValueError):
        parse_budget(budget)


def test_nonempty_output_and_odd_games_rejected_without_overwrite(tmp_path):
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("user data")
    with pytest.raises(SystemExit):
        T.main(["--agents", "random,heuristic", "--out-dir", str(tmp_path)])
    assert sentinel.read_text() == "user data"
    with pytest.raises(SystemExit):
        T.main(["--agents", "random,heuristic", "--games", "3", "--out-dir", str(tmp_path / "odd")])
    assert not (tmp_path / "odd").exists()


def test_checkpoint_hash_and_aliases(tmp_path):
    checkpoint = tmp_path / "renju.pt"
    checkpoint.write_bytes(b"checkpoint bytes")
    assert T.resolve_agent("mcts_pure") == "mcts-pure"
    assert T.resolve_agent("az_renju") == "az_v0"
    with pytest.raises(ValueError):
        T.resolve_agent("az_v1")
    result = T.checkpoint_manifest({"az_renju": str(checkpoint)})
    assert result["az_renju"]["sha256"] == hashlib.sha256(b"checkpoint bytes").hexdigest()


def test_raw_telemetry_matches_integer_moves(tmp_path):
    run_dir, _ = run(tmp_path, games=2, run_id="raw")
    records = [json.loads(line) for line in (run_dir / "games.jsonl").read_text().splitlines()]
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
    full = [json.loads(line) for line in (full_dir / "games.jsonl").read_text().splitlines()]
    pair = [json.loads(line) for line in (pair_dir / "games.jsonl").read_text().splitlines()]
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
    out = tmp_path / "resume"
    prefix = (out / "games.jsonl").read_bytes()
    config = (out / "config.json").read_bytes()
    seen = []
    def resumed(task, builder=None):
        seen.append(task["game_id"])
        return original(task, builder)
    monkeypatch.setattr(T, "play_game_task", resumed)
    _, summary = run(tmp_path, games=4, run_id="resume", seed=13, pair="first,seeded", resume=True)
    assert seen == [2, 3]
    assert (out / "games.jsonl").read_bytes().startswith(prefix)
    assert (out / "config.json").read_bytes() == config
    assert summary["games_played"] == 4
    with pytest.raises(ValueError, match="configuration"):
        run(tmp_path, games=4, run_id="resume", seed=14, pair="first,seeded", resume=True)


def test_resume_rejects_corrupted_identity(tmp_path):
    out, _ = run(tmp_path, games=2, run_id="corrupt", pair="first,seeded")
    records = [json.loads(line) for line in (out / "games.jsonl").read_text().splitlines()]
    records[0]["seeds"]["black"] += 1
    (out / "games.jsonl").write_text("\n".join(json.dumps(record) for record in records) + "\n")
    before = (out / "games.jsonl").read_bytes()
    with pytest.raises(ValueError, match="identity|seed"):
        run(tmp_path, games=2, run_id="corrupt", pair="first,seeded", resume=True)
    assert (out / "games.jsonl").read_bytes() == before


def test_merge_uses_raw_games_and_rejects_missing_duplicate_or_mismatched_shards(tmp_path):
    import shutil
    from dataclasses import asdict
    from itertools import combinations
    from az.game_log import GameRecord, MoveRecord
    from scripts.tournament import merge_shards

    root = tmp_path / "shards"
    agents = ["random", "heuristic", "alphabeta"]
    config = dict(agents=agents, rules="normal", budget="sims:2", depth=2, seed=7,
                  games=2, loaded_config={}, threads=1)
    for pair_index, pair in enumerate(combinations(agents, 2)):
        source = root / str(pair_index)
        source.mkdir(parents=True)
        manifest = {"config": {**config, "pair": ",".join(pair)}, "checkpoints": {}}
        (source / "manifest.json").write_text(json.dumps(manifest))
        # Deliberately false shard summary: merge must never trust it.
        (source / "summary.json").write_text('{"total_games": 999}')
        records = []
        seed = 7 + pair_index * 100000
        for index in range(2):
            black, white = pair if index == 0 else pair[::-1]
            record = GameRecord(index, "now", "normal", black, white, "draw", False,
                                "board_full", 1, .1,
                                config=dict(pair=list(pair), pair_seed=seed, seed_a=seed + index * 2,
                                            seed_b=seed + index * 2 + 1, budget="sims:2", depth=2),
                                moves=[MoveRecord(0, 1, 0, 0, 0.0, [], 0, latency_sec=.1)])
            records.append(json.dumps(asdict(record)))
        (source / "games.jsonl").write_text("\n".join(records) + "\n")
    original = (root / "0" / "games.jsonl").read_bytes()
    out = tmp_path / "frozen"
    assert main(["--merge-root", str(root), "--out-dir", str(out)]) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["total_games"] == 6
    assert all(pair["score"] == .5 and pair["draws"] == 2 for pair in summary["pairs"])
    assert [r["game_id"] for r in map(json.loads, (out / "games.jsonl").read_text().splitlines())] == list(range(6))
    assert len(json.loads((out / "manifest.json").read_text())["sources"]) == 3
    assert (root / "0" / "games.jsonl").read_bytes() == original
    with pytest.raises(ValueError, match="empty"):
        merge_shards(root, out)
    shutil.copytree(root / "0", root / "duplicate")
    with pytest.raises(ValueError, match="duplicate"):
        merge_shards(root, tmp_path / "duplicate-output")
    shutil.rmtree(root / "duplicate")
    manifest_path = root / "2" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["config"]["depth"] = 3
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="configuration"):
        merge_shards(root, tmp_path / "bad-config")
    manifest["config"]["depth"] = 2
    manifest_path.write_text(json.dumps(manifest))
    shutil.rmtree(root / "2")
    with pytest.raises(ValueError, match="missing pair"):
        merge_shards(root, tmp_path / "missing")



def test_unified_shards_merge_raw_records_and_reject_bad_inputs(tmp_path):
    import shutil
    from itertools import combinations

    root = tmp_path / "unified"
    names = ("first", "seeded", "cheater")
    for index, pair in enumerate(combinations(names, 2)):
        source, _ = run(root, names=names, games=2, run_id=str(index), seed=9, pair=",".join(pair))
        (source / "summary.json").write_text('{"games_played": 999}')
    before = (root / "0" / "games.jsonl").read_bytes()
    out = tmp_path / "merged"
    assert T.merge_shards(root, out) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["games_played"] == 6
    assert len(summary["pairs_summary"]) == 3
    records = [json.loads(line) for line in (out / "games.jsonl").read_text().splitlines()]
    assert len({record["game_id"] for record in records}) == 6
    assert all(record["raw_moves"] for record in records)
    assert (root / "0" / "games.jsonl").read_bytes() == before
    with pytest.raises((ValueError, FileExistsError)):
        T.merge_shards(root, out)
    shutil.copytree(root / "0", root / "duplicate")
    with pytest.raises(ValueError, match="duplicate"):
        T.merge_shards(root, tmp_path / "duplicate-merged")
    shutil.rmtree(root / "duplicate")
    config_path = root / "2" / "config.json"
    config = json.loads(config_path.read_text())
    config["seed"] += 1
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="configuration|incompatible"):
        T.merge_shards(root, tmp_path / "mismatch-merged")
    config["seed"] -= 1
    config_path.write_text(json.dumps(config))
    shutil.rmtree(root / "2")
    with pytest.raises(ValueError, match="missing"):
        T.merge_shards(root, tmp_path / "missing-merged")
