import csv
import hashlib
import json

import pytest

from scripts.tournament import main, parse_budget


def test_round_robin_outputs_balanced_games_and_manifest(tmp_path):
    out = tmp_path / "run"
    assert main(["--agents", "random,heuristic", "--games", "20", "--budget", "sims:1",
                 "--seed", "17", "--out-dir", str(out)]) == 0
    records = [json.loads(line) for line in (out / "games.jsonl").read_text().splitlines()]
    assert len(records) == 20
    assert sum(r["first_player"] == "random" for r in records) == 10
    assert all(len(r["moves"]) == r["moves_count"] for r in records)
    assert all(m["latency_sec"] >= 0 for r in records for m in r["moves"])
    summary = json.loads((out / "summary.json").read_text())
    pair = summary["pairs"][0]
    assert pair["seed"] == 17
    assert pair["wins"] + pair["draws"] + pair["losses"] == 20
    assert 0 <= pair["score_ci_95"][0] <= pair["score"] <= pair["score_ci_95"][1] <= 1
    assert pair["as_black"]["games"] == pair["as_white"]["games"] == 10
    rows = list(csv.reader((out / "matrix.csv").open()))
    assert len(rows) == 3 and all(len(row) == 3 for row in rows)
    assert float(rows[1][2]) + float(rows[2][1]) == pytest.approx(1)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["config"]["games"] == 20
    assert manifest["config"]["budget"] == "sims:1"
    assert manifest["checkpoints"] == {}


@pytest.mark.parametrize("budget", ["sims:0", "sims:1.5", "time:0", "time:nan", "time:inf", "other:1"])
def test_invalid_budgets_rejected(budget):
    with pytest.raises(ValueError):
        parse_budget(budget)


def test_nonempty_output_and_odd_games_rejected_without_overwrite(tmp_path):
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("user data")
    with pytest.raises(SystemExit):
        main(["--agents", "random,heuristic", "--out-dir", str(tmp_path)])
    assert sentinel.read_text() == "user data"
    with pytest.raises(SystemExit):
        main(["--agents", "random,heuristic", "--games", "3", "--out-dir", str(tmp_path / "odd")])
    assert not (tmp_path / "odd").exists()


def test_checkpoint_hash_and_aliases(tmp_path):
    from scripts.tournament import checkpoint_manifest, resolve_agent

    checkpoint = tmp_path / "renju.pt"
    checkpoint.write_bytes(b"checkpoint bytes")
    assert resolve_agent("mcts_pure") == "mcts-pure"
    assert resolve_agent("az_v1") == "az_v1_threat"
    assert resolve_agent("az_renju") == "az_v0"
    result = checkpoint_manifest({"az_renju": str(checkpoint)})
    assert result["az_renju"]["sha256"] == hashlib.sha256(b"checkpoint bytes").hexdigest()


def test_filtered_pair_keeps_full_lineup_seed_and_budget(tmp_path, monkeypatch):
    import numpy as np
    from scripts import tournament

    calls = []

    class FirstLegal:
        def select_move(self, board):
            return divmod(int(np.flatnonzero(board.grid == 0)[0]), 11)

    def factory(agent, rules, **kwargs):
        calls.append((agent, kwargs))
        return FirstLegal()

    monkeypatch.setattr(tournament, "create_player", factory)
    out = tmp_path / "filtered"
    main(["--agents", "random,heuristic,alphabeta", "--pair", "heuristic,alphabeta",
          "--games", "20", "--budget", "time:0.25", "--depth", "3", "--seed", "7", "--out-dir", str(out)])
    records = [json.loads(line) for line in (out / "games.jsonl").read_text().splitlines()]
    assert len(records) == 20
    assert sum(r["first_player"] == "heuristic" for r in records) == 10
    assert all(r["config"]["pair_seed"] == 200007 for r in records)
    assert calls[0][1]["seed"] == 200007
    assert all(kwargs["time_budget_sec"] == .25 and kwargs["depth"] == 3 for _, kwargs in calls)
    assert all(kwargs["n_simulations"] is None for _, kwargs in calls)


def test_simulation_budget_and_alias_reach_factory(monkeypatch):
    from types import SimpleNamespace
    from scripts import tournament

    received = {}

    def factory(agent, rules, **kwargs):
        received.update(agent=agent, **kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(tournament, "create_player", factory)
    player = tournament.make_named_player("az_v1", object(), config="configs/normal.yaml",
                                          checkpoint="model.pt", seed=4, budget=("sims", 200), depth=3)
    assert player.name == "az_v1"
    assert received["agent"] == "az_v1_threat"
    assert received["n_simulations"] == received["rollouts"] == 200
    assert received["depth"] == 3
    assert received["time_budget_sec"] is None


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


def test_resume_preserves_prefix_and_skips_logged_games(tmp_path, monkeypatch):
    import scripts.tournament as tournament
    out = tmp_path / "resume"
    args = ["--agents", "random,heuristic", "--pair", "random,heuristic", "--games", "4",
            "--budget", "sims:1", "--seed", "13", "--out-dir", str(out)]
    original = tournament.play_game
    def interrupted(*values):
        if values[3] == 2:
            raise RuntimeError("interrupted")
        return original(*values)
    monkeypatch.setattr(tournament, "play_game", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        main(args)
    prefix = (out / "games.jsonl").read_bytes()
    manifest = (out / "manifest.json").read_bytes()
    seen = []
    def resumed(*values):
        seen.append(values[3])
        return original(*values)
    monkeypatch.setattr(tournament, "play_game", resumed)
    assert main(args + ["--resume"]) == 0
    assert seen == [2, 3]
    assert (out / "games.jsonl").read_bytes().startswith(prefix)
    assert (out / "manifest.json").read_bytes() == manifest
    assert json.loads((out / "summary.json").read_text())["total_games"] == 4
    with pytest.raises(SystemExit):
        main(args + ["--resume", "--depth", "3"])
