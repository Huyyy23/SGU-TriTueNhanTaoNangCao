import json

from az.model import AZNet, save_checkpoint
from gomoku.board import Board
from gomoku.rules.normal import NormalRules
from scripts.arena import main, make_player


def test_same_checkpoint_with_separate_tactical_configs(tmp_path):
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(AZNet(n_blocks=1, n_filters=4), checkpoint)
    board = Board()
    for move in [(5, 1), (0, 0), (5, 2), (0, 2), (5, 3), (0, 4), (5, 4), (1, 0)]:
        board.play(*move)
    player = make_player(str(checkpoint), NormalRules(), {"tactical": {"enabled": True}}, seed=0)
    assert player.select_move(board) in {(5, 0), (5, 5)}
    assert player.last_override_reason == "win_now"

    base = tmp_path / "base.yaml"
    tactical = tmp_path / "tactical.yaml"
    base.write_text("n_simulations: 1\n", encoding="utf-8")
    tactical.write_text("n_simulations: 1\ntactical:\n  enabled: true\n", encoding="utf-8")
    out = tmp_path / "arena"

    assert main(["--a", str(checkpoint), "--b", str(checkpoint), "--games", "1",
                 "--config-a", str(base), "--config-b", str(tactical),
                 "--out-dir", str(out)]) == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    record = json.loads((out / "games.jsonl").read_text(encoding="utf-8"))
    assert summary["total_games"] == 1
    assert summary["config_a"].get("tactical", {}).get("enabled", False) is False
    assert summary["config_b"]["tactical"]["enabled"] is True
    assert record["first_player"] != record["second_player"]
    assert record["moves_count"] >= 9
