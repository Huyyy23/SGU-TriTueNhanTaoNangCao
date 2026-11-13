from az.model import AZNet
from az.selfplay import _play_chunk
from gomoku.board import BLACK, SIZE, Board
from gomoku.features import board_to_planes
from gomoku.rules import get_rules



def test_renju_forbidden_plane_marks_exact_forbidden_cells():
    rules = get_rules("renju")
    board = Board()
    for row, col in ((5, 4), (5, 6), (4, 5), (6, 5)):
        board.grid[row * SIZE + col] = BLACK
    board.current_player = BLACK

    forbidden = rules.forbidden_moves(board)
    features = board_to_planes(board, rules, BLACK)

    assert forbidden
    assert features[2].sum() == len(forbidden)
    assert all(features[2].flat[move] == 1 for move in forbidden)


def test_parallel_worker_builds_renju_rules(monkeypatch):
    import az.selfplay as selfplay

    captured = {}
    model = AZNet(n_blocks=1, n_filters=8)
    cfg = {"n_blocks": 1, "n_filters": 8}

    def fake_run_games(model, rules, cfg, n_games, **kwargs):
        captured["rule_name"] = rules.name
        return iter(())

    monkeypatch.setattr(selfplay, "run_games", fake_run_games)
    records, samples = _play_chunk((model.state_dict(), "renju", cfg, 1, 0, 0))

    assert captured["rule_name"] == "renju"
    assert records == []
    assert samples == []
