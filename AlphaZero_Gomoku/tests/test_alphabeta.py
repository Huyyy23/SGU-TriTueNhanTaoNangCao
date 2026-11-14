from gomoku.board import BLACK, SIZE, WHITE, Board
from gomoku.rules import get_rules
from players.alphabeta import AlphaBetaPlayer


def test_alphabeta_takes_immediate_win():
    board = Board()
    for col in range(2, 6):
        board.grid[5 * SIZE + col] = BLACK

    assert AlphaBetaPlayer(get_rules("normal"), depth=1).select_move(board) in ((5, 1), (5, 6))


def test_pruning_and_full_minimax_return_same_move_and_visit_counts():
    board = Board()
    for row in range(SIZE):
        for col in range(SIZE):
            board.grid[row * SIZE + col] = (
                BLACK if (row + 2 * col) % 5 < 2 else WHITE
            )
    for row, col in ((2, 2), (2, 8), (5, 5), (5, 6), (8, 2), (8, 8)):
        board.grid[row * SIZE + col] = 0
    rules = get_rules("normal")
    pruned = AlphaBetaPlayer(rules, depth=2, pruning=True)
    plain = AlphaBetaPlayer(rules, depth=2, pruning=False)

    assert pruned.select_move(board) == plain.select_move(board)
    assert pruned.nodes_visited < plain.nodes_visited
    assert board.history == []


def test_deadline_restores_board_after_interrupted_iteration(monkeypatch):
    import numpy as np
    board = Board()
    board.play(5, 5)
    before = board.copy()
    player = AlphaBetaPlayer(get_rules("normal"), depth=1, time_budget_sec=1)
    monkeypatch.setattr(player, "_score_after", lambda *_: 0)
    def expire(*_):
        raise TimeoutError("expired inside child")
    monkeypatch.setattr(player, "_search", expire)
    move = player.select_move(board)
    assert move[0] * SIZE + move[1] in player.rules.legal_moves(board)
    assert np.array_equal(board.grid, before.grid)
    assert board.history == before.history
    assert board.current_player == before.current_player


def test_time_mode_deepens_beyond_fixed_depth(monkeypatch):
    import players.alphabeta as module
    player = AlphaBetaPlayer(get_rules("normal"), depth=1, time_budget_sec=1)
    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(player, "_score_after", lambda *_: 0)
    def evaluate(_board, depth, *_):
        clock[0] += 0.002
        return depth
    monkeypatch.setattr(player, "_search", evaluate)
    player.select_move(Board())
    assert player.completed_depth >= 2
