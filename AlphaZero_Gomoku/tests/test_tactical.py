from gomoku.board import BLACK, SIZE, WHITE, Board
from gomoku.rules import RenjuRules, get_rules
from az.player_tactical import TacticalAZPlayer


class FixedPlayer:
    name = "fixed"

    def __init__(self, rules, move):
        self.rules = rules
        self.move = move
        self.calls = 0

    def select_move(self, board):
        self.calls += 1
        return divmod(self.move, SIZE)


def board_with(stones: dict[tuple[int, int], int], player: int) -> Board:
    board = Board()
    for (r, c), stone in stones.items():
        board.grid[r * SIZE + c] = stone
    board.current_player = player
    return board


def test_disabled_flag_delegates_to_base():
    rules = get_rules("normal")
    board = board_with({(5, c): BLACK for c in (2, 3, 4, 5)}, BLACK)
    base = FixedPlayer(rules, 0)

    player = TacticalAZPlayer(base, cfg={"tactical": {"enabled": False}})

    assert player.select_move(board) == (0, 0)
    assert base.calls == 1
    assert player.last_override_applied is False


def test_enabled_picks_win_in_one():
    rules = get_rules("normal")
    board = board_with({(5, c): BLACK for c in (2, 3, 4, 5)}, BLACK)
    base = FixedPlayer(rules, 0)

    player = TacticalAZPlayer(base, enabled=True)

    assert player.select_move(board) in ((5, 1), (5, 6))
    assert player.last_override_reason == "win_now"
    assert base.calls == 0


def test_enabled_blocks_opponent_win_in_one():
    rules = get_rules("normal")
    board = board_with({(5, c): WHITE for c in (2, 3, 4, 5)}, BLACK)
    base = FixedPlayer(rules, 0)

    player = TacticalAZPlayer(base, enabled=True)

    assert player.select_move(board) in ((5, 1), (5, 6))
    assert player.last_override_reason == "block_win"


def test_enabled_without_tactic_delegates_to_base():
    rules = get_rules("normal")
    board = Board()
    base = FixedPlayer(rules, 0)

    player = TacticalAZPlayer(base, enabled=True)

    assert player.select_move(board) == (0, 0)
    assert base.calls == 1
    assert player.last_override_applied is False


def test_override_restores_board():
    rules = get_rules("normal")
    board = board_with({(5, c): BLACK for c in (2, 3, 4, 5)}, BLACK)
    before = board.grid.copy()

    player = TacticalAZPlayer(FixedPlayer(rules, 0), enabled=True)
    player.select_move(board)

    assert (board.grid == before).all()
    assert board.history == []
    assert board.current_player == BLACK


def test_never_returns_renju_forbidden_move():
    rules = RenjuRules()
    # Black would make six in a row at (5, 5); overline is forbidden for Black.
    board = board_with({(5, c): BLACK for c in (1, 2, 3, 4, 6)}, BLACK)
    before = board.grid.copy()

    player = TacticalAZPlayer(FixedPlayer(rules, 0), enabled=True)
    move = player.select_move(board)
    flat = move[0] * SIZE + move[1]

    assert move not in ((5, 5),)
    assert flat in rules.legal_moves(board)
    assert flat not in rules.forbidden_moves(board)
    assert (board.grid == before).all()


def test_opponent_candidates_use_opponent_legal_set():
    class AsymmetricRules:
        def legal_moves(self, board):
            return [0, 1] if board.current_player == WHITE else [1]

        def is_win(self, board):
            return board.grid[0] == BLACK

    board = board_with({}, WHITE)
    base = FixedPlayer(AsymmetricRules(), 1)
    player = TacticalAZPlayer(base, enabled=True)
    assert player.select_move(board) == (0, 1)
    assert not player.last_override_applied
    assert base.calls == 1
    assert board.current_player == WHITE
    assert not board.history


def test_override_clears_stale_counters_and_thinking_budget(monkeypatch):
    rules = get_rules("normal")
    base = FixedPlayer(rules, 0)
    base.time_budget_sec = 1.0
    base.nn_calls = 3
    base.simulations_run = 3
    base.nodes_visited = 4
    clock = [0.0]
    monkeypatch.setattr("az.player_tactical.time.monotonic", lambda: clock[0])
    player = TacticalAZPlayer(base, enabled=False)
    original = base.select_move

    def select_move(board):
        assert base.time_budget_sec == 1.0
        return original(board)

    monkeypatch.setattr(base, "select_move", select_move)
    player.select_move(Board())
    assert player.nn_calls == player.simulations_run == 3
    assert player.nodes_visited == 4
    player.enabled = True
    board = board_with({(5, c): BLACK for c in (2, 3, 4, 5)}, BLACK)
    player.select_move(board)
    assert player.last_override_reason == "win_now"
    assert player.nn_calls == player.simulations_run == player.nodes_visited == 0
    assert base.time_budget_sec == 1.0
