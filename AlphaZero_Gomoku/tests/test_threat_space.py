from gomoku.board import BLACK, SIZE, WHITE, Board
from gomoku.patterns import find_patterns
from gomoku.rules import get_rules
from az.threat_space import ThreatSpaceAZPlayer, ThreatSpaceSearcher


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


def forced_win_board() -> Board:
    # Black: horizontal three (blocked left) plus vertical three. Playing (9, 5)
    # makes an open four; playing (6, 7) makes a blocked four. Both force a win.
    return board_with(
        {
            (6, 4): BLACK,
            (6, 5): BLACK,
            (6, 6): BLACK,
            (7, 5): BLACK,
            (8, 5): BLACK,
            (6, 3): WHITE,
        },
        BLACK,
    )


def blocked_four_board() -> Board:
    # Only forcing move is (6, 7); the line resolves after one forced block.
    return board_with({(6, 4): BLACK, (6, 5): BLACK, (6, 6): BLACK, (6, 3): WHITE}, BLACK)


def double_threat_board() -> Board:
    return board_with({(5, c): WHITE for c in (3, 4, 5, 6)}, BLACK)


def test_finds_forced_win_from_four_chain():
    board = forced_win_board()
    before = board.grid.copy()
    rules = get_rules("normal")

    result = ThreatSpaceSearcher().search(board, rules, BLACK)

    assert result.triggered is True
    assert result.value == 1.0
    assert result.move is not None
    assert result.move in rules.legal_moves(board)
    r, c = divmod(result.move, SIZE)
    assert any(p.kind in ("open_four", "blocked_four") for p in find_patterns(board, r, c, BLACK))
    assert (board.grid == before).all()
    assert board.history == []


def test_reports_loss_against_two_simultaneous_threats():
    board = double_threat_board()

    result = ThreatSpaceSearcher().search(board, get_rules("normal"), BLACK)

    assert result.triggered is True
    assert result.value == -1.0
    assert result.move is None


def test_respects_node_limit():
    board = blocked_four_board()

    result = ThreatSpaceSearcher(max_nodes=1).search(board, get_rules("normal"), BLACK)

    assert result.limit_reached is True
    assert result.nodes == 1


def test_no_threat_does_not_trigger():
    board = board_with({(5, 5): BLACK, (5, 6): WHITE}, BLACK)

    result = ThreatSpaceSearcher().search(board, get_rules("normal"), BLACK)

    assert result.triggered is False
    assert result.value == 0.0


def test_disabled_flag_delegates_to_base():
    rules = get_rules("normal")
    board = forced_win_board()
    base = FixedPlayer(rules, 0)

    player = ThreatSpaceAZPlayer(base, cfg={"threat_space": {"enabled": False}})

    assert player.select_move(board) == (0, 0)
    assert base.calls == 1
    assert player.activations == 0


def test_enabled_uses_forced_win_and_counts_activation():
    rules = get_rules("normal")
    board = forced_win_board()
    base = FixedPlayer(rules, 0)

    player = ThreatSpaceAZPlayer(base, cfg={"threat_space": {"enabled": True}})
    move = player.select_move(board)
    flat = move[0] * SIZE + move[1]

    assert flat in rules.legal_moves(board)
    assert any(p.kind in ("open_four", "blocked_four") for p in find_patterns(board, move[0], move[1], BLACK))
    assert player.last_override_reason == "threat_space"
    assert player.activations == 1
    assert base.calls == 0
    assert board.history == []


def test_pattern_alone_does_not_prove_forced_win(monkeypatch):
    from types import SimpleNamespace
    import az.threat_space as module

    class Rules:
        def legal_moves(self, board):
            return [i for i in (0, 1, 2) if board.grid[i] == 0]

        def is_win(self, board):
            return False

    monkeypatch.setattr(module, "find_patterns", lambda *args: [SimpleNamespace(kind="open_four", completion_cells={(0, 1), (0, 2)})])
    board = Board()
    result = ThreatSpaceSearcher().search(board, Rules())
    assert result.value == 0
    assert result.move is None
    assert not board.grid.any()
    assert not board.history
    assert board.current_player == BLACK


def test_threat_and_base_telemetry_and_budget_are_per_move(monkeypatch):
    from az.threat_space import ThreatResult
    import az.player_tactical as tactical

    rules = get_rules("normal")
    base = FixedPlayer(rules, 0)
    base.time_budget_sec = 1.0
    base.nn_calls = 2
    base.simulations_run = 2
    base.nodes_visited = 3
    player = ThreatSpaceAZPlayer(base, cfg={"threat_space": {"enabled": True}})
    clock = [0.0]
    monkeypatch.setattr(tactical.time, "monotonic", lambda: clock[0])

    def search(*args):
        clock[0] += 0.4
        return ThreatResult(None, 0.0, 5, True, False)

    def select_move(board):
        assert abs(base.time_budget_sec - 0.6) < 1e-8
        return (0, 0)

    monkeypatch.setattr(player.searcher, "search", search)
    monkeypatch.setattr(base, "select_move", select_move)
    player.select_move(Board())
    assert player.nodes_visited == 8
    assert player.simulations_run == player.nn_calls == 2
    assert base.time_budget_sec == 1.0
    monkeypatch.setattr(player.searcher, "search", lambda *args: ThreatResult(1, 1.0, 4, True, False))
    assert player.select_move(Board()) == (0, 1)
    assert player.nodes_visited == 4
    assert player.simulations_run == player.nn_calls == 0
    assert player.last_override_reason == "threat_space"
    player.threat_enabled = False
    clock[0] = 0.0
    monkeypatch.setattr(base, "select_move", lambda board: (0, 0))
    player.select_move(Board())
    assert player.last_threat_value == 0.0
    assert player.last_override_reason is None
