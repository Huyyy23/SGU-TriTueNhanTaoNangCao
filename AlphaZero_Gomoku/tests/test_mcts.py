import sys
import time
from pathlib import Path
import numpy as np
import pytest

# Ensure AlphaZero_Gomoku root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from az.mcts import MCTS, LeafRequest, Node


class MockBoard:
    """11x11 Gomoku board stub with in-place play/undo for testing MCTS."""

    def __init__(self, size: int = 11) -> None:
        self.size = size
        self.grid = np.zeros(size * size, dtype=np.int8)
        self.current_player: int = 1
        self.history: list[int] = []

    def play(self, r: int, c: int | None = None) -> None:
        idx = r if c is None else r * self.size + c
        if self.grid[idx] != 0:
            raise ValueError(f"Cell {idx} is already occupied.")
        self.grid[idx] = self.current_player
        self.history.append(idx)
        self.current_player = 3 - self.current_player

    def undo(self) -> None:
        if not self.history:
            raise ValueError("No moves to undo.")
        idx = self.history.pop()
        self.grid[idx] = 0
        self.current_player = 3 - self.current_player


class MockRules:
    """Rules stub supporting custom win condition, legal moves, and forbidden moves."""

    def __init__(self, forbidden: set[int] | None = None) -> None:
        self.forbidden = forbidden if forbidden is not None else set()

    def legal_moves(self, board: MockBoard) -> list[int]:
        return list(np.flatnonzero(board.grid == 0))

    def forbidden_moves(self, board: MockBoard) -> list[int]:
        return list(self.forbidden)

    def is_win(self, board: MockBoard) -> bool:
        if not board.history:
            return False
        last_move = board.history[-1]
        player = board.grid[last_move]
        r, c = divmod(last_move, board.size)
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dr, dc in directions:
            count = 1
            # forward
            nr, nc = r + dr, c + dc
            while 0 <= nr < board.size and 0 <= nc < board.size and board.grid[nr * board.size + nc] == player:
                count += 1
                nr += dr
                nc += dc
            # backward
            nr, nc = r - dr, c - dc
            while 0 <= nr < board.size and 0 <= nc < board.size and board.grid[nr * board.size + nc] == player:
                count += 1
                nr -= dr
                nc -= dc
            if count >= 5:
                return True
        return False

    def is_terminal(self, board: MockBoard) -> bool:
        return self.is_win(board) or len(self.legal_moves(board)) == 0


class StubEvaluator:
    """Stub evaluator returning uniform policy and fixed value."""

    def __init__(self, value: float = 0.0) -> None:
        self.value = float(value)

    def predict(self, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        batch_size = planes.shape[0]
        probs = np.ones((batch_size, 121), dtype=np.float32) / 121.0
        values = np.full((batch_size,), self.value, dtype=np.float32)
        return probs, values

    def predict_batch(self, requests: list[LeafRequest]) -> list[tuple[np.ndarray, float]]:
        res = []
        for _ in requests:
            res.append((np.ones(121, dtype=np.float32) / 121.0, self.value))
        return res


# Gate Test 1: Win-in-1
def test_win_in_1():
    board = MockBoard()
    rules = MockRules()
    # Place 4 Black stones in a row at (5, 1), (5, 2), (5, 3), (5, 4)
    # White blocks (5, 0), leaving (5, 5) as the unique winning move for Black
    board.play(5, 1)  # Black (1)
    board.play(5, 0)  # White (2) blocks left end
    board.play(5, 2)
    board.play(0, 0)
    board.play(5, 3)
    board.play(0, 1)
    board.play(5, 4)
    board.play(0, 2)

    # Next move is Black's turn. Unique winning move is (5, 5) -> idx 60.
    winning_move = 5 * 11 + 5
    mcts = MCTS(rules=rules, n_simulations=150, c_puct=1.5, root_noise=False)
    evaluator = StubEvaluator(value=0.0)

    counts = mcts.search(board, evaluator=evaluator)
    best_move = mcts.select_move(temperature=0.0)

    assert best_move == winning_move
    assert counts[winning_move] > 0


# Gate Test 2: Backup negates child value (Hard Rule 2)
def test_backup_negates_child_value():
    board = MockBoard()
    rules = MockRules()
    mcts = MCTS(rules=rules, n_simulations=2, c_puct=1.5, root_noise=False)
    mcts.reset(board)

    # Simulation 1: root request
    req1 = mcts.request()
    assert req1 is not None
    assert req1.node is mcts.root
    # Root evaluated with value = +0.6 (Black advantage)
    uniform_policy = np.ones(121, dtype=np.float32) / 121.0
    mcts.receive(uniform_policy, 0.6)

    assert mcts.root.visit_count == 1
    assert pytest.approx(mcts.root.value_sum) == 0.6
    assert pytest.approx(mcts.root.q_value) == 0.6

    # Simulation 2: descends to child node
    req2 = mcts.request()
    assert req2 is not None
    child_node = req2.node
    assert child_node is not mcts.root
    assert child_node.parent is mcts.root

    # Child is evaluated with value = +0.8 (meaning White is advantageous at child)
    mcts.receive(uniform_policy, 0.8)

    # Verification:
    # 1. Child node stores value from perspective of player at child (White)
    assert child_node.visit_count == 1
    assert pytest.approx(child_node.value_sum) == 0.8
    assert pytest.approx(child_node.q_value) == 0.8

    # 2. Root (parent) received -0.8 (negated child value!)
    # Root had 0.6 from sim 1, now + (-0.8) = -0.2
    assert mcts.root.visit_count == 2
    assert pytest.approx(mcts.root.value_sum) == 0.6 - 0.8
    assert pytest.approx(mcts.root.q_value) == (0.6 - 0.8) / 2

    # 3. Action value Q(root, child.move) from parent's perspective is -child.q_value = -0.8
    assert pytest.approx(-child_node.q_value) == -0.8


# Gate Test 3: nn_calls equals expanded leaves, excluding terminal nodes
def test_nn_calls_excludes_terminal_leaves():
    board = MockBoard()
    rules = MockRules()
    # Setup board where Black has 4 in a row, (5, 0) blocked, next move (5, 5) wins
    board.play(5, 1)  # Black (1)
    board.play(5, 0)  # White (2) blocks left end
    board.play(5, 2)
    board.play(0, 0)
    board.play(5, 3)
    board.play(0, 1)
    board.play(5, 4)
    board.play(0, 2)

    mcts = MCTS(rules=rules, n_simulations=150, c_puct=1.5, root_noise=False)
    evaluator = StubEvaluator(value=0.0)
    mcts.search(board, evaluator=evaluator)

    # Winning move (5, 5) was visited
    win_move = 5 * 11 + 5
    assert win_move in mcts.root.children
    winning_child = mcts.root.children[win_move]
    assert winning_child.is_terminal is True
    assert winning_child.is_expanded is True
    # The terminal child was never sent to NN, so nn_calls count is exactly
    # the number of non-terminal leaves that called receive()
    # Let's count non-terminal expanded nodes in the tree
    def count_expanded_non_terminal(node: Node) -> int:
        c = 1 if node.is_expanded and not node.is_terminal else 0
        for child in node.children.values():
            c += count_expanded_non_terminal(child)
        return c

    expected_nn_calls = count_expanded_non_terminal(mcts.root)
    assert mcts.nn_calls == expected_nn_calls


# Gate Test 4: Forbidden moves have prior 0 and are never expanded (Hard Rule 4)
def test_forbidden_moves_masked_and_never_expanded():
    board = MockBoard()
    forbidden_move = 60  # cell (5, 5)
    rules = MockRules(forbidden={forbidden_move})

    mcts = MCTS(rules=rules, n_simulations=30, c_puct=1.5, root_noise=True)
    mcts.reset(board)

    # Root request
    req = mcts.request()
    assert req.legal_mask[forbidden_move] == False

    # Receive uniform policy where forbidden_move had high probability
    policy = np.zeros(121, dtype=np.float32)
    policy[forbidden_move] = 0.99
    policy[0] = 0.01
    mcts.receive(policy, 0.0)

    # Check root children: forbidden_move must NOT be among children
    assert forbidden_move not in mcts.root.children
    assert 0 in mcts.root.children

    # Complete search
    evaluator = StubEvaluator()
    counts = mcts.search(board, evaluator)
    assert counts[forbidden_move] == 0


# Gate Test 5: request() returns None when simulation budget is exhausted
def test_request_returns_none_when_budget_exhausted():
    board = MockBoard()
    rules = MockRules()
    n_sims = 10
    mcts = MCTS(rules=rules, n_simulations=n_sims, c_puct=1.5)
    mcts.reset(board)

    evaluator = StubEvaluator()
    for _ in range(n_sims):
        req = mcts.request()
        if req is None:
            break
        policy, val = evaluator.predict_batch([req])[0]
        mcts.receive(policy, val)

    assert mcts.simulations_done == n_sims
    # Budget exhausted -> request() must return None
    assert mcts.request() is None


# CPU Benchmark: 200 simulations with stub evaluator
def test_benchmark_200_sims():
    board = MockBoard()
    rules = MockRules()
    mcts = MCTS(rules=rules, n_simulations=200, c_puct=1.5, root_noise=False)
    evaluator = StubEvaluator()

    start_time = time.perf_counter()
    counts = mcts.search(board, evaluator=evaluator)
    elapsed_sec = time.perf_counter() - start_time

    assert np.sum(counts) > 0
    print(f"\n[BENCHMARK] MCTS 200 simulations with StubEvaluator: {elapsed_sec:.4f} s/move")
