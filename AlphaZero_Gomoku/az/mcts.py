from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class LeafRequest:
    """Request for neural network evaluation on a leaf node."""
    node: Node
    planes: np.ndarray          # (4, 11, 11) uint8
    legal_mask: np.ndarray      # (121,) bool


class Node:
    """A node in the MCTS search tree."""

    def __init__(self, parent: Node | None = None, move: int = -1, prior: float = 0.0) -> None:
        self.parent = parent
        self.move = move
        self.prior = float(prior)
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.children: dict[int, Node] = {}
        self.is_expanded: bool = False
        self.is_terminal: bool = False
        self.terminal_value: float = 0.0

    @property
    def q_value(self) -> float:
        """Mean action value from the perspective of the player to move at this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTS:
    """AlphaZero Monte Carlo Tree Search with step-by-step request/receive API."""

    def __init__(
        self,
        rules,
        evaluator=None,
        n_simulations: int = 200,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.15,
        dirichlet_eps: float = 0.25,
        root_noise: bool = False,
        feature_fn=None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.rules = rules
        self.evaluator = evaluator
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.root_noise = root_noise
        self.feature_fn = feature_fn
        self.rng = rng if rng is not None else np.random.default_rng()

        self.board = None
        self.root: Node | None = None
        self.simulations_done: int = 0
        self.nn_calls: int = 0

        self.pending_request: LeafRequest | None = None
        self.pending_path: list[tuple[Node, int, Node]] | None = None

    def reset(self, board) -> None:
        """Reset the search tree for a new move decision on the given board."""
        self.board = board
        self.root = Node()
        self.simulations_done = 0
        self.pending_request = None
        self.pending_path = None

        if hasattr(self.rules, "is_terminal") and self.rules.is_terminal(self.board):
            self.root.is_terminal = True
            if hasattr(self.rules, "is_win") and self.rules.is_win(self.board):
                self.root.terminal_value = -1.0
            else:
                self.root.terminal_value = 0.0

    def request(self) -> LeafRequest | None:
        """Advance search until a non-terminal leaf requires evaluation, or budget exhausted."""
        if self.pending_request is not None:
            raise RuntimeError("Cannot request() while a LeafRequest is already pending. Call receive() first.")
        if self.board is None or self.root is None:
            raise RuntimeError("MCTS must be reset(board) before calling request().")
        if self.root.is_terminal:
            return None

        while self.simulations_done < self.n_simulations:
            current = self.root
            path: list[tuple[Node, int, Node]] = []

            # 1. Traversal downwards
            while current.is_expanded and not current.is_terminal:
                move, next_child = self._select_child(current)
                self._play_move(self.board, move)
                path.append((current, move, next_child))
                current = next_child

            # 2. Terminal check
            if current.is_terminal:
                self._backup_terminal(path, current.terminal_value)
                self._undo_path(path)
                self.simulations_done += 1
                continue

            # Check if leaf state on the board is terminal
            is_win = hasattr(self.rules, "is_win") and self.rules.is_win(self.board)
            is_term = hasattr(self.rules, "is_terminal") and self.rules.is_terminal(self.board)

            if is_win:
                current.is_terminal = True
                current.terminal_value = -1.0
                current.is_expanded = True
                self._backup_terminal(path, -1.0)
                self._undo_path(path)
                self.simulations_done += 1
                continue
            elif is_term:
                current.is_terminal = True
                current.terminal_value = 0.0
                current.is_expanded = True
                self._backup_terminal(path, 0.0)
                self._undo_path(path)
                self.simulations_done += 1
                continue

            # 3. Leaf requires evaluation
            planes = self._get_planes(self.board)
            legal_mask = self._get_legal_mask(self.board)

            if not np.any(legal_mask):
                current.is_terminal = True
                current.terminal_value = 0.0
                current.is_expanded = True
                self._backup_terminal(path, 0.0)
                self._undo_path(path)
                self.simulations_done += 1
                continue

            self.pending_path = path
            self._undo_path(path)

            req = LeafRequest(node=current, planes=planes, legal_mask=legal_mask)
            self.pending_request = req
            return req

        return None

    def receive(self, policy: np.ndarray, value: float) -> None:
        """Receive evaluation for the pending leaf, expand its children, and backup value."""
        if self.pending_request is None:
            raise RuntimeError("No pending LeafRequest to receive().")

        req = self.pending_request
        node = req.node
        legal_mask = req.legal_mask

        # Mask and normalize policy priors
        masked_policy = policy * legal_mask
        p_sum = float(np.sum(masked_policy))
        if p_sum > 1e-8:
            priors = masked_policy / p_sum
        else:
            legal_count = int(np.sum(legal_mask))
            priors = np.where(legal_mask, 1.0 / legal_count, 0.0) if legal_count > 0 else np.zeros_like(masked_policy)

        # Dirichlet noise at root if enabled
        if node is self.root and self.root_noise:
            legal_indices = np.flatnonzero(legal_mask)
            if len(legal_indices) > 0:
                noise = self.rng.dirichlet([self.dirichlet_alpha] * len(legal_indices))
                for idx, move in enumerate(legal_indices):
                    priors[move] = (1.0 - self.dirichlet_eps) * priors[move] + self.dirichlet_eps * noise[idx]

        # Expand node children
        for move in np.flatnonzero(legal_mask):
            node.children[int(move)] = Node(parent=node, move=int(move), prior=float(priors[move]))
        node.is_expanded = True
        self.nn_calls += 1

        # Backup: value is from perspective of player at leaf `node`
        node.visit_count += 1
        node.value_sum += float(value)

        # Walk up path, alternating perspective
        v = float(value)
        if self.pending_path:
            for parent, _, child in reversed(self.pending_path):
                v = -v
                parent.visit_count += 1
                parent.value_sum += v

        self.simulations_done += 1
        self.pending_request = None
        self.pending_path = None

    def visit_counts(self) -> np.ndarray:
        """Return visit counts array of shape (121,) for moves at root."""
        counts = np.zeros(121, dtype=np.int32)
        if self.root is not None:
            for move, child in self.root.children.items():
                counts[move] = child.visit_count
        return counts

    def select_move(self, temperature: float = 1.0) -> int:
        """Select a move based on root visit counts and temperature."""
        counts = self.visit_counts()
        total_visits = int(np.sum(counts))
        if total_visits == 0:
            raise ValueError("Cannot select_move with 0 visit counts.")

        if temperature == 0.0:
            max_c = np.max(counts)
            candidates = np.flatnonzero(counts == max_c)
            return int(self.rng.choice(candidates))

        # Sample proportionally to visits^(1/tau)
        visits = counts.astype(np.float64)
        probs = visits ** (1.0 / float(temperature))
        p_sum = float(np.sum(probs))
        if p_sum > 0:
            probs = probs / p_sum
            return int(self.rng.choice(len(probs), p=probs))

        return int(self.rng.choice(np.flatnonzero(counts > 0)))

    def root_value(self) -> float:
        """Estimated value from perspective of current player at root."""
        return self.root.q_value if self.root is not None else 0.0

    def search(self, board, evaluator=None) -> np.ndarray:
        """Deterministic convenience runner: run search to completion and return visit counts."""
        evaluator = evaluator or self.evaluator
        if evaluator is None:
            raise ValueError("An evaluator must be provided to run search().")

        self.reset(board)
        while True:
            req = self.request()
            if req is None:
                break
            if hasattr(evaluator, "predict_batch"):
                batch_res = evaluator.predict_batch([req])
                policy, val = batch_res[0]
            elif hasattr(evaluator, "predict"):
                probs, vals = evaluator.predict(req.planes[np.newaxis, ...])
                policy, val = probs[0], float(vals[0])
            elif callable(evaluator):
                policy, val = evaluator(req)
            else:
                raise TypeError(f"Unsupported evaluator type: {type(evaluator)}")
            self.receive(policy, val)

        return self.visit_counts()

    def _select_child(self, node: Node) -> tuple[int, Node]:
        """Select child maximizing PUCT score with random tie-breaking."""
        best_score = -float("inf")
        best_candidates: list[tuple[int, Node]] = []
        total_n = max(1, node.visit_count)
        sqrt_total_n = np.sqrt(total_n)

        for move, child in node.children.items():
            # Q is from parent perspective: -child.q_value
            q_val = -child.q_value
            u_val = self.c_puct * child.prior * (sqrt_total_n / (1.0 + child.visit_count))
            score = q_val + u_val

            if score > best_score + 1e-9:
                best_score = score
                best_candidates = [(move, child)]
            elif abs(score - best_score) <= 1e-9:
                best_candidates.append((move, child))

        idx = self.rng.integers(len(best_candidates))
        return best_candidates[idx]

    def _backup_terminal(self, path: list[tuple[Node, int, Node]], terminal_value: float) -> None:
        """Backup terminal value up the path with alternating perspective."""
        v = float(terminal_value)
        for parent, _, child in reversed(path):
            child.visit_count += 1
            child.value_sum += v
            v = -v
        self.root.visit_count += 1
        self.root.value_sum += v

    def _play_move(self, board, move: int) -> None:
        try:
            board.play(move // 11, move % 11)
        except TypeError:
            board.play(move)

    def _undo_move(self, board) -> None:
        board.undo()

    def _undo_path(self, path: list[tuple[Node, int, Node]]) -> None:
        for _ in path:
            self._undo_move(self.board)

    def _get_planes(self, board) -> np.ndarray:
        if self.feature_fn is not None:
            curr_player = getattr(board, "current_player", 1)
            return self.feature_fn(board, self.rules, curr_player)
        try:
            from gomoku.features import board_to_planes
            curr_player = getattr(board, "current_player", 1)
            return board_to_planes(board, self.rules, curr_player)
        except (ImportError, ModuleNotFoundError, AttributeError):
            # ponytail: mock fallback planes, remove when gomoku.features ready
            if hasattr(board, "get_planes"):
                return board.get_planes()
            return np.zeros((4, 11, 11), dtype=np.uint8)

    def _get_legal_mask(self, board) -> np.ndarray:
        mask = np.zeros(121, dtype=bool)
        if hasattr(self.rules, "legal_moves"):
            legals = self.rules.legal_moves(board)
            for m in legals:
                if isinstance(m, (tuple, list)):
                    mask[m[0] * 11 + m[1]] = True
                else:
                    mask[int(m)] = True

        if hasattr(self.rules, "forbidden_moves"):
            for m in self.rules.forbidden_moves(board):
                if isinstance(m, (tuple, list)):
                    mask[m[0] * 11 + m[1]] = False
                else:
                    mask[int(m)] = False

        return mask
