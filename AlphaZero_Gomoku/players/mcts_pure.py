"""UCT search with random rollouts and no neural network."""

from __future__ import annotations

import math
import random

from gomoku.board import BLACK, EMPTY, SIZE, WHITE, Board
from players.base import Player


def _other(player: int) -> int:
    return BLACK if player == WHITE else WHITE


class _Node:
    __slots__ = ("parent", "move", "player_to_move", "children", "visits", "value_sum")

    def __init__(self, parent, move: int, player_to_move: int) -> None:
        self.parent = parent
        self.move = move
        self.player_to_move = player_to_move
        self.children: dict[int, _Node] = {}
        self.visits = 0
        self.value_sum = 0.0


class MCTSPure(Player):
    """Choose moves with UCT and random rollouts."""

    name = "mcts-pure"

    def __init__(
        self,
        rules,
        n_rollouts: int = 200,
        *,
        c_uct: float = math.sqrt(2.0),
        max_playouts: int = SIZE * SIZE,
        rng: random.Random | None = None,
    ) -> None:
        if n_rollouts <= 0:
            raise ValueError("n_rollouts must be positive")
        if c_uct < 0:
            raise ValueError("c_uct must be non-negative")
        if max_playouts <= 0:
            raise ValueError("max_playouts must be positive")
        self.rules = rules
        self.n_rollouts = int(n_rollouts)
        self.c_uct = float(c_uct)
        self.max_playouts = int(max_playouts)
        self.rng = rng if rng is not None else random.Random()

    def select_move(self, board: Board) -> tuple[int, int]:
        move, _ = self.search(board)
        return divmod(move, SIZE)

    def search(self, board: Board) -> tuple[int, list[tuple[int, int]]]:
        legal = self.rules.legal_moves(board)
        if not legal or self.rules.is_win(board):
            raise RuntimeError("MCTSPure asked to move on a terminal board")

        work = Board()
        work.grid[:] = board.grid
        work.current_player = board.current_player
        work.history[:] = board.history
        mover = board.current_player

        for move in legal:
            work.play(*divmod(move, SIZE))
            won = self.rules.is_win(work) and int(work.grid[work.history[-1]]) == mover
            work.undo()
            if won:
                return move, [(candidate, self.n_rollouts if candidate == move else 0) for candidate in legal]

        root = _Node(parent=None, move=-1, player_to_move=mover)
        root_history_len = len(work.history)
        for _ in range(self.n_rollouts):
            node = root
            path = [root]
            while True:
                if self.rules.is_win(work):
                    winner = self._winner(work)
                    self._backup(path, winner)
                    break

                current_legal = self.rules.legal_moves(work)
                if not current_legal:
                    self._backup(path, EMPTY)
                    break
                untried = [move for move in current_legal if move not in node.children]
                if untried:
                    move = self.rng.choice(untried)
                    work.play(*divmod(move, SIZE))
                    child = _Node(node, move, work.current_player)
                    node.children[move] = child
                    path.append(child)
                    winner = (
                        self._winner(work)
                        if self.rules.is_win(work)
                        else self._rollout(work)
                    )
                    self._backup(path, winner)
                    break

                move, child = self._select_child(node)
                work.play(*divmod(move, SIZE))
                node = child
                path.append(node)

            while len(work.history) > root_history_len:
                work.undo()

        visited = [move for move in legal if move in root.children]
        best = max(visited, key=lambda move: root.children[move].visits)
        counts = [
            (move, root.children[move].visits if move in root.children else 0)
            for move in legal
        ]
        return best, counts

    def _select_child(self, node: _Node) -> tuple[int, _Node]:
        best_move = -1
        best_child = None
        best_score = -math.inf
        log_parent_visits = math.log(max(1, node.visits))

        for move, child in node.children.items():
            if child.visits == 0:
                score = math.inf
            else:
                child_value = child.value_sum / child.visits
                score = (
                    1.0 - child_value
                    + self.c_uct
                    * math.sqrt(log_parent_visits / child.visits)
                )
            if score > best_score:
                best_move, best_child, best_score = move, child, score
        return best_move, best_child

    def _rollout(self, board: Board) -> int:
        for _ in range(self.max_playouts):
            legal = self.rules.legal_moves(board)
            if not legal:
                return EMPTY
            board.play(*divmod(self.rng.choice(legal), SIZE))
            if self.rules.is_win(board):
                return int(board.grid[board.history[-1]])
        return EMPTY

    def _backup(self, path: list[_Node], winner: int) -> None:
        for node in reversed(path):
            node.visits += 1
            if winner == EMPTY:
                node.value_sum += 0.5
            elif winner == node.player_to_move:
                node.value_sum += 1.0

    def _winner(self, board: Board) -> int:
        if not self.rules.is_win(board) or not board.history:
            return EMPTY
        return int(board.grid[board.history[-1]])
