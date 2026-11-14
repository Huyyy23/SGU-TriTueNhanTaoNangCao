"""Forced-threat search over four-threat chains (VCF), off by default."""

from __future__ import annotations

import time
from dataclasses import dataclass

from gomoku.board import BLACK, SIZE, WHITE
from gomoku.patterns import find_patterns

from az.player_tactical import TacticalAZPlayer


@dataclass(frozen=True)
class ThreatResult:
    move: int | None
    value: float  # from the perspective of the side to move: +1 win, -1 loss
    nodes: int
    triggered: bool
    limit_reached: bool


class _Budget(Exception):
    pass


class ThreatSpaceSearcher:
    """Search only forced four-threat chains; return +1/-1 or 0 when unknown."""

    def __init__(self, max_nodes: int = 5000, time_limit_ms: float = 150.0) -> None:
        if max_nodes < 1:
            raise ValueError("max_nodes must be positive")
        if time_limit_ms <= 0:
            raise ValueError("time_limit_ms must be positive")
        self.max_nodes = max_nodes
        self.time_limit_ms = time_limit_ms

    def search(self, board, rules, player: int | None = None) -> ThreatResult:
        if not (board.grid == 0).any():
            return ThreatResult(None, 0.0, 0, False, False)
        player = board.current_player if player is None else player
        self._board = board
        self._rules = rules
        self._player = player
        self._nodes = 0
        self._deadline = time.monotonic() + self.time_limit_ms / 1000.0
        self.triggered = False
        self.limit_reached = False

        try:
            value, move = self._attack(player)
        except _Budget:
            self.limit_reached = True
            value, move = 0.0, None

        return ThreatResult(move, value, self._nodes, self.triggered, self.limit_reached)

    # -- search -----------------------------------------------------------

    def _attack(self, player: int) -> tuple[float, int | None]:
        if self._nodes >= self.max_nodes or time.monotonic() >= self._deadline:
            raise _Budget
        self._nodes += 1

        legal = self._legal_for(player)
        win = self._first_win(legal, player)
        if win is not None:
            self.triggered = True
            return 1.0, win

        opponent = WHITE if player == BLACK else BLACK
        opponent_legal = self._legal_for(opponent)
        opponent_wins = [move for move in opponent_legal if self._wins_after(move, opponent)]
        if len(opponent_wins) >= 2:
            # Two immediate wins cannot both be blocked: the side to move loses.
            self.triggered = True
            return -1.0, None

        forcing = []
        for move in legal:
            if time.monotonic() >= self._deadline:
                raise _Budget
            r, c = divmod(move, SIZE)
            fours = [p for p in find_patterns(self._board, r, c, player) if p.kind in ("open_four", "blocked_four")]
            if fours:
                forcing.append((move, fours))
        if not forcing:
            return 0.0, None
        self.triggered = True
        forcing.sort(key=lambda item: 0 if any(p.kind == "open_four" for p in item[1]) else 1)

        for move, fours in forcing:
            saved = self._place(move, player)
            try:
                if self._rules.is_win(self._board):
                    return 1.0, move
                defender_legal = self._legal_for(opponent)
                if self._first_win(defender_legal, opponent) is not None:
                    continue  # opponent wins before blocking.
                # Patterns nominate moves; only legal winning continuations prove threats.
                candidates = {r * SIZE + c for pattern in fours for r, c in pattern.completion_cells}
                attacker_legal = set(self._legal_for(player))
                completions = [cell for cell in candidates if cell in attacker_legal and self._wins_after(cell, player)]
                if len(completions) >= 2:
                    return 1.0, move
                if not completions:
                    continue
                playable = [cell for cell in completions if cell in defender_legal]
                if not playable:
                    return 1.0, move
                forced_win = True
                for cell in playable:
                    saved_block = self._place(cell, opponent)
                    try:
                        if self._rules.is_win(self._board):
                            forced_win = False
                            break
                        value, _ = self._attack(player)
                    finally:
                        self._unplace(saved_block)
                    if value <= 0:
                        forced_win = False
                        break
                if forced_win:
                    return 1.0, move
            finally:
                self._unplace(saved)
        return 0.0, None

    def _legal_for(self, player: int) -> list[int]:
        saved = self._board.current_player
        self._board.current_player = player
        try:
            return self._rules.legal_moves(self._board)
        finally:
            self._board.current_player = saved

    def _first_win(self, legal: list[int], player: int) -> int | None:
        for move in legal:
            if self._wins_after(move, player):
                return move
        return None

    def _wins_after(self, move: int, player: int) -> bool:
        if time.monotonic() >= self._deadline:
            raise _Budget
        saved = self._place(move, player)
        try:
            return self._rules.is_win(self._board)
        finally:
            self._unplace(saved)

    def _place(self, move: int, player: int) -> int:
        saved = self._board.current_player
        self._board.current_player = player
        self._board.play(*divmod(move, SIZE))
        return saved

    def _unplace(self, saved_player: int) -> None:
        self._board.undo()
        self._board.current_player = saved_player


class ThreatSpaceAZPlayer(TacticalAZPlayer):
    """Tactical wrapper plus forced-threat search; both layers can be disabled."""

    def __init__(
        self,
        base_player,
        tactical_enabled: bool = False,
        cfg: dict | None = None,
        max_nodes: int = 5000,
        time_limit_ms: float = 150.0,
    ) -> None:
        super().__init__(base_player, enabled=tactical_enabled, cfg=cfg)
        threat = (cfg or {}).get("threat_space") or {}
        self.threat_enabled = bool(threat.get("enabled", False))
        self.searcher = ThreatSpaceSearcher(max_nodes=max_nodes, time_limit_ms=time_limit_ms)
        self.activations = 0
        self.last_threat_value = 0.0

    def select_move(self, board) -> tuple[int, int]:
        self.last_threat_value = 0.0
        return super().select_move(board)

    def _fallback_move(self, board) -> tuple[int, int]:
        if self.threat_enabled and (self._deadline is None or time.monotonic() < self._deadline):
            limit = self.searcher.time_limit_ms
            if self._deadline is not None:
                self.searcher.time_limit_ms = min(limit, max(0.0, self._deadline - time.monotonic()) * 1000.0)
            try:
                result = self.searcher.search(board, self.rules, board.current_player)
            finally:
                self.searcher.time_limit_ms = limit
            self.nodes_visited += result.nodes
            self.last_threat_value = result.value
            if result.triggered:
                self.activations += 1
            if result.value > 0 and result.move is not None:
                self.last_override_applied = True
                self.last_override_reason = "threat_space"
                return divmod(result.move, SIZE)
        return super()._fallback_move(board)
