import os
import time
from datetime import datetime, timezone

import numpy as np
import pytest

from az.game_log import GameLogger, GameRecord
from az.model import AZNet, save_checkpoint
from gomoku.board import BLACK, SIZE, WHITE, Board
from gomoku.rules.normal import NormalRules
from players.az_player import AZPlayer

RUN_SLOW = os.environ.get("RUN_SLOW") == "1"


def tiny_checkpoint(path) -> str:
    save_checkpoint(AZNet(n_blocks=1, n_filters=8), path, meta={"rule_set": "normal"})
    return str(path)


class RandomOpponent:
    """Stand-in for A's RandomPlayer (players/random.py, T6) until that file lands."""

    def __init__(self, rules, seed: int = 0) -> None:
        self.rules = rules
        self.rng = np.random.default_rng(seed)

    def select_move(self, board) -> tuple[int, int]:
        return divmod(int(self.rng.choice(self.rules.legal_moves(board))), SIZE)


def legal_cells(rules, board) -> set[tuple[int, int]]:
    return {divmod(move, SIZE) for move in rules.legal_moves(board)}


def test_select_move_is_legal_and_leaves_the_board_untouched(tmp_path):
    rules = NormalRules()
    board = Board()
    board.play(5, 5)
    player = AZPlayer(tiny_checkpoint(tmp_path / "model.pt"), rules, cfg={"n_simulations": 4}, seed=0)
    before = board.grid.copy()

    move = player.select_move(board)

    assert move in legal_cells(rules, board)
    assert np.array_equal(board.grid, before)
    assert not board.history[1:]


def test_time_budget_still_returns_a_legal_move(tmp_path):
    rules = NormalRules()
    board = Board()
    player = AZPlayer(
        tiny_checkpoint(tmp_path / "model.pt"), rules, cfg={"n_simulations": 200}, time_budget_sec=1e-6, seed=0
    )

    started = time.monotonic()
    move = player.select_move(board)
    elapsed = time.monotonic() - started

    assert move in legal_cells(rules, board)
    assert elapsed < 2.0, f"budget did not cut the move short: {elapsed:.2f}s"


@pytest.mark.skipif(not RUN_SLOW, reason="GD4 gate check; set RUN_SLOW=1 to run 20 games")
def test_twenty_games_against_random_are_valid_and_logged(tmp_path):
    rules = NormalRules()
    az = AZPlayer(tiny_checkpoint(tmp_path / "model.pt"), rules, cfg={"n_simulations": 8}, seed=0)
    random_player = RandomOpponent(rules, seed=1)
    logger = GameLogger(tmp_path / "games.jsonl")

    for game_id in range(20):
        board = Board()
        az_is_black = game_id % 2 == 0
        players = {BLACK: az if az_is_black else random_player, WHITE: random_player if az_is_black else az}
        while not rules.is_terminal(board):
            r, c = players[board.current_player].select_move(board)
            board.play(r, c)

        winner = int(board.grid[board.history[-1]]) if rules.is_win(board) else None
        logger.log_game(
            GameRecord(
                game_id=game_id,
                timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                rule_set=rules.name,
                black_player="az" if az_is_black else "random",
                white_player="random" if az_is_black else "az",
                winner="draw" if winner is None else ("black" if winner == BLACK else "white"),
                first_player_won=winner == BLACK,
                end_reason="five_in_row" if winner is not None else "board_full",
                num_moves=len(board.history),
                duration_sec=0.0,
            )
        )

    records = GameLogger.read_records(tmp_path / "games.jsonl")
    assert len(records) == 20
    assert all(record["num_moves"] > 0 for record in records)
