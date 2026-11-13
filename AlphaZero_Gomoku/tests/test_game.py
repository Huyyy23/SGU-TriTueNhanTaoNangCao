import pytest

from gomoku.board import BLACK, EMPTY, SIZE, WHITE
from gomoku.game import Game


def test_reset_starts_an_empty_game_with_black_to_move():
    game = Game()
    game.step(5 * SIZE + 5)

    game.reset()

    assert not game.board.history
    assert game.board.current_player == BLACK
    assert (game.board.grid == EMPTY).all()
    assert len(game.legal_actions()) == SIZE * SIZE


def test_legal_actions_exclude_renju_forbidden_moves():
    game = Game("renju")
    game.board.grid[[5 * SIZE + 4, 5 * SIZE + 6, 4 * SIZE + 5, 6 * SIZE + 5]] = BLACK

    assert 5 * SIZE + 5 not in game.legal_actions()


@pytest.mark.parametrize(
    ("action", "message"),
    [
        (-1, "outside the board"),
        (SIZE * SIZE, "outside the board"),
        (True, "integer board index"),
        ("12", "integer board index"),
    ],
)
def test_step_rejects_malformed_or_out_of_range_actions(action, message):
    with pytest.raises(ValueError, match=message):
        Game().step(action)


def test_step_rejects_occupied_and_renju_forbidden_actions():
    game = Game()
    game.step(0)
    with pytest.raises(ValueError, match="already occupied"):
        game.step(0)

    renju = Game("renju")
    renju.board.grid[[5 * SIZE + 4, 5 * SIZE + 6, 4 * SIZE + 5, 6 * SIZE + 5]] = BLACK
    with pytest.raises(ValueError, match="forbidden"):
        renju.step(5 * SIZE + 5)


def test_step_rejects_moves_after_game_ends():
    game = Game()
    for col in range(5):
        game.step(5 * SIZE + col)
        if col < 4:
            game.step(col)

    with pytest.raises(ValueError, match="terminal game"):
        game.step(SIZE)


def test_clone_has_independent_board_and_history():
    game = Game()
    game.step(5 * SIZE + 5)

    clone = game.clone()
    clone.step(5 * SIZE + 6)

    assert game.board.history == [5 * SIZE + 5]
    assert clone.board.history == [5 * SIZE + 5, 5 * SIZE + 6]
    assert game.board.grid[5 * SIZE + 6] == EMPTY
    assert clone.board.grid[5 * SIZE + 6] == WHITE


def test_result_reports_winner_and_draw_using_game_log_values():
    game = Game()
    for col in range(5):
        game.step(5 * SIZE + col)
        if col < 4:
            game.step(col)

    assert game.result() == ("black", "five_in_row")

    drawn = Game()
    for row in range(SIZE):
        for col in range(SIZE):
            drawn.board.grid[row * SIZE + col] = (
                BLACK if (row + 2 * col) % 5 < 2 else WHITE
            )
    assert drawn.is_terminal()
    assert drawn.result() == ("draw", "board_full")


def test_result_is_none_before_terminal_and_render_is_ascii():
    game = Game()

    assert game.result() is None
    text = game.render()
    assert len(text.splitlines()) == SIZE + 1
    assert "." in text
    assert "X" not in text
    assert "O" not in text
