import pygame

from gomoku.board import BLACK, SIZE, WHITE
from gomoku.rules import get_rules
from players.alphabeta import AlphaBetaPlayer
from players.heuristic import HeuristicPlayer
from ui import pygame_app
from ui.pygame_app import PygameApp


def test_pygame_headless_event_places_human_move(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = PygameApp(rules=get_rules("normal"), ai_delay_ms=0)
    click = pygame.event.Event(
        pygame.MOUSEBUTTONDOWN,
        button=1,
        pos=(
            app.margin + 5 * app.cell,
            app.margin + 5 * app.cell,
        ),
    )
    close = pygame.event.Event(pygame.QUIT)
    events = iter(([click], [close]))
    monkeypatch.setattr(pygame.event, "get", lambda: next(events, []))

    app.run()

    assert app.board.grid[5 * SIZE + 5] == BLACK
    assert len(app.board.history) == 2


def test_ai_vs_ai_pause_and_restart_controls():
    app = PygameApp(human_color=None)
    pause = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE)
    app._handle_event(pause, pygame)
    assert app.paused
    app.board.play(0, 0)
    app._restart_rect = pygame.Rect(10, 10, 50, 30)
    restart = pygame.event.Event(
        pygame.MOUSEBUTTONDOWN, button=1, pos=(20, 20)
    )
    app._handle_event(restart, pygame)
    assert not app.paused
    assert app.board.history == []


def test_ui_rejects_click_on_occupied_intersection():
    app = PygameApp(rules=get_rules("normal"))
    assert app._play((3, 3))

    assert not app._play((3, 3))
    assert len(app.board.history) == 1
    assert app.message == "That intersection is occupied or forbidden"


def test_ui_displays_result_and_restart_starts_a_new_game():
    app = PygameApp(rules=get_rules("normal"))
    moves = [
        (5, 3), (0, 0), (5, 4), (0, 1), (5, 5),
        (0, 2), (5, 6), (0, 3), (5, 7),
    ]
    for move in moves:
        assert app._play(move)
    assert app.result == "Black wins"

    app._restart_rect = pygame.Rect(10, 10, 50, 30)
    restart = pygame.event.Event(
        pygame.MOUSEBUTTONDOWN, button=1, pos=(20, 20)
    )
    app._handle_event(restart, pygame)

    assert app.result is None
    assert app.board.history == []


def test_ui_cli_selects_distinct_agents_and_human_color(monkeypatch):
    observed = {}

    class StubApp:
        def __init__(self, **kwargs):
            observed.update(kwargs)

        def run(self):
            pass

    monkeypatch.setattr(pygame_app, "PygameApp", StubApp)

    pygame_app.main(
        [
            "--mode", "ai-vs-ai",
            "--black-agent", "heuristic",
            "--white-agent", "alphabeta",
            "--depth", "1",
        ]
    )

    assert isinstance(observed["black_player"], HeuristicPlayer)
    assert isinstance(observed["white_player"], AlphaBetaPlayer)
    assert observed["black_player"] is not observed["white_player"]

    pygame_app.main(
        [
            "--mode", "human-vs-ai",
            "--human-color", "white",
            "--agent", "heuristic",
        ]
    )

    assert observed["human_color"] == WHITE
    assert isinstance(observed["ai"], HeuristicPlayer)
