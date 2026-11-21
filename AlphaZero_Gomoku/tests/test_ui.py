import pygame
import time

from gomoku.board import BLACK, SIZE, WHITE
from gomoku.rules import get_rules
from players.alphabeta import AlphaBetaPlayer
from players.heuristic import HeuristicPlayer
from ui import pygame_app
from ui.pygame_app import PygameApp
from ui.pygame_frontend import PygameFrontend
from players.factory import AGENT_CHOICES


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


def test_frontend_setup_uses_factory_for_color_and_rules(monkeypatch):
    app = PygameFrontend()
    app.mode = "ai-vs-ai"
    app.ruleset = "renju"
    created = []

    class StubPlayer:
        def __init__(self, name):
            self.name = name

    def factory(agent, rules, **kwargs):
        created.append((agent, rules.name, kwargs.get("checkpoint")))
        return StubPlayer(agent)

    monkeypatch.setattr("ui.pygame_frontend.create_player", factory)
    app._start_game()

    assert app.rules.name == "renju"
    assert [app.players[side].name for side in (BLACK, WHITE)] == ["heuristic", "heuristic"]
    assert [item[1] for item in created] == ["renju", "renju"]
    assert app.paused


def test_frontend_discovers_valid_checkpoint_pointers():
    app = PygameFrontend()
    assert app.checkpoint_options
    assert all(path.endswith(".json") or path.endswith(".pt") for path in app.checkpoint_options)
    assert any(path.endswith("latest.json") for path in app.checkpoint_options)
    assert app.checkpoints[BLACK] in app.checkpoint_options


def test_frontend_real_az_loads_and_makes_legal_move():
    app = PygameFrontend()
    pointer = app.checkpoints[BLACK]
    app.mode = "ai-vs-ai"
    app.agent_indices[BLACK] = AGENT_CHOICES.index("az_v0")
    app.agent_indices[WHITE] = AGENT_CHOICES.index("heuristic")
    app.checkpoints[BLACK] = pointer
    app._start_game()
    assert app.page == "game", app.message
    app._request_ai_move()
    deadline = time.monotonic() + 30
    while app._results.empty() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not app._results.empty(), "AlphaZero worker did not return a move"
    generation, side, move, elapsed, simulations, nn_calls, error = app._results.get_nowait()
    assert error is None, error
    assert side == BLACK and generation == app._generation
    assert move[0] * SIZE + move[1] in app.rules.legal_moves(app.board)
    app._play_move(move, elapsed)
    app.last_search = (simulations, nn_calls)
    assert len(app.board.history) == 1
    assert app.last_search[0] == 200


def test_frontend_rejects_wrong_ruleset_checkpoint():
    app = PygameFrontend()
    app.ruleset = "renju"
    app.agent_indices[WHITE] = AGENT_CHOICES.index("az_v0")
    app.checkpoints[WHITE] = "checkpoints/normal/latest.json"
    app._start_game()
    assert app.page == "menu"
    assert "rules" in app.message.lower()


def test_spectator_seed_modes_replay_or_vary_three_games(monkeypatch):
    app = PygameFrontend()
    app.mode = "ai-vs-ai"
    app.agent_indices[BLACK] = AGENT_CHOICES.index("heuristic")
    app.agent_indices[WHITE] = AGENT_CHOICES.index("heuristic")
    app.spectator_mode = "deterministic"
    deterministic = []
    for _ in range(3):
        app._start_game()
        while app.result is None:
            side = app.board.current_player
            app._play_move(app.players[side].select_move(app.board.copy()), 0.0)
        deterministic.append((tuple(app.board.history), app.result))
    assert deterministic[0] == deterministic[1] == deterministic[2]
    assert app.game_seed == 0
    seeds = iter((101, 202, 303))
    monkeypatch.setattr("ui.pygame_frontend.secrets.randbits", lambda _: next(seeds))
    app.agent_indices[BLACK] = AGENT_CHOICES.index("random")
    app.agent_indices[WHITE] = AGENT_CHOICES.index("random")
    app.spectator_mode = "exploration"
    exploratory = []
    game_seeds = []
    for _ in range(3):
        app._start_game()
        game_seeds.append(app.game_seed)
        while app.result is None:
            side = app.board.current_player
            app._play_move(app.players[side].select_move(app.board.copy()), 0.0)
        exploratory.append(tuple(app.board.history))
    assert game_seeds == [101, 202, 303]
    assert len(set(exploratory)) == 3


def test_start_game_assigns_mode_dependent_seeds(monkeypatch):
    app = PygameFrontend()
    app.mode = "ai-vs-ai"
    app.agent_indices[BLACK] = AGENT_CHOICES.index("random")
    app.agent_indices[WHITE] = AGENT_CHOICES.index("random")
    captured = []
    from ui import pygame_frontend

    original = pygame_frontend.create_player

    def capture_seed(agent, rules, **kwargs):
        captured.append(kwargs["seed"])
        return original(agent, rules, **kwargs)

    monkeypatch.setattr(pygame_frontend, "create_player", capture_seed)
    app.spectator_mode = "deterministic"
    app._start_game()
    assert app.game_seed == 0 and captured[-2:] == [BLACK, WHITE]
    app.spectator_mode = "exploration"
    monkeypatch.setattr("ui.pygame_frontend.secrets.randbits", lambda _: 500)
    app._start_game()
    assert app.game_seed == 500 and captured[-2:] == [500 + BLACK, 500 + WHITE]


def test_spectator_mode_buttons_select_deterministic_or_exploration():
    app = PygameFrontend()
    app.page = "setup"
    app.mode = "ai-vs-ai"
    app._rects = {
        "exploration": pygame.Rect(0, 0, 30, 30),
        "deterministic": pygame.Rect(40, 0, 30, 30),
    }
    app._event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(5, 5)))
    assert app.spectator_mode == "exploration"
    app._event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(45, 5)))
    assert app.spectator_mode == "deterministic"


def test_exploration_reset_recreates_agents_with_fresh_seed(monkeypatch):
    app = PygameFrontend()
    app.mode = "ai-vs-ai"
    app.spectator_mode = "exploration"
    app.agent_indices[BLACK] = AGENT_CHOICES.index("random")
    app.agent_indices[WHITE] = AGENT_CHOICES.index("random")
    seeds = iter((77, 88))
    monkeypatch.setattr("ui.pygame_frontend.secrets.randbits", lambda _: next(seeds))
    app._start_game()
    first_player = app.players[BLACK]
    app._reset()
    assert app.players[BLACK] is not first_player
    assert app.game_seed == 88
    assert app.board.history == []


def test_frontend_requires_checkpoint_for_selected_az():
    app = PygameFrontend()
    app.agent_indices[WHITE] = AGENT_CHOICES.index("az_v0")
    app.checkpoints[WHITE] = ""
    app._start_game()
    assert app.page == "menu"
    assert "checkpoint" in app.message.lower()


def test_frontend_step_pause_and_reset(monkeypatch):
    app = PygameFrontend()
    app.mode = "ai-vs-ai"
    app.page = "game"
    app.paused = True
    app.players = {BLACK: object(), WHITE: object()}
    app._rects = {
        "play": pygame.Rect(0, 0, 20, 20),
        "step": pygame.Rect(30, 0, 20, 20),
        "reset": pygame.Rect(60, 0, 20, 20),
    }
    app._event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(35, 5)))
    assert app.paused and app.step_requested
    app.board.play(0, 0)
    app._event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(65, 5)))
    assert app.board.history == []
    assert app.paused and not app.step_requested


def test_frontend_menu_exit_and_back_to_menu():
    app = PygameFrontend()
    app.page = "menu"
    app._rects = {"human": pygame.Rect(0, 0, 20, 20), "exit": pygame.Rect(30, 0, 20, 20)}
    assert app._event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(5, 5)))
    assert app.page == "setup" and app.mode == "human-vs-ai"
    app.page = "menu"
    assert not app._event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(35, 5)))


def test_frontend_ai_worker_does_not_mutate_board_until_event_pump():
    app = PygameFrontend()
    app.mode = "ai-vs-ai"
    app.page = "game"

    class SlowPlayer:
        name = "slow"

        def select_move(self, board):
            time.sleep(0.01)
            return (5, 5)

    import time

    app.players[BLACK] = SlowPlayer()
    app._request_ai_move()
    assert app.board.history == []
    deadline = time.monotonic() + 1
    while app._results.empty() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert app.board.history == []
    generation, side, move, elapsed, sims, calls, error = app._results.get_nowait()
    assert generation == app._generation and side == BLACK and move == (5, 5)
    assert elapsed >= 0 and error is None
    app._play_move(move, elapsed)
    assert app.board.history == [5 * SIZE + 5]


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
