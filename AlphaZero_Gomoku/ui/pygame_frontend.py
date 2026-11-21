from __future__ import annotations

import queue
import secrets
import threading
import time
from pathlib import Path

import pygame

from gomoku.board import BLACK, EMPTY, SIZE, WHITE, Board
from gomoku.rules import get_rules
from players.factory import AGENT_CHOICES, create_player


BG = (246, 242, 231)
PANEL = (255, 252, 244)
PANEL_ALT = (255, 252, 244)
TEXT = (39, 43, 40)
MUTED = (91, 99, 92)
ACCENT = (125, 77, 28)
ACCENT_DARK = (67, 87, 65)
WOOD = (232, 181, 79)
GRID = (107, 70, 30)
ORANGE = (142, 75, 24)
RED = (173, 49, 37)
BORDER = (189, 184, 166)
DISABLED = (222, 221, 211)


class PygameFrontend:
    """Small menu/setup/spectator shell around the existing game and player APIs."""

    def __init__(self) -> None:
        self.page = "menu"
        self.mode = "human-vs-ai"
        self.ruleset = "normal"
        self.spectator_mode = "deterministic"
        self.game_seed = None
        self.human_color = BLACK
        self.agent_indices = {BLACK: AGENT_CHOICES.index("heuristic"), WHITE: AGENT_CHOICES.index("heuristic")}
        self.checkpoint_options = self._discover_checkpoints()
        self.checkpoints = {side: self._default_checkpoint() for side in (BLACK, WHITE)}
        self.config = ""
        self.message = ""
        self.board = Board()
        self.rules = get_rules(self.ruleset)
        self.players = {BLACK: None, WHITE: None}
        self.last_move = None
        self.result = None
        self.move_history: list[tuple[int, int, int, float]] = []
        self.paused = True
        self.step_requested = False
        self.speed = 1
        self.thinking_side: int | None = None
        self.thinking_since = 0.0
        self.last_duration: float | None = None
        self.last_search: tuple[int | None, int | None] = (None, None)
        self._generation = 0
        self._results: queue.Queue = queue.Queue()
        self._worker_busy = False
        self._checkpoint_focus: int | None = None
        self._text_input = ""
        self._rects: dict[str, pygame.Rect] = {}

    def _default_checkpoint(self):
        if self.checkpoint_options:
            return self.checkpoint_options[0]
        return str(Path(__file__).resolve().parents[1] / "models" / "az_normal_weights.pt")

    @staticmethod
    def _discover_checkpoints():
        try:
            from az.model import resolve_checkpoint_path
        except ModuleNotFoundError:
            return []

        root = Path(__file__).resolve().parents[1]
        candidates = [
            root / "checkpoints" / "normal" / "latest.json",
            root / "models" / "az_normal_weights.pt",
        ]
        valid = []
        for path in candidates:
            try:
                resolved = resolve_checkpoint_path(path) if path.suffix.lower() == ".json" else path
                if resolved.is_file():
                    valid.append(str(path))
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return valid

    @staticmethod
    def _label(surface, font, text, pos, color=TEXT):
        surface.blit(font.render(str(text), True, color), pos)

    def _button(self, surface, font, key, text, rect, selected=False, enabled=True):
        color = ACCENT_DARK if selected else PANEL_ALT
        if not enabled:
            color = DISABLED
        shadow = rect.move(0, 2)
        pygame.draw.rect(surface, (205, 201, 186), shadow, border_radius=6)
        pygame.draw.rect(surface, color, rect, border_radius=6)
        pygame.draw.rect(surface, ACCENT_DARK if selected else BORDER, rect, 1, border_radius=6)
        rendered = font.render(text, True, (255, 255, 255) if selected and enabled else TEXT if enabled else MUTED)
        surface.blit(rendered, rendered.get_rect(center=rect.center))
        self._rects[key] = rect

    def _start_game(self):
        self.rules = get_rules(self.ruleset)
        self.game_seed = (0 if self.spectator_mode == "deterministic" else secrets.randbits(63)) if self.mode == "ai-vs-ai" else 0
        agents = {}
        try:
            for side in (BLACK, WHITE):
                if self.mode == "human-vs-ai" and side == self.human_color:
                    agents[side] = None
                    continue
                agent = AGENT_CHOICES[self.agent_indices[side]]
                checkpoint = self.checkpoints[side] or None
                if agent.startswith("az") and checkpoint is None:
                    raise ValueError("Select a discovered checkpoint or enter a checkpoint path for AlphaZero.")
                if agent.startswith("az") and not Path(checkpoint).exists():
                    raise FileNotFoundError(f"AlphaZero checkpoint does not exist: {checkpoint}")
                if agent.startswith("az"):
                    from az.model import load_checkpoint

                    _, checkpoint_meta = load_checkpoint(checkpoint, device="cpu")
                    checkpoint_rules = checkpoint_meta.get("rule_set")
                    if checkpoint_rules and checkpoint_rules != self.ruleset:
                        raise ValueError(f"Checkpoint uses {checkpoint_rules} rules; selected rules are {self.ruleset}.")
                agents[side] = create_player(
                    agent,
                    self.rules,
                    checkpoint=checkpoint if agent.startswith("az") else None,
                    config=self.config or None,
                    seed=self.game_seed + side,
                )
        except ModuleNotFoundError as exc:
            dependency = exc.name or "unknown"
            self.message = f"Missing Python dependency '{dependency}' required by {agent}: install project requirements."
            return
        except Exception as exc:
            self.message = f"Could not create {agent}: {exc}"
            return
        self.players = {BLACK: agents[BLACK], WHITE: agents[WHITE]}
        self.board = Board()
        self.last_move = None
        self.result = None
        self.move_history.clear()
        self.paused = self.mode == "ai-vs-ai"
        self.step_requested = False
        self.last_duration = None
        self.last_search = (None, None)
        self.message = ""
        self._generation += 1
        self.page = "game"

    def _reset(self):
        if self.mode == "ai-vs-ai":
            self._start_game()
            return
        self.board = Board()
        self.last_move = None
        self.result = None
        self.move_history.clear()
        self.message = ""
        self.paused = self.mode == "ai-vs-ai"
        self.step_requested = False
        self.last_duration = None
        self.last_search = (None, None)
        self._generation += 1

    def _play_move(self, move, elapsed):
        side = self.board.current_player
        legal = self.rules.legal_moves(self.board)
        if not isinstance(move, tuple) or len(move) != 2:
            self.message = "Agent returned an invalid move"
            self.paused = True
            return
        row, col = map(int, move)
        if not (0 <= row < SIZE and 0 <= col < SIZE) or row * SIZE + col not in legal:
            self.message = "Agent returned an illegal move"
            self.paused = True
            return
        self.board.play(row, col)
        self.last_move = (row, col)
        self.last_duration = elapsed
        self.move_history.append((side, row, col, elapsed))
        self.message = ""
        if self.rules.is_win(self.board):
            self.result = f"{'Black' if side == BLACK else 'White'} wins"
            self.paused = True
        elif self.rules.is_terminal(self.board):
            self.result = "Draw"
            self.paused = True

    def _request_ai_move(self):
        if self._worker_busy or self.result is not None:
            return
        side = self.board.current_player
        player = self.players[side]
        if player is None:
            return
        snapshot = self.board.copy()
        generation = self._generation
        self._worker_busy = True
        self.thinking_side = side
        self.thinking_since = time.perf_counter()
        started = self.thinking_since

        def think():
            try:
                move = player.select_move(snapshot)
                payload = (generation, side, move, time.perf_counter() - started,
                           getattr(player, "simulations_run", None), getattr(player, "nn_calls", None), None)
            except Exception as exc:
                payload = (generation, side, None, time.perf_counter() - started,
                           None, None, str(exc))
            self._results.put(payload)

        threading.Thread(target=think, name="gomoku-ai-turn", daemon=True).start()

    def _event(self, event):
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN and self._checkpoint_focus is not None:
            if event.key == pygame.K_RETURN:
                if self._checkpoint_focus == -1:
                    self.config = self._text_input.strip()
                else:
                    self.checkpoints[self._checkpoint_focus] = self._text_input.strip()
                self._text_input = ""
                self._checkpoint_focus = None
            elif event.key == pygame.K_ESCAPE:
                self._checkpoint_focus = None
                self._text_input = ""
            elif event.key == pygame.K_BACKSPACE:
                self._text_input = self._text_input[:-1]
            else:
                self._text_input += event.unicode
            return True
        if event.type == pygame.KEYDOWN and self.page == "game":
            if event.key == pygame.K_SPACE and self.mode == "ai-vs-ai":
                self.paused = not self.paused
                self.step_requested = False
            elif event.key == pygame.K_r:
                self._reset()
            elif event.key == pygame.K_ESCAPE:
                self._generation += 1
                self.page = "menu"
            return True
        if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return True
        pos = event.pos
        if self.page == "menu":
            for key, mode in (("human", "human-vs-ai"), ("spectator", "ai-vs-ai")):
                if self._rects.get(key) and self._rects[key].collidepoint(pos):
                    self.mode = mode
                    self.page = "setup"
            if self._rects.get("exit") and self._rects["exit"].collidepoint(pos):
                return False
        elif self.page == "setup":
            if self._rects.get("normal") and self._rects["normal"].collidepoint(pos):
                self.ruleset = "normal"
            elif self._rects.get("renju") and self._rects["renju"].collidepoint(pos):
                self.ruleset = "renju"
            elif self._rects.get("deterministic") and self._rects["deterministic"].collidepoint(pos):
                self.spectator_mode = "deterministic"
            elif self._rects.get("exploration") and self._rects["exploration"].collidepoint(pos):
                self.spectator_mode = "exploration"
            elif self._rects.get("color") and self._rects["color"].collidepoint(pos):
                self.human_color = WHITE if self.human_color == BLACK else BLACK
            elif next((key for key in ("agent-prev-black", "agent-prev-white") if self._rects.get(key) and self._rects[key].collidepoint(pos)), None):
                key = next(key for key in ("agent-prev-black", "agent-prev-white") if self._rects.get(key) and self._rects[key].collidepoint(pos))
                side = BLACK if key.endswith("black") else WHITE
                self.agent_indices[side] = (self.agent_indices[side] - 1) % len(AGENT_CHOICES)
            elif next((key for key in ("agent-next-black", "agent-next-white") if self._rects.get(key) and self._rects[key].collidepoint(pos)), None):
                key = next(key for key in ("agent-next-black", "agent-next-white") if self._rects.get(key) and self._rects[key].collidepoint(pos))
                side = BLACK if key.endswith("black") else WHITE
                self.agent_indices[side] = (self.agent_indices[side] + 1) % len(AGENT_CHOICES)
            elif self._rects.get("checkpoint-black") and self._rects["checkpoint-black"].collidepoint(pos):
                self._cycle_checkpoint(BLACK)
            elif self._rects.get("checkpoint-white") and self._rects["checkpoint-white"].collidepoint(pos):
                self._cycle_checkpoint(WHITE)
            elif self._rects.get("checkpoint-path-black") and self._rects["checkpoint-path-black"].collidepoint(pos):
                self._checkpoint_focus = BLACK
                self._text_input = self.checkpoints[BLACK]
            elif self._rects.get("checkpoint-path-white") and self._rects["checkpoint-path-white"].collidepoint(pos):
                self._checkpoint_focus = WHITE
                self._text_input = self.checkpoints[WHITE]
            elif self._rects.get("config") and self._rects["config"].collidepoint(pos):
                self._checkpoint_focus = -1
                self._text_input = self.config
            elif self._rects.get("back") and self._rects["back"].collidepoint(pos):
                self.page = "menu"
            elif self._rects.get("start") and self._rects["start"].collidepoint(pos):
                if self._checkpoint_focus == -1:
                    self.config = self._text_input.strip()
                    self._checkpoint_focus = None
                self._start_game()
        elif self.page == "game":
            if self._rects.get("back") and self._rects["back"].collidepoint(pos):
                self._generation += 1
                self.page = "menu"
            elif self._rects.get("reset") and self._rects["reset"].collidepoint(pos):
                self._reset()
            elif self._rects.get("play") and self._rects["play"].collidepoint(pos):
                self.paused = not self.paused
                self.step_requested = False
            elif self._rects.get("step") and self._rects["step"].collidepoint(pos):
                self.paused = True
                self.step_requested = True
            elif self._rects.get("speed") and self._rects["speed"].collidepoint(pos):
                self.speed = self.speed % 4 + 1
            else:
                cell = self._pixel_to_cell(pos)
                side = self.board.current_player
                if cell is not None and self.result is None and self.players[side] is None:
                    self._play_move(cell, 0.0)
        return True

    def _cycle_checkpoint(self, side):
        if not self.checkpoint_options:
            self.message = "No checkpoint found under checkpoints/. Enter a path below."
            return
        try:
            index = self.checkpoint_options.index(self.checkpoints[side])
        except ValueError:
            index = -1
        self.checkpoints[side] = self.checkpoint_options[(index + 1) % len(self.checkpoint_options)]
        self.message = ""

    def _agent_side(self, pos):
        if self.mode == "human-vs-ai":
            return WHITE if self.human_color == BLACK else BLACK
        return BLACK if pos[1] < 464 else WHITE

    def _pixel_to_cell(self, pos):
        x, y = pos
        left, top, cell = 42, 70, 48
        col, row = round((x - left) / cell), round((y - top) / cell)
        if 0 <= row < SIZE and 0 <= col < SIZE:
            if abs(x - left - col * cell) < cell // 2 and abs(y - top - row * cell) < cell // 2:
                return row, col
        return None

    def _draw_menu(self, screen, fonts):
        title, body, small = fonts
        self._label(screen, title, "GOMOKU", (92, 116), ACCENT)
        self._label(screen, body, "11 × 11  /  AlphaZero arena", (96, 170), MUTED)
        self._label(screen, body, "Choose a game mode", (96, 258))
        self._button(screen, body, "human", "HUMAN  VS  AI", pygame.Rect(96, 310, 300, 64))
        self._button(screen, body, "spectator", "AI  VS  AI  ·  SPECTATE", pygame.Rect(96, 390, 300, 64))
        self._button(screen, body, "exit", "EXIT", pygame.Rect(96, 470, 300, 54))
        self._label(screen, small, "11 × 11 board  •  Normal / Renju", (96, 560), MUTED)

    def _draw_setup(self, screen, fonts):
        title, body, small = fonts
        self._label(screen, title, "GAME SETUP", (72, 44), ACCENT)
        self._label(screen, body, "RULESET", (72, 124), MUTED)
        self._button(screen, body, "normal", "NORMAL", pygame.Rect(72, 160, 150, 48), self.ruleset == "normal")
        self._button(screen, body, "renju", "RENJU", pygame.Rect(236, 160, 150, 48), self.ruleset == "renju")
        self._label(screen, body, f"MODE   {'Human vs AI' if self.mode == 'human-vs-ai' else 'AI vs AI spectator'}", (72, 232))
        if self.mode == "ai-vs-ai":
            self._label(screen, body, "SPECTATOR MOVE SELECTION", (520, 232), MUTED)
            self._label(screen, small, "Uses selected agents' RNG only; deterministic agents stay deterministic.", (520, 250), MUTED)
            self._button(screen, small, "deterministic", "DETERMINISTIC", pygame.Rect(520, 278, 190, 42), self.spectator_mode == "deterministic")
            self._button(screen, small, "exploration", "EXPLORATION", pygame.Rect(724, 278, 190, 42), self.spectator_mode == "exploration")
        if self.mode == "human-vs-ai":
            self._button(screen, body, "color", f"Human plays {'Black' if self.human_color == BLACK else 'White'} · change", pygame.Rect(72, 278, 420, 48))
            sides = [WHITE if self.human_color == BLACK else BLACK]
            labels = ["Opponent agent"]
        else:
            sides, labels = [BLACK, WHITE], ["Black agent", "White agent"]
        for index, (side, label) in enumerate(zip(sides, labels)):
            y = 366 + index * 98
            self._label(screen, body, label, (72, y), MUTED)
            self._button(screen, small, f"agent-prev-{ 'black' if side == BLACK else 'white' }", "‹", pygame.Rect(72, y + 30, 46, 42))
            self._label(screen, body, f"{AGENT_CHOICES[self.agent_indices[side]]}{' · AZ' if AGENT_CHOICES[self.agent_indices[side]].startswith('az') else ''}", (136, y + 39), TEXT)
            self._button(screen, small, f"agent-next-{ 'black' if side == BLACK else 'white' }", "›", pygame.Rect(350, y + 30, 46, 42))
        self._label(screen, small, f"AZ checkpoints detected: {len(self.checkpoint_options)} · Normal/Renju checkpoint must match selected rules.", (72, 570), MUTED)
        self._button(screen, small, "checkpoint-black", "Black checkpoint  ▼", pygame.Rect(72, 626, 240, 38))
        self._button(screen, small, "checkpoint-white", "White checkpoint  ▼", pygame.Rect(322, 626, 240, 38))
        self._button(screen, small, "config", "Config (optional)", pygame.Rect(572, 626, 200, 38))
        self._button(screen, small, "checkpoint-path-black", "Edit path", pygame.Rect(72, 670, 92, 32))
        self._button(screen, small, "checkpoint-path-white", "Edit path", pygame.Rect(322, 670, 92, 32))
        self._label(screen, small, f"{Path(self.checkpoints[BLACK]).name if self.checkpoints[BLACK] else 'No checkpoint selected'}", (174, 678), MUTED)
        self._label(screen, small, f"{Path(self.checkpoints[WHITE]).name if self.checkpoints[WHITE] else 'No checkpoint selected'}", (424, 678), MUTED)
        if self._checkpoint_focus is not None:
            self._label(screen, small, f"Path: {self._text_input}_", (72, 714), ORANGE)
        elif self.config:
            self._label(screen, small, f"Config: {self.config}", (492, 678), MUTED)
        self._button(screen, body, "back", "BACK", pygame.Rect(742, 690, 120, 44))
        self._button(screen, body, "start", "START GAME", pygame.Rect(876, 690, 174, 44), True)
        if self.message:
            self._label(screen, small, self.message[:100], (72, 752), RED)

    def _draw_game(self, screen, fonts):
        title, body, small = fonts
        self._rects = {}
        pygame.draw.rect(screen, (227, 220, 203), pygame.Rect(0, 0, 600, 780))
        pygame.draw.rect(screen, (161, 111, 45), pygame.Rect(14, 42, 536, 536), border_radius=4)
        pygame.draw.rect(screen, WOOD, pygame.Rect(18, 46, 528, 528), border_radius=3)
        left, top, cell = 42, 70, 48
        for idx in range(SIZE):
            p = left + idx * cell
            pygame.draw.line(screen, GRID, (left, top + idx * cell), (left + (SIZE - 1) * cell, top + idx * cell), 1)
            pygame.draw.line(screen, GRID, (p, top), (p, top + (SIZE - 1) * cell), 1)
        for row in (2, SIZE // 2, SIZE - 3):
            for col in (2, SIZE // 2, SIZE - 3):
                pygame.draw.circle(screen, GRID, (left + col * cell, top + row * cell), 3)
        for index, value in enumerate(self.board.grid):
            if value == EMPTY:
                continue
            row, col = divmod(index, SIZE)
            center = (left + col * cell, top + row * cell)
            pygame.draw.circle(screen, (143, 100, 46), (center[0] + 2, center[1] + 3), 20)
            color = (30, 34, 33) if int(value) == BLACK else (252, 251, 244)
            pygame.draw.circle(screen, color, center, 20)
            pygame.draw.circle(screen, (8, 12, 11) if int(value) == BLACK else (151, 145, 128), center, 20, 2)
            if (row, col) == self.last_move:
                pygame.draw.circle(screen, (207, 45, 34), center, 5)
                pygame.draw.circle(screen, (255, 237, 211), center, 5, 1)
        self._label(screen, body, "Gomoku  /  11 × 11", (20, 10), TEXT)
        self._label(screen, small, "Red marker: last move", (20, 596), MUTED)
        pygame.draw.rect(screen, PANEL, pygame.Rect(600, 0, 520, 780))
        self._label(screen, title, "GOMOKU", (684, 34), TEXT)
        selection = self.spectator_mode.upper() if self.mode == "ai-vs-ai" else "STANDARD"
        self._label(screen, small, f"{self.ruleset.upper()}  ·  11×11  ·  {self.mode.upper()}  ·  {selection}", (686, 76), MUTED)
        if self.mode == "ai-vs-ai":
            self._label(screen, small, f"Game seed: {self.game_seed}  ·  selection RNG seed: {self.game_seed + self.board.current_player}", (686, 98), MUTED)
        if self.result:
            status, color = self.result, ACCENT
        elif self.thinking_side is not None:
            status, color = f"{'Black' if self.thinking_side == BLACK else 'White'} is thinking…", ORANGE
        elif self.mode == "ai-vs-ai" and self.paused:
            status, color = "Spectator paused", ORANGE
        else:
            status, color = f"{'Black' if self.board.current_player == BLACK else 'White'} to move", TEXT
        self._label(screen, body, status, (686, 114), color)
        self._label(screen, small, f"BLACK   {getattr(self.players[BLACK], 'name', 'Human') if self.players[BLACK] else 'Human'}", (686, 154), MUTED)
        self._label(screen, small, f"WHITE   {getattr(self.players[WHITE], 'name', 'Human') if self.players[WHITE] else 'Human'}", (686, 178), MUTED)
        self._label(screen, small, f"Move time: {self.last_duration:.3f}s" if self.last_duration is not None else "Move time: —", (686, 212), TEXT)
        sims, calls = self.last_search
        self._label(screen, small, f"MCTS simulations: {sims if sims is not None else '—'}   NN calls: {calls if calls is not None else '—'}", (686, 238), MUTED)
        if self.mode == "ai-vs-ai":
            self._label(screen, small, f"Current agent RNG seed: {self.game_seed + self.board.current_player}", (686, 260), MUTED)
        self._button(screen, small, "reset", "RESET", pygame.Rect(686, 288, 112, 36))
        self._button(screen, small, "back", "BACK TO MENU", pygame.Rect(810, 288, 150, 36))
        if self.mode == "ai-vs-ai":
            self._button(screen, small, "play", "PLAY" if self.paused else "PAUSE", pygame.Rect(686, 328, 100, 38), self.paused)
            self._button(screen, small, "step", "STEP", pygame.Rect(796, 328, 88, 38))
            self._button(screen, small, "speed", f"SPEED  {self.speed}×", pygame.Rect(894, 328, 130, 38))
        self._label(screen, body, "MOVE HISTORY", (686, 394), TEXT)
        for i, (side, row, col, elapsed) in enumerate(self.move_history[-11:]):
            y = 430 + i * 24
            self._label(screen, small, f"{len(self.move_history) - min(11, len(self.move_history)) + i + 1:02d}  {'B' if side == BLACK else 'W'}  ({row + 1},{col + 1})  {elapsed:.2f}s", (690, y), MUTED)
        if self.message:
            self._label(screen, small, self.message[:70], (686, 710), RED)

    def run(self):
        pygame.init()
        screen = pygame.display.set_mode((1120, 780))
        pygame.display.set_caption("Gomoku · AlphaZero Arena")
        fonts = (
            pygame.font.SysFont("segoeui", 34, bold=True),
            pygame.font.SysFont("segoeui", 20, bold=True),
            pygame.font.SysFont("segoeui", 15),
        )
        clock = pygame.time.Clock()
        running = True
        last_move_time = 0.0
        try:
            while running:
                for event in pygame.event.get():
                    running = self._event(event) and running
                while not self._results.empty():
                    generation, side, move, elapsed, sims, calls, error = self._results.get()
                    self._worker_busy = False
                    self.thinking_side = None
                    if generation == self._generation and side == self.board.current_player:
                        if error:
                            self.message = f"Agent error: {error}"
                            self.paused = True
                        else:
                            self._play_move(move, elapsed)
                            self.last_search = (sims, calls)
                            last_move_time = time.perf_counter()
                        self.step_requested = False
                screen.fill(BG)
                if self.page == "menu":
                    self._rects = {}
                    self._draw_menu(screen, fonts)
                elif self.page == "setup":
                    self._rects = {}
                    self._draw_setup(screen, fonts)
                else:
                    self._draw_game(screen, fonts)
                    side = self.board.current_player
                    ai_turn = self.players[side] is not None
                    ready = time.perf_counter() - last_move_time >= max(0.08, 0.8 / self.speed)
                    if ai_turn and self.result is None and not self._worker_busy and (not self.paused or self.step_requested) and ready:
                        self._request_ai_move()
                if self._checkpoint_focus is not None and self.page == "setup":
                    self._label(screen, fonts[2], f"Type path: {self._text_input}_", (72, 714), ORANGE)
                pygame.display.flip()
                clock.tick(60)
        finally:
            pygame.quit()
