"""Pygame UI for human-vs-AI and AI-vs-AI Gomoku games."""

from __future__ import annotations

from numbers import Integral

from gomoku.board import BLACK, EMPTY, SIZE, WHITE, Board
from gomoku.rules import get_rules
from players.base import IllegalMoveError, Player
from players.factory import AGENT_CHOICES, create_player
from players.random import RandomPlayer


class PygameApp:
    """Display an 11x11 game while delegating game rules to a rule set."""

    def __init__(
        self,
        rules=None,
        human_color: int | None = BLACK,
        ai: Player | None = None,
        *,
        black_player: Player | None = None,
        white_player: Player | None = None,
        cell: int = 44,
        margin: int = 36,
        ai_delay_ms: int = 120,
    ) -> None:
        if human_color not in (None, BLACK, WHITE):
            raise ValueError("human_color must be BLACK, WHITE, or None")
        if ai_delay_ms < 0:
            raise ValueError("ai_delay_ms must be non-negative")

        self.rules = rules if rules is not None else get_rules("normal")
        self.human_color = human_color
        self.cell = cell
        self.margin = margin
        self.ai_delay_ms = ai_delay_ms
        self.board = Board()
        self.last_move: tuple[int, int] | None = None
        self.result: str | None = None
        self.message = ""
        self.paused = False
        self._restart_rect = None
        self._pause_rect = None

        if human_color is None:
            self.players: dict[int, Player | None] = {
                BLACK: black_player or RandomPlayer(self.rules),
                WHITE: white_player or RandomPlayer(self.rules),
            }
        else:
            opponent = WHITE if human_color == BLACK else BLACK
            opponent_player = ai or (
                white_player if opponent == WHITE else black_player
            )
            if opponent_player is None:
                opponent_player = RandomPlayer(self.rules)
            self.players = {BLACK: None, WHITE: None}
            self.players[opponent] = opponent_player

    def _pixel_to_cell(self, pos: tuple[int, int]) -> tuple[int, int] | None:
        x, y = pos
        col = round((x - self.margin) / self.cell)
        row = round((y - self.margin) / self.cell)
        if not (0 <= row < SIZE and 0 <= col < SIZE):
            return None
        px = self.margin + col * self.cell
        py = self.margin + row * self.cell
        if abs(x - px) <= self.cell // 2 and abs(y - py) <= self.cell // 2:
            return row, col
        return None

    def _play(self, move: tuple[int, int], *, by_player: bool = False) -> bool:
        if self.result is not None:
            return False
        if (
            not isinstance(move, tuple)
            or len(move) != 2
            or not all(isinstance(value, Integral) for value in move)
        ):
            if by_player:
                raise IllegalMoveError(f"player returned malformed move {move!r}")
            self.message = "Invalid move"
            return False

        row, col = map(int, move)
        if not (0 <= row < SIZE and 0 <= col < SIZE):
            if by_player:
                raise IllegalMoveError(f"player moved out of range: ({row}, {col})")
            self.message = "Click an intersection on the board"
            return False

        index = row * SIZE + col
        if index not in self.rules.legal_moves(self.board):
            if by_player:
                raise IllegalMoveError(f"player returned illegal move ({row}, {col})")
            self.message = "That intersection is occupied or forbidden"
            return False

        side = self.board.current_player
        self.board.play(row, col)
        self.last_move = (row, col)
        self.message = ""
        if self.rules.is_win(self.board):
            color = "Black" if side == BLACK else "White"
            self.result = f"{color} wins"
        elif self.rules.is_terminal(self.board):
            self.result = "Draw"
        return True

    def _reset(self) -> None:
        self.board = Board()
        self.last_move = None
        self.result = None
        self.message = ""
        self.paused = False

    def _toggle_pause(self) -> None:
        if self.human_color is None and self.result is None:
            self.paused = not self.paused

    def _handle_event(self, event, pygame) -> bool:
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                self._reset()
            elif event.key == pygame.K_SPACE:
                self._toggle_pause()
            return True
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self._restart_rect is not None and self._restart_rect.collidepoint(event.pos):
                self._reset()
            elif self._pause_rect is not None and self._pause_rect.collidepoint(event.pos):
                self._toggle_pause()
            elif (
                self.result is None
                and self.players[self.board.current_player] is None
            ):
                move = self._pixel_to_cell(event.pos)
                if move is not None:
                    self._play(move)
                else:
                    self.message = "Click an intersection on the board"
        return True

    def run(self) -> None:
        import pygame
        import pygame.gfxdraw

        pygame.init()
        try:
            display = pygame.display.Info()
            panel_height = 100
            available_board = min(
                int(display.current_w * 0.92),
                int(display.current_h * 0.85) - panel_height,
            )
            available_board = max(180, available_board)
            self.margin = min(self.margin, max(18, available_board // 16))
            max_cell = max(
                16, (available_board - self.margin * 2) // (SIZE - 1)
            )
            self.cell = min(self.cell, max_cell)
            board_size = self.margin * 2 + (SIZE - 1) * self.cell
            screen = pygame.display.set_mode(
                (board_size, board_size + panel_height)
            )
            pygame.display.set_caption("Gomoku 11x11")
            title_font = pygame.font.SysFont("segoeui", 25, bold=True)
            status_font = pygame.font.SysFont("segoeui", 19)
            small_font = pygame.font.SysFont("segoeui", 14)
            button_font = pygame.font.SysFont("segoeui", 14, bold=True)
            clock = pygame.time.Clock()

            wood = (222, 177, 112)
            ink = (67, 45, 28)
            background = (38, 43, 49)
            shadow = (112, 84, 55)
            stone_radius = max(5, self.cell // 2 - 4)
            grid_width = 1 if self.cell < 38 else 2
            board_surface = pygame.Surface((board_size, board_size))
            board_surface.fill(wood)
            for index in range(SIZE):
                position = self.margin + index * self.cell
                pygame.draw.line(
                    board_surface, ink, (self.margin, position),
                    (board_size - self.margin, position), grid_width,
                )
                pygame.draw.line(
                    board_surface, ink, (position, self.margin),
                    (position, board_size - self.margin), grid_width,
                )
            for row in (2, SIZE // 2, SIZE - 3):
                for col in (2, SIZE // 2, SIZE - 3):
                    pygame.draw.circle(
                        board_surface, ink,
                        (
                            self.margin + col * self.cell,
                            self.margin + row * self.cell,
                        ),
                        max(2, self.cell // 10),
                    )

            button_width, button_height = 112, 32
            panel_y = board_size
            self._restart_rect = pygame.Rect(
                board_size - self.margin - button_width,
                panel_y + 12,
                button_width,
                button_height,
            )
            self._pause_rect = pygame.Rect(
                board_size - self.margin - button_width,
                panel_y + 52,
                button_width,
                button_height,
            )

            def draw_stone(surface, center, player, highlighted):
                x, y = center

                def smooth_circle(color, point, radius):
                    pygame.gfxdraw.filled_circle(
                        surface, point[0], point[1], radius, color
                    )
                    pygame.gfxdraw.aacircle(
                        surface, point[0], point[1], radius, color
                    )

                smooth_circle(shadow, (x + 2, y + 4), stone_radius + 1)
                if player == BLACK:
                    smooth_circle((20, 22, 25), center, stone_radius)
                    highlight = (67, 71, 77)
                else:
                    smooth_circle((250, 248, 239), center, stone_radius)
                    pygame.gfxdraw.aacircle(
                        surface, x, y, stone_radius, (214, 211, 199)
                    )
                    highlight = (255, 255, 255)
                smooth_circle(
                    highlight,
                    (x - stone_radius // 3, y - stone_radius // 3),
                    max(3, stone_radius // 4),
                )
                if highlighted:
                    pygame.draw.circle(surface, (226, 79, 63), center, 4)

            running = True
            next_ai_move = 0
            while running:
                for event in pygame.event.get():
                    running = self._handle_event(event, pygame) and running

                now = pygame.time.get_ticks()
                if (
                    running
                    and self.result is None
                    and not self.paused
                    and self.players[self.board.current_player] is not None
                    and now >= next_ai_move
                ):
                    player = self.players[self.board.current_player]
                    self._play(player.select_move(self.board), by_player=True)
                    next_ai_move = now + self.ai_delay_ms

                screen.fill(background)
                screen.blit(board_surface, (0, 0))
                for index, value in enumerate(self.board.grid):
                    if value != EMPTY:
                        row, col = divmod(index, SIZE)
                        center = (
                            self.margin + col * self.cell,
                            self.margin + row * self.cell,
                        )
                        draw_stone(
                            screen, center, int(value),
                            (row, col) == self.last_move,
                        )

                screen.blit(
                    title_font.render("GOMOKU", True, (244, 241, 234)),
                    (self.margin, panel_y + 12),
                )
                if self.result is not None:
                    status = self.result
                    status_color = (130, 224, 164)
                elif self.paused:
                    status = "AI game paused"
                    status_color = (236, 205, 143)
                else:
                    side = self.board.current_player
                    status = f"{'Black' if side == BLACK else 'White'} to move"
                    status_color = (236, 205, 143)
                screen.blit(
                    status_font.render(status, True, status_color),
                    (self.margin, panel_y + 46),
                )
                rule_name = getattr(self.rules, "name", "custom").title()
                screen.blit(
                    small_font.render(
                        f"11x11 | {rule_name} rules | "
                        f"{'AI vs AI' if self.human_color is None else 'Human vs AI'}",
                        True, (184, 190, 197),
                    ),
                    (self.margin, panel_y + 76),
                )
                if self.message:
                    screen.blit(
                        small_font.render(self.message, True, (245, 144, 126)),
                        (self.margin, panel_y + 28),
                    )

                pygame.draw.rect(screen, (77, 89, 104), self._restart_rect, border_radius=5)
                restart = button_font.render("New game", True, (244, 241, 234))
                screen.blit(
                    restart,
                    restart.get_rect(center=self._restart_rect.center),
                )
                if self.human_color is None:
                    pygame.draw.rect(screen, (77, 89, 104), self._pause_rect, border_radius=5)
                    label = "Resume" if self.paused else "Pause"
                    pause = button_font.render(label, True, (244, 241, 234))
                    screen.blit(pause, pause.get_rect(center=self._pause_rect.center))
                pygame.display.flip()
                clock.tick(60)
        finally:
            pygame.quit()


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Play Gomoku with a Pygame UI.")
    parser.add_argument("--mode", choices=("human-vs-ai", "ai-vs-ai"), default="human-vs-ai")
    parser.add_argument("--ruleset", choices=("normal", "renju"), default="normal")
    parser.add_argument("--agent", choices=AGENT_CHOICES)
    parser.add_argument("--black-agent", choices=AGENT_CHOICES)
    parser.add_argument("--white-agent", choices=AGENT_CHOICES)
    parser.add_argument("--human-color", choices=("black", "white"), default="black")
    parser.add_argument("--rollouts", type=int, default=50)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--no-pruning", action="store_true")
    parser.add_argument("--checkpoint", help="Optional AlphaZero checkpoint for the AI.")
    parser.add_argument("--black-checkpoint")
    parser.add_argument("--white-checkpoint")
    parser.add_argument("--config", help="AlphaZero config; defaults to the ruleset config.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-budget", type=float, help="AlphaZero move budget in seconds.")
    args = parser.parse_args(argv)
    if args.rollouts <= 0:
        parser.error("--rollouts must be positive")
    if args.depth <= 0:
        parser.error("--depth must be positive")
    if args.time_budget is not None and args.time_budget <= 0:
        parser.error("--time-budget must be positive")
    rules = get_rules(args.ruleset)
    default_agent = args.agent or ("az" if args.checkpoint else "random")

    def make_side_agent(side: str) -> Player:
        explicit = args.black_agent if side == "black" else args.white_agent
        side_checkpoint = args.black_checkpoint if side == "black" else args.white_checkpoint
        agent = explicit or ("az" if side_checkpoint else default_agent)
        checkpoint = side_checkpoint or args.checkpoint
        return create_player(
            agent,
            rules,
            checkpoint=checkpoint if agent.startswith("az") else None,
            config=args.config,
            seed=args.seed + (1 if side == "white" else 0),
            rollouts=args.rollouts,
            depth=args.depth,
            pruning=not args.no_pruning,
            time_budget_sec=args.time_budget,
        )

    if args.mode == "ai-vs-ai":
        app = PygameApp(
            rules=rules,
            human_color=None,
            black_player=make_side_agent("black"),
            white_player=make_side_agent("white"),
        )
    else:
        human_color = BLACK if args.human_color == "black" else WHITE
        ai_color = WHITE if human_color == BLACK else BLACK
        app = PygameApp(
            rules=rules,
            human_color=human_color,
            ai=make_side_agent("white" if ai_color == WHITE else "black"),
        )
    app.run()


if __name__ == "__main__":
    main()
