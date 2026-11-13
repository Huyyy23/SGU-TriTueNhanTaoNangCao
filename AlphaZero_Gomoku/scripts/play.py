"""Play a Gomoku game in the terminal."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gomoku.board import BLACK, EMPTY, SIZE, WHITE
from gomoku.rules import get_rules
from gomoku.runner import REASON_DRAW, Runner, render_board
from players.factory import AGENT_CHOICES, create_player
from players.human import HumanPlayer


def _input_prompt(_board, hint: str | None) -> str | None:
    if hint:
        print(f"  {hint}; try again.")
    try:
        return input("  move (row,col): ")
    except EOFError:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("human-vs-ai", "human-vs-random", "ai-vs-ai"),
    )
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

    def side_agent(side: str) -> str:
        explicit = args.black_agent if side == "black" else args.white_agent
        side_checkpoint = args.black_checkpoint if side == "black" else args.white_checkpoint
        return explicit or ("az" if side_checkpoint else default_agent)

    def make_side_agent(side: str):
        agent = side_agent(side)
        checkpoint = (
            args.black_checkpoint if side == "black" else args.white_checkpoint
        ) or args.checkpoint
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

    if args.mode.startswith("human-vs-"):
        human_color = BLACK if args.human_color == "black" else WHITE
        black = (
            HumanPlayer(rules, prompt=_input_prompt)
            if human_color == BLACK
            else make_side_agent("black")
        )
        white = (
            HumanPlayer(rules, prompt=_input_prompt)
            if human_color == WHITE
            else make_side_agent("white")
        )
    else:
        black = make_side_agent("black")
        white = make_side_agent("white")

    show_board = args.mode != "ai-vs-ai"

    def on_move(board, player, side):
        if show_board:
            symbol = "X" if side == BLACK else "O"
            row, col = divmod(board.history[-1], SIZE)
            print(f"\n{player.name} ({symbol}) played ({row}, {col}):")
            print(render_board(board))

    result = Runner(rules).play(black, white, on_move=on_move)
    if result.reason == REASON_DRAW:
        print("Draw.")
    else:
        winner = "Black (X)" if result.winner == BLACK else "White (O)"
        print(f"{winner} wins after {len(result.moves)} moves.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
