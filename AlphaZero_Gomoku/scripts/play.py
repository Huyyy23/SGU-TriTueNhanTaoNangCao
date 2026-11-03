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
from players.human import HumanPlayer
from players.random import RandomPlayer


def _input_prompt(_board, hint: str | None) -> str | None:
    if hint:
        print(f"  {hint}; try again.")
    try:
        return input("  move (row,col): ")
    except EOFError:
        return None


def _make_agent(
    rules, agent: str, checkpoint: str | None, config: str, seed: int, rollouts: int
):
    if checkpoint is None:
        if agent == "mcts-pure":
            from players.mcts_pure import MCTSPure

            return MCTSPure(rules, n_rollouts=rollouts)
        return RandomPlayer(rules, seed=seed)

    from config import load_config
    from players.az_player import AZPlayer

    return AZPlayer(checkpoint, rules, cfg=load_config(config), seed=seed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("human-vs-ai", "human-vs-random", "ai-vs-ai"),
    )
    parser.add_argument("--ruleset", choices=("normal", "renju"), default="normal")
    parser.add_argument("--agent", choices=("random", "mcts-pure"), default="random")
    parser.add_argument("--rollouts", type=int, default=50)
    parser.add_argument("--checkpoint", help="Optional AlphaZero checkpoint for the AI.")
    parser.add_argument("--config", default="configs/normal.yaml")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.rollouts <= 0:
        parser.error("--rollouts must be positive")

    rules = get_rules(args.ruleset)
    if args.mode.startswith("human-vs-"):
        black = HumanPlayer(rules, prompt=_input_prompt)
        white = _make_agent(
            rules, args.agent, args.checkpoint, args.config, args.seed, args.rollouts
        )
    else:
        black = _make_agent(
            rules, args.agent, args.checkpoint, args.config, args.seed, args.rollouts
        )
        white = _make_agent(
            rules, args.agent, args.checkpoint, args.config, args.seed + 1, args.rollouts
        )

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
