"""Score a checkpoint's value head on completed games from a tournament database.

Example (from the AlphaZero_Gomoku directory):

    python scripts/renju_value_probe.py \
        --checkpoint checkpoints/normal/latest.json \
        --checkpoint checkpoints/renju_finetune_1500/latest.json \
        --games-db data/renju_diagnostic_screen_20261123.sqlite --min-plies 4

Reads only; plays no games and trains nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from az.model import load_checkpoint, resolve_checkpoint_path
from az.value_probe import value_metrics
from gomoku.board import BLACK, SIZE, WHITE, Board
from gomoku.features import board_to_planes
from gomoku.rules import get_rules


def positions_from_games(
    db_path: Path, *, rule_set: str, min_plies: int, max_games: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect (planes, z, mover_black) for every stored position at or after `min_plies`."""
    rules = get_rules(rule_set)
    planes: list[np.ndarray] = []
    z_values: list[float] = []
    movers: list[bool] = []
    uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        db.row_factory = sqlite3.Row
        games = db.execute(
            "SELECT winner, payload_json FROM games WHERE rule_set=? ORDER BY game_id LIMIT ?",
            (rule_set, int(max_games)),
        ).fetchall()
    for game in games:
        payload = json.loads(game["payload_json"] or "{}")
        if game["winner"] not in ("black", "white") or payload.get("illegal_by"):
            continue
        winner = BLACK if game["winner"] == "black" else WHITE
        board = Board()
        for index, action in enumerate(payload.get("moves") or []):
            if index >= int(min_plies):
                mover = board.current_player
                planes.append(board_to_planes(board, rules, mover))
                z_values.append(1.0 if mover == winner else -1.0)
                movers.append(mover == BLACK)
            if isinstance(action, dict):
                row, col = int(action["row"]), int(action["col"])
            else:
                row, col = divmod(int(action), SIZE)
            board.play(row, col)
    if not planes:
        raise SystemExit("no decided positions found in that database")
    return np.asarray(planes), np.asarray(z_values), np.asarray(movers)


def score_checkpoint(checkpoint: Path, positions) -> dict:
    planes, z_values, movers = positions
    resolved = Path(resolve_checkpoint_path(checkpoint))
    model, _ = load_checkpoint(resolved)
    model = model.to("cpu").eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(planes), 512):
            batch = torch.as_tensor(planes[start:start + 512], dtype=torch.float32)
            _, values = model(batch)
            predictions.append(values.reshape(-1).cpu().numpy())
    metrics = value_metrics(np.concatenate(predictions), z_values, movers)
    return {
        "checkpoint": str(resolved),
        "checkpoint_sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
        "metrics": metrics,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", action="append", required=True,
                        help="generation pointer or checkpoint file; repeat to compare")
    parser.add_argument("--games-db", default="data/renju_diagnostic_screen_20261123.sqlite")
    parser.add_argument("--ruleset", default="renju")
    parser.add_argument("--min-plies", type=int, default=0,
                        help="skip opening plies that were random, not searched")
    parser.add_argument("--max-games", type=int, default=200)
    args = parser.parse_args(argv)
    if args.min_plies < 0 or args.max_games <= 0:
        raise SystemExit("--min-plies must be >= 0 and --max-games must be positive")

    positions = positions_from_games(
        Path(args.games_db), rule_set=args.ruleset,
        min_plies=args.min_plies, max_games=args.max_games,
    )
    results = [score_checkpoint(Path(path), positions) for path in args.checkpoint]
    output = results[0] if len(results) == 1 else results
    print(json.dumps({
        "games_db": str(Path(args.games_db)),
        "rule_set": args.ruleset,
        "min_plies": args.min_plies,
        "positions": int(len(positions[1])),
        "results": output,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
