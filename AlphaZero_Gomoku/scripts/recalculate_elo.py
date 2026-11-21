"""Recalculate Elo from immutable games in a tournament SQLite database."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from az.elo import recalculate_elo  # noqa: E402
from az.logging_v2 import TOURNAMENT_DB, connect, initialize  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=TOURNAMENT_DB)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--k-factor", type=float, default=32.0)
    parser.add_argument("--initial-rating", type=float, default=1200.0)
    args = parser.parse_args(argv)
    initialize(args.db, "tournament")
    with connect(args.db) as connection:
        result = recalculate_elo(
            connection, args.experiment_id, k_factor=args.k_factor,
            initial_rating=args.initial_rating,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
