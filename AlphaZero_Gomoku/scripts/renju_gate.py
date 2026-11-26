"""Apply the Renju objective-success screen to a finished tournament database.

Example (from the AlphaZero_Gomoku directory):

    python scripts/renju_gate.py \
        --db data/tournament.sqlite \
        --experiment renju-screen-20261124 \
        --agent az_renju --min-games 100 --min-score 0.70

Exit codes: 0 pass, 1 fail, 2 invalid input. Reads a database; plays no games.
This is the AlphaZero-vs-baseline screen, not the candidate-vs-best promotion gate.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from az.screening import gate_passes, score_summary


def agent_outcomes(db_path: Path, experiment: str, agent_label: str, rule_set: str) -> list[float]:
    with sqlite3.connect(db_path) as db:
        db.row_factory = sqlite3.Row
        agent = db.execute(
            "SELECT agent_id FROM agents WHERE label=?", (agent_label,)
        ).fetchone()
        if agent is None:
            raise SystemExit(f"agent '{agent_label}' is not in that database")
        agent_id = agent["agent_id"]
        games = db.execute(
            "SELECT winner, black_agent_id, white_agent_id, payload_json FROM games "
            "WHERE experiment_id=? AND rule_set=?",
            (experiment, rule_set),
        ).fetchall()
    outcomes: list[float] = []
    for game in games:
        payload = json.loads(game["payload_json"] or "{}")
        if payload.get("illegal_by"):
            continue
        if game["black_agent_id"] == agent_id:
            agent_color = "black"
        elif game["white_agent_id"] == agent_id:
            agent_color = "white"
        else:
            continue
        if game["winner"] == "draw":
            outcomes.append(0.5)
        elif game["winner"] in ("black", "white"):
            outcomes.append(1.0 if game["winner"] == agent_color else 0.0)
    return outcomes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", default="data/tournament.sqlite")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--agent", default="az_renju")
    parser.add_argument("--ruleset", default="renju")
    parser.add_argument("--min-games", type=int, default=100)
    parser.add_argument("--min-score", type=float, default=0.70)
    parser.add_argument("--basis", choices=("score", "wilson_low"), default="score",
                        help="statistic compared with --min-score")
    args = parser.parse_args(argv)
    if args.min_games <= 0 or not 0.0 < args.min_score <= 1.0:
        raise SystemExit("--min-games must be positive and --min-score must be in (0, 1]")
    if not Path(args.db).is_file():
        raise SystemExit(f"database not found: {args.db}")

    outcomes = agent_outcomes(Path(args.db), args.experiment, args.agent, args.ruleset)
    if not outcomes:
        print(json.dumps({"status": "invalid", "reason": "no scored games for that experiment"}))
        return 2
    summary = score_summary(outcomes)
    passed = gate_passes(summary, min_games=args.min_games, min_score=args.min_score, basis=args.basis)
    print(json.dumps({
        "status": "pass" if passed else "fail",
        "experiment": args.experiment,
        "agent": args.agent,
        "rule_set": args.ruleset,
        "required": {"min_games": args.min_games, "min_score": args.min_score,
                     "statistic": args.basis},
        **summary,
    }, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
