"""Ablation experiments for T32.

Two modes write one JSON file each, logs/ablation/<run_id>.json, and a third draws it:

    # Alpha-Beta: pruning on vs off at the same depth, on the same positions.
    python scripts/ablation.py alphabeta --depths 2 --positions 3 --position-plies 100

    # MCTS Pure: several simulation budgets against one opponent, colours split evenly.
    python scripts/ablation.py mcts --sims 50,200,800 --games 100 --opponent heuristic

    # PNG straight from the JSON (needs matplotlib).
    python scripts/ablation.py plot logs/ablation/<run_id>.json

Alpha-Beta notes: at depth 1 no cut-off can happen (the root only raises alpha and depth-0
nodes return the evaluation), so pruning only shows from depth 2. Positions come from random
legal play of --position-plies plies with a fixed seed; on an empty 11x11 board depth 2 without
pruning costs minutes per move, which is why the smoke default uses crowded positions.

Nothing here changes players/ or any checkpoint; existing JSON files are never overwritten.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gomoku.board import SIZE, Board
from gomoku.rules import get_rules
from scripts.tournament import (
    AgentSpec,
    Budget,
    Defaults,
    _git_state,
    _mean,
    _versions,
    agent_settings,
    aggregate,
    derive_seed,
    play_game_task,
    random_opening,
    schedule,
)


def _int_list(text: str) -> list[int]:
    try:
        values = [int(part) for part in str(text).split(",") if part.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} must be comma-separated integers") from None
    if not values or any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError(f"{text!r} must list positive integers")
    return values


def _output_path(out_root: str | Path, run_id: str) -> Path:
    path = Path(out_root) / f"{run_id}.json"
    if path.exists():
        raise FileExistsError(f"{path} already exists; logs are never overwritten, pick another --run-id")
    return path


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _common(mode: str, run_id: str, seed: int, ruleset: str) -> dict:
    return {
        "mode": mode,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join([Path(sys.argv[0]).name] + sys.argv[1:]),
        "seed": seed,
        "ruleset": ruleset,
        "git": _git_state(),
        "versions": _versions(),
    }


# -- Alpha-Beta pruning on / off ------------------------------------------------


def make_positions(ruleset: str, count: int, plies: int, seed: int) -> list[list[int]]:
    return [random_opening(ruleset, plies, derive_seed(seed, "position", i)) for i in range(count)]


def run_alphabeta(
    *,
    depths: list[int],
    positions: list[list[int]],
    ruleset: str = "normal",
    seed: int = 0,
    run_id: str | None = None,
    out_root: str | Path = "logs/ablation",
    position_plies: int | None = None,
    log: Callable[[str], None] | None = None,
) -> tuple[Path, dict]:
    from players.alphabeta import AlphaBetaPlayer

    log = log or (lambda message: print(message, flush=True))
    run_id = run_id or f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}_alphabeta_{ruleset}_s{seed}"
    path = _output_path(out_root, run_id)
    rules = get_rules(ruleset)
    rows = []
    for p_index, moves in enumerate(positions):
        board = Board()
        for move in moves:
            board.play(*divmod(move, SIZE))
        if rules.is_terminal(board):
            raise ValueError(f"position {p_index} is already finished")
        legal = len(rules.legal_moves(board))
        for depth in depths:
            row = {"position": p_index, "stones": len(moves), "legal_moves": legal, "depth": depth}
            for pruning in (True, False):
                player = AlphaBetaPlayer(rules, depth=depth, pruning=pruning)
                probe = board.copy()
                started = time.perf_counter()
                move = player.select_move(probe)
                elapsed = time.perf_counter() - started
                key = "on" if pruning else "off"
                row[f"move_{key}"] = list(move)
                row[f"nodes_{key}"] = player.nodes_visited
                row[f"time_{key}_sec"] = round(elapsed, 6)
            row["same_move"] = row["move_on"] == row["move_off"]
            row["node_ratio_off_over_on"] = round(row["nodes_off"] / row["nodes_on"], 4) if row["nodes_on"] else None
            row["time_ratio_off_over_on"] = (round(row["time_off_sec"] / row["time_on_sec"], 4)
                                             if row["time_on_sec"] else None)
            rows.append(row)
            log(f"position {p_index} depth {depth}: nodes {row['nodes_on']} (on) vs {row['nodes_off']} (off), "
                f"{row['time_on_sec']:.2f}s vs {row['time_off_sec']:.2f}s, same move: {row['same_move']}")

    by_depth = []
    for depth in depths:
        subset = [r for r in rows if r["depth"] == depth]
        by_depth.append({
            "depth": depth,
            "positions": len(subset),
            "same_move_rate": round(sum(r["same_move"] for r in subset) / len(subset), 4),
            "mean_nodes_on": _mean(r["nodes_on"] for r in subset),
            "mean_nodes_off": _mean(r["nodes_off"] for r in subset),
            "mean_time_on_sec": _mean(r["time_on_sec"] for r in subset),
            "mean_time_off_sec": _mean(r["time_off_sec"] for r in subset),
        })
    payload = {
        **_common("alphabeta", run_id, seed, ruleset),
        "config": {"depths": depths, "positions": len(positions), "position_plies": position_plies,
                   "position_rule": "uniform random legal moves, seed crc32(seed|position|i)"},
        "positions": positions,
        "results": rows,
        "summary": by_depth,
    }
    _write(path, payload)
    log(f"wrote {path}")
    return path, payload


# -- MCTS Pure simulation budgets ----------------------------------------------


def run_mcts(
    *,
    sims: list[int],
    games: int,
    opponent: str = "heuristic",
    ruleset: str = "normal",
    seed: int = 0,
    run_id: str | None = None,
    out_root: str | Path = "logs/ablation",
    alphabeta_depth: int = 1,
    builder: Callable | None = None,
    log: Callable[[str], None] | None = None,
) -> tuple[Path, dict]:
    log = log or (lambda message: print(message, flush=True))
    if games < 2 or games % 2:
        raise ValueError("--games must be an even number >= 2 so colours split evenly")
    run_id = run_id or f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}_mcts_{ruleset}_s{seed}"
    path = _output_path(out_root, run_id)
    budget = Budget("sims", 1)
    defaults = Defaults(alphabeta_depth=alphabeta_depth)
    opponent_spec = AgentSpec.parse(opponent)
    if opponent_spec.kind.startswith("az"):
        raise ValueError("the MCTS ablation opponent must not need a checkpoint")

    levels, all_games = [], []
    for n in sims:
        label = f"mcts_pure@sims={n}"
        if label == opponent_spec.label:
            raise ValueError(f"opponent {opponent!r} is the same as level {label!r}")
        entries = {label: agent_settings(AgentSpec.parse(label), budget, defaults),
                   opponent_spec.label: agent_settings(opponent_spec, budget, defaults)}
        tasks = schedule(list(entries), games, derive_seed(seed, "level", n), ruleset, 0)
        records = []
        for task in tasks:
            task.update(run_id=run_id, settings=entries, time_budget=None)
            record = play_game_task(task, builder)
            records.append(record)
            log(f"sims {n} game {len(records)}/{games}: {record['first_player']} (black) vs "
                f"{record['second_player']} (white): {record['winner_agent'] or 'draw'}")
        pair = aggregate(records, list(entries), None)["pairs"][0]
        levels.append({
            "sims": n,
            "games": pair["games"],
            "wins": pair["wins_a"], "draws": pair["draws"], "losses": pair["losses_a"],
            "score": pair["score_a"],
            "wilson95": pair["wilson95_a"],
            "games_black": pair["games_a_black"], "games_white": pair["games_a_white"],
            "score_black": pair["score_a_black"], "score_white": pair["score_a_white"],
            "illegal_moves": pair["illegal_moves_a"] + pair["illegal_moves_b"],
            "mean_moves_per_game": pair["mean_moves_per_game"],
            "mcts": pair["stats_a"],
            "opponent": pair["stats_b"],
        })
        for record in records:
            record = dict(record)
            record["stats"] = {side: {k: v for k, v in s.items() if k not in ("times", "nodes", "sims", "nn_calls")}
                               for side, s in record["stats"].items()}
            all_games.append({"sims": n, **record})

    payload = {
        **_common("mcts", run_id, seed, ruleset),
        "config": {"sims": sims, "games_per_level": games, "opponent": opponent_spec.label,
                   "score_rule": "score of MCTS Pure: win 1, draw 0.5, loss 0; Wilson 95% interval",
                   "seed_rule": "tournament schedule seeded with crc32(seed|level|sims)"},
        "results": levels,
        "games": all_games,
    }
    _write(path, payload)
    log(f"wrote {path}")
    return path, payload


# -- plot ------------------------------------------------------------------------


def plot(json_path: str | Path, out: str | Path | None = None) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("plot needs matplotlib: pip install matplotlib") from None
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    out = Path(out) if out else Path(json_path).with_suffix(".png")
    if data["mode"] == "alphabeta":
        rows = data["summary"]
        labels = [f"depth {r['depth']}" for r in rows]
        x = range(len(rows))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        width = 0.38
        ax1.bar([i - width / 2 for i in x], [r["mean_nodes_on"] for r in rows], width, label="pruning on")
        ax1.bar([i + width / 2 for i in x], [r["mean_nodes_off"] for r in rows], width, label="pruning off")
        ax1.set_ylabel("mean nodes_visited per move")
        ax2.bar([i - width / 2 for i in x], [r["mean_time_on_sec"] for r in rows], width, label="pruning on")
        ax2.bar([i + width / 2 for i in x], [r["mean_time_off_sec"] for r in rows], width, label="pruning off")
        ax2.set_ylabel("mean seconds per move")
        for ax in (ax1, ax2):
            ax.set_xticks(list(x), labels)
            ax.legend()
        fig.suptitle(f"Alpha-Beta pruning ablation ({rows[0]['positions']} positions) - {data['run_id']}")
    elif data["mode"] == "mcts":
        rows = data["results"]
        sims = [r["sims"] for r in rows]
        scores = [r["score"] for r in rows]
        low = [r["score"] - r["wilson95"][0] for r in rows]
        high = [r["wilson95"][1] - r["score"] for r in rows]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar(sims, scores, yerr=[low, high], marker="o", capsize=4)
        ax.set_xscale("log")
        ax.set_xticks(sims, [str(s) for s in sims])
        ax.set_ylim(0, 1)
        ax.set_xlabel("MCTS Pure rollouts per move")
        ax.set_ylabel(f"score vs {data['config']['opponent']} (Wilson 95%)")
        ax.set_title(f"{rows[0]['games']} games per level - {data['run_id']}")
    else:
        raise SystemExit(f"unknown ablation mode {data.get('mode')!r}")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")
    return out


# -- CLI -----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="T32 ablations: Alpha-Beta pruning, MCTS Pure budgets.")
    sub = parser.add_subparsers(dest="mode", required=True)

    ab = sub.add_parser("alphabeta", help="Pruning on vs off at the same depth.")
    ab.add_argument("--depths", type=_int_list, default=[2], help="Comma-separated depths (default 2).")
    ab.add_argument("--positions", type=int, default=3, help="Number of random positions (default 3).")
    ab.add_argument("--position-plies", type=int, default=100,
                    help="Stones per position (default 100: crowded board keeps depth 2 fast).")

    mc = sub.add_parser("mcts", help="MCTS Pure at several rollout budgets vs one opponent.")
    mc.add_argument("--sims", type=_int_list, default=[50, 200, 800], help="Comma-separated rollouts per move.")
    mc.add_argument("--games", type=int, default=100, help="Games per level, even (default 100).")
    mc.add_argument("--opponent", default="heuristic", help="Agent spec as in tournament.py (default heuristic).")
    mc.add_argument("--alphabeta-depth", type=int, default=1, help="Depth if the opponent is alphabeta.")

    for p in (ab, mc):
        p.add_argument("--ruleset", choices=("normal", "renju"), default="normal")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--run-id")
        p.add_argument("--out-root", default="logs/ablation")
        p.add_argument("--plot", action="store_true", help="Also write the PNG next to the JSON.")

    pl = sub.add_parser("plot", help="Draw a PNG from an ablation JSON.")
    pl.add_argument("json")
    pl.add_argument("--out")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mode == "plot":
        plot(args.json, args.out)
        return 0
    try:
        if args.mode == "alphabeta":
            if args.positions <= 0 or args.position_plies < 0:
                raise ValueError("--positions must be positive and --position-plies >= 0")
            positions = make_positions(args.ruleset, args.positions, args.position_plies, args.seed)
            path, _ = run_alphabeta(depths=args.depths, positions=positions, ruleset=args.ruleset, seed=args.seed,
                                    run_id=args.run_id, out_root=args.out_root, position_plies=args.position_plies)
        else:
            path, _ = run_mcts(sims=args.sims, games=args.games, opponent=args.opponent, ruleset=args.ruleset,
                               seed=args.seed, run_id=args.run_id, out_root=args.out_root,
                               alphabeta_depth=args.alphabeta_depth)
    except (FileExistsError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    if args.plot:
        plot(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
