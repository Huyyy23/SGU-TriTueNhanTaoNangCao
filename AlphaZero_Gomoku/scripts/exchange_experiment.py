"""Export or additively import a tournament experiment SQLite bundle."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from az.experiment_exchange import export_experiment, import_experiment  # noqa: E402
from az.logging_v2 import TOURNAMENT_DB  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="create a new experiment bundle; existing output is never overwritten")
    export.add_argument("--db", type=Path, default=TOURNAMENT_DB)
    export.add_argument("--experiment-id", required=True)
    export.add_argument("--out", type=Path, required=True)
    importer = commands.add_parser("import", help="add an experiment to the local database without replacing rows")
    importer.add_argument("--db", type=Path, default=TOURNAMENT_DB)
    importer.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "export":
        checksum = export_experiment(args.db, args.out, args.experiment_id)
        print(json.dumps({"experiment_id": args.experiment_id, "bundle": str(args.out), "sha256": checksum}))
    else:
        print(json.dumps(import_experiment(args.db, args.bundle), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
