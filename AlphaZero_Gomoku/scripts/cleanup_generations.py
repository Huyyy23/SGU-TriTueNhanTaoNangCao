"""Conservative generation retention CLI (dry-run by default)."""

import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from az.generation_retention import apply, format_report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generation retention dry-run/apply")
    parser.add_argument("--out-dir", default="checkpoints/normal")
    parser.add_argument("--apply", action="store_true", help="delete candidates (default dry-run)")
    parser.add_argument("--protected", nargs="*", default=(), help="extra files/dirs to scan for references")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    protected = [Path(p) for p in args.protected]
    result = apply(args.out_dir, dry_run=not args.apply, protected_paths=protected)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(format_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
