"""Conservative retention for immutable AlphaZero generation directories.

The policy keeps the active/latest generation, the evaluation-approved best
generation, latest's immediate parent ("one previous"), and recovery outboxes
that are actually needed by those generations. Everything else is only a
candidate until a dry-run is reviewed; apply mode re-checks before deleting.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

_GENERATION_RE = re.compile(r"^[0-9a-f]{32}$")


def _size_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path, followlinks=False):
        for name in files:
            candidate = Path(root) / name
            if candidate.is_symlink():
                continue
            try:
                total += candidate.stat().st_size
            except OSError:
                return -1
    return total


def _read_pointer(path: Path) -> str | None:
    if not path.is_file() or path.is_symlink():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))["generation"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return value if isinstance(value, str) and _GENERATION_RE.fullmatch(value) else None


def _checkpoint_meta(directory: Path) -> dict[str, Any]:
    checkpoint = directory / "checkpoint.pt"
    if not checkpoint.is_file() or checkpoint.is_symlink():
        return {}
    try:
        import torch

        loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception:
        return {}
    if not isinstance(loaded, dict):
        return {}
    meta = loaded.get("meta", {})
    return meta if isinstance(meta, dict) else {}


def _recovery_dir_for_identity(root: Path, identity: dict[str, Any]) -> Path | None:
    try:
        import hashlib
        import json

        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    except Exception:
        return None
    return root / "recovery" / key


def plan(out_dir: str | Path, *, dry_run: bool = True, protected_paths=()) -> dict[str, Any]:
    """Build a conservative retention plan. Never deletes anything."""
    out_dir = Path(out_dir)
    generations_dir = out_dir / "generations"
    result: dict[str, Any] = {
        "out_dir": str(out_dir),
        "dry_run": dry_run,
        "kept": {},
        "candidates": [],
        "candidate_bytes": 0,
        "freed_bytes": 0,
        "deleted": [],
        "skipped": [],
        "errors": [],
    }

    if not generations_dir.is_dir() or generations_dir.is_symlink():
        result["errors"].append("generations directory is missing or invalid")
        return result

    latest_ptr = out_dir / "latest.json"
    best_ptr = out_dir / "best.json"
    latest_gen = _read_pointer(latest_ptr)
    best_gen = _read_pointer(best_ptr)

    kept: dict[str, str] = {}
    if latest_gen is not None:
        kept[latest_gen] = "active latest pointer"
    if best_gen is not None and best_gen != latest_gen:
        kept[best_gen] = "best pointer"
    # One immediate previous of latest, plus the abandoned branch it may point to.
    if latest_gen is not None:
        directory = generations_dir / latest_gen
        manifest_path = directory / "manifest.json"
        if manifest_path.is_file() and not manifest_path.is_symlink():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                for key, label in (("previous", "one previous of latest"),
                                   ("commit_previous", "abandoned branch of latest")):
                    value = manifest.get(key)
                    if (isinstance(value, str) and _GENERATION_RE.fullmatch(value)
                            and (generations_dir / value).is_dir()):
                        kept.setdefault(value, label)
            except (OSError, ValueError, TypeError):
                result["skipped"].append(f"{latest_gen}: unreadable manifest")

    # Recovery outboxes needed by kept generations.
    active_run_ids: set[str] = set()
    kept_checkpoint_shas: set[str] = set()
    for gen, reason in list(kept.items()):
        directory = generations_dir / gen
        meta = _checkpoint_meta(directory)
        recovery = meta.get("recovery")
        if isinstance(recovery, dict):
            identity = recovery.get("identity")
            if isinstance(identity, dict):
                rd = _recovery_dir_for_identity(out_dir, identity)
                if rd is not None and rd.is_dir() and not rd.is_symlink():
                    kept[f"recovery:{rd.name}"] = f"recovery needed by {gen}"
        run_id = meta.get("run_id")
        if isinstance(run_id, str):
            active_run_ids.add(run_id)
        checkpoint = directory / "checkpoint.pt"
        if checkpoint.is_file() and not checkpoint.is_symlink():
            try:
                import hashlib

                kept_checkpoint_shas.add(hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            except OSError:
                pass

    # External references in pointer-like JSON files and checkpoint metadata.
    for json_path in out_dir.glob("*.json"):
        if json_path.is_symlink():
            continue
        try:
            text = json_path.read_text(encoding="utf-8")
        except OSError:
            continue
        # Keep any generation id that appears as a whole token.
        for match in re.finditer(r"\b([0-9a-f]{32})\b", text):
            gid = match.group(1)
            if (generations_dir / gid).is_dir():
                kept.setdefault(gid, f"referenced by {json_path.relative_to(out_dir)}")
        # Keep recovery directories named in JSON.
        for match in re.finditer(r"recovery/([0-9a-f]{64})", text):
            rname = match.group(1)
            if (out_dir / "recovery" / rname).is_dir():
                kept.setdefault(f"recovery:{rname}", f"referenced by {json_path.relative_to(out_dir)}")

    for protected in protected_paths:
        protected = Path(protected)
        if protected.is_file():
            candidates = [protected]
        elif protected.is_dir():
            candidates = [p for p in protected.rglob("*") if p.is_file()]
        else:
            candidates = []
        for path in candidates:
            if path.is_symlink():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for match in re.finditer(r"\b([0-9a-f]{32})\b", text):
                gid = match.group(1)
                if (generations_dir / gid).is_dir():
                    kept.setdefault(gid, f"referenced by protected {path}")
            for match in re.finditer(r"recovery/([0-9a-f]{64})", text):
                rname = match.group(1)
                if (out_dir / "recovery" / rname).is_dir():
                    kept.setdefault(f"recovery:{rname}", f"referenced by protected {path}")
            # SQLite may store generation ids in blob/text; scan raw bytes too.
            try:
                data = path.read_bytes()
            except OSError:
                continue
            for match in re.finditer(rb"([0-9a-f]{32})", data):
                gid = match.group(1).decode("ascii")
                if (generations_dir / gid).is_dir():
                    kept.setdefault(gid, f"referenced by protected {path}")

    # Recovery directories needed by identity/source checkpoint/run.
    recovery_root = out_dir / "recovery"
    if recovery_root.is_dir() and not recovery_root.is_symlink():
        for recovery_dir in recovery_root.iterdir():
            if not recovery_dir.is_dir() or recovery_dir.is_symlink():
                result["skipped"].append(f"{recovery_dir}: invalid recovery directory")
                continue
            manifest_path = recovery_dir / "manifest.json"
            if not manifest_path.is_file() or manifest_path.is_symlink():
                # Fail closed: unknown recovery directory is retained.
                kept.setdefault(f"recovery:{recovery_dir.name}", "unverified recovery directory")
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                kept.setdefault(f"recovery:{recovery_dir.name}", "unreadable recovery manifest")
                continue
            identity = manifest.get("identity")
            if not isinstance(identity, dict):
                kept.setdefault(f"recovery:{recovery_dir.name}", "recovery manifest without identity")
                continue
            source_sha = identity.get("source_checkpoint_sha256")
            run_id = identity.get("run_id")
            if source_sha in kept_checkpoint_shas:
                kept.setdefault(f"recovery:{recovery_dir.name}", "recovery needed by kept generation")

    # Build candidate list. Anything not explicitly kept is a candidate.
    candidate_bytes = 0
    candidates: list[str] = []
    for directory in sorted(generations_dir.iterdir()):
        if not directory.is_dir() or directory.is_symlink():
            if directory.name not in (".", ".."):
                result["skipped"].append(f"{directory}: not a deletable generation directory")
            continue
        if not _GENERATION_RE.fullmatch(directory.name):
            result["skipped"].append(f"{directory.name}: non-generation directory")
            continue
        if directory.name in kept:
            continue
        size = _size_bytes(directory)
        if size < 0:
            result["skipped"].append(f"{directory.name}: unreadable size")
            continue
        # Fail closed on symlink/reparse points anywhere inside.
        has_symlink = False
        for root, dirs, files in os.walk(directory, followlinks=False):
            if any((Path(root) / name).is_symlink() for name in dirs + files):
                has_symlink = True
                break
        if has_symlink:
            result["skipped"].append(f"{directory.name}: contains symlink")
            continue
        candidates.append(directory.name)
        candidate_bytes += size

    result["kept"] = kept
    result["candidates"] = candidates
    result["candidate_bytes"] = candidate_bytes
    result["freed_bytes"] = 0 if dry_run else 0
    return result


def apply(out_dir: str | Path, *, dry_run: bool = True, protected_paths=()) -> dict[str, Any]:
    """Apply retention. Defaults to dry-run."""
    out_dir = Path(out_dir)
    first = plan(out_dir, dry_run=True, protected_paths=protected_paths)
    if dry_run:
        return first

    # Re-check immediately before mutation.
    second = plan(out_dir, dry_run=True, protected_paths=protected_paths)
    if second["candidates"] != first["candidates"] or second["candidate_bytes"] != first["candidate_bytes"]:
        result = dict(second)
        result["errors"].append("retention plan changed between dry-run and apply; kept all candidates")
        result["candidates"] = []
        result["candidate_bytes"] = 0
        result["freed_bytes"] = 0
        return result

    generations_dir = out_dir / "generations"
    freed = 0
    deleted: list[str] = []
    errors: list[str] = []
    for name in second["candidates"]:
        directory = generations_dir / name
        size = _size_bytes(directory)
        try:
            shutil.rmtree(directory)
            deleted.append(name)
            freed += max(size, 0)
        except OSError as exc:
            errors.append(f"{name}: {exc}")
            # Fail closed: stop deleting after first failure.
            break

    result = dict(second)
    result["dry_run"] = False
    result["deleted"] = deleted
    result["freed_bytes"] = freed
    result["errors"].extend(errors)
    if deleted:
        result["candidates"] = [name for name in second["candidates"] if name not in deleted]
    return result


def format_report(result: dict[str, Any]) -> str:
    lines = [
        f"retention: kept={len(result.get('kept', {}))} "
        f"candidates={len(result.get('candidates', []))} "
        f"deleted={len(result.get('deleted', []))} "
        f"freed_bytes={result.get('freed_bytes', 0)}",
    ]
    if result.get("candidates"):
        lines.append("candidates: " + ", ".join(result["candidates"]))
    if result.get("deleted"):
        lines.append("deleted: " + ", ".join(result["deleted"]))
    if result.get("skipped"):
        lines.append("skipped: " + "; ".join(result["skipped"]))
    if result.get("errors"):
        lines.append("errors: " + "; ".join(result["errors"]))
    return "\n".join(lines)
