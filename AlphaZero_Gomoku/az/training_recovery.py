from __future__ import annotations

import base64
import hashlib
import json
import os
import uuid
from dataclasses import asdict
from pathlib import Path

from az.game_log import GameRecord, MoveRecord
from az.replay_buffer import ReplayBuffer, _file_sha256
from az.trainer import _sync_directory, _sync_file


class Capture:
    def __init__(self, buffer, games):
        self.buffer = buffer
        self.samples = ReplayBuffer(max(1, int(games) * 121))

    def add(self, features, pi, z, rule_set, game_id=""):
        self.samples.add(features, pi, z, rule_set, game_id=game_id)
        self.buffer.add(features, pi, z, rule_set, game_id=game_id)


def _identity(run_id, iteration, source):
    return {"run_id": run_id, "iteration": iteration, "source_checkpoint_sha256": source}


def _directory(root, identity):
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return Path(root) / "recovery" / key



def reject_other_source(root, identity):
    parent = Path(root) / "recovery"
    if not parent.exists():
        return
    for directory in parent.iterdir():
        if directory.name.startswith(".pending-"):
            continue
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        other = manifest["identity"]
        if (other["run_id"] == identity["run_id"] and other["iteration"] == identity["iteration"]
                and other != identity):
            raise ValueError("Recovery source checkpoint collision")


def _validate_records(identity, records):
    seen = set()
    for record in records:
        provenance = record.config.get("provenance") or {}
        if (any(provenance.get(key) != value for key, value in identity.items())
                or not provenance.get("game_id") or provenance["game_id"] in seen):
            raise ValueError("Recovery game provenance mismatch or collision")
        seen.add(provenance["game_id"])


def _write_new(path, data):
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def prepare(root, identity, buffer, records, samples=None):
    _validate_records(identity, records)
    if samples is not None:
        expected = {record.config["provenance"]["game_id"]: record.moves_count for record in records}
        actual = {game_id: 0 for game_id in expected}
        for game_id in samples.game_ids[:len(samples)]:
            if game_id not in actual:
                raise ValueError("Recovery sample provenance mismatch")
            actual[game_id] += 1
        if actual != expected:
            raise ValueError("Recovery sample count mismatch")
    directory = _directory(root, identity)
    parent = directory.parent
    parent.mkdir(exist_ok=True)
    temporary = parent / (".pending-" + uuid.uuid4().hex)
    temporary.mkdir()
    buffer.save_npz(temporary / "replay.npz")
    _sync_file(temporary / "replay.npz")
    if samples is not None:
        samples.save_npz(temporary / "samples.npz")
        _sync_file(temporary / "samples.npz")
    payload = []
    for record in records:
        item = asdict(record)
        for root_data in item.get("root_data") or []:
            root_data["data"] = base64.b64encode(root_data["data"]).decode("ascii")
        payload.append(item)
    _write_new(temporary / "records.json", json.dumps(payload, sort_keys=True, allow_nan=False).encode())
    manifest = {"identity": identity, "replay_sha256": _file_sha256(temporary / "replay.npz"),
                "records_sha256": _file_sha256(temporary / "records.json")}
    if samples is not None:
        manifest["samples_sha256"] = _file_sha256(temporary / "samples.npz")
    _write_new(temporary / "manifest.json", json.dumps(manifest, sort_keys=True).encode())
    _sync_directory(temporary)
    os.rename(temporary, directory)
    _sync_directory(parent)
    return directory


def load(root, identity):
    directory = _directory(root, identity)
    if not directory.exists():
        return None
    if directory.is_symlink():
        raise ValueError("Invalid recovery directory")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("identity") != identity or
            manifest.get("replay_sha256") != _file_sha256(directory / "replay.npz") or
            manifest.get("records_sha256") != _file_sha256(directory / "records.json")):
        raise ValueError("Recovery outbox identity or hash mismatch")
    if "samples_sha256" in manifest and manifest["samples_sha256"] != _file_sha256(directory / "samples.npz"):
        raise ValueError("Recovery samples hash mismatch")
    records = []
    for item in json.loads((directory / "records.json").read_text(encoding="utf-8")):
        if item.get("moves") is not None:
            item["moves"] = [MoveRecord(**move) for move in item["moves"]]
        for root_data in item.get("root_data") or []:
            root_data["data"] = base64.b64decode(root_data["data"], validate=True)
        records.append(GameRecord(**item))
    _validate_records(identity, records)
    return directory, ReplayBuffer.load_npz(directory / "replay.npz"), records


def save_metrics(directory, row, recorded_at):
    name = "metrics-" + uuid.uuid4().hex + ".json"
    path = directory / name
    data = json.dumps({"row": row, "recorded_at": recorded_at}, sort_keys=True, allow_nan=False).encode()
    _write_new(path, data)
    _sync_directory(directory)
    return name, _file_sha256(path)


def load_metrics(directory, name, digest):
    if not isinstance(name, str) or not name.startswith("metrics-") or Path(name).name != name:
        raise ValueError("Invalid recovery metrics path")
    path = directory / name
    if not digest or _file_sha256(path) != digest:
        raise ValueError("Recovery metrics hash mismatch")
    return json.loads(path.read_text(encoding="utf-8"))
