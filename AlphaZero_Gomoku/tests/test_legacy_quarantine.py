import hashlib
import sqlite3

import pytest

from az import logging_v2
from scripts.quarantine_legacy import quarantine_source, restore_source


def test_quarantine_preserves_bytes_and_is_idempotent(tmp_path):
    path = tmp_path / 'history.jsonl'
    raw = b'\xef\xbb\xbf{"game_id":1}\r\n\x00unparsed\xff'
    path.write_bytes(raw)
    database = tmp_path / 'selfplay.sqlite'
    logging_v2.initialize(database, 'selfplay')
    with sqlite3.connect(database) as db:
        assert quarantine_source(db, path, 'history.jsonl', 'identity conflict') is True
        assert quarantine_source(db, path, 'history.jsonl', 'identity conflict') is False
        assert restore_source(db, 'history.jsonl', hashlib.sha256(raw).hexdigest()) == raw
        assert db.execute('select count(*) from games').fetchone()[0] == 0
        assert db.execute("select count(*) from events where event='legacy_quarantine'").fetchone()[0] == 1


def test_quarantine_does_not_mark_uncommitted_changes_as_saved(tmp_path):
    path = tmp_path / 'history.jsonl'
    path.write_bytes(b'old')
    database = tmp_path / 'selfplay.sqlite'
    logging_v2.initialize(database, 'selfplay')
    with sqlite3.connect(database) as db:
        quarantine_source(db, path, 'history.jsonl', 'conflict')
        path.write_bytes(b'new')
        quarantine_source(db, path, 'history.jsonl', 'changed source')
        assert db.execute("select count(*) from events where event='legacy_quarantine'").fetchone()[0] == 2
        with pytest.raises(KeyError):
            restore_source(db, 'history.jsonl', 'missing')
