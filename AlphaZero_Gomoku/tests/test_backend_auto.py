import json
import sqlite3
import sys
import warnings
from pathlib import Path

import pytest

from az import backend
from az.backend_logging import log_backend_selection


class FakeNative:
    __version__ = "fake-native"

    @staticmethod
    def Tree(_c_puct):
        return object()

    @staticmethod
    def find_patterns(*_args):
        return []


@pytest.fixture(autouse=True)
def restore_pattern_backend():
    import gomoku.patterns as patterns

    native, active = patterns._native, patterns._active_backend
    yield
    patterns._native, patterns._active_backend = native, active


def reset_detection(monkeypatch):
    backend.find_cpp_compiler.cache_clear()
    monkeypatch.setattr(backend, "_native_modules", {})
    monkeypatch.setattr(backend, "_build_forbidden", False)
    monkeypatch.setattr(backend, "_warned_auto", set())

def test_auto_without_compiler_selects_python_without_loading_native(monkeypatch):
    reset_detection(monkeypatch)
    monkeypatch.setattr(backend.shutil, "which", lambda _name: None)

    def unexpected_load(*_args, **_kwargs):
        pytest.fail("auto without a compiler must not import or build native modules")

    monkeypatch.setattr(backend, "_ensure_native_module", unexpected_load)
    with pytest.warns(RuntimeWarning, match="compiler.*Python"):
        result = backend.configure_backends("auto")

    assert result["mcts"] == result["patterns"] == "python"


@pytest.mark.parametrize("compiler", ("g++", "clang++", "cl"))
def test_auto_recognizes_supported_compilers(monkeypatch, compiler):
    reset_detection(monkeypatch)
    monkeypatch.setattr(backend.shutil, "which", lambda name: f"/toolchain/{compiler}" if name == compiler else None)
    monkeypatch.delenv("CXX", raising=False)

    assert backend.find_cpp_compiler() == f"/toolchain/{compiler}"


def test_auto_uses_native_when_compiler_and_modules_are_available(monkeypatch):
    reset_detection(monkeypatch)
    monkeypatch.setattr(backend.shutil, "which", lambda name: "/toolchain/g++" if name == "g++" else None)
    loaded = []
    monkeypatch.setattr(backend, "_ensure_native_module",
                        lambda component, **_kwargs: loaded.append(component) or FakeNative())
    import gomoku.patterns as patterns
    monkeypatch.setattr(patterns, "set_backend", lambda selected: setattr(patterns, "_active_backend", selected))

    result = backend.configure_backends("auto")

    assert result["mcts"] == result["patterns"] == "cpp"
    assert loaded == ["mcts", "patterns"]
    assert result["compiler"] == "/toolchain/g++"


def test_auto_build_failure_falls_back_but_explicit_cpp_fails(monkeypatch):
    reset_detection(monkeypatch)
    monkeypatch.setattr(backend.shutil, "which", lambda name: "/toolchain/g++" if name == "g++" else None)
    monkeypatch.setattr(backend, "_ensure_native_module",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("compiler failed")))
    import gomoku.patterns as patterns
    monkeypatch.setattr(patterns, "set_backend", lambda selected: setattr(patterns, "_active_backend", selected))

    with pytest.warns(RuntimeWarning, match="build.*Python"):
        result = backend.configure_backends("auto")
    assert result["mcts"] == result["patterns"] == "python"

    with pytest.raises(ImportError, match="backend 'cpp'.*unavailable"):
        backend.configure_backends("cpp")


def test_python_selection_does_not_probe_or_load_native(monkeypatch):
    reset_detection(monkeypatch)

    def unexpected(*_args, **_kwargs):
        pytest.fail("Python backend must not probe or load C++")

    monkeypatch.setattr(backend, "find_cpp_compiler", unexpected)
    monkeypatch.setattr(backend, "_ensure_native_module", unexpected)
    import gomoku.patterns as patterns
    monkeypatch.setattr(patterns, "set_backend", lambda selected: setattr(patterns, "_active_backend", selected))

    result = backend.configure_backends("python")
    assert result["mcts"] == result["patterns"] == "python"


def test_worker_resolution_never_builds(monkeypatch):
    reset_detection(monkeypatch)
    calls = []
    monkeypatch.setattr(backend, "_ensure_native_module",
                        lambda component, **kwargs: calls.append((component, kwargs["allow_build"])) or FakeNative())
    import gomoku.patterns as patterns
    monkeypatch.setattr(patterns, "set_backend", lambda selected: setattr(patterns, "_active_backend", selected))

    result = backend.configure_backends("cpp", allow_build=False)

    assert result["mcts"] == result["patterns"] == "cpp"
    assert calls == [("mcts", False), ("patterns", False)]


def test_backend_selection_is_recorded_once_in_primary_sqlite(tmp_path):
    path = tmp_path / "selfplay.sqlite"
    for _ in range(2):
        log_backend_selection(
            run_id="train-001",
            workload="selfplay",
            requested_backend="auto",
            backends={"mcts": "cpp", "patterns": "cpp", "compiler": "g++"},
            components_used={"mcts": "cpp", "patterns": "not_used"},
            primary_path=path,
        )

    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "SELECT event, payload_json FROM events WHERE event='backend_selected'"
        ).fetchall()

    assert len(rows) == 1
    assert rows[0][0] == "backend_selected"
    payload = json.loads(rows[0][1])
    assert payload["run_id"] == "train-001"
    assert payload["backends"]["compiler"] == "g++"

def test_train_defers_backend_default_and_tournament_uses_auto():
    from scripts.tournament import build_parser as tournament_parser
    from scripts.train import build_parser as train_parser

    assert train_parser().parse_args([]).backend is None
    assert tournament_parser().parse_args([]).backend == "auto"


def test_msvc_build_command_targets_python_extension():
    from scripts.build_gomoku_native import build_command

    command = build_command("cl.exe", Path("module.cpp"), Path("module.pyd"), Path("pybind"))

    assert "/std:c++17" in command
    assert "/LD" in command
    assert any(argument.endswith(f"python{sys.version_info.major}{sys.version_info.minor}.lib") for argument in command)
