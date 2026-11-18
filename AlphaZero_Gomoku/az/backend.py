"""Resolve optional native components once per process, outside search loops."""
from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
from functools import lru_cache
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys
import warnings

_ROOT = Path(__file__).resolve().parents[1]
_BUILD_SCRIPT = _ROOT / "scripts" / "build_gomoku_native.py"
_NATIVE = {
    "mcts": ("gomoku_mcts_native", _ROOT / "native" / "gomoku_mcts_native.cpp"),
    "patterns": ("gomoku_native", _ROOT / "native" / "gomoku_native.cpp"),
}
_native_modules = {}
_warned_auto: set[str] = set()
_build_forbidden = False


@lru_cache(maxsize=1)
def find_cpp_compiler() -> str | None:
    """Return the first supported compiler executable found in PATH."""
    configured = os.environ.get("CXX", "").strip().strip('"')
    for name in dict.fromkeys(filter(None, (configured, "g++", "clang++", "cl"))):
        compiler = shutil.which(name)
        if compiler:
            return compiler
    return None


def _module_is_current(module_name: str, source: Path) -> bool:
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ValueError):
        return False
    if spec is None or not spec.origin:
        return False
    if not any(spec.origin.endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES):
        return False
    try:
        newest_input = source.stat().st_mtime_ns
        return Path(spec.origin).stat().st_mtime_ns >= newest_input
    except OSError:
        return False


def _ensure_native_module(component: str, *, compiler: str | None, allow_build: bool):
    module_name, source = _NATIVE[component]
    if component in _native_modules:
        return _native_modules[component]
    if not _module_is_current(module_name, source):
        if not allow_build:
            raise RuntimeError(f"{module_name} is missing or stale; workers are not allowed to build it")
        if module_name in sys.modules:
            raise RuntimeError(f"{module_name} was loaded before its source changed; restart before rebuilding")
        compiler = compiler or find_cpp_compiler()
        if compiler is None:
            raise RuntimeError("no compatible C++ compiler found in PATH")
        try:
            subprocess.run(
                [sys.executable, str(_BUILD_SCRIPT), "--module", component, "--cxx", compiler],
                cwd=_ROOT, check=True, capture_output=True, text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            output = (getattr(exc, "stderr", "") or getattr(exc, "stdout", "") or str(exc)).strip()
            raise RuntimeError(f"build failed: {output[-400:]}") from exc
        importlib.invalidate_caches()
        if not _module_is_current(module_name, source):
            raise RuntimeError(f"build did not produce a current {module_name} module")
    try:
        module = importlib.import_module(module_name)
        if component == "mcts":
            module.Tree(1.5)
        elif not callable(module.find_patterns):
            raise AttributeError("find_patterns is missing")
    except (ImportError, OSError, AttributeError, RuntimeError, TypeError) as exc:
        raise RuntimeError(f"{module_name} is incompatible: {exc}") from exc
    _native_modules[component] = module
    return module


def _warn_auto(message: str) -> None:
    if message not in _warned_auto:
        warnings.warn(message, RuntimeWarning, stacklevel=3)
        _warned_auto.add(message)


def configure_backends(backend: str = "auto", *, mcts_backend: str | None = None,
                       patterns_backend: str | None = None, allow_build: bool | None = None) -> dict[str, str | None]:
    """Resolve choices once; a process marked as a worker can only load, never build."""
    global _build_forbidden
    worker_process = multiprocessing.current_process().name != "MainProcess"
    if worker_process:
        _build_forbidden = True
    allow_build = bool(allow_build) and not _build_forbidden if allow_build is not None else not _build_forbidden
    if backend not in ("auto", "python", "cpp"):
        raise ValueError("backend must be auto, cpp, or python")
    requested = {
        "mcts": backend if mcts_backend is None else mcts_backend,
        "patterns": backend if patterns_backend is None else patterns_backend,
    }
    if any(value not in ("auto", "python", "cpp") for value in requested.values()):
        raise ValueError("component backends must be auto, python, or cpp")

    automatic = {name for name, value in requested.items() if value == "auto"}
    unresolved_worker_auto = worker_process and bool(automatic)
    compiler = find_cpp_compiler() if automatic and not unresolved_worker_auto else None
    selected = {name: ("cpp" if compiler else "python") if value == "auto" else value
                for name, value in requested.items()}
    if unresolved_worker_auto:
        _warn_auto("Worker received unresolved auto backend; using Python backend.")
    elif automatic and compiler is None:
        _warn_auto("No C++ compiler (g++, clang++, or cl) in PATH; using Python backend.")

    # Explicit C++ requests fail closed; only auto-selected components may fall back.
    for component in ("mcts", "patterns"):
        if selected[component] != "cpp" or component in automatic:
            continue
        try:
            _ensure_native_module(component, compiler=compiler, allow_build=allow_build)
        except Exception as exc:
            raise ImportError(
                f"{'MCTS' if component == 'mcts' else 'Patterns'} backend 'cpp' is unavailable or incompatible: {exc}. "
                "Install a compatible C++ compiler or use --backend python."
            ) from exc

    auto_cpp = [name for name in ("mcts", "patterns")
                if name in automatic and selected[name] == "cpp"]
    for component in auto_cpp:
        try:
            _ensure_native_module(component, compiler=compiler, allow_build=allow_build)
        except Exception as exc:
            for name in automatic:
                selected[name] = "python"
            _warn_auto(f"Native C++ build/load failed ({type(exc).__name__}); using Python backend.")
            break

    from gomoku import patterns as pattern_module
    if pattern_module._active_backend != selected["patterns"]:
        try:
            pattern_module.set_backend(selected["patterns"])
        except (ImportError, OSError, AttributeError, RuntimeError, TypeError) as exc:
            if "patterns" not in automatic:
                raise ImportError(f"patterns backend 'cpp' is unavailable or incompatible: {exc}") from exc
            selected["patterns"] = "python"
            for name in automatic:
                selected[name] = "python"
            pattern_module.set_backend("python")
            _warn_auto(f"Native patterns backend could not be loaded ({type(exc).__name__}); using Python backend.")

    mcts_module = _native_modules.get("mcts") if selected["mcts"] == "cpp" else None
    pattern_module_native = pattern_module._native if selected["patterns"] == "cpp" else None
    return {
        "requested_backend": backend,
        "mcts": selected["mcts"],
        "patterns": pattern_module._active_backend,
        "compiler": compiler,
        "native_mcts_version": (str(getattr(mcts_module, "__version__", None)
                                    or getattr(mcts_module, "__file__", "unknown"))
                                if mcts_module is not None else None),
        "native_patterns_version": (str(getattr(pattern_module_native, "__version__", None)
                                        or getattr(pattern_module_native, "__file__", "unknown"))
                                    if pattern_module_native is not None else None),
    }
