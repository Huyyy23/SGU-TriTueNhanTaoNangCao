"""Build the optional pybind11 extension with a supported C++ compiler."""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import time

ROOT = Path(__file__).resolve().parents[1]


def find_compiler() -> str | None:
    configured = os.environ.get("CXX", "").strip().strip('"')
    for name in dict.fromkeys(filter(None, (configured, "g++", "clang++", "cl"))):
        if compiler := shutil.which(name):
            return compiler
    return None


def build_command(cxx: str, source: Path, output: Path, include: Path) -> list[str]:
    python_include = sysconfig.get_path("include")
    if Path(cxx).name.lower() in {"cl", "cl.exe"}:
        python_lib = Path(sys.base_prefix) / "libs"
        return [
            cxx, "/nologo", "/O2", "/std:c++17", "/EHsc", "/MD", "/LD",
            "/I" + python_include, "/I" + str(include), str(source), "/link",
            "/OUT:" + str(output), "/LIBPATH:" + str(python_lib),
            f"python{sys.version_info.major}{sys.version_info.minor}.lib",
        ]
    cmd = [cxx, "-O3", "-std=c++17", "-shared", "-I" + python_include,
           "-I" + str(include), str(source), "-o", str(output)]
    if sys.platform == "win32":
        cmd += ["-static", str(Path(sys.base_prefix) / "libs" /
                               f"python{sys.version_info.major}{sys.version_info.minor}.lib")]
    else:
        cmd.insert(1, "-fPIC")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cxx")
    parser.add_argument("--module", choices=("patterns", "mcts"), default="patterns")
    args = parser.parse_args()
    compiler = args.cxx or find_compiler()
    if not compiler:
        raise SystemExit("No C++ compiler found in PATH (g++, clang++, or cl).")
    spec = importlib.util.find_spec("pybind11")
    if spec is not None:
        import pybind11
        include = Path(pybind11.get_include())
    else:
        torch = importlib.util.find_spec("torch")
        if torch is None or not torch.submodule_search_locations:
            raise SystemExit("Install pybind11 or provide its headers via the existing PyTorch installation.")
        include = Path(next(iter(torch.submodule_search_locations))) / "include"
    if not (include / "pybind11" / "pybind11.h").is_file():
        raise SystemExit(f"pybind11 headers not found under {include}")
    suffix = "gomoku_native" if args.module == "patterns" else "gomoku_mcts_native"
    source_name = "gomoku_native.cpp" if args.module == "patterns" else "gomoku_mcts_native.cpp"
    output = ROOT / (suffix + sysconfig.get_config_var("EXT_SUFFIX"))
    source = ROOT / "native" / source_name
    cmd = build_command(compiler, source, output, include)
    env = os.environ.copy()
    if sys.platform == "win32":
        compiler_dir = str(Path(compiler).resolve().parent) if Path(compiler).is_file() else ""
        if compiler_dir:
            env["PATH"] = compiler_dir + os.pathsep + env.get("PATH", "")
        env["TEMP"] = env["TMP"] = str(ROOT)
    print("Compiling:", subprocess.list2cmdline(cmd), flush=True)
    started = time.perf_counter()
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    print(f"Built {output} in {time.perf_counter() - started:.3f}s", flush=True)
    started = time.perf_counter()
    subprocess.run([sys.executable, "-c", f"import {suffix}; print({suffix}.__file__)"], cwd=ROOT, check=True)
    print(f"Import subprocess: {time.perf_counter() - started:.3f}s", flush=True)


if __name__ == "__main__":
    main()
