import os
import subprocess
import sys

import numpy as np
import pytest

import gomoku.patterns as patterns
from gomoku.board import BLACK, Board
from gomoku.rules.normal import NormalRules
from gomoku.rules.renju import RenjuRules

@pytest.fixture
def native():
    return pytest.importorskip("gomoku_native")


def test_differential_patterns_on_random_boards(monkeypatch, native):
    rng = np.random.default_rng(42)
    count = 0
    for _ in range(120):
        board = Board()
        board.grid[:] = rng.choice([0, 1, 2], 121, p=[0.65, 0.175, 0.175])
        before = board.grid.copy()
        for index in np.flatnonzero(board.grid == 0):
            r, c = divmod(int(index), 11)
            for player in (1, 2):
                monkeypatch.setattr(patterns, "_native", None)
                expected = patterns.find_patterns(board, r, c, player)
                monkeypatch.setattr(patterns, "_native", native)
                assert patterns.find_patterns(board, r, c, player) == expected
                count += 1
        np.testing.assert_array_equal(board.grid, before)
    assert count >= 17790


def test_invalid_inputs_match_and_board_unchanged(monkeypatch, native):
    board = Board()
    board.play(5, 5)
    before = board.grid.copy()
    for args in ((-1, 0, BLACK), (11, 0, BLACK), (0, -1, BLACK), (0, 11, BLACK), (0, 0, 0), (0, 0, 3), (5, 5, BLACK)):
        errors = []
        for backend in (None, native):
            monkeypatch.setattr(patterns, "_native", backend)
            with pytest.raises(ValueError) as exc:
                patterns.find_patterns(board, *args)
            errors.append(str(exc.value))
        assert errors[0] == errors[1]
    np.testing.assert_array_equal(board.grid, before)


def test_invalid_board_rejected_by_both_backends(monkeypatch, native):
    for backend in (None, native):
        monkeypatch.setattr(patterns, "_native", backend)
        with pytest.raises(ValueError, match="board must be a Board"):
            patterns.find_patterns(None, 0, 0, BLACK)


def test_rule_legal_moves_unchanged(monkeypatch, native):
    board = Board()
    for r, c in [(5, 4), (0, 0), (5, 6), (0, 2), (4, 5), (0, 4), (6, 5), (0, 6)]:
        board.play(r, c)
    for rules in (NormalRules(), RenjuRules()):
        before = board.grid.copy()
        monkeypatch.setattr(patterns, "_native", None)
        expected = rules.legal_moves(board)
        for backend in (None, native):
            monkeypatch.setattr(patterns, "_native", backend)
            for i in expected[:5]:
                patterns.find_patterns(board, *divmod(int(i), 11), board.current_player)
            assert rules.legal_moves(board) == expected
            np.testing.assert_array_equal(board.grid, before)


def test_pattern_order_and_edge_completions(monkeypatch, native):
    board = Board()
    for row, col in ((0, 1), (0, 2), (1, 0), (2, 0), (1, 1), (2, 2)):
        board.grid[row * 11 + col] = BLACK
    before = board.grid.copy()
    expected = [
        patterns.Pattern("blocked_three", (0, 1), ((0, 3),)),
        patterns.Pattern("blocked_three", (1, 0), ((3, 0),)),
        patterns.Pattern("blocked_three", (1, 1), ((3, 3),)),
    ]
    for backend in (None, native):
        monkeypatch.setattr(patterns, "_native", backend)
        assert patterns.find_patterns(board, 0, 0, BLACK) == expected
        np.testing.assert_array_equal(board.grid, before)


def test_forced_backends_really_selected():
    for backend in ("python", "native"):
        env = {**os.environ, "GOMOKU_PATTERN_BACKEND": backend}
        code = "import gomoku.patterns as p; assert (p._native is not None) == " + str(backend == "native")
        subprocess.run([sys.executable, "-c", code], env=env, check=True)


def test_forced_native_missing_extension_fails():
    env = {**os.environ, "GOMOKU_PATTERN_BACKEND": "native"}
    code = '''
import builtins
original_import = builtins.__import__
def without_native(name, *args, **kwargs):
    if name == "gomoku_native":
        raise ImportError("extension unavailable")
    return original_import(name, *args, **kwargs)
builtins.__import__ = without_native
import gomoku.patterns
'''
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "GOMOKU_PATTERN_BACKEND=native requires the gomoku_native extension" in result.stderr


def test_invalid_backend_fails():
    env = {**os.environ, "GOMOKU_PATTERN_BACKEND": "invalid"}
    result = subprocess.run(
        [sys.executable, "-c", "import gomoku.patterns"],
        env=env, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "GOMOKU_PATTERN_BACKEND must be python, native, or auto" in result.stderr
