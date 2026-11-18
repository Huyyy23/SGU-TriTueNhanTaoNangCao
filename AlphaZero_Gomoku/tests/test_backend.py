import pytest

from az.backend import configure_backends


def test_python_backend_and_component_override():
    result = configure_backends("python")
    assert result["mcts"] == "python"
    assert result["patterns"] == "python"
    result = configure_backends("python", patterns_backend="python")
    assert result["patterns"] == "python"
    result = configure_backends("cpp", mcts_backend="python", patterns_backend="python")
    assert result["mcts"] == "python"

def test_cpp_backend_initializes_native_mcts():
    try:
        result = configure_backends("cpp", patterns_backend="python")
    except ImportError as exc:
        pytest.skip(str(exc))
    assert result["mcts"] == "cpp"

def test_missing_native_mcts_fails_closed(monkeypatch):
    from az import backend

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("extension missing")

    monkeypatch.setattr(backend, "_ensure_native_module", unavailable)
    with pytest.raises(ImportError, match="MCTS backend 'cpp' is unavailable"):
        configure_backends("cpp")

def test_native_tree_lifecycle_and_backup():
    native = pytest.importorskip("gomoku_mcts_native")
    tree = native.Tree(1.5)
    node, moves = tree.request(1, lambda n: 0)
    assert node == 0 and moves == []
    tree.receive(node, list(range(121)), [1 / 121] * 121, 0.25)
    node, moves = tree.request(2, lambda n: 0)
    assert len(moves) == 1
    tree.receive(node, list(range(121)), [1 / 121] * 121, 0.6)
    assert tree.simulations_done == 2
    assert tree.visit_counts().sum() == 1
    assert tree.root_value == pytest.approx(-0.175)
    for _ in range(200):
        tree.reset()
        node, _ = tree.request(1, lambda n: 0)
        tree.receive(node, [0], [1.0], 0.0)
        assert tree.size == 2


def test_invalid_backend_is_rejected():
    with pytest.raises(ValueError, match="backend must be auto, cpp, or python"):
        configure_backends("bogus")
