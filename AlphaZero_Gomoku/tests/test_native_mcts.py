import numpy as np
import pytest

from az.mcts import MCTS, NativeMCTS
from gomoku.board import Board
from gomoku.rules.normal import NormalRules
from gomoku.rules.renju import RenjuRules


def test_native_standard_root_telemetry_captures_one_search_without_researching():
    import time
    from az.logging_v2 import decode_mcts_root
    from az.selfplay import _Game, _advance

    board = Board()
    rules = NormalRules()
    native, _ = run_search(NativeMCTS, board, rules, simulations=2)
    game = _Game(
        game_id=0, provenance={"game_id": "test"}, board=board, mcts=native, started=time.perf_counter(),
        mcts_logging="STANDARD", search_started=time.perf_counter(),
        move_records=[], root_data=[],
    )
    config = {"pi_target_temp": 1.0, "temp_moves": 15, "n_simulations": 2}
    expected_visits = int(native.visit_counts().sum())
    assert _advance(game, rules, config, None, 0) is None
    assert len(game.root_data) == len(game.move_records) == 1
    root = decode_mcts_root(game.root_data[0]["encoding"], game.root_data[0]["data"])
    assert int(root["visit_counts"].sum()) == expected_visits
    assert game.move_records[0].simulations_run == 2


class FixedEvaluator:
    def __init__(self):
        self.policy = np.linspace(1.0, 2.0, 121, dtype=np.float32)

    def __call__(self, request):
        return self.policy, 0.2


def run_search(search, board, rules, simulations=20, noise=False):
    rng = np.random.default_rng(13)
    searcher = search(rules, n_simulations=simulations, root_noise=noise, rng=rng)
    searcher.reset(board)
    evaluator = FixedEvaluator()
    requests = []
    while True:
        request = searcher.request()
        if request is None:
            break
        requests.append((request.planes.copy(), request.legal_mask.copy()))
        policy, value = evaluator(request)
        searcher.receive(policy, value)
    return searcher, requests


def test_native_mcts_matches_python_requests_visits_and_backup():
    board = Board()
    rules = NormalRules()
    python, py_requests = run_search(MCTS, board.copy(), rules, noise=True)
    native, cpp_requests = run_search(NativeMCTS, board.copy(), rules, noise=True)
    assert len(py_requests) == len(cpp_requests) == 20
    for (py_planes, py_mask), (cpp_planes, cpp_mask) in zip(py_requests, cpp_requests):
        np.testing.assert_array_equal(cpp_planes, py_planes)
        np.testing.assert_array_equal(cpp_mask, py_mask)
    np.testing.assert_array_equal(native.visit_counts(), python.visit_counts())
    for native_stats, python_stats in zip(native.root_statistics(), python.root_statistics()):
        np.testing.assert_allclose(native_stats, python_stats, rtol=0, atol=1e-12)
    assert native.root_value() == pytest.approx(python.root_value())
    assert native.simulations_done == python.simulations_done


def test_native_mcts_masks_renju_forbidden_moves():
    board = Board()
    rules = RenjuRules()
    native = NativeMCTS(rules, n_simulations=2, root_noise=True, rng=np.random.default_rng(3))
    native.reset(board)
    request = native.request()
    assert request is not None
    forbidden = rules.forbidden_moves(board)
    assert not request.legal_mask[forbidden].any()
    native.receive(np.ones(121, dtype=np.float32), 0.0)
    assert not set(forbidden).intersection(native.visit_counts().nonzero()[0])
    board.grid[[5 * 11 + c for c in (4, 6)] + [r * 11 + 5 for r in (4, 6)]] = 1
    native.reset(board)
    request = native.request()
    assert request is not None
    assert not request.legal_mask[5 * 11 + 5]


def test_native_mcts_matches_python_on_renju_forbidden_fixture():
    board = Board()
    board.grid[[5 * 11 + c for c in (4, 6)] + [r * 11 + 5 for r in (4, 6)]] = 1
    rules = RenjuRules()
    python, py_requests = run_search(MCTS, board.copy(), rules, simulations=8, noise=True)
    native, cpp_requests = run_search(NativeMCTS, board.copy(), rules, simulations=8, noise=True)
    assert len(py_requests) == len(cpp_requests)
    for (py_planes, py_mask), (cpp_planes, cpp_mask) in zip(py_requests, cpp_requests):
        np.testing.assert_array_equal(cpp_planes, py_planes)
        np.testing.assert_array_equal(cpp_mask, py_mask)
    np.testing.assert_array_equal(native.visit_counts(), python.visit_counts())
    for native_stats, python_stats in zip(native.root_statistics(), python.root_statistics()):
        np.testing.assert_allclose(native_stats, python_stats, rtol=0, atol=1e-12)
    assert native.root_value() == pytest.approx(python.root_value())

def test_native_renju_exact_five_legal_and_overline_forbidden():
    rules = RenjuRules()
    exact_five = Board()
    exact_five.grid[[5 * 11 + c for c in (2, 3, 4, 5)]] = 1
    assert 5 * 11 + 6 not in rules.forbidden_moves(exact_five)
    native = NativeMCTS(rules, n_simulations=3, rng=np.random.default_rng(5))
    native.search(exact_five, lambda _r: (np.eye(1, 121, 5 * 11 + 6, dtype=np.float32)[0], 0.0))
    assert native.visit_counts()[5 * 11 + 6] > 0

    overline = Board()
    overline.play(0, 0)
    overline.grid[[5 * 11 + c for c in (1, 2, 3, 4, 5)]] = 1
    overline.current_player = 1
    native.reset(overline)
    request = native.request()
    assert request is not None
    assert 5 * 11 + 6 in rules.forbidden_moves(overline)
    assert not request.legal_mask[5 * 11 + 6]

def test_native_mcts_win_in_one_and_terminal_root():
    board = Board()
    rules = NormalRules()
    for move in ((5, 1), (0, 0), (5, 2), (0, 1), (5, 3), (0, 2), (5, 4), (0, 3)):
        board.play(*move)
    native = NativeMCTS(rules, n_simulations=64, root_noise=False, rng=np.random.default_rng(2))
    native.search(board, lambda _request: (np.ones(121, dtype=np.float32), 0.0))
    assert native.visit_counts()[5 * 11 + 5] > 0

    won = board.copy()
    won.play(5, 5)
    native.reset(won)
    assert native.request() is None


def test_native_tree_repeated_search_releases_nodes():
    rules = NormalRules()
    native = NativeMCTS(rules, n_simulations=1, rng=np.random.default_rng(1))
    evaluator = lambda _request: (np.ones(121, dtype=np.float32), 0.0)
    for _ in range(100):
        native.search(Board(), evaluator)
        assert native.tree_size() == 122
        native.reset(Board())
        assert native.tree_size() == 1

def test_native_selfplay_keeps_batching_and_sample_targets():
    from az.selfplay import run_games

    class Model:
        def predict(self, features):
            return np.ones((len(features), 121), dtype=np.float32), np.zeros(len(features), dtype=np.float32)

    cfg = {"rule_set": "normal", "parallel_games": 2, "n_simulations": 2,
           "c_puct": 1.5, "dirichlet_alpha": 0.15, "dirichlet_eps": 0.25,
           "pi_target_temp": 1.0, "temp_moves": 15, "log_moves": False,
           "mcts_backend": "cpp", "patterns_backend": "python"}
    samples = []
    records = list(run_games(Model(), NormalRules(), cfg, 2, seed=7, samples_out=samples))
    assert len(records) == 2
    assert len(samples) == sum(record.moves_count for record in records)
    assert all(record.mean_batch > 1 for record in records)
    assert all(np.isclose(pi.sum(), 1.0) and z in (-1.0, 0.0, 1.0) for _, pi, z in samples)

def test_native_selfplay_parallel_workers_store_samples():
    from concurrent.futures import ProcessPoolExecutor

    from az.model import AZNet
    from az.replay_buffer import ReplayBuffer
    from az.selfplay import run_games_parallel

    cfg = {"rule_set": "normal", "parallel_games": 1, "n_simulations": 2,
           "c_puct": 1.5, "dirichlet_alpha": 0.15, "dirichlet_eps": 0.25,
           "pi_target_temp": 1.0, "temp_moves": 2, "log_moves": False,
           "mcts_backend": "cpp", "patterns_backend": "python",
           "n_blocks": 1, "n_filters": 8,
           "components_used": {"mcts": "cpp", "patterns": "not_used"}}
    model = AZNet(n_blocks=1, n_filters=8)
    buffer = ReplayBuffer(5000)
    with ProcessPoolExecutor(max_workers=2) as pool:
        records = run_games_parallel(pool, model, NormalRules(), cfg, 2,
                                     seed=19, buffer=buffer, workers=2)
    assert len(records) == 2
    assert len(buffer) == sum(record.moves_count for record in records)
    assert all(record.config["components_used"]["mcts"] == "cpp" for record in records)


