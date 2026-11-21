from collections import Counter

import numpy as np
import pytest

from gomoku.board import Board, SIZE
from gomoku.rules.normal import NormalRules
from dataclasses import replace
from scripts.tactical_benchmark import FIXTURES, board_for, oracle, winning_moves, run_case, split_integrity


def test_fixed_fixtures_balanced_legal_and_oracle_expected():
    assert len({f.name for f in FIXTURES}) == len(FIXTURES)
    assert Counter((f.split, f.category, f.player) for f in FIXTURES) == {
        (split, category, player): 3
        for split in ('train', 'dev', 'holdout')
        for category in ('win', 'block', 'fork', 'forced_defense')
        for player in (1, 2)
    }
    rules = NormalRules()
    for fixture in FIXTURES:
        board = board_for(fixture)
        replay = Board()
        for action in board.history:
            assert not rules.is_terminal(replay)
            assert action in rules.legal_moves(replay)
            replay.play(*divmod(action, SIZE))
        assert not rules.is_terminal(replay)
        assert board.current_player == fixture.player
        answers = oracle(board.grid, fixture.player, fixture.category)
        assert fixture.expected in answers
        assert all(board.grid[a] == 0 for a in answers)
        assert len(fixture.black) == len(fixture.white) + (fixture.player == 2)


def test_splits_are_disjoint_by_family_and_color_aware_d4_state():
    report = split_integrity(FIXTURES)
    assert report['ok']
    assert report['counts'] == {'train': 24, 'dev': 24, 'holdout': 24}
    assert not report['family_leakage']
    assert not report['duplicate_states']


def test_integrity_detects_family_and_d4_leakage():
    original = FIXTURES[0]
    def rotate(action):
        row, col = divmod(action, SIZE)
        return col * SIZE + SIZE - 1 - row
    duplicate = replace(original, name='duplicate', split='holdout',
                        black=tuple(map(rotate, original.black)),
                        white=tuple(map(rotate, original.white)))
    report = split_integrity((original, duplicate))
    assert not report['ok']
    assert report['family_leakage']
    assert report['duplicate_states']


def test_integrity_detects_duplicate_names_even_for_different_states():
    report = split_integrity((FIXTURES[0], replace(FIXTURES[1], name=FIXTURES[0].name)))
    assert not report['ok']
    assert report['duplicate_names']


def test_fork_and_defense_horizon():
    fork = next(f for f in FIXTURES if f.group == 'gap_three' and f.player == 1)
    defense = next(f for f in FIXTURES if f.group == 'defend_gap_three' and f.player == 1)
    assert 60 in oracle(board_for(fork).grid, 1, 'fork')
    answers = oracle(board_for(defense).grid, 1, 'forced_defense')
    assert 60 in answers
    assert 1 not in answers


def test_negative_controls_have_no_immediate_win_or_fork():
    grid = np.zeros(121, dtype=np.int8)
    grid[[58, 60]] = 1
    grid[[0, 14]] = 2
    for player in (1, 2):
        assert not oracle(grid, player, 'win')
        assert not oracle(grid, player, 'fork')


def test_window_wins_match_normal_rules_for_both_colors():
    rules = NormalRules()
    for fixture in FIXTURES:
        board = board_for(fixture)
        for player in (1, 2):
            expected = set()
            for action in rules.legal_moves(board):
                trial = board.copy()
                trial.current_player = player
                trial.play(*divmod(action, SIZE))
                if rules.is_win(trial):
                    expected.add(action)
            assert winning_moves(board.grid, player) == expected


def test_oracle_finds_all_immediate_wins_and_blocks():
    grid = np.zeros(121, dtype=np.int8)
    grid[[36, 37, 38, 39]] = 1
    assert winning_moves(grid, 1) == {35, 40}
    assert oracle(grid, 1, 'win') == {35, 40}
    # Two simultaneous threats cannot be stopped in one move.
    assert oracle(grid, 2, 'block') == set()


def test_oracle_does_not_mutate_input_and_rejects_unknown_category():
    board = board_for(FIXTURES[-1])
    before = board.grid.copy()
    oracle(board.grid, board.current_player, 'forced_defense')
    assert np.array_equal(board.grid, before)
    with pytest.raises(ValueError):
        oracle(board.grid, board.current_player, 'unknown')


def test_real_search_telemetry_with_fixed_evaluator():
    class Uniform:
        def predict(self, planes):
            return np.full((len(planes), 121), 1 / 121), np.zeros(len(planes))

    result = run_case(FIXTURES[0], Uniform(), 'python', 42)
    assert result['simulations'] == 200
    assert sum(result['visits']) == 199
    assert result['selected'] in result['legal']
    assert result['correct'] == (result['selected'] in result['answers'])
    assert 0 <= result['answer_prior_mass'] <= 1
    assert 0 <= result['answer_visit_mass'] <= 1
