"""Fixed Normal-rule tactical probes; not a verified training holdout.

Train, development and holdout use different construction families and exclude
D4-equivalent states. Historical 25k training exposure remains unknown. Variants
within a family are correlated; counts are not independent game samples.
The independent five-cell-window oracle certifies only a shallow tactical horizon.
Use --check-integrity for a model-free split and duplicate audit.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from az.mcts import MCTS, NativeMCTS
from az.model import load_checkpoint
from gomoku.board import Board, SIZE
from gomoku.rules.normal import NormalRules


@dataclass(frozen=True)
class Fixture:
    name: str
    category: str
    player: int
    black: tuple[int, ...]
    white: tuple[int, ...]
    expected: int
    split: str
    group: str

# Families are assigned before any D4 transform; no transformed variant crosses splits.
# Black's extra stone in White-to-move positions makes the histories alternating.
_CASES = (
    ('train', 'inside_four', 'win', (58, 59, 60, 62), (0, 14, 98, 27), 61),
    ('train', 'block_inside_four', 'block', (57, 0, 14, 98), (58, 59, 60, 62), 61),
    ('train', 'cross_three', 'fork', (58, 59, 61, 38, 49, 71), (0, 14, 27, 98, 10, 100), 60),
    ('train', 'defend_cross_three', 'forced_defense', (0, 14, 27, 98, 10, 100), (58, 59, 61, 38, 49, 71), 60),
    ('dev', 'open_four', 'win', (58, 59, 60, 61), (0, 13, 26, 99), 62),
    ('dev', 'blocked_four', 'block', (57, 0, 13, 99), (58, 59, 60, 61), 62),
    ('dev', 'gap_three', 'fork', (58, 59, 61), (0, 13, 99), 60),
    ('dev', 'defend_gap_three', 'forced_defense', (0, 13, 99), (58, 59, 61), 60),
    ('holdout', 'broken_four', 'win', (58, 59, 61, 62), (0, 14, 27, 98), 60),
    ('holdout', 'block_broken_four', 'block', (57, 0, 14, 98), (58, 59, 61, 62), 60),
    ('holdout', 'straight_three', 'fork', (58, 59, 60), (0, 14, 98), 61),
    ('holdout', 'defend_straight_three', 'forced_defense', (0, 14, 98), (58, 59, 60), 61),
)
FIXTURES = tuple(
    Fixture(f'{split}_{group}_{"black" if player == 1 else "white"}'
            + (f'_variant{variant}' if variant else ''),
            category, player,
            (own if player == 1 else opponent + (110,)) + extra_black,
            (opponent if player == 1 else own) + extra_white,
            expected, split, group)
    for split, group, category, own, opponent, expected in _CASES
    for player in (1, 2)
    for variant, extra_black, extra_white in ((0, (), ()), (1, (3,), (107,)),
                                               (2, (5,), (105,)))
)

WINDOWS = tuple(
    tuple((r + k * dr) * SIZE + c + k * dc for k in range(5))
    for r in range(SIZE) for c in range(SIZE)
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1))
    if 0 <= r + 4 * dr < SIZE and 0 <= c + 4 * dc < SIZE
)


def board_for(fixture: Fixture) -> Board:
    board = Board()
    for index in range(len(fixture.black)):
        board.play(*divmod(fixture.black[index], SIZE))
        if index < len(fixture.white):
            board.play(*divmod(fixture.white[index], SIZE))
    return board


def split_integrity(fixtures=FIXTURES) -> dict:
    """Report family overlap and color-aware D4 duplicates, including within splits."""
    names = defaultdict(list)
    families = defaultdict(set)
    states = defaultdict(list)
    counts = defaultdict(int)
    for fixture in fixtures:
        names[fixture.name].append(fixture.name)
        families[fixture.group].add(fixture.split)
        counts[fixture.split] += 1
        grid = board_for(fixture).grid.reshape(SIZE, SIZE)
        rotations = [np.rot90(grid, k) for k in range(4)]
        key = (fixture.player, min(v.tobytes() for v in rotations
                                   + [np.fliplr(v) for v in rotations]))
        states[key].append(fixture.name)
    duplicate_names = sorted(name for name, entries in names.items() if len(entries) > 1)
    family_leakage = {group: sorted(splits) for group, splits in sorted(families.items())
                      if len(splits) > 1}
    duplicate_states = sorted(sorted(entries) for entries in states.values() if len(entries) > 1)
    return {'ok': not (duplicate_names or family_leakage or duplicate_states),
            'counts': dict(sorted(counts.items())), 'families': len(families),
            'duplicate_names': duplicate_names, 'family_leakage': family_leakage,
            'duplicate_states': duplicate_states,
            'historical_25k_exposure': 'unknown'}


def winning_moves(grid, player: int) -> set[int]:
    """All one-ply wins from independent five-cell windows (Normal >=5)."""
    result = set()
    for window in WINDOWS:
        empty = -1
        count = 0
        for action in window:
            value = grid[action]
            if value == player:
                count += 1
            elif value == 0:
                empty = action
            else:
                break
        else:
            if count == 4 and empty >= 0:
                result.add(empty)
    return result


def fork_moves(grid, player: int) -> set[int]:
    """All moves creating two distinct next-ply wins, absent an immediate reply win."""
    candidates = set()
    for window in WINDOWS:
        values = [grid[a] for a in window]
        if values.count(player) >= 3 and 3 - player not in values:
            candidates.update(a for a in window if grid[a] == 0)
    result = set()
    for action in candidates:
        grid[action] = player
        if not winning_moves(grid, 3 - player) and len(winning_moves(grid, player)) >= 2:
            result.add(action)
        grid[action] = 0
    return result


def oracle(grid, player: int, category: str) -> set[int]:
    """Exhaustive Normal-rule tactical answers, not a full-game solver.

    A fork wins in at most three plies if the opponent cannot win immediately.
    Forced defense means preventing opponent wins in one or forks in two plies;
    it does not establish a draw or win beyond that horizon.
    """
    grid = np.array(grid, copy=True)
    opponent = 3 - player
    wins = winning_moves(grid, player)
    if category == 'win':
        return wins
    if category == 'block':
        threats = winning_moves(grid, opponent)
        return wins or (threats if len(threats) == 1 else set())
    if category == 'fork':
        return fork_moves(grid, player) - wins
    if category != 'forced_defense':
        raise ValueError(f'Unknown category: {category}')
    result = set(wins)
    for action in np.flatnonzero(grid == 0):
        action = int(action)
        if action in wins:
            continue
        grid[action] = player
        if not winning_moves(grid, opponent) and not fork_moves(grid, opponent):
            result.add(action)
        grid[action] = 0
    return result


def run_case(fixture: Fixture, model, backend: str, seed: int) -> dict:
    board = board_for(fixture)
    answers = sorted(oracle(board.grid, fixture.player, fixture.category))
    if not answers or fixture.expected not in answers:
        raise ValueError(f'Invalid fixture or oracle: {fixture.name}')
    search_class = {'python': MCTS, 'cpp': NativeMCTS}[backend]
    search = search_class(NormalRules(), n_simulations=200, root_noise=False,
                          rng=np.random.default_rng(seed))
    search.reset(board)
    root_nn_value = None
    while (request := search.request()) is not None:
        policy, value = model.predict(request.planes[None])
        if root_nn_value is None:
            root_nn_value = float(value[0])
        search.receive(policy[0], float(value[0]))
    visits, priors, q_values, legal = search.root_statistics()
    selected = search.select_move(0)
    total = int(visits.sum())
    return {
        'name': fixture.name, 'group': fixture.group, 'split': fixture.split,
        'category': fixture.category,
        'color': 'black' if fixture.player == 1 else 'white',
        'black': list(fixture.black), 'white': list(fixture.white),
        'answers': answers, 'selected': selected, 'correct': selected in answers,
        'simulations': search.simulations_done, 'nn_calls': search.nn_calls,
        'root_nn_value': root_nn_value, 'root_value': float(search.root_value()),
        'legal': np.flatnonzero(legal).tolist(), 'visits': visits.tolist(),
        'priors': priors.tolist(), 'q_values': q_values.tolist(),
        'answer_prior_mass': float(priors[answers].sum()),
        'answer_visit_mass': float(visits[answers].sum() / total) if total else 0.0,
    }


def benchmark(checkpoint: str, backend: str, seed: int, split: str = 'holdout') -> dict:
    torch.set_num_threads(1)
    digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    model, metadata = load_checkpoint(checkpoint, device='cpu')
    if metadata.get('rule_set', 'normal') != 'normal':
        raise ValueError('Tactical benchmark requires a Normal-rule checkpoint')
    if split not in ('train', 'dev', 'holdout', 'all'):
        raise ValueError(f'Unknown split: {split}')
    integrity = split_integrity()
    if not integrity['ok']:
        raise ValueError(f'Tactical split integrity failed: {integrity}')
    cases = [run_case(fixture, model, backend, seed) for fixture in FIXTURES
             if split == 'all' or fixture.split == split]
    groups = defaultdict(list)
    for case in cases:
        groups['overall'].append(case['correct'])
        groups['category:' + case['category']].append(case['correct'])
        groups['color:' + case['color']].append(case['correct'])
        groups['category_color:' + case['category'] + ':' + case['color']].append(case['correct'])
        groups['split:' + case['split']].append(case['correct'])
        groups['group:' + case['group']].append(case['correct'])
    return {
        'checkpoint': str(checkpoint), 'checkpoint_sha256': digest,
        'backend': backend, 'seed': seed, 'split': split, 'rules': 'normal', 'simulations': 200,
        'root_noise': False, 'temperature': 0, 'device': 'cpu',
        'fixture_provenance': 'Fixed constructions; historical 25k training exposure is unknown. Variants within construction families are correlated.',
        'split_integrity': integrity,
        'oracle_scope': 'Normal rules: immediate win/block and fork; forced defense prevents immediate wins and forks within two opponent plies, not a full-game result.',
        'accuracy': {name: {'correct': sum(values), 'count': len(values),
                            'accuracy': sum(values) / len(values)}
                     for name, values in sorted(groups.items())},
        'cases': cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint')
    parser.add_argument('--backend', choices=('python', 'cpp'), default='python')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split', choices=('train', 'dev', 'holdout', 'all'), default='holdout')
    parser.add_argument('--check-integrity', action='store_true',
                        help='Audit all fixture splits without loading a checkpoint')
    args = parser.parse_args()
    if args.check_integrity:
        report = split_integrity()
        print(json.dumps(report, indent=2))
        if not report['ok']:
            raise SystemExit(1)
    else:
        if not args.checkpoint:
            parser.error('--checkpoint is required unless --check-integrity is used')
        print(json.dumps(benchmark(args.checkpoint, args.backend, args.seed, args.split), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
