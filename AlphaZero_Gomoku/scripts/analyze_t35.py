"""Validate and summarize the completed, selected T35 evidence without replay."""
import hashlib
import json
from itertools import combinations
from collections import Counter, defaultdict
from pathlib import Path
import statistics
import sys

import numpy as np


def analyze(root):
    completion = json.loads((root / 'colab_completed.json').read_text())
    for name, digest in completion['files'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
    selection = json.loads((root / 'evidence/depth1_selection.json').read_text())
    groups = defaultdict(list)
    paths = [(root / x['path'], 'normal_fixed' if 'normal' in x['path'] else 'renju_fixed') for x in selection['pairs']]
    paths += [(p.parent, 'normal_time') for p in (root / 'logs/tournament/t35_normal_time1_colab_parallel3').glob('*/manifest.json')]
    assert len(paths) == 24
    for path, group in paths:
        manifest = json.loads((path / 'manifest.json').read_text())
        cfg = manifest['config']
        a, b = cfg['pair'].split(',')
        records = [json.loads(line) for line in (path / 'games.jsonl').read_text().splitlines() if line.strip()]
        pair_index = list(combinations(cfg['agents'], 2)).index((a, b))
        assert len(records) == 100 and [r['game_id'] for r in records] == list(range(100)), path
        assert cfg['seed'] == 20261113
        scores = Counter()
        colors = Counter()
        moves = defaultdict(list)
        for i, r in enumerate(records):
            black, white = (a, b) if i % 2 == 0 else (b, a)
            assert (r['first_player'], r['second_player']) == (black, white)
            assert r['winner'] in {'black', 'white', 'draw'}
            assert r['rule_set'] == cfg['rules'] and len(r['moves']) == r['moves_count']
            settings = r['config']
            assert settings['pair'] == [a, b] and settings['budget'] == cfg['budget']
            assert settings['seed_a'] == cfg['seed'] + pair_index * 100000 + i * 2
            assert settings['seed_b'] == settings['seed_a'] + 1
            if 'alphabeta' in (a, b) and group.endswith('fixed'):
                assert settings['depth'] == cfg['depth'] == 1
            winner = black if r['winner'] == 'black' else white if r['winner'] == 'white' else None
            scores['W' if winner == a else 'D' if winner is None else 'L'] += 1
            colors['black_wins' if r['winner'] == 'black' else 'white_wins' if r['winner'] == 'white' else 'draws'] += 1
            if winner == a:
                colors['a_black_wins' if black == a else 'a_white_wins'] += 1
            for move in r['moves']:
                moves[black if move['player'] == 1 else white].append(move)
        w, d, loss = (scores[k] for k in ('W', 'D', 'L'))
        p = (w + d / 2) / 100
        denominator = 1 + 1.96 ** 2 / 100
        center = (p + 1.96 ** 2 / 200) / denominator
        half = 1.96 * ((p * (1 - p) / 100 + 1.96 ** 2 / 40000) ** .5) / denominator
        telemetry = {}
        for agent, ms in moves.items():
            item = {'moves': len(ms), 'mean_ms': statistics.mean(m['latency_sec'] for m in ms) * 1000,
                    'p95_ms': float(np.percentile([m['latency_sec'] for m in ms], 95)) * 1000,
                    'over_1s': sum(m['latency_sec'] > 1 for m in ms),
                    'override_count': sum(m.get('override_applied', False) for m in ms)}
            for field in ('nodes_visited', 'simulations_run', 'nn_calls'):
                measured = [m[field] for m in ms if m.get(field) is not None]
                item[field + '_mean'] = statistics.mean(measured) if measured else None
            telemetry[agent] = item
        groups[group].append({'a': a, 'b': b, 'W': w, 'D': d, 'L': loss, 'score': p,
                              'wilson95': [center - half, center + half], 'colors': dict(colors),
                              'mean_game_sec': statistics.mean(r['duration_sec'] for r in records),
                              'mean_moves': statistics.mean(r['moves_count'] for r in records),
                              'telemetry': telemetry, 'path': str(path),
                              'games_sha256': hashlib.sha256((path / 'games.jsonl').read_bytes()).hexdigest()})
    assert sum(len(v) for v in groups.values()) * 100 == 2400
    rankings = {}
    for group, pairs in groups.items():
        stats = defaultdict(Counter)
        for pair in pairs:
            for a, w, d, loss in ((pair['a'], pair['W'], pair['D'], pair['L']), (pair['b'], pair['L'], pair['D'], pair['W'])):
                stats[a].update(W=w, D=d, L=loss)
        rankings[group] = [{'agent': a, **v, 'score': (v['W'] + v['D'] / 2) / sum(v.values())}
                           for a, v in sorted(stats.items(), key=lambda kv: -(kv[1]['W'] + kv[1]['D'] / 2) / sum(kv[1].values()))]
    return {'games': 2400, 'groups': dict(groups), 'rankings': rankings}


if __name__ == '__main__':
    result = analyze(Path(sys.argv[1]))
    output = Path(sys.argv[2])
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
