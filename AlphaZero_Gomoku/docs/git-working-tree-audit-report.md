# Gomoku AlphaZero — Git Working Tree Audit & Cleanup Report

Date: 2026-11-21. Branch: `main`. Scope: read-only audit; no deletions, commits, or code changes.

## 1. Snapshot

- Tracked modified: 53 files (core AZ, config, players, scripts, tests, ui, docs, tasks).
- Untracked paths: 68 (source/tests/scripts/docs + run dirs + artifacts + backups).
- Total project size: ~29G (dominated by `checkpoints/` 21G, `data/` 3.7G, backup 2.4G).
- `.gitignore` (project): already ignores `checkpoints/`, `logs/`, `evidence/`, `*.sqlite*`, `pytest-of-*/`, `__pycache__/`, `.venv/`, `*.pt`, `*.npz`, `*.pyd`/`*.so`. Yet `data/`, experiment dirs, `native/`, `docs/analysis`, raw perf JSON remain unignored.

## 2. Modified tracked files audit (53)

| Area | File | Type | Evidence | Risk |
|---|---|---|---|---|
| core | `az/backend_logging.py` | refactor | ±76 lines | medium (logging contract) |
| core | `az/evaluator.py` | bugfix | `evaluate(..., logger=...)` removed | **high (public signature)** |
| core | `az/game_log.py` | refactor | −140 lines | **high (GameLogger API)** |
| core | `az/mcts.py` | feature | NativeMCTS + root stats +132 | **high (search behavior)** |
| core | `az/model.py` | refactor | +21 | medium |
| core | `az/renju_finetune.py` | feature | +75 | medium |
| core | `az/replay_buffer.py` | feature | `add(..., game_id=...)` optional | low-medium (backward compat) |
| core | `az/selfplay.py` | feature | provenance, `_new_game(..., provenance)` | **high (data lineage)** |
| core | `az/trainer.py` | feature/refactor | `train_steps` returns dict; `save(pointer_path)`; bootstrap | **high (checkpoint contract)** |
| config | `config.py` | feature | +10 | medium (schema) |
| config | `configs/normal.yaml` | feature | 12 lines | medium |
| config | `configs/normal_5000.yaml` | **deleted** | 5 lines removed | **high (budget config removed)** |
| config | `configs/renju.yaml` | feature | +3 | low |
| players | `players/az_player.py` | refactor | 6 lines | low |
| players | `players/factory.py` | refactor | 6 lines | medium (defaults) |
| scripts | `ablation.py` | refactor | 122 | medium |
| scripts | `analyze_t35.py` | refactor | 258 | medium |
| scripts | `arena.py` | refactor | 110 | medium |
| scripts | `bench_batching.py` | refactor | 37 | low |
| scripts | `bench_selfplay_round2.py` | refactor | 62 | low |
| scripts | `spike_mp.py` | refactor | 9 | low |
| scripts | `tournament.py` | refactor/feature | 443 | **high (CLI/storage)** |
| scripts | `train.py` | feature | 578 | **high (pipeline)** |
| tests | 22 `test_*.py` | test changes | many added/rewritten assertions | **high (expectations changed)** |
| ui | `ui/pygame_app.py` | feature/refactor | 39 | medium |
| docs | `README.md`, `SPEC.md`, `GLOSSARY.md`, `report.md`, `tournament_guide.md`, ADRs 0004/0005 | docs | documentation | low |
| repo | `README.md` | docs | 25 | low |
| repo | `.gitignore` | config | 7 | low |
| tasks | `tasks/*` (4) | docs | task tracking | low |

**Public/API/config/test changes flagged:**
- `evaluate` lost `logger` param.
- `GameLogger` API reduced.
- `Trainer.train_steps` return type changed (`dict[str,float]` → `dict`).
- `Trainer.save` signature changed to pointer path.
- `ReplayBuffer.add` gained optional `game_id`.
- `normal_5000.yaml` deleted (budget variant).
- Many tests renamed/rewritten → behavior/expectation changes.

**Risk summary:** highest: trainer checkpoint/save contract, tournament storage, selfplay provenance, evaluator logger removal, `normal_5000.yaml` deletion. Medium: MCTS native path, config schema. Low: docs/tasks.

## 3. Untracked files audit (68)

### KEEP / COMMIT (source, tests, docs, scripts)
- Core: `az/elo.py`, `az/experiment_exchange.py`, `az/logging_v2.py`, `az/training_recovery.py` (referenced by train.py/selfplay/tests).
- Tests: 11 new `tests/test_*.py` (elo, experiment_exchange, legacy_quarantine, logging_v2, migrate_logs_to_sqlite, native_mcts, native_patterns, tactical_benchmark, tactical_oracle, train_resume_logging, …).
- Scripts: 9 new (bench_native_patterns, bench_real_world, exchange_experiment, investigate_phase_timing, migrate_logs_to_sqlite, profile_cpp_backend, quarantine_legacy, recalculate_elo, tactical_benchmark).
- UI: `ui/pygame_frontend.py`. Native: `native/gomoku_native.cpp`, `native/gomoku_mcts_native.cpp`. Root: `requirements.txt`.
- Docs: `docs/analysis/` (22 md), `docs/exports/`, `docs/intent/`, many `docs/bao-cao-*.md`, `docs/cpp-*.md`, `docs/az-*-report.md`, `docs/logging-v2-implementation-report.md`, `tasks/progress-review-2026-11-11.md`.

### IGNORE (already gitignored generated)
- `data/` (3.7G), `checkpoints_selfplay_20k/`, `pygame_gui.log`, `native/*.pyd` (none project-level), `pytest-of-*` (none at project root).

### SAFE DELETE (experiment run dirs + backup)
- `az_100pilot_botfozch/` 337M, `az_contpilot_wzdps5fu/` 101M, `az_finalprobe_71723je1/` 35M, `az_readiness_native_wo5zl7ml/` 18M, `az_crashproof_mkxtuk9c/` 7.9M.
- `final_50k_pilot/` 51M, `defensive_*` 1.5M, `tactical_*` 8.5M, `value_calibration_*` 1.0M.
- `checkpoints_selfplay_20k/` 1.9M.
- `gomoku_logging_backup_20261119T173723541431Z/` (root) 2.4G.
- `"lab05 - game/*.zip/.docx"` (unrelated lab deliverable) small.
- Total SAFE DELETE ≈ 3.0G.

### NEED REVIEW
- `docs/exports/`, `docs/performance/raw-*.json` — raw perf captures; verify referenced by benchmark reports.
- `docs/untracked-and-junk-inventory.md` — self-audit doc ownership.
- `"lab05 - game/"` files ownership.

## 4. Data artifact audit (checkpoints / SQLite / logs / evidence)

| Artifact | Size | Referenced by | Needed for resume/train/tournament? | Class |
|---|---:|---|---|---|
| `checkpoints/` (normal pointer + generations) | 21G | `configs/normal.yaml`, README, SPEC, `az/renju_finetune.py`, `--checkpoint` defaults | **Yes** — resume needs `checkpoints/normal/latest.json` + generation `checkpoint.pt`/`replay.npz`; tournament AZ agents need checkpoint | KEEP (core, not committed) |
| `checkpoints_selfplay_20k/` | 1.9M | only historical/experiment docs | No for core train/tournament | NEED REVIEW / SAFE DELETE candidate |
| `data/*.sqlite` (normal/selfplay/tournament/eval) | ~4.9G+ | `az/logging_v2.py` defaults, scripts `--db`, README/SPEC | **Yes** — self-play/training records (`normal.sqlite`), tournament/Elo (`tournament.sqlite`); resume needs run IDs | KEEP |
| `data/az_25k_continuation_v1*` + eval DBs | ~1.1G + 16M | experiment/analysis scripts only | Historical evidence, not core resume | KEEP as evidence / NEED REVIEW |
| `evidence/` (colab_return, depth1_cutover, paused) | 149M | README migration dry-run scans `evidence/`; docs | Not needed for train/tournament runtime | KEEP (evidence, do not delete) |
| `logs/` + `logs/backend_runs.sqlite3` | 20M | SPEC: runtime artifacts; migration scans `logs/` | No for train/tournament resume (telemetry only) | KEEP/IGNORE (runtime, gitignored) |
| `pytest-of-User/` (root + project) | 30M+ | pytest temp; no tracked references | No | SAFE DELETE |
| Experiment dirs (`final_50k_pilot`, `defensive_*`, `tactical_*`, `az_*pilot*`, `az_*probe*`) | ~80M+ | only their own `analyze.py`/`build.py` | No for core train/tournament | NEED REVIEW (own analyze.py referencing checkpoints/data) |
| `gomoku-selfplay-renju-bench-e_wotby6/` | small | experiment only | No | NEED REVIEW |
| `*.pyd` (project) | none project-level | `.venv` only | No | IGNORE (venv, gitignored) |

**Key conclusions:** core runtime requires `checkpoints/normal/` pointer + generation files and `data/normal.sqlite` / `data/tournament.sqlite`. Experiment/evidence dirs are historical, not needed for resume/train/tournament, but must not be auto-deleted without run-id/hash verification. `pytest-of-User` is safe to delete. No DBs modified; sqlite inspected read-only.

**Reference verification (grep):**
- `checkpoints_selfplay_20k` referenced only by docs (az-80k-checkpoint-forensics, az-final-cleanup-report, az-normal-migration-report) and the self-audit doc — no code/config references.
- Experiment dirs (`final_50k_pilot`, `defensive_*`, `value_calibration_*`) referenced by their own `analyze.py`/`build.py`/`evaluate.py` and analysis docs — no core pipeline references.
- `evidence/` referenced by README, SPEC, benchmarks, integration-2026-11-14, logging-v2 report.

## 5. Classification summary

| Class | Examples | Size |
|---|---|---:|
| KEEP (core runtime) | `checkpoints/`, `data/*.sqlite`, `evidence/`, `logs/` | ~26G |
| COMMIT (source/tests/docs) | 4 az modules, 9 scripts, 11 tests, ui, native cpp, docs, requirements | ~ few MB |
| IGNORE (gitignored runtime) | `data/` is *not* ignored (!), `logs/`, `pytest-of-*`, `.venv/`, `*.pyd` | mixed |
| SAFE DELETE | pytest temp, experiment run dirs, backup, checkpoints_selfplay_20k, lab05 deliverables | ≈ 3.0G |
| NEED REVIEW | raw perf JSON, exports, experiment dirs, self-audit doc, lab05 files, selfplay_20k checkpoint | ≈ 1.2G+ |

## 6. Recommended commit grouping

1. **Core AZ pipeline + tests + docs** (single commit): all KEEP/COMMIT untracked source/test/script/ui/native/docs + the 53 modified tracked files. Review high-risk signature changes first (evaluator logger removal, GameLogger simplify, trainer.save contract, normal_5000.yaml deletion).
2. **Separate commit:** `configs/normal_5000.yaml` deletion with migration note (budget config removed; update SPEC/README references).
3. Do not commit runtime artifacts (`data/`, `logs/`, `checkpoints/`, experiment dirs).

## 7. .gitignore additions (proposed, not applied)

```
# Experiment run directories (dated/pilot)
az_*pilot*/
az_*probe*/
az_*native*/
*_50k_*/
tactical_experiment_*/
value_calibration_*/
defensive_*_50k*/
final_*_pilot/
# Backups
gomoku_logging_backup_*/
# Pygame
pygame_gui.log
# Docs generated exports/perf captures
docs/exports/
docs/performance/raw-*.json
# Root-level caches
pip-build-tracker-*/
pip-ephem-wheel-cache-*/
pip-install-*/
torchinductor_*/
pytest-of-*/
```

Note: `data/` should remain NOT ignored? Currently not ignored. Recommend keeping `data/` tracked as runtime (ignored via existing `*.sqlite*` only) — i.e., ignore `*.sqlite*` but keep directory; add `data/*.log` and `data/*.jsonl` if not needed. Verify before applying.

## 8. SAFE DELETE list with reclaimable bytes

| Path | Size | Reason |
|---|---:|---|
| `AlphaZero_Gomoku/pytest-of-User/` + root `pytest-of-User/` | 30M+ | pytest temp, no references |
| `AlphaZero_Gomoku/az_100pilot_botfozch/` | 337M | experiment run |
| `AlphaZero_Gomoku/az_contpilot_wzdps5fu/` | 101M | experiment run |
| `AlphaZero_Gomoku/az_finalprobe_71723je1/` | 35M | probe |
| `AlphaZero_Gomoku/az_readiness_native_wo5zl7ml/` | 18M | readiness run |
| `AlphaZero_Gomoku/az_crashproof_mkxtuk9c/` | 7.9M | crashproof run |
| `AlphaZero_Gomoku/final_50k_pilot/` | 51M | pilot |
| `AlphaZero_Gomoku/defensive_gate_ablation_50k/` | 0.4M | ablation |
| `AlphaZero_Gomoku/defensive_holdout_50k_data/` | 0.15M | holdout data |
| `AlphaZero_Gomoku/defensive_holdout_50k_eval/` | 1.0M | holdout eval |
| `AlphaZero_Gomoku/tactical_50k_analysis_20261120/` | 0.3M | analysis |
| `AlphaZero_Gomoku/tactical_experiment_20261120/` | 8.2M | experiment |
| `AlphaZero_Gomoku/value_calibration_50k_data/` | 0.67M | calibration |
| `AlphaZero_Gomoku/value_calibration_50k_mcts/` | 0.23M | calibration |
| `AlphaZero_Gomoku/value_calibration_50k_stats/` | 0.08M | calibration |
| `AlphaZero_Gomoku/checkpoints_selfplay_20k/` | 1.9M | historical checkpoint |
| `gomoku_logging_backup_20261119T173723541431Z/` (root) | 2.4G | log backup |
| `"lab05 - game/*.zip/.docx"` | small | unrelated lab deliverable |
| **Total SAFE DELETE** | **≈ 3.0G** | |

Do not delete `checkpoints/` (21G) or `data/` (3.7G) — core runtime needed for resume/train/tournament.

## 9. NEED REVIEW list (missing evidence before action)

| Item | Missing evidence |
|---|---|
| `docs/exports/`, `docs/performance/raw-*.json` | Verify referenced by benchmark reports (`docs/performance/10-optimization-round2.md`, `11-production-benchmark.md`). |
| `checkpoints_selfplay_20k/` | Verify no run-id dependency in experiment scripts (grep shows only docs refs — likely safe, but confirm run IDs). |
| `data/az_25k_continuation_v1*` + eval DBs (~1.1G) | Confirm run IDs not needed for resume; check `az/logging_v2.py` run-id lookups. |
| Experiment dirs (`az_*`, `defensive_*`, `tactical_*`, `value_calibration_*`, `final_50k_pilot`) | Each has own `analyze.py`/`build.py`/`evaluate.py` referencing checkpoints/data; verify artifacts can be regenerated. |
| `docs/untracked-and-junk-inventory.md` | Ownership — self-audit doc; keep or fold into this report. |
| `"lab05 - game/"` files | Ownership — unrelated lab deliverable; confirm student wants to keep. |

## 10. Migration notes to single AZ pipeline

- **Single checkpoint dir:** `checkpoints/normal/` with `latest.json` pointer + generation dirs (`checkpoint.pt`, `replay.npz`). Retire `checkpoints_selfplay_20k/`.
- **Single log dir:** `logs/` (telemetry only) + SQLite `data/normal.sqlite` (self-play/training), `data/tournament.sqlite` (Elo/tournament). Retire `gomoku_logging_backup_*` and scattered experiment logs.
- **Single experiment runner:** unify `scripts/train.py` (`--config`, `--checkpoint`, `--db`) as sole entry; migrate `exchange_experiment.py`, `tactical_benchmark.py` to read from `data/*.sqlite` via `az/logging_v2.py`. Remove per-experiment `analyze.py` duplication.
- **Config:** single `configs/normal.yaml`; delete `normal_5000.yaml` (budget variant) and document removal in SPEC/README. Reconcile `config.py` schema changes.
- **Pointer contract:** `Trainer.save(pointer_path=...)` and `latest.json` must be the single resume mechanism; update tournament/arena consumers.
- **Tests:** update affected public signatures (evaluator.evaluate logger param, GameLogger API, train_steps return dict) before commit.

## 11. Notes

- No deletions performed. All commands read-only.
- `native/*.cpp` source should be committed; only `*.pyd` binaries are junk (none at project level currently).
- `.venv/` (~4.1G) already ignored and not shown in untracked.
- `graphify-out/` (~31M) is ignored per root `.gitignore`.
- Total reclaimable without risk: ≈ 3.0G (SAFE DELETE). Additional ~1.2G+ contingent on NEED REVIEW verification. Core runtime (~26G: checkpoints + data + evidence + logs) must be retained.
