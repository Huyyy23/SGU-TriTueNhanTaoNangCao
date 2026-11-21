# Gomoku AlphaZero — Cleanup Execution Report

Date: 2026-11-21. Branch: `main`.

## 1. Regression tests
- `test_mcts`, `test_selfplay`, `test_trainer`, `test_replay_buffer`, `test_logging`, `test_logging_v2`, `test_tournament`, `test_arena`: **131 passed** (32.2s). No failures.

## 2. `.gitignore` changes
- Root: added `lab05 - game/`, `gomoku_logging_backup_*/`.
- Project: added experiment run dir patterns, smoke dirs, `pygame_gui.log`, `native/*.pyd`/`*.so` (kept `.cpp` source), `docs/exports/`, `docs/performance/raw-*.json`.
- Verified `native/gomoku_native.cpp` and `native/gomoku_mcts_native.cpp` are NOT ignored (exit 1 from check-ignore, correct).

## 3. Commits (11 new commits on main)
| SHA | Subject |
|---|---|
| de4de35 | feat: native MCTS backend and batching benchmark |
| b81c9b2 | chore: gitignore patterns and cleanup audit docs |
| 2badc22 | feat: self-play and trainer loop with replay buffer provenance |
| 9c52667 | feat: logging v2 and experiment exchange with migration |
| 2566cfa | feat: Elo ranking and tournament/arena infrastructure |
| 70b7b8d | feat: Renju fine-tune, model and rule changes |
| 3c881fa | feat: player/agent factory and pygame UI |
| de94261 | chore: config, docs, tasks and remaining script updates |
| 6b04cc7 | chore: remove normal_5000 budget config variant |
| e80581e | feat: training recovery, native pattern benchmarks and remaining tests |
| 7aa00f1 | docs: analysis reports and audit documentation |

## 4. Safe deletes performed (≈ 3.0G freed)
- `gomoku_logging_backup_20261119T173723541431Z/` 2.4G (log backup; no code refs)
- Experiment run dirs: `az_100pilot_botfozch` 337M, `az_contpilot_wzdps5fu` 101M, `az_finalprobe_71723je1` 35M, `az_readiness_native_wo5zl7ml` 18M, `az_crashproof_mkxtuk9c` 7.9M, `tactical_experiment_20261120` 8.2M, `defensive_holdout_50k_eval` 1.0M, `value_calibration_50k_*` 1.0M, `defensive_holdout_50k_data` 152K, `defensive_gate_ablation_50k` 395K, `tactical_50k_analysis_20261120` 336K.
- Smoke artifacts: `checkpoints/_check` 6.9M, `_smoke` 7.0M, `_smoke2` 8.1M.
- pytest temp: `AlphaZero_Gomoku/pytest-of-User` 28M, root `pytest-of-User` 1.5M.

## 5. Retention policy proposed (NOT executed — pending approval)
Active checkpoint chain (latest.json → 2156cbadc… → 6054ad97… → 7183e8b8… → 6874e562… → a1acb3ac… → c556edfe…). Proposed keep: active + 3 most recent rollback generations (≈ 4 × 54M = 216M) + recovery snapshots referenced by `training_recovery.py`/`trainer.py`. Remaining ≈ 200 generations (~10.8G) + `recovery/` (9.5G) candidate for pruning. Requires resume test before deletion.

## 6. Still pending (NEED REVIEW — not deleted)
- `final_50k_pilot/` 51M (has own tests)
- `checkpoints_selfplay_20k/` 1.9M
- `docs/exports/` 2.3M, `docs/performance/raw-*.json` 329K (referenced by benchmark reports)
- `data/az_25k_continuation_v1*` + eval DBs ~1.1G
- Top-level legacy pointers `latest.pt` 5.7M, `buffer_latest.npz` 37M
- Renju artifacts `renju_last.pt` 5.7M, `renju_buffer_last.npz` 8.7M
- `data/` 3.7G (runtime, KEEP)
- `lab05 - game/` (preserved per instruction)

## 7. Remaining untracked
Only `AlphaZero_Gomoku/data/` (3.7G runtime, gitignored via `*.sqlite*`). All source/test/script/doc committed.
