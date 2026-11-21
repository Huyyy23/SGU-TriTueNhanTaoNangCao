# Inventory: Untracked / Modified files & junk artifacts

Generated 2026-11-21. Working tree has 53 tracked-modified + 68 untracked entries.

## 1. Modified tracked files (53)

### Core AZ (`AlphaZero_Gomoku/az/`)
- `backend_logging.py`, `evaluator.py`, `game_log.py`, `mcts.py`, `model.py`
- `renju_finetune.py`, `replay_buffer.py`, `selfplay.py`, `trainer.py`

### Config
- `config.py`, `configs/normal.yaml`, `configs/renju.yaml`
- `configs/normal_5000.yaml` — deleted

### Players
- `players/az_player.py`, `players/factory.py`

### Scripts
- `scripts/ablation.py`, `analyze_t35.py`, `arena.py`, `bench_batching.py`
- `scripts/bench_selfplay_round2.py`, `spike_mp.py`, `tournament.py`, `train.py`

### Tests
- `test_ablation.py`, `test_agent_factory.py`, `test_arena.py`, `test_az_player.py`
- `test_az_player_factory.py`, `test_backend_auto.py`, `test_config.py`, `test_logging.py`
- `test_mcts.py`, `test_patterns.py`, `test_replay_buffer.py`, `test_rules_normal.py`
- `test_selfplay.py`, `test_tournament.py`, `test_trainer.py`, `test_ui.py`

### UI / Docs / Tasks
- `ui/pygame_app.py`
- `docs/adr/0004-structured-logging-and-benchmarking.md`, `docs/adr/0005-budget-5000-games-for-az-v0b.md`
- `docs/report.md`, `docs/tournament_guide.md`
- `tasks/Lo_trinh_cai_tien_AlphaZero_Gomoku.md`, `tasks/Phan_viec_cua_B_khong_dung_A_va_C.md`
- `tasks/plan.md`, `tasks/todo.md`
- `GLOSSARY.md`, `README.md`, `SPEC.md`
- root `README.md`

## 2. Untracked new files (keep candidates)

### Core
- `az/elo.py`, `az/experiment_exchange.py`, `az/logging_v2.py`, `az/training_recovery.py`

### Native C++ source
- `native/gomoku_native.cpp`, `native/gomoku_mcts_native.cpp`

### Scripts
- `scripts/bench_native_patterns.py`, `scripts/bench_real_world.py`
- `scripts/exchange_experiment.py`, `scripts/investigate_phase_timing.py`
- `scripts/migrate_logs_to_sqlite.py`, `scripts/profile_cpp_backend.py`
- `scripts/quarantine_legacy.py`, `scripts/recalculate_elo.py`
- `scripts/tactical_benchmark.py`

### Tests
- `tests/test_elo.py`, `test_experiment_exchange.py`, `test_legacy_quarantine.py`
- `tests/test_logging_v2.py`, `test_migrate_logs_to_sqlite.py`, `test_native_mcts.py`
- `tests/test_native_patterns.py`, `test_tactical_benchmark.py`, `test_tactical_oracle.py`
- `tests/test_train_resume_logging.py`

### UI / Docs / Tasks
- `ui/pygame_frontend.py`
- `docs/analysis/` (22 md), `docs/exports/`, `docs/intent/`
- `docs/bao-cao-*.md`, `docs/cpp-*.md`, `docs/az-*-report.md`, `docs/logging-v2-implementation-report.md`
- `tasks/progress-review-2026-11-11.md`
- root `requirements.txt`
- `"lab05 - game/..."` (zip/docx deliverable)

## 3. Untracked junk / runtime artifacts (candidates to delete)

| Path | Size | Note |
|---|---:|---|
| `AlphaZero_Gomoku/checkpoints/` | 21G | checkpoint/model/npz |
| `AlphaZero_Gomoku/data/` | 3.7G | sqlite/log experiments |
| `gomoku_logging_backup_20261119T173723541431Z/` | 2.4G | log backup |
| `AlphaZero_Gomoku/az_100pilot_botfozch/` | 337M | experiment run |
| `AlphaZero_Gomoku/az_contpilot_wzdps5fu/` | 101M | experiment run |
| `AlphaZero_Gomoku/evidence/` | 149M | evidence runs |
| `AlphaZero_Gomoku/final_50k_pilot/` | 51M | pilot |
| `AlphaZero_Gomoku/az_finalprobe_71723je1/` | 35M | probe |
| `AlphaZero_Gomoku/az_readiness_native_wo5zl7ml/` | 18M | readiness run |
| `AlphaZero_Gomoku/az_crashproof_mkxtuk9c/` | 7.9M | crashproof run |
| `AlphaZero_Gomoku/tactical_experiment_20261120/` | 8.2M | experiment |
| `AlphaZero_Gomoku/logs/` | 20M | training logs |
| `AlphaZero_Gomoku/pytest-of-User/` | 30M | pytest temp |
| `AlphaZero_Gomoku/checkpoints_selfplay_20k/` | 1.9M | selfplay checkpoint |
| `AlphaZero_Gomoku/defensive_*`, `value_calibration_*`, `tactical_50k_analysis_*` | ~2MB | analysis artifacts |
| `AlphaZero_Gomoku/native/*.pyd` (4 files) | ~3.5MB | compiled binary, rebuildable |
| `AlphaZero_Gomoku/pygame_gui.log` | 115B | runtime log |
| `AlphaZero_Gomoku/__pycache__/`, `.pytest_cache/` | small | cache |
| `AlphaZero_Gomoku/docs/performance/raw-round2-*.json` (untracked) | ~120KB | raw benchmark output |

Potential reclaim: ~27GB (mostly checkpoints/data/backup).

## 4. Inspection commands

```bash
find AlphaZero_Gomoku -type f -size +50M -printf '%s %p\n' | sort -nr | head -30
git status --ignored --short | head -50
```

## 5. Cleanup commands (run only after backup if needed)

```bash
# Cache/test artifacts
find . -type d -name __pycache__ -prune -exec rm -rf {} +
rm -rf AlphaZero_Gomoku/.pytest_cache AlphaZero_Gomoku/pytest-of-User
rm -f AlphaZero_Gomoku/pygame_gui.log AlphaZero_Gomoku/*.pyd

# Runtime/experiment artifacts (DANGEROUS: loses checkpoints/data)
rm -rf AlphaZero_Gomoku/checkpoints AlphaZero_Gomoku/data AlphaZero_Gomoku/logs AlphaZero_Gomoku/evidence
rm -rf AlphaZero_Gomoku/az_100pilot_botfozch AlphaZero_Gomoku/az_contpilot_wzdps5fu \
  AlphaZero_Gomoku/az_crashproof_mkxtuk9c AlphaZero_Gomoku/az_finalprobe_71723je1 \
  AlphaZero_Gomoku/az_readiness_native_wo5zl7ml AlphaZero_Gomoku/final_50k_pilot \
  AlphaZero_Gomoku/checkpoints_selfplay_20k AlphaZero_Gomoku/tactical_experiment_20261120 \
  AlphaZero_Gomoku/defensive_gate_ablation_50k AlphaZero_Gomoku/defensive_holdout_50k_data \
  AlphaZero_Gomoku/defensive_holdout_50k_eval AlphaZero_Gomoku/value_calibration_50k_data \
  AlphaZero_Gomoku/value_calibration_50k_mcts AlphaZero_Gomoku/value_calibration_50k_stats \
  AlphaZero_Gomoku/tactical_50k_analysis_20261120
rm -rf gomoku_logging_backup_20261119T173723541431Z
```

## 6. Suggested .gitignore additions

```gitignore
AlphaZero_Gomoku/az_*pilot*/
AlphaZero_Gomoku/az_*probe*/
AlphaZero_Gomoku/az_*native_*/
AlphaZero_Gomoku/defensive_*_50k*/
AlphaZero_Gomoku/tactical_*_50k*/
AlphaZero_Gomoku/value_calibration_50k_*/
AlphaZero_Gomoku/final_50k_pilot/
AlphaZero_Gomoku/checkpoints_selfplay_20k/
AlphaZero_Gomoku/data/
AlphaZero_Gomoku/native/*.pyd
AlphaZero_Gomoku/pygame_gui.log
gomoku_logging_backup_*/
```

## 7. Notes
- No deletions performed. Review/commit valuable code & docs before removing artifacts.
- `native/*.cpp` source should be committed; only `*.pyd` binaries are junk.
- `.venv/` (~4.1G) already ignored and not shown in untracked.
