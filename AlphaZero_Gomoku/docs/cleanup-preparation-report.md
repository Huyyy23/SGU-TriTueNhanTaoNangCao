# Gomoku AlphaZero — Cleanup Preparation Report (Final)

Date: 2026-11-21. Branch: `main`. Read-only audit; no deletions, commits, resets, or code changes.

## 1. Decision summary

| Bucket | Size | Action |
|---|---:|---|
| KEEP (core runtime) | ~26G | Retain: `checkpoints/` active, `data/*.sqlite`, `evidence/`, `logs/` |
| COMMIT (source/tests/docs) | ~few MB | Commit in 7 feature groups (§5) |
| IGNORE (gitignored runtime) | mixed | Already ignored; add patterns (§4) |
| SAFE DELETE (verified) | ≈ 3.0G | Delete after commit (§3) |
| NEED REVIEW | ≈ 1.2G+ (checkpoint history ≈ 20.6G extra) | Require explicit approval (§6) |

## 2. Checkpoint audit (`checkpoints/`, 21G)

- Active resume target: `checkpoints/normal/latest.json` → generation `2156cbadc8c546ebac27ec0925f42a82` (~55M: `checkpoint.pt` + `manifest.json` + `replay.npz`).
- `normal/generations/`: 204 dirs × ~54M ≈ 11.0G. Only the active one is needed for resume; the other 203 are history/rollback.
- `normal/recovery/`: 201 dirs, 9.5G. Referenced by `az/trainer.py`, `az/training_recovery.py`, `scripts/train.py`, `tests/test_train_resume_logging.py` — **do not delete without training-recovery policy review**.
- Smoke/test dirs: `_check/` 6.9M, `_smoke/` 7.0M, `_smoke2/` 8.1M. No project code references (only `.venv` matches). SAFE DELETE candidate.
- Top-level legacy pointers: `latest.pt` 5.7M, `buffer_latest.npz` 37M. NEED REVIEW (confirm no config/script references).
- Renju variant: `renju_last.pt` 5.7M, `renju_buffer_last.npz` 8.7M. KEEP if renju training active; else NEED REVIEW.

**Classification:**
- KEEP: `normal/latest.json` + active generation dir + config-referenced paths.
- SAFE DELETE (likely): `_check/`, `_smoke/`, `_smoke2/` (~22M) after test-dependency check.
- NEED REVIEW: 203 non-active generations (~11G), `recovery/` (9.5G), top-level legacy pointers (~43M), renju artifacts (~14M). Retention policy decision required.

## 3. SAFE DELETE list (verified, ≈ 3.0G)

Evidence: all candidates are untracked (`git ls-files` returns none); core pipeline (`az/`, `scripts/`, `players/`, `configs/`) has zero references; only self-contained experiment `analyze.py`/`build.py` or audit docs reference them.

| Path | Size | Verdict | Justification |
|---|---:|---|---|
| `gomoku_logging_backup_20261119T173723541431Z/` (root) | 2.4G | SAFE DELETE | Log backup; no code/config/runtime refs |
| `AlphaZero_Gomoku/az_100pilot_botfozch/` | 337M | SAFE DELETE | Self-contained experiment |
| `AlphaZero_Gomoku/az_contpilot_wzdps5fu/` | 101M | SAFE DELETE | Cross-ref only by `az_finalprobe/experiment.py`; delete together |
| `AlphaZero_Gomoku/az_finalprobe_71723je1/` | 35M | SAFE DELETE | Self-contained probe |
| `AlphaZero_Gomoku/az_readiness_native_wo5zl7ml/` | 18M | SAFE DELETE | Self-contained readiness run |
| `AlphaZero_Gomoku/az_crashproof_mkxtuk9c/` | 7.9M | SAFE DELETE | Self-contained |
| `AlphaZero_Gomoku/tactical_experiment_20261120/` | 8.2M | SAFE DELETE | Self-contained |
| `AlphaZero_Gomoku/defensive_holdout_50k_eval/` | 1.0M | SAFE DELETE | Self-contained |
| `AlphaZero_Gomoku/value_calibration_50k_data/` | 669K | SAFE DELETE | Self-contained |
| `AlphaZero_Gomoku/value_calibration_50k_mcts/` | 225K | SAFE DELETE | Self-contained |
| `AlphaZero_Gomoku/value_calibration_50k_stats/` | 83K | SAFE DELETE | Self-contained |
| `AlphaZero_Gomoku/defensive_holdout_50k_data/` | 152K | SAFE DELETE | Self-contained |
| `AlphaZero_Gomoku/defensive_gate_ablation_50k/` | 395K | SAFE DELETE | Only referenced by `final_50k_pilot/data/build_holdout.py` |
| `AlphaZero_Gomoku/tactical_50k_analysis_20261120/` | 336K | SAFE DELETE | Self-contained |
| `AlphaZero_Gomoku/checkpoints_selfplay_20k/` | 1.9M | NEED REVIEW | No code refs; only historical docs; confirm run IDs |
| `AlphaZero_Gomoku/pytest-of-User/` | 30M | SAFE DELETE | pytest temp, no references, gitignored |
| root `pytest-of-User/` | 1.5M | SAFE DELETE | pytest temp |
| `AlphaZero_Gomoku/_check/`, `_smoke/`, `_smoke2/` (under checkpoints/) | 22M | SAFE DELETE (likely) | Smoke artifacts; verify no test depends |

**Total SAFE DELETE ≈ 3.0G** (backup 2.4G dominates).

## 4. `.gitignore` additions (proposed, not applied)

Currently ignored: `checkpoints/`, `logs/`, `evidence/`, `*.sqlite*`, `pytest-of-*/`, `__pycache__/`, `.venv/`, `*.pt`, `*.npz`, `*.pyd`/`*.so`. Gap: experiment dirs, raw perf JSON, exports, backup, lab05, native dir.

Add to **root** `.gitignore`:
```
lab05 - game/
gomoku_logging_backup_*/
```

Add to **`AlphaZero_Gomoku/.gitignore`**:
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
checkpoints_selfplay_20k/
# Backups
gomoku_logging_backup_*/
# Pygame
pygame_gui.log
# Native build artifacts (dir-level)
native/
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

Risk check: `git check-ignore` confirms `.sqlite`, `.pyd`, `checkpoints/` already ignored. `lab05 - game/` and `raw-*.json` are NOT ignored → accidental commit risk. No tracked binary/DB artifacts exist (`git ls-files` returned none).

## 5. Commit grouping (7 feature groups)

| # | Group | Core files | Risk | Tests |
|---|---|---|---|---|
| 1 | MCTS native backend + batching | `az/mcts.py`, `scripts/bench_batching.py`, `tests/test_mcts.py`, `tests/test_native_mcts.py` (new), `native/*` | HIGH (search correctness) | test_mcts, test_native_mcts |
| 2 | Self-play / trainer loop | `az/selfplay.py`, `az/trainer.py`, `az/replay_buffer.py`, tests | HIGH (training correctness) | test_selfplay (+237), test_trainer (+316), test_replay_buffer (+75) |
| 3 | Logging v2 + experiment exchange | `az/logging_v2.py` (new), `az/experiment_exchange.py` (new), `az/backend_logging.py`, `az/game_log.py`, `scripts/migrate_logs_to_sqlite.py`, `scripts/exchange_experiment.py`, tests | Medium (data ingestion) | test_logging (rewritten), test_logging_v2 (new) |
| 4 | Elo + tournament/arena | `az/elo.py` (new), `scripts/recalculate_elo.py`, `scripts/tournament.py`, `scripts/arena.py`, `players/factory.py`, tests | Medium | test_elo (new), test_tournament (+342), test_arena (+66) |
| 5 | Renju fine-tune + model | `az/renju_finetune.py`, `az/model.py`, `config*.yaml`, tests | Medium-high (game rules) | test_rules_normal (+12), test_patterns (+11) |
| 6 | Player/agent factory + UI | `players/az_player.py`, `ui/pygame_app.py`, `ui/pygame_frontend.py` (new), tests | Low-medium | test_ui (+221 new) |
| 7 | Config/docs/tasks/housekeeping | `config.py`, configs, docs/*, tasks/*, README, `.gitignore` | Low | test_config (+21) |

**Test-weakening check:** No assertion weakening detected. `test_logging.py` assertions rewritten to new logging-v2 schema (semantic, tied to feature). `test_az_player.py` added `@pytest.mark.skipif(not RUN_SLOW)` gates (pre-existing pattern, not weakening). `test_trainer.py`/`test_selfplay.py` are large net additions. New untracked tests all additive.

**High-risk signature changes to verify before commit:** evaluator `evaluate` lost `logger` param; `GameLogger` API reduced; `Trainer.train_steps` return type changed; `Trainer.save` signature changed to pointer path; `ReplayBuffer.add` gained optional `game_id`; `normal_5000.yaml` deleted.

## 6. NEED REVIEW (insufficient evidence)

| Item | Size | Missing evidence |
|---|---:|---|
| `final_50k_pilot/` | 51M | Own `analyze.py`/`build.py`/`evaluate.py`/`test_pilot.py`; verify regenerable |
| `checkpoints_selfplay_20k/` | 1.9M | No code refs; only historical docs; confirm run IDs not needed for resume |
| `docs/exports/` | 2.3M | Verify referenced by benchmark reports |
| `docs/performance/raw-*.json` (12 files) | 329K | Referenced by `09-deep-profiling.md` and `11-production-benchmark.md` (keep for reproducibility) |
| `checkpoints/normal/` non-active 203 generations | ~11G | Training retention/rollback policy |
| `checkpoints/normal/recovery/` | 9.5G | Referenced by trainer/training_recovery; confirm retention policy |
| Top-level legacy pointers (`latest.pt`, `buffer_latest.npz`) | ~43M | Confirm no config/script references |
| Renju artifacts (`renju_last.pt`, `renju_buffer_last.npz`) | ~14M | Confirm renju training inactive |
| `data/az_25k_continuation_v1*` + eval DBs | ~1.1G | Confirm run IDs not needed for resume (`az/logging_v2.py`) |
| Experiment dirs (`final_50k_pilot`, `defensive_*`, `tactical_*`, `value_calibration_*`) | ~80M+ | Each has own analyze/build/evaluate; verify regenerable |
| `"lab05 - game/"` | small | Ownership (unrelated lab deliverable) |
| `docs/untracked-and-junk-inventory.md` | small | Self-audit doc ownership |

## 7. Cleanup order (recommended)

1. **Run regression tests** for groups 1–2 (`test_mcts`, `test_selfplay`, `test_trainer`, `test_replay_buffer`) and group 3 (`test_logging`, `test_logging_v2`) before any commit.
2. **Commit** in 7 feature groups (§5); separate commit for `normal_5000.yaml` deletion with migration note.
3. **Apply `.gitignore` additions** (§4) — commit as part of group 7 or separate housekeeping commit.
4. **SAFE DELETE** (§3) ≈ 3.0G: pytest temp, experiment run dirs, backup, smoke dirs. Run `du -sh` before/after to confirm.
5. **NEED REVIEW** (§6): resolve each item with explicit approval; estimated additional reclaimable up to ~20.6G if checkpoint history approved for pruning.

## 8. Estimated reclaimable

- SAFE DELETE (no risk): ≈ 3.0G.
- NEED REVIEW (contingent): ≈ 1.2G+ (experiment data + docs) + up to ~20.6G (checkpoint history if approved).
- Core runtime retained: ~26G (`checkpoints/` active + `data/` + `evidence/` + `logs/`).
- `.venv/` (~4.1G) already ignored, not counted.

## 9. Notes

- No deletions, commits, resets, or code changes performed. All audits read-only.
- `native/*.cpp` source should be committed; only `*.pyd`/`*.so` binaries are junk (none at project level currently).
- `graphify-out/` (~31M) already ignored per root `.gitignore`.
- Cross-reference dependency: `az_contpilot_wzdps5fu` is referenced by `az_finalprobe_71723je1/experiment.py` — delete together if approved.
- `final_50k_pilot/data/build_holdout.py` references `defensive_gate_ablation_50k/` — delete together if approved.
