- Gomoku AlphaZero — Kiểm kê: tệp chưa theo dõi/đã sửa đổi và tệp rác

Tạo ngày 2026-11-21. Cây làm việc có 53 tệp được theo dõi đã sửa đổi + 68 mục chưa được theo dõi.

## 1. Các tệp được theo dõi đã sửa đổi (53)

### AZ cốt lõi (`AlphaZero_Gomoku/az/`)
- `backend_logging.py`, `evaluator.py`, `game_log.py`, `mcts.py`, `model.py`
- `renju_finetune.py`, `replay_buffer.py`, `selfplay.py`, `trainer.py`

### Cấu hình
- `config.py`, `configs/normal.yaml`, `configs/renju.yaml`
- `configs/normal_5000.yaml` — đã xóa

### Người chơi
- `players/az_player.py`, `players/factory.py`

### Tập lệnh
- `scripts/bench_selfplay_round2.py`, `spike_mp.py`, `tournament.py`, `train.py`

### Kiểm thử
- `test_az_player_factory.py`, `test_backend_auto.py`, `test_config.py`, `test_logging.py`
- `test_mcts.py`, `test_patterns.py`, `test_replay_buffer.py`, `test_rules_normal.py`
- `test_selfplay.py`, `test_tournament.py`, `test_trainer.py`, `test_ui.py`

### UI / Tài liệu / Công việc
- `ui/pygame_app.py`
- `docs/adr/0004-structured-logging-and-benchmarking.md`, `docs/adr/0005-budget-5000-games-for-az-v0b.md`
- `docs/report.md`, `docs/tournament_guide.md`
- `tasks/Lo_trinh_cai_tien_AlphaZero_Gomoku.md`, `tasks/Phan_viec_cua_B_khong_dung_A_va_C.md`
- `tasks/plan.md`, `tasks/todo.md`
- `GLOSSARY.md`, `README.md`, `SPEC.md`
- `README.md` ở gốc

## 2. Tệp mới chưa được theo dõi (ứng viên cần giữ)

### Cốt lõi
- `az/elo.py`, `az/experiment_exchange.py`, `az/logging_v2.py`, `az/training_recovery.py`

### Mã nguồn C++ native
- `native/gomoku_native.cpp`, `native/gomoku_mcts_native.cpp`

### Tập lệnh
- `scripts/bench_native_patterns.py`, `scripts/bench_real_world.py`
- `scripts/exchange_experiment.py`, `scripts/investigate_phase_timing.py`
- `scripts/migrate_logs_to_sqlite.py`, `scripts/profile_cpp_backend.py`
- `scripts/quarantine_legacy.py`, `scripts/recalculate_elo.py`
- `scripts/tactical_benchmark.py`

### Kiểm thử
- `tests/test_elo.py`, `test_experiment_exchange.py`, `test_legacy_quarantine.py`
- `tests/test_logging_v2.py`, `test_migrate_logs_to_sqlite.py`, `test_native_mcts.py`
- `tests/test_native_patterns.py`, `test_tactical_benchmark.py`, `test_tactical_oracle.py`
- `tests/test_train_resume_logging.py`

### UI / Tài liệu / Công việc
- `ui/pygame_frontend.py`
- `docs/analysis/` (22 md), `docs/exports/`, `docs/intent/`
- `docs/bao-cao-*.md`, `docs/cpp-*.md`, `docs/az-*-report.md`, `docs/logging-v2-implementation-report.md`
- `tasks/progress-review-2026-11-11.md`
- tệp `requirements.txt` ở gốc
- `"lab05 - game/..."` (tệp zip/docx bàn giao)

## 3. Tệp rác chưa theo dõi / tệp runtime (ứng viên xóa)

| Đường dẫn | Dung lượng | Ghi chú |
|---|---:|---|
| `AlphaZero_Gomoku/checkpoints/` | 21G | checkpoint/model/npz |
| `AlphaZero_Gomoku/data/` | 3.7G | sqlite/log thử nghiệm |
| `gomoku_logging_backup_20261119T173723541431Z/` | 2.4G | bản sao lưu log |
| `AlphaZero_Gomoku/az_100pilot_botfozch/` | 337M | lần chạy thử nghiệm |
| `AlphaZero_Gomoku/az_contpilot_wzdps5fu/` | 101M | lần chạy thử nghiệm |
| `AlphaZero_Gomoku/evidence/` | 149M | lần chạy thu thập bằng chứng |
| `AlphaZero_Gomoku/final_50k_pilot/` | 51M | lần chạy thử nghiệm |
| `AlphaZero_Gomoku/az_finalprobe_71723je1/` | 35M | lần chạy thăm dò |
| `AlphaZero_Gomoku/az_readiness_native_wo5zl7ml/` | 18M | lần chạy kiểm tra mức độ sẵn sàng |
| `AlphaZero_Gomoku/az_crashproof_mkxtuk9c/` | 7.9M | lần chạy kiểm tra khả năng chống crash |
| `AlphaZero_Gomoku/tactical_experiment_20261120/` | 8.2M | thử nghiệm |
| `AlphaZero_Gomoku/logs/` | 20M | log huấn luyện |
| `AlphaZero_Gomoku/pytest-of-User/` | 30M | tệp tạm pytest |
| `AlphaZero_Gomoku/checkpoints_selfplay_20k/` | 1.9M | checkpoint selfplay |
| `AlphaZero_Gomoku/defensive_*`, `value_calibration_*`, `tactical_50k_analysis_*` | ~2MB | tệp phân tích |
| `AlphaZero_Gomoku/native/*.pyd` (4 tệp) | ~3.5MB | mã nhị phân đã biên dịch, có thể tạo lại |
| `AlphaZero_Gomoku/pygame_gui.log` | 115B | log runtime |
| `AlphaZero_Gomoku/__pycache__/`, `.pytest_cache/` | nhỏ | bộ nhớ đệm |
| `AlphaZero_Gomoku/docs/performance/raw-round2-*.json` (chưa được theo dõi) | ~120KB | kết quả benchmark thô |

Dung lượng có thể thu hồi: ~27GB (chủ yếu là checkpoint, dữ liệu và bản sao lưu).

## 4. Các lệnh kiểm tra

```bash
find AlphaZero_Gomoku -type f -size +50M -printf '%s %p\n' | sort -nr | head -30
git status --ignored --short | head -50
```

## 5. Lệnh dọn dẹp (chỉ chạy sau khi sao lưu nếu cần)

```bash
# Tệp cache/kiểm thử
find . -type d -name __pycache__ -prune -exec rm -rf {} +
rm -rf AlphaZero_Gomoku/.pytest_cache AlphaZero_Gomoku/pytest-of-User
rm -f AlphaZero_Gomoku/pygame_gui.log AlphaZero_Gomoku/*.pyd

# Tệp runtime/thử nghiệm (DANGEROUS: loses checkpoints/data)
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

## 6. Các mục đề xuất thêm vào .gitignore

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

## 7. Ghi chú
- Không thực hiện xóa. Hãy xem xét/commit mã và tài liệu có giá trị trước khi xóa tệp phát sinh.
- `.venv/` (~4.1G) đã được ignore và không xuất hiện trong danh sách chưa được theo dõi.
