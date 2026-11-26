# Gomoku AlphaZero — Báo cáo kiểm tra cây làm việc Git và dọn dẹp

Ngày: 2026-11-21. Nhánh: `main`. Phạm vi: kiểm tra chỉ đọc; không xóa, commit hay thay đổi mã.

## 1. Ảnh chụp hiện trạng

- Có 53 tệp được theo dõi đã sửa đổi (AZ cốt lõi, cấu hình, người chơi, tập lệnh, bài kiểm thử, UI, tài liệu, công việc).
- Có 68 đường dẫn chưa được theo dõi (mã nguồn/kiểm thử/tập lệnh/tài liệu + thư mục chạy thử + tệp phát sinh + bản sao lưu).
- Tổng dung lượng dự án: ~29G (chủ yếu là `checkpoints/` 21G, `data/` 3.7G, bản sao lưu 2.4G).
- `.gitignore` (dự án): đã ignore `checkpoints/`, `logs/`, `evidence/`, `*.sqlite*`, `pytest-of-*/`, `__pycache__/`, `.venv/`, `*.pt`, `*.npz`, `*.pyd`/`*.so`. Tuy nhiên, `data/`, thư mục thử nghiệm, `native/`, `docs/analysis`, JSON hiệu năng thô vẫn chưa được ignore.

## 2. Kiểm tra các tệp được theo dõi đã sửa đổi (53)

| Khu vực | Tệp | Loại | Bằng chứng | Rủi ro |
|---|---|---|---|---|
| cốt lõi | `az/backend_logging.py` | tái cấu trúc | ±76 dòng | trung bình (hợp đồng logging) |
| cốt lõi | `az/evaluator.py` | sửa lỗi | đã bỏ `evaluate(..., logger=...)` | **cao (chữ ký public)** |
| cốt lõi | `az/game_log.py` | tái cấu trúc | −140 dòng | **cao (API GameLogger)** |
| cốt lõi | `az/mcts.py` | tính năng | NativeMCTS + thống kê gốc +132 | **cao (hành vi tìm kiếm)** |
| cốt lõi | `az/model.py` | tái cấu trúc | +21 | trung bình |
| cốt lõi | `az/renju_finetune.py` | tính năng | +75 | trung bình |
| cốt lõi | `az/replay_buffer.py` | tính năng | `add(..., game_id=...)` tùy chọn | thấp-trung bình (tương thích ngược) |
| cốt lõi | `az/selfplay.py` | tính năng | provenance, `_new_game(..., provenance)` | **cao (nguồn gốc dữ liệu)** |
| cốt lõi | `az/trainer.py` | tính năng/tái cấu trúc | `train_steps` trả về dict; `save(pointer_path)`; bootstrap | **cao (hợp đồng checkpoint)** |
| cấu hình | `config.py` | tính năng | +10 | trung bình (schema) |
| cấu hình | `configs/normal.yaml` | tính năng | 12 dòng | trung bình |
| cấu hình | `configs/normal_5000.yaml` | **đã xóa** | đã xóa 5 dòng | **cao (đã xóa cấu hình budget)** |
| cấu hình | `configs/renju.yaml` | tính năng | +3 | thấp |
| người chơi | `players/az_player.py` | tái cấu trúc | 6 dòng | thấp |
| người chơi | `players/factory.py` | tái cấu trúc | 6 dòng | trung bình (mặc định) |
| tập lệnh | `ablation.py` | tái cấu trúc | 122 | trung bình |
| tập lệnh | `analyze_t35.py` | tái cấu trúc | 258 | trung bình |
| tập lệnh | `arena.py` | tái cấu trúc | 110 | trung bình |
| tập lệnh | `bench_batching.py` | tái cấu trúc | 37 | thấp |
| tập lệnh | `bench_selfplay_round2.py` | tái cấu trúc | 62 | thấp |
| tập lệnh | `spike_mp.py` | tái cấu trúc | 9 | thấp |
| tập lệnh | `tournament.py` | tái cấu trúc/tính năng | 443 | **cao (CLI/lưu trữ)** |
| tập lệnh | `train.py` | tính năng | 578 | **cao (pipeline)** |
| kiểm thử | 22 `test_*.py` | thay đổi bài kiểm thử | nhiều assertion được thêm/viết lại | **cao (thay đổi kỳ vọng)** |
| giao diện | `ui/pygame_app.py` | tính năng/tái cấu trúc | 39 | trung bình |
| tài liệu | `README.md`, `SPEC.md`, `GLOSSARY.md`, `report.md`, `tournament_guide.md`, ADRs 0004/0005 | tài liệu | tài liệu | thấp |
| kho mã | `README.md` | tài liệu | 25 | thấp |
| kho mã | `.gitignore` | cấu hình | 7 | thấp |
| công việc | `tasks/*` (4) | tài liệu | theo dõi công việc | thấp |

**Các thay đổi public/API/config/test cần lưu ý:**
- `evaluate` đã bỏ tham số `logger`.
- API `GameLogger` đã được thu gọn.
- Kiểu trả về của `Trainer.train_steps` đã thay đổi (`dict[str,float]` → `dict`).
- Chữ ký `Trainer.save` đã đổi thành đường dẫn pointer.
- `ReplayBuffer.add` đã thêm `game_id` tùy chọn.
- `normal_5000.yaml` đã bị xóa (biến thể budget).
- Nhiều bài kiểm thử đã được đổi tên/viết lại → thay đổi hành vi/kỳ vọng.

**Tóm tắt rủi ro:** rủi ro cao nhất là hợp đồng lưu checkpoint của trainer, lưu trữ tournament, provenance selfplay, bỏ logger khỏi evaluator và xóa `normal_5000.yaml`. Rủi ro trung bình: nhánh MCTS native, schema cấu hình. Rủi ro thấp: tài liệu/công việc.

## 3. Kiểm tra tệp chưa được theo dõi (68)

### GIỮ / ĐƯA VÀO COMMIT (mã nguồn, kiểm thử, tài liệu, tập lệnh)
- Cốt lõi: `az/elo.py`, `az/experiment_exchange.py`, `az/logging_v2.py`, `az/training_recovery.py` (được train.py/selfplay/tests tham chiếu).
- Kiểm thử: 11 tệp `tests/test_*.py` mới (elo, experiment_exchange, legacy_quarantine, logging_v2, migrate_logs_to_sqlite, native_mcts, native_patterns, tactical_benchmark, tactical_oracle, train_resume_logging, …).
- Tập lệnh: 9 script mới (bench_native_patterns, bench_real_world, exchange_experiment, investigate_phase_timing, migrate_logs_to_sqlite, profile_cpp_backend, quarantine_legacy, recalculate_elo, tactical_benchmark).
- UI: `ui/pygame_frontend.py`. Native: `native/gomoku_native.cpp`, `native/gomoku_mcts_native.cpp`. Gốc: `requirements.txt`.
- Tài liệu: `docs/analysis/` (22 md), `docs/exports/`, `docs/intent/`, nhiều `docs/bao-cao-*.md`, `docs/cpp-*.md`, `docs/az-*-report.md`, `docs/logging-v2-implementation-report.md`, `tasks/progress-review-2026-11-11.md`.

### BỎ QUA (tệp phát sinh đã gitignore)
- `data/` (3.7G), `checkpoints_selfplay_20k/`, `pygame_gui.log`, `native/*.pyd` (không có ở cấp dự án), `pytest-of-*` (không có ở gốc dự án).

### XÓA AN TOÀN (thư mục chạy thử + bản sao lưu)
- `az_100pilot_botfozch/` 337M, `az_contpilot_wzdps5fu/` 101M, `az_finalprobe_71723je1/` 35M, `az_readiness_native_wo5zl7ml/` 18M, `az_crashproof_mkxtuk9c/` 7.9M.
- `final_50k_pilot/` 51M, `defensive_*` 1.5M, `tactical_*` 8.5M, `value_calibration_*` 1.0M.
- `checkpoints_selfplay_20k/` 1.9M.
- `gomoku_logging_backup_20261119T173723541431Z/` (gốc) 2.4G.
- `"lab05 - game/*.zip/.docx"` (sản phẩm bàn giao lab không liên quan) dung lượng nhỏ.
- Tổng XÓA AN TOÀN ≈ 3.0G.

### CẦN XEM XÉT
- `docs/exports/`, `docs/performance/raw-*.json` — dữ liệu hiệu năng thô; xác minh báo cáo benchmark có tham chiếu.
- `docs/untracked-and-junk-inventory.md` — quyền sở hữu tài liệu tự kiểm tra.
- Quyền sở hữu các tệp trong `"lab05 - game/"`.

## 4. Kiểm tra tệp dữ liệu (checkpoints / SQLite / logs / evidence)

| Tệp | Dung lượng | Được tham chiếu bởi | Cần cho resume/train/tournament? | Phân loại |
|---|---:|---|---|---|
| `checkpoints/` (normal pointer + generations) | 21G | `configs/normal.yaml`, README, SPEC, `az/renju_finetune.py`, mặc định `--checkpoint` | **Có** — resume cần `checkpoints/normal/latest.json` + `checkpoint.pt`/`replay.npz` của thế hệ; agent AZ trong tournament cần checkpoint | GIỮ (cốt lõi, không commit) |
| `checkpoints_selfplay_20k/` | 1.9M | chỉ tài liệu lịch sử/thử nghiệm | Không cần cho train/tournament cốt lõi | Ứng viên CẦN XEM XÉT / XÓA AN TOÀN |
| `data/*.sqlite` (normal/selfplay/tournament/eval) | ~4.9G+ | mặc định `az/logging_v2.py`, scripts `--db`, README/SPEC | **Có** — bản ghi self-play/huấn luyện (`normal.sqlite`), tournament/Elo (`tournament.sqlite`); resume cần run ID | GIỮ |
| `data/az_25k_continuation_v1*` + eval DBs | ~1.1G + 16M | chỉ script thử nghiệm/phân tích | Bằng chứng lịch sử, không cần cho resume cốt lõi | GIỮ làm bằng chứng / CẦN XEM XÉT |
| `evidence/` (colab_return, depth1_cutover, paused) | 149M | README chạy thử migration quét `evidence/`; docs | Không cần cho runtime train/tournament | GIỮ (bằng chứng, không xóa) |
| `logs/` + `logs/backend_runs.sqlite3` | 20M | SPEC: tệp runtime; migration quét `logs/` | Không cần cho resume train/tournament (chỉ telemetry) | GIỮ/BỎ QUA (runtime, gitignored) |
| `pytest-of-User/` (gốc + dự án) | 30M+ | tệp tạm pytest; không có tham chiếu được theo dõi | Không | XÓA AN TOÀN |
| Thư mục thử nghiệm (`final_50k_pilot`, `defensive_*`, `tactical_*`, `az_*pilot*`, `az_*probe*`) | ~80M+ | chỉ `analyze.py`/`build.py` riêng của từng thư mục | Không cần cho train/tournament cốt lõi | CẦN XEM XÉT (`analyze.py` riêng tham chiếu checkpoints/data) |
| `gomoku-selfplay-renju-bench-e_wotby6/` | nhỏ | chỉ thử nghiệm | Không | CẦN XEM XÉT |
| `*.pyd` (dự án) | không có ở cấp dự án | chỉ `.venv` | Không | BỎ QUA (venv, gitignored) |

**Kết luận chính:** runtime cốt lõi cần pointer `checkpoints/normal/` cùng các tệp thế hệ và `data/normal.sqlite` / `data/tournament.sqlite`. Thư mục thử nghiệm/bằng chứng là dữ liệu lịch sử, không cần cho resume/train/tournament, nhưng không được tự động xóa nếu chưa xác minh run-id/hash. Có thể xóa `pytest-of-User` an toàn. Không sửa DB; sqlite được kiểm tra ở chế độ chỉ đọc.

**Xác minh tham chiếu (grep):**
- `checkpoints_selfplay_20k` chỉ được tài liệu tham chiếu (az-80k-checkpoint-forensics, az-final-cleanup-report, az-normal-migration-report) cùng tài liệu tự kiểm tra; không có tham chiếu mã/cấu hình.
- Thư mục thử nghiệm (`final_50k_pilot`, `defensive_*`, `value_calibration_*`) được `analyze.py`/`build.py`/`evaluate.py` riêng và tài liệu phân tích tham chiếu; pipeline cốt lõi không tham chiếu.
- `evidence/` được README, SPEC, benchmarks, integration-2026-11-14, báo cáo logging-v2 tham chiếu.

## 5. Tóm tắt phân loại

| Nhóm | Ví dụ | Dung lượng |
|---|---|---:|
| GIỮ (runtime cốt lõi) | `checkpoints/`, `data/*.sqlite`, `evidence/`, `logs/` | ~26G |
| ĐƯA VÀO COMMIT (mã nguồn/kiểm thử/tài liệu) | 4 mô-đun az, 9 tập lệnh, 11 bài kiểm thử, UI, mã C++ native, tài liệu, `requirements.txt` | ~ vài MB |
| BỎ QUA (runtime gitignored) | `data/` KHÔNG bị ignore (!), `logs/`, `pytest-of-*`, `.venv/`, `*.pyd` | hỗn hợp |
| XÓA AN TOÀN | tệp tạm pytest, thư mục chạy thử, bản sao lưu, checkpoints_selfplay_20k, tệp lab05 | ≈ 3.0G |
| CẦN XEM XÉT | JSON hiệu năng thô, exports, thư mục thử nghiệm, tài liệu tự kiểm tra, tệp lab05, checkpoint selfplay_20k | ≈ 1.2G+ |
## 6. Nhóm commit đề xuất

1. **Pipeline AZ cốt lõi + kiểm thử + tài liệu** (một commit): toàn bộ mã nguồn/kiểm thử/tập lệnh/UI/native/tài liệu chưa được theo dõi thuộc GIỮ/ĐƯA VÀO COMMIT + 53 tệp được theo dõi đã sửa đổi. Trước tiên xem xét các thay đổi chữ ký rủi ro cao (bỏ logger khỏi evaluator, đơn giản hóa GameLogger, hợp đồng trainer.save, xóa normal_5000.yaml).
2. **Commit riêng:** xóa `configs/normal_5000.yaml` kèm ghi chú di chuyển (đã xóa cấu hình budget; cập nhật tham chiếu trong SPEC/README).
3. Không commit tệp runtime (`data/`, `logs/`, `checkpoints/`, thư mục thử nghiệm).

## 7. Các mục thêm vào .gitignore (đề xuất, chưa áp dụng)

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

Lưu ý: `data/` nên tiếp tục KHÔNG bị ignore? Hiện chưa được ignore. Đề xuất giữ `data/` được theo dõi như runtime (chỉ ignore bằng `*.sqlite*` hiện có), tức là ignore `*.sqlite*` nhưng giữ thư mục; thêm `data/*.log` và `data/*.jsonl` nếu không cần. Xác minh trước khi áp dụng.

## 8. Danh sách XÓA AN TOÀN và dung lượng có thể giải phóng

| Đường dẫn | Dung lượng | Lý do |
|---|---:|---|
| `AlphaZero_Gomoku/pytest-of-User/` + `pytest-of-User/` ở gốc | 30M+ | tệp tạm pytest, không có tham chiếu |
| `AlphaZero_Gomoku/az_100pilot_botfozch/` | 337M | lần chạy thử nghiệm |
| `AlphaZero_Gomoku/az_contpilot_wzdps5fu/` | 101M | lần chạy thử nghiệm |
| `AlphaZero_Gomoku/az_finalprobe_71723je1/` | 35M | thăm dò |
| `AlphaZero_Gomoku/az_readiness_native_wo5zl7ml/` | 18M | lần chạy kiểm tra mức độ sẵn sàng |
| `AlphaZero_Gomoku/az_crashproof_mkxtuk9c/` | 7.9M | lần chạy kiểm tra khả năng chống crash |
| `AlphaZero_Gomoku/final_50k_pilot/` | 51M | thử nghiệm |
| `AlphaZero_Gomoku/defensive_gate_ablation_50k/` | 0.4M | thử nghiệm loại bỏ thành phần |
| `AlphaZero_Gomoku/defensive_holdout_50k_data/` | 0.15M | dữ liệu kiểm định riêng |
| `AlphaZero_Gomoku/defensive_holdout_50k_eval/` | 1.0M | đánh giá tập kiểm định riêng |
| `AlphaZero_Gomoku/tactical_50k_analysis_20261120/` | 0.3M | phân tích |
| `AlphaZero_Gomoku/tactical_experiment_20261120/` | 8.2M | thử nghiệm |
| `AlphaZero_Gomoku/value_calibration_50k_data/` | 0.67M | hiệu chỉnh |
| `AlphaZero_Gomoku/value_calibration_50k_mcts/` | 0.23M | hiệu chỉnh |
| `AlphaZero_Gomoku/value_calibration_50k_stats/` | 0.08M | hiệu chỉnh |
| `AlphaZero_Gomoku/checkpoints_selfplay_20k/` | 1.9M | checkpoint lịch sử |
| `gomoku_logging_backup_20261119T173723541431Z/` (gốc) | 2.4G | bản sao lưu log |
| `"lab05 - game/*.zip/.docx"` | nhỏ | sản phẩm bàn giao lab không liên quan |
| **Tổng XÓA AN TOÀN** | **≈ 3.0G** | |

Không xóa `checkpoints/` (21G) hoặc `data/` (3.7G) — runtime cốt lõi cần cho resume/train/tournament.

## 9. Danh sách CẦN XEM XÉT (cần bằng chứng trước khi hành động)

| Mục | Bằng chứng còn thiếu |
|---|---|
| `docs/exports/`, `docs/performance/raw-*.json` | Xác minh được báo cáo benchmark tham chiếu (`docs/performance/10-optimization-round2.md`, `11-production-benchmark.md`). |
| `checkpoints_selfplay_20k/` | Xác minh script thử nghiệm không phụ thuộc run-id (grep chỉ thấy tham chiếu tài liệu, có vẻ an toàn nhưng cần xác nhận run ID). |
| `data/az_25k_continuation_v1*` + eval DBs (~1.1G) | Xác nhận run ID không cần cho resume; kiểm tra tra cứu run-id trong `az/logging_v2.py`. |
| Thư mục thử nghiệm (`az_*`, `defensive_*`, `tactical_*`, `value_calibration_*`, `final_50k_pilot`) | Mỗi thư mục có `analyze.py`/`build.py`/`evaluate.py` riêng tham chiếu checkpoints/data; xác minh có thể tạo lại tệp phát sinh. |
| `docs/untracked-and-junk-inventory.md` | Quyền sở hữu — tài liệu tự kiểm tra; giữ lại hoặc gộp vào báo cáo này. |
| Các tệp `"lab05 - game/"` | Quyền sở hữu — sản phẩm bàn giao lab không liên quan; xác nhận sinh viên muốn giữ. |

## 10. Ghi chú di chuyển sang pipeline AZ đơn

- **Thư mục checkpoint duy nhất:** `checkpoints/normal/` với pointer `latest.json` + thư mục thế hệ (`checkpoint.pt`, `replay.npz`). Ngừng dùng `checkpoints_selfplay_20k/`.
- **Thư mục log duy nhất:** `logs/` (chỉ telemetry) + SQLite `data/normal.sqlite` (self-play/huấn luyện), `data/tournament.sqlite` (Elo/tournament). Ngừng dùng `gomoku_logging_backup_*` và log thử nghiệm rải rác.
- **Trình chạy thử nghiệm duy nhất:** hợp nhất `scripts/train.py` (`--config`, `--checkpoint`, `--db`) thành entry point duy nhất; chuyển `exchange_experiment.py`, `tactical_benchmark.py` sang đọc `data/*.sqlite` qua `az/logging_v2.py`. Xóa phần `analyze.py` trùng lặp trong từng thử nghiệm.
- **Cấu hình:** chỉ dùng `configs/normal.yaml`; xóa `normal_5000.yaml` (biến thể budget) và ghi lại việc xóa trong SPEC/README. Đối chiếu các thay đổi schema `config.py`.
- **Hợp đồng pointer:** `Trainer.save(pointer_path=...)` và `latest.json` phải là cơ chế resume duy nhất; cập nhật các thành phần tiêu thụ ở tournament/arena.
- **Kiểm thử:** cập nhật các chữ ký public bị ảnh hưởng (tham số logger của evaluator.evaluate, API GameLogger, dict trả về của train_steps) trước khi commit.

## 11. Ghi chú

- Không thực hiện xóa. Tất cả lệnh chỉ đọc.
- Nên commit mã nguồn `native/*.cpp`; chỉ các tệp nhị phân `*.pyd` là rác (hiện không có ở cấp dự án).
- `.venv/` (~4.1G) đã được ignore và không xuất hiện trong danh sách chưa được theo dõi.
- `graphify-out/` (~31M) đã được ignore theo `.gitignore` ở gốc.
- Dung lượng có thể giải phóng không rủi ro: ≈ 3.0G (XÓA AN TOÀN). Có thể giải phóng thêm ~1.2G+ nếu xác minh CẦN XEM XÉT. Phải giữ runtime cốt lõi (~26G: checkpoints + data + evidence + logs).
