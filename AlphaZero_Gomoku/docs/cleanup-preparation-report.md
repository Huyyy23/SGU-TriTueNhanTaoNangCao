# Gomoku AlphaZero — Báo cáo chuẩn bị dọn dẹp (Bản cuối)
Ngày: 2026-11-21. Nhánh: `main`. Kiểm tra chỉ đọc; không xóa, commit, reset hay thay đổi mã.
## 1. Tóm tắt quyết định
| Nhóm | Dung lượng | Hành động |
|---|---:|---|
| GIỮ (runtime cốt lõi) | ~26G | Giữ lại: `checkpoints/` đang hoạt động, `data/*.sqlite`, `evidence/`, `logs/` |
| ĐƯA VÀO COMMIT (mã nguồn/kiểm thử/tài liệu) | ~vài MB | Commit theo 7 nhóm tính năng (§5) |
| BỎ QUA (runtime đã bị gitignore) | hỗn hợp | Đã được ignore; thêm các mẫu (§4) |
| XÓA AN TOÀN (đã xác minh) | ≈ 3.0G | Xóa sau khi commit (§3) |
| CẦN XEM XÉT | ≈ 1.2G+ (lịch sử checkpoint ≈ 20.6G bổ sung) | Cần phê duyệt rõ ràng (§6) |
## 2. Kiểm tra checkpoint (`checkpoints/`, 21G)
- Đích resume đang hoạt động: `checkpoints/normal/latest.json` → thế hệ `2156cbadc8c546ebac27ec0925f42a82` (~55M: `checkpoint.pt` + `manifest.json` + `replay.npz`).
- `normal/generations/`: 204 thư mục × ~54M ≈ 11.0G. Chỉ cần thế hệ đang hoạt động để resume; 203 thế hệ còn lại dùng làm lịch sử/khôi phục.
- `normal/recovery/`: 201 thư mục, 9.5G. Được `az/trainer.py`, `az/training_recovery.py`, `scripts/train.py`, `tests/test_train_resume_logging.py` tham chiếu — **không xóa nếu chưa xem xét chính sách khôi phục huấn luyện**.
- Thư mục smoke/kiểm thử: `_check/` 6.9M, `_smoke/` 7.0M, `_smoke2/` 8.1M. Không có tham chiếu trong mã dự án (chỉ có kết quả khớp trong `.venv`). ỨNG VIÊN XÓA AN TOÀN.
- Các liên kết cũ ở cấp cao nhất: `latest.pt` 5.7M, `buffer_latest.npz` 37M. CẦN XEM XÉT (xác nhận không có tham chiếu trong cấu hình/script).
- Biến thể Renju: `renju_last.pt` 5.7M, `renju_buffer_last.npz` 8.7M. GIỮ nếu còn huấn luyện Renju; nếu không thì CẦN XEM XÉT.
**Phân loại:**
- GIỮ: `normal/latest.json` + thư mục thế hệ đang hoạt động + các đường dẫn được cấu hình tham chiếu.
- XÓA AN TOÀN (có khả năng): `_check/`, `_smoke/`, `_smoke2/` (~22M) sau khi kiểm tra phụ thuộc của bài kiểm thử.
- CẦN XEM XÉT: 203 thế hệ không hoạt động (~11G), `recovery/` (9.5G), các liên kết cũ cấp cao nhất (~43M), tệp Renju (~14M). Cần quyết định chính sách lưu giữ.
## 3. Danh sách XÓA AN TOÀN (đã xác minh, ≈ 3.0G)

Bằng chứng: không ứng viên nào được theo dõi (`git ls-files` không trả về kết quả); pipeline cốt lõi (`az/`, `scripts/`, `players/`, `configs/`) không có tham chiếu nào; chỉ `analyze.py`/`build.py` của các thử nghiệm độc lập hoặc tài liệu kiểm tra có tham chiếu đến chúng.
| Đường dẫn | Dung lượng | Kết luận | Lý do |
|---|---:|---|---|
| `gomoku_logging_backup_20261119T173723541431Z/` (gốc) | 2.4G | XÓA AN TOÀN | Bản sao lưu log; không có tham chiếu mã/cấu hình/runtime |
| `AlphaZero_Gomoku/az_100pilot_botfozch/` | 337M | XÓA AN TOÀN | Thử nghiệm độc lập |
| `AlphaZero_Gomoku/az_contpilot_wzdps5fu/` | 101M | XÓA AN TOÀN | Chỉ được `az_finalprobe/experiment.py` tham chiếu chéo; xóa cùng nhau |
| `AlphaZero_Gomoku/az_finalprobe_71723je1/` | 35M | XÓA AN TOÀN | Thử nghiệm độc lập |
| `AlphaZero_Gomoku/az_readiness_native_wo5zl7ml/` | 18M | XÓA AN TOÀN | Lần chạy kiểm tra mức độ sẵn sàng độc lập |
| `AlphaZero_Gomoku/az_crashproof_mkxtuk9c/` | 7.9M | XÓA AN TOÀN | Độc lập |
| `AlphaZero_Gomoku/tactical_experiment_20261120/` | 8.2M | XÓA AN TOÀN | Độc lập |
| `AlphaZero_Gomoku/defensive_holdout_50k_eval/` | 1.0M | XÓA AN TOÀN | Độc lập |
| `AlphaZero_Gomoku/value_calibration_50k_data/` | 669K | XÓA AN TOÀN | Độc lập |
| `AlphaZero_Gomoku/value_calibration_50k_mcts/` | 225K | XÓA AN TOÀN | Độc lập |
| `AlphaZero_Gomoku/value_calibration_50k_stats/` | 83K | XÓA AN TOÀN | Độc lập |
| `AlphaZero_Gomoku/defensive_holdout_50k_data/` | 152K | XÓA AN TOÀN | Độc lập |
| `AlphaZero_Gomoku/defensive_gate_ablation_50k/` | 395K | XÓA AN TOÀN | Chỉ được `final_50k_pilot/data/build_holdout.py` tham chiếu |
| `AlphaZero_Gomoku/tactical_50k_analysis_20261120/` | 336K | XÓA AN TOÀN | Độc lập |
| `AlphaZero_Gomoku/checkpoints_selfplay_20k/` | 1.9M | CẦN XEM XÉT | Không có tham chiếu mã; chỉ có trong tài liệu lịch sử; xác nhận ID lần chạy |
| `AlphaZero_Gomoku/pytest-of-User/` | 30M | XÓA AN TOÀN | Tệp tạm pytest, không có tham chiếu, đã gitignore |
| `pytest-of-User/` ở gốc | 1.5M | XÓA AN TOÀN | Tệp tạm pytest |
| `AlphaZero_Gomoku/_check/`, `_smoke/`, `_smoke2/` (trong checkpoints/) | 22M | XÓA AN TOÀN (có khả năng) | Tệp smoke; xác minh không có bài kiểm thử phụ thuộc |

**Tổng XÓA AN TOÀN ≈ 3.0G** (bản sao lưu 2.4G chiếm phần lớn).
## 4. Các mục thêm vào `.gitignore` (đề xuất, chưa áp dụng)
Hiện đã ignore: `checkpoints/`, `logs/`, `evidence/`, `*.sqlite*`, `pytest-of-*/`, `__pycache__/`, `.venv/`, `*.pt`, `*.npz`, `*.pyd`/`*.so`. Còn thiếu: thư mục thử nghiệm, JSON hiệu năng thô, exports, bản sao lưu, lab05, thư mục native.
Thêm các dòng sau vào `.gitignore` ở gốc:

```gitignore
lab05 - game/
gomoku_logging_backup_*/
```

Thêm các dòng sau vào `AlphaZero_Gomoku/.gitignore`:

```gitignore
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
Thêm vào **`AlphaZero_Gomoku/.gitignore`**:
Kiểm tra rủi ro: `git check-ignore` xác nhận `.sqlite`, `.pyd`, `checkpoints/` đã được ignore. `lab05 - game/` và `raw-*.json` CHƯA được ignore → có nguy cơ commit nhầm. Không có tệp nhị phân/cơ sở dữ liệu nào được theo dõi (`git ls-files` không trả về kết quả).
## 5. Nhóm commit (7 nhóm tính năng)

| # | Nhóm | Tệp cốt lõi | Rủi ro | Kiểm thử |
|---|---|---|---|---|
| 1 | Backend MCTS native + batching | `az/mcts.py`, `scripts/bench_batching.py`, `tests/test_mcts.py`, `tests/test_native_mcts.py` (mới), `native/*` | CAO (tính đúng đắn của tìm kiếm) | test_mcts, test_native_mcts |
| 2 | Vòng lặp self-play / trainer | `az/selfplay.py`, `az/trainer.py`, `az/replay_buffer.py`, các bài kiểm thử | CAO (tính đúng đắn của huấn luyện) | test_selfplay (+237), test_trainer (+316), test_replay_buffer (+75) |
| 3 | Logging v2 + trao đổi thử nghiệm | `az/logging_v2.py` (mới), `az/experiment_exchange.py` (mới), `az/backend_logging.py`, `az/game_log.py`, `scripts/migrate_logs_to_sqlite.py`, `scripts/exchange_experiment.py`, các bài kiểm thử | Trung bình (nạp dữ liệu) | test_logging (viết lại), test_logging_v2 (mới) |
| 4 | Elo + tournament/arena | `az/elo.py` (mới), `scripts/recalculate_elo.py`, `scripts/tournament.py`, `scripts/arena.py`, `players/factory.py`, các bài kiểm thử | Trung bình | test_elo (mới), test_tournament (+342), test_arena (+66) |
| 5 | Tinh chỉnh Renju + model | `az/renju_finetune.py`, `az/model.py`, `config*.yaml`, các bài kiểm thử | Trung bình-cao (luật chơi) | test_rules_normal (+12), test_patterns (+11) |
| 6 | Factory player/agent + UI | `players/az_player.py`, `ui/pygame_app.py`, `ui/pygame_frontend.py` (mới), các bài kiểm thử | Thấp-trung bình | test_ui (+221 mới) |
| 7 | Cấu hình/tài liệu/tasks/bảo trì | `config.py`, configs, docs/*, tasks/*, README, `.gitignore` | Thấp | test_config (+21) |

**Kiểm tra việc làm yếu bài kiểm thử:** Không phát hiện làm yếu assertion. Assertion trong `test_logging.py` được viết lại theo schema logging-v2 mới (ngữ nghĩa gắn với tính năng). `test_az_player.py` thêm các điều kiện `@pytest.mark.skipif(not RUN_SLOW)` (mẫu có từ trước, không phải làm yếu kiểm thử). `test_trainer.py`/`test_selfplay.py` có lượng bổ sung ròng lớn. Các bài kiểm thử mới chưa được theo dõi đều bổ sung thêm.

**Các thay đổi chữ ký rủi ro cao cần xác minh trước khi commit:** evaluator `evaluate` bỏ tham số `logger`; API `GameLogger` được thu gọn; kiểu trả về của `Trainer.train_steps` thay đổi; chữ ký `Trainer.save` đổi thành đường dẫn pointer; `ReplayBuffer.add` thêm `game_id` tùy chọn; xóa `normal_5000.yaml`.
## 6. CẦN XEM XÉT (chưa đủ bằng chứng)

| Mục | Dung lượng | Bằng chứng còn thiếu |
|---|---:|---|
| `final_50k_pilot/` | 51M | Có `analyze.py`/`build.py`/`evaluate.py`/`test_pilot.py` riêng; xác minh có thể tạo lại |
| `checkpoints_selfplay_20k/` | 1.9M | Không có tham chiếu mã; chỉ có trong tài liệu lịch sử; xác nhận không cần ID lần chạy để resume |
| `docs/exports/` | 2.3M | Xác minh các báo cáo benchmark có tham chiếu |
| `docs/performance/raw-*.json` (12 tệp) | 329K | Được `09-deep-profiling.md` và `11-production-benchmark.md` tham chiếu (giữ để tái lập kết quả) |
| 203 thế hệ không hoạt động trong `checkpoints/normal/` | ~11G | Chính sách lưu giữ/khôi phục huấn luyện |
| `checkpoints/normal/recovery/` | 9.5G | Được trainer/training_recovery tham chiếu; xác nhận chính sách lưu giữ |
| Các pointer cũ ở cấp cao nhất (`latest.pt`, `buffer_latest.npz`) | ~43M | Xác nhận không có tham chiếu cấu hình/script |
| Tệp Renju (`renju_last.pt`, `renju_buffer_last.npz`) | ~14M | Xác nhận việc huấn luyện Renju đã dừng |
| `data/az_25k_continuation_v1*` + cơ sở dữ liệu eval | ~1.1G | Xác nhận không cần ID lần chạy để resume (`az/logging_v2.py`) |
| Thư mục thử nghiệm (`final_50k_pilot`, `defensive_*`, `tactical_*`, `value_calibration_*`) | ~80M+ | Mỗi thư mục có analyze/build/evaluate riêng; xác minh có thể tạo lại |
| `"lab05 - game/"` | nhỏ | Quyền sở hữu (sản phẩm bàn giao lab không liên quan) |
| `docs/untracked-and-junk-inventory.md` | nhỏ | Quyền sở hữu tài liệu tự kiểm tra |

## 7. Thứ tự dọn dẹp (đề xuất)

1. **Chạy kiểm thử hồi quy** cho nhóm 1–2 (`test_mcts`, `test_selfplay`, `test_trainer`, `test_replay_buffer`) và nhóm 3 (`test_logging`, `test_logging_v2`) trước mọi commit.
2. **Commit** theo 7 nhóm tính năng (§5); tạo commit riêng để xóa `normal_5000.yaml` kèm ghi chú di chuyển.
3. **Áp dụng các mục thêm vào `.gitignore`** (§4), commit trong nhóm 7 hoặc commit bảo trì riêng.
4. **XÓA AN TOÀN** (§3) ≈ 3.0G: tệp tạm pytest, thư mục chạy thử nghiệm, bản sao lưu, thư mục smoke. Chạy `du -sh` trước/sau để xác nhận.
5. **CẦN XEM XÉT** (§6): giải quyết từng mục sau khi có phê duyệt rõ ràng; ước tính có thể giải phóng thêm tối đa ~20.6G nếu được duyệt cắt bỏ lịch sử checkpoint.

## 8. Ước tính dung lượng có thể giải phóng

- XÓA AN TOÀN (không rủi ro): ≈ 3.0G.
- CẦN XEM XÉT (tùy điều kiện): ≈ 1.2G+ (dữ liệu thử nghiệm + tài liệu) + tối đa ~20.6G (lịch sử checkpoint nếu được duyệt).
- Runtime cốt lõi được giữ lại: ~26G (`checkpoints/` đang hoạt động + `data/` + `evidence/` + `logs/`).
- `.venv/` (~4.1G) đã được ignore, không tính vào đây.

## 9. Ghi chú

- Không xóa, commit, reset hay thay đổi mã. Toàn bộ kiểm tra chỉ đọc.
- Nên commit mã nguồn `native/*.cpp`; chỉ các tệp nhị phân `*.pyd`/`*.so` là rác (hiện không có ở cấp dự án).
- `graphify-out/` (~31M) đã được ignore theo `.gitignore` ở gốc.
- Phụ thuộc tham chiếu chéo: `az_contpilot_wzdps5fu` được `az_finalprobe_71723je1/experiment.py` tham chiếu — nếu được duyệt thì xóa cùng nhau.
- `final_50k_pilot/data/build_holdout.py` tham chiếu `defensive_gate_ablation_50k/` — nếu được duyệt thì xóa cùng nhau.
