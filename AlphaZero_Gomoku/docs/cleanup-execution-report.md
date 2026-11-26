# Gomoku AlphaZero — Báo cáo thực hiện dọn dẹp

Ngày: 2026-11-21. Nhánh: `main`.
## 1. Kiểm thử hồi quy
- `test_mcts`, `test_selfplay`, `test_trainer`, `test_replay_buffer`, `test_logging`, `test_logging_v2`, `test_tournament`, `test_arena`: **131 passed** (32.2s). Không có lỗi.
## 2. Thay đổi `.gitignore`
- Gốc: thêm `lab05 - game/`, `gomoku_logging_backup_*/`.
- Dự án: thêm các mẫu thư mục chạy thử nghiệm, thư mục smoke, `pygame_gui.log`, `native/*.pyd`/`*.so` (giữ mã nguồn `.cpp`), `docs/exports/`, `docs/performance/raw-*.json`.
- Đã xác minh `native/gomoku_native.cpp` và `native/gomoku_mcts_native.cpp` KHÔNG bị ignore (mã thoát 1 từ check-ignore, đúng).
## 3. Các commit (11 commit mới trên main)
| SHA | Tiêu đề |
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
## 4. Đã xóa an toàn (giải phóng ≈ 3.0G)
- `gomoku_logging_backup_20261119T173723541431Z/` 2.4G (bản sao lưu log; không có tham chiếu mã nguồn)
- Các thư mục chạy thử nghiệm: `az_100pilot_botfozch` 337M, `az_contpilot_wzdps5fu` 101M, `az_finalprobe_71723je1` 35M, `az_readiness_native_wo5zl7ml` 18M, `az_crashproof_mkxtuk9c` 7.9M, `tactical_experiment_20261120` 8.2M, `defensive_holdout_50k_eval` 1.0M, `value_calibration_50k_*` 1.0M, `defensive_holdout_50k_data` 152K, `defensive_gate_ablation_50k` 395K, `tactical_50k_analysis_20261120` 336K.
- Tệp smoke: `checkpoints/_check` 6.9M, `_smoke` 7.0M, `_smoke2` 8.1M.
- Tệp tạm pytest: `AlphaZero_Gomoku/pytest-of-User` 28M, `pytest-of-User` ở gốc 1.5M.
## 5. Chính sách lưu giữ đề xuất (CHƯA thực hiện — chờ phê duyệt)
Chuỗi checkpoint đang hoạt động (latest.json → 2156cbadc… → 6054ad97… → 7183e8b8… → 6874e562… → a1acb3ac… → c556edfe…). Đề xuất giữ: bản đang hoạt động + 3 thế hệ gần nhất để khôi phục (≈ 4 × 54M = 216M) + các snapshot khôi phục được `training_recovery.py`/`trainer.py` tham chiếu. Khoảng 200 thế hệ còn lại (~10.8G) + `recovery/` (9.5G) là ứng viên để cắt bỏ. Cần kiểm thử resume trước khi xóa.
## 6. Còn chờ xử lý (CẦN XEM XÉT — chưa xóa)
- `final_50k_pilot/` 51M (có bộ kiểm thử riêng)
- `checkpoints_selfplay_20k/` 1.9M
- `docs/exports/` 2.3M, `docs/performance/raw-*.json` 329K (được các báo cáo benchmark tham chiếu)
- `data/az_25k_continuation_v1*` + cơ sở dữ liệu eval ~1.1G
- Các liên kết cũ ở cấp cao nhất `latest.pt` 5.7M, `buffer_latest.npz` 37M
- `data/` 3.7G (dữ liệu runtime, GIỮ LẠI)
- `lab05 - game/` (giữ lại theo chỉ dẫn)
## 7. Mục còn chưa được theo dõi
Chỉ còn `AlphaZero_Gomoku/data/` (3.7G dữ liệu runtime, được ignore bằng `*.sqlite*`). Toàn bộ mã nguồn/kiểm thử/script/tài liệu đã commit.
