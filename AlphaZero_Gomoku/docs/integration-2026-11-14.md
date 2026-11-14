# Tích hợp T31/T32 và kết quả T35 — 2026-11-14

## Phạm vi

Branch: `integration/t35-upstream-20261114`.

- Giữ commit local `7d7d267` chứa triển khai đánh giá và báo cáo T35; tích hợp upstream `e64c6f33df446c170e76eaa56de8fb4a33fa71c3`.
- Factory giữ validation nghiêm ngặt cho simulations/deadline, hỗ trợ ba mốc `az_v0`, `az_v1_tactical`, `az_v1_threat` dùng chung checkpoint.
- Tournament dùng một scheduler/backend cho CLI và các hợp đồng ablation upstream. Giữ budget sims/time, agent spec `@...`, nhiều worker, khai cuộc chung, thống kê theo màu, Wilson CI, timing, tactical/threat telemetry.
- Thêm resume append-only, shard theo cặp, gộp từ raw JSONL, checksum checkpoint và hai seed scheme. `az_v1` bị từ chối vì alias mơ hồ; dùng tên tactical/threat rõ ràng.
- Giữ nguyên toàn bộ ablation upstream: pruning on/off, MCTS nhiều rollout budget, JSON và plot. Bổ sung kiểm thử tích hợp.
- Counter không tồn tại giữ `null`, không chuyển thành 0; sửa lỗi `int(None)` phát hiện bằng smoke AlphaZero thật.

## Bằng chứng kiểm chứng

| Kiểm chứng | Kết quả đã quan sát |
|---|---|
| `python -m pytest -q tests` trước bản vá nullable counter cuối | 234 passed, 2 skipped |
| `python -m pytest -q tests/test_tournament.py tests/test_ablation.py` sau bản vá cuối | 37 passed |
| Factory, arena, CLI, UI contract, AlphaBeta, MCTS, evaluator, logging, tactical, threat tests | 69 passed |
| Tournament thật `sims:2`, random/MCTS | 2 ván hoàn tất |
| Tournament thật `time:0.01`, random/AlphaBeta/MCTS | 6 ván hoàn tất |
| Resume giải đã hoàn tất | 0 ván chơi lại |
| Ba shard thật, 2 ván/cặp, rồi merge | 6 ván gộp |
| Ba mốc AZ thật, cùng `checkpoints/last.pt`, `sims:2` | 6 ván hoàn tất; checkpoint không đổi |
| Ablation AlphaBeta depth 1, một vị trí | JSON và PNG tạo thành công |
| Ablation MCTS budgets 1/2, hai ván/mức | JSON và PNG tạo thành công |
| SHA-256 tập dữ liệu bảo vệ | 1.216 file không đổi hoặc mất |
| Graphify AST update | Hoàn tất sau bản vá cuối |

Smoke nằm trong `logs/tournament/integration_*_20261114`, `logs/integration_shards_20261114`, `logs/integration_merged_20261114`, `logs/ablation/integration_*_20261114.*`. Các file này giữ local, không đưa vào commit.

Backup nằm ngoài thư mục package: `../gomoku_integration_backup_20261114_173341/`, gồm bản sao file, patch và inventory checksum. Backup ban đầu nằm trong package gây pytest import mismatch; đã chuyển ra ngoài, không xóa dữ liệu.

## Giới hạn và rủi ro

- Không chạy lại 2.400 ván T35, không train/fine-tune, không sửa checkpoint hoặc bằng chứng lịch sử. Smoke chỉ chứng minh pipeline, không chứng minh sức chơi.
- Deadline là mềm; một rollout hoặc thao tác đang chạy có thể vượt ngân sách.
- Log lịch sử `manifest.json` vẫn đọc/gộp được, nhưng runner mới không resume trực tiếp vào đó để tránh trộn phiên bản thuật toán. Resume mới dùng `config.json`.
- Hai test skip giữ nguyên; không tuyên bố đã kiểm chứng phần bị skip. Không mở Pygame thủ công; UI chỉ qua contract tests.
- Graphify cảnh báo 366 file không có extractor hỗ trợ; graph chỉ là công cụ định hướng.
- `evidence/`, log/checkpoint local và file untracked ngoài phạm vi không được tự động stage/push. Báo cáo T35 vẫn tham chiếu bằng chứng local.
