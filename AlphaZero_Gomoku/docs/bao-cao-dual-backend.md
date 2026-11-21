# Báo cáo triển khai hai backend Python/C++

## Kết luận

Native MCTS C++17/pybind11 đã chạy trong self-play và tournament. C++ sở hữu cây bằng arena `std::vector<Node>`; cha/con dùng index, không có Python-visible node hay vòng sở hữu. Python tiếp tục quản lý board, rules, RNG, feature generation và neural inference. Python vẫn là mặc định.

Đúng môi trường project venv: full suite CPU **260 passed, 3 skipped**; Native MCTS + Native Patterns build CPython 3.10; native self-play spawn, training/checkpoint/resume CPU và tournament CLI thật đã chạy. Benchmark bốn tổ hợp đã chạy lại sau khi sửa lỗi propagation Patterns. Trong workload ngắn này, Native MCTS thay đổi tối đa khoảng +0,6%, thấp hơn ngưỡng 10%; chưa khuyến nghị production hoặc 5.000 ván.

## Trạng thái theo gate

| Gate | Trạng thái | Bằng chứng / phần còn thiếu |
|---|---|---|
| 1 — Audit/hợp đồng | Đạt | Giữ request/receive, PUCT, prior, tie-breaking RNG, backup/sign, root noise, visit counts, temperature và terminal. Rules/inference chạy trong Python; Normal/Renju differential đạt. |
| 2 — Native safety | Một phần | Arena/index ownership và `reset()` giải phóng vùng cây; lifecycle tests đạt. Chưa chạy sanitizer hoặc RSS soak dài hạn. |
| 3 — Backend integration | Smoke đạt | Native self-play ProcessPool spawn; CPU training 2 workers; checkpoint/resume; tournament CLI thật với checkpoint và 2 workers. GPU training tiny smoke lỗi ở optimizer. |
| 4 — Correctness/stability | Một phần | Full suite CPU: **260 passed, 3 skipped**. Chưa xác nhận GPU training, sanitizer hay soak memory dài hạn. |
| 5 — Benchmark | Đo thử bốn tổ hợp; chưa nghiệm thu production | Tournament CPU, 3 lượt xen kẽ/config, mỗi lượt 2 ván, 2 simulations/move, AlphaBeta depth 1. MCTS C++ chưa tăng 10%; workload ngắn, không đại diện budget vận hành. |
| 6 — Training quality | Chưa đạt | Training smoke chỉ kiểm tra pipeline/checkpoint; chưa có arena/strength evaluation có kiểm soát. |

## Thay đổi chính

- Thêm `native/gomoku_mcts_native.cpp`: PUCT, expansion, terminal/nonterminal backup, visit counts và arena theo index.
- Sửa `scripts/build_gomoku_native.py`: `--module patterns|mcts`.
- Sửa `az/mcts.py`, `az/backend.py`, `az/selfplay.py`: Native adapter, backend fail-closed, self-play và worker propagation.
- Sửa `players/az_player.py`, `players/factory.py`, `scripts/tournament.py`: chọn MCTS/Patterns độc lập, truyền cả hai selectors vào AZ player/workers, ghi manifest/game metadata. Sửa lỗi factory từng reset Patterns selection khi tạo AZ player.
- Sửa `scripts/train.py`: truyền backend tới workers, ghi backend đã chọn và `components_used` vào training JSONL.
- Thêm/mở rộng `tests/test_backend.py`, `tests/test_native_mcts.py`, `tests/test_agent_factory.py`, `tests/test_tournament.py`.

## Môi trường, build và test

Interpreter đúng: `.venv/Scripts/python.exe` trong `AlphaZero_Gomoku`, CPython 3.10.6, PyTorch 2.14.1+cu126, CUDA 12.6/RTX 4060, NumPy 2.2.6, pytest 9.1.1, PyYAML 6.0.3, psutil 7.2.2. Package `pybind11` không cài; build dùng headers của PyTorch. Trước đó đã kiểm tra nhầm Python 3.15 toàn cục và venv ở thư mục cha.

```powershell
.venv\Scripts\python.exe scripts\build_gomoku_native.py --module mcts --cxx 'D:\Jetbrain tools\CLion 2025.1.1\bin\mingw\bin\g++.exe'
.venv\Scripts\python.exe scripts\build_gomoku_native.py --module patterns --cxx 'D:\Jetbrain tools\CLion 2025.1.1\bin\mingw\bin\g++.exe'
CUDA_VISIBLE_DEVICES=-1 .venv\Scripts\python.exe -m pytest -q
```

Build/import thành công `gomoku_mcts_native.cp310-win_amd64.pyd` và `gomoku_native.cp310-win_amd64.pyd`. Full suite CPU-only đạt **260 passed, 3 skipped in 31.90s**. GPU tiny training chạy self-play/inference nhưng optimizer lỗi `CUBLAS_STATUS_EXECUTION_FAILED`/allocation; GPU training chưa đạt.

## Smoke integration đã chạy

- Native self-play ProcessPool spawn: 2 workers/2 games; ReplayBuffer sample count khớp moves.
- Native training CPU: 2 workers, 2 games, 2 simulations/move, 1 optimizer step, model 1 block/8 filters. Tạo `last.pt` và `buffer_last.npz`; resume hoàn tất iteration 2, log giữ backend `cpp`.
- Tournament CLI với `checkpoints/last.pt`: AZ+Native MCTS đấu AlphaBeta, 2 games, MCTS/Patterns C++; chạy `--workers 1` và `--workers 2`. Manifest/game records ghi đúng backend/component usage.
- Tournament resume giữ nguyên manifest/log và không chạy lại games đã lưu.
- Full tests bao gồm differential Normal/Renju, exact-five, overline, terminal, RNG/noise, visit/backup, worker samples và lifecycle.

## Benchmark bốn cấu hình

Tournament game-task trên CPU: `CUDA_VISIBLE_DEVICES=-1`, Torch một thread; checkpoint `checkpoints/last.pt`, SHA-256 `6e29a4e3b8c235449898450de0f44b520829eda097ee74f97ef599238e52ffbc`; Normal; AZ 2 simulations/move; AlphaBeta depth 1; seeds 900/901/902; 2 color-swapped games/lượt; warm-up 2 games/config; ba lượt interleaved/config; workers=1. Đo wall quanh `run_tournament` (preflight + game tasks), RSS peak lấy mẫu 20 ms, inference latency wrapper quanh `model.predict`. CUDA tắt nên VRAM 0. Mỗi lượt ra cùng 71 moves, 70 simulations và 70 inference calls; không phải quality test.

Median ba lượt:

| MCTS | Patterns | Games/min | Moves/s | Sims/s | Wall s/2 games | CPU % một lõi | RSS peak MiB | Inference ms/call | VRAM MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Python | Python | 7,23 | 4,279 | 4,218 | 16,594 | 29,64 | 548,38 | 2,095 | 0 |
| Python | C++ | 34,48 | 20,403 | 20,115 | 3,480 | 30,98 | 547,22 | 2,119 | 0 |
| C++ | Python | 7,22 | 4,273 | 4,213 | 16,614 | 27,73 | 546,59 | 2,097 | 0 |
| C++ | C++ | 34,68 | 20,521 | 20,232 | 3,460 | 40,45 | 548,41 | 2,106 | 0 |

Trong workload depth-1 này, Patterns C++ tăng moves/s median khoảng **4,77×–4,80×**. MCTS C++ thay đổi khoảng **−0,1%** với Patterns Python và **+0,6%** với Patterns C++; không đạt ngưỡng +10%, do đó vẫn tùy chọn và không thành mặc định. Patterns result giới hạn ở workload CPU/tournament ngắn này; self-play AZ chuẩn không gọi `find_patterns()`.

Benchmark bốn-way đầu tiên không hợp lệ: tạo AZ player đã reset Patterns selector về Python. Đã sửa `players/factory.py`/tournament propagation và chạy lại toàn bộ bảng trên. Không dùng số liệu lượt đầu.

## Lệnh sử dụng

```powershell
.venv\Scripts\python.exe scripts\build_gomoku_native.py --module mcts --cxx 'D:\Jetbrain tools\CLion 2025.1.1\bin\mingw\bin\g++.exe'
.venv\Scripts\python.exe scripts\train.py --backend cpp --patterns-backend python --workers 2
.venv\Scripts\python.exe scripts\tournament.py --agents az,alphabeta --backend cpp --patterns-backend cpp --games 2 --budget sims:2 --workers 2
```

Python mặc định. Ép chọn C++ nhưng thiếu extension/không tương thích sẽ lỗi rõ, không fallback. Tournament ghi backend đã chọn và `components_used`; Patterns ghi `not_used` trong self-play AZ chuẩn.

## Chưa nghiệm thu

1. GPU training chưa qua optimizer smoke; CPU training/checkpoint/resume đã chạy.
2. Chưa có RSS soak dài hạn hoặc sanitizer.
3. Four-way benchmark còn quá nhỏ để production acceptance; Native MCTS chưa đạt +10% end-to-end.
4. Chưa đánh giá training quality bằng arena có kiểm soát.
5. Chưa khuyến nghị backend/worker cho 5.000 ván. Giữ Python mặc định tới khi stability, workload đại diện và quality gate đạt.
