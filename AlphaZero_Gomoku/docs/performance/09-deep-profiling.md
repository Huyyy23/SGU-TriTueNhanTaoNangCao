# 09 — Profiling sâu self-play

## Môi trường và cách đo

- Windows 11 x64, Python 3.10.6, PyTorch 2.14.1+cu126, RTX 4060 Laptop GPU; 16 logical CPU, RAM 16.94 GB.
- Checkpoint `checkpoints/last.pt`, SHA-256 `6e29a4e3b8c235449898450de0f44b520829eda097ee74f97ef599238e52ffbc`.
- Profile workload: Normal, 4 game instances hoạt động song song, 200 simulations/move, seed 230000, 111 moves, 22,200 nominal simulations, 21,149 neural calls; cùng trajectories giữa backend. `raw-round2-profile-before.json` và `raw-round2-profile-after.json` giữ toàn bộ output.
- cProfile làm tăng đáng kể wall time. Các số cProfile dùng phân bổ hotspot, không dùng làm throughput benchmark. Bộ timer dùng `perf_counter_ns`, ghi inclusive/exclusive riêng và p50/p95 cho mỗi lần gọi. Parent + child RSS và GPU được lấy mẫu. Các span lồng nhau không được cộng; tỷ lệ dưới đây so từng inclusive span với wall time, nên không phải phân hoạch thời gian độc lập.

## Wall time của profile có instrumentation

| Backend / trạng thái | Wall | NN predict inclusive | Tỷ trọng wall | MCTS receive inclusive | Tỷ trọng wall |
|---|---:|---:|---:|---:|---:|
| Python, trước | 25.099 s | 14.872 s | 59.3% | `MCTS.receive` 4.873 s | 19.4% |
| C++, trước | 21.358 s | 13.464 s | 63.0% | adapter `NativeMCTS.receive` 5.556 s | 26.0% |
| Python, sau | 24.307 s | 13.920 s | 57.3% | `MCTS.receive` 5.012 s | 20.6% |
| C++, sau | 14.821 s | 11.671 s | 78.7% | adapter `NativeMCTS.receive` 0.876 s | 5.9% |

Phần lớn `predict` nằm trong model forward và chuyển tensor CPU/GPU. Trong profile sau, 6,709 lần Python `predict` chiếm 13.920 s, trong đó `AZNet.forward` 9.249 s, `.cpu()` 2.275 s, `.to()` 1.840 s. C++ tương ứng `predict` 11.671 s, forward 8.572 s, `.cpu()` 1.742 s, `.to()` 0.829 s. C++ MCTS vẫn gọi cùng Python inference; không có inference native.

## MCTS và boundary C++/Python

Timer C++ đo trực tiếp method pybind, nên tách được arena native khỏi adapter Python:

| Span / trạng thái | Calls | Inclusive tổng | Exclusive tổng | p50 | p95 |
|---|---:|---:|---:|---:|---:|
| `pybind.Tree.receive`, trước | 21,149 | 4,822.2 ms | 4,822.2 ms | 0.2492 ms | 0.4657 ms |
| `NativeMCTS.receive`, trước | 21,149 | 5,556.2 ms | 734.0 ms ngoài span con | 0.2826 ms | 0.5061 ms |
| `pybind.Tree.receive`, sau | 21,149 | 269.0 ms | 269.0 ms | 0.0076 ms | 0.0188 ms |
| `NativeMCTS.receive`, sau | 21,149 | 875.8 ms | 606.8 ms ngoài span con | 0.0301 ms | 0.0709 ms |
| `pybind.Tree.request`, trước → sau | 21,458 | 271.8 → 258.8 ms | 138.0 → 132.6 ms | 0.0110 → 0.0109 ms | 0.0239 → 0.0209 ms |
| Python tie callback, trước → sau | 43,272 | 133.8 → 126.2 ms | body only | 0.0020 → 0.0021 ms | 0.0088 → 0.0070 ms |
| `NativeMCTS.request`, trước → sau | 21,260 | 1,967.3 → 1,930.0 ms | 575.7 → 567.1 ms | 0.0841 → 0.0849 ms | 0.1381 → 0.1256 ms |

Đây là bằng chứng định lượng root cause: `Tree.receive` trước đó mất 22.6% wall profile; Python callback thân chỉ khoảng 0.6%. Boundary/callback có chi phí nhưng không giải thích hàng giây bị mất. Sau khi sửa arena growth, `Tree.receive` còn khoảng 1.8% wall profile; bottleneck chuyển sang model inference.

## Board, luật chơi và feature generation

| Thành phần | Python trước (calls; tổng; p50/p95) | C++ trước | C++ sau |
|---|---|---|---|
| `Board.play` | 43,383; 82.6 ms; 0.0017/0.0029 ms | 42,379; 77.8 ms; 0.0016/0.0034 ms | 42,379; 75.6 ms; 0.0016/0.0030 ms |
| `Board.undo` | 43,272; 65.6 ms; 0.0014/0.0020 ms | 42,268; 63.6 ms; 0.0014/0.0021 ms | 42,268; 63.0 ms; 0.0014/0.0020 ms |
| Normal `is_win` | 42,920; 414.5 ms; 0.0070/0.0110 ms | 42,920; 323.0 ms; 0.0068/0.0114 ms | 42,920; 318.5 ms; 0.0068/0.0108 ms |
| Normal `legal_moves` | 21,149; 147.4 ms; 0.0066/0.0085 ms | 21,149; 149.6 ms; 0.0066/0.0093 ms | 21,149; 145.6 ms; 0.0065/0.0082 ms |
| Feature planes | 21,149; 145.9 ms; 0.0061/0.0093 ms | 21,149; 142.2 ms; 0.0059/0.0094 ms | 21,149; 141.1 ms; 0.0060/0.0086 ms |

Trong workload Normal đã đo, thao tác board, sinh nước hợp lệ và tạo feature đều thấp hơn đáng kể so với inference và expansion native. Số liệu không ủng hộ việc viết lại toàn bộ board/rules bằng C++. Renju có test correctness, nhưng profile self-play này chỉ dùng Normal; không suy rộng chi phí luật Renju từ kết quả đó.

## Các thành phần phụ và giới hạn

- Self-play Python tạo một sample mỗi nước đi; worker serializes records/samples qua IPC. Profile một process không đi qua replay buffer/IPC và tắt logging. `raw-round2-workers.json` đo throughput/RSS thực với worker; `raw-round2-auxiliary.json` chứa microbenchmark riêng cho replay/logger/Renju, không phải tỷ lệ trong self-play.
- Allocation hotspot của Python MCTS: cProfile ghi nhận 2,233,110 lần `az.mcts.Node.__init__`, 2.301 s exclusive (9.5% wall profile có instrumentation). Đây là các child được tạo cho legal moves; C++ tree không tạo Python `Node` cho arena. Không lưu p50/p95 từng lần vì hơn 2.2 triệu mẫu sẽ làm profile tăng mạnh bộ nhớ.
- `Board.copy` không được gọi trong self-play profile này; MCTS traversal dùng `Board.play/undo` (đã đo ở bảng trên), không copy board ở mỗi simulation.

- Microbenchmark phụ trợ (`raw-round2-auxiliary.json`, workload tổng hợp): Renju forbidden-cache miss 24 lần, tổng 426.0 ms, p50/p95 17.65/23.09 ms; cache hit 0.0015/0.0018 ms; `legal_moves` 0.0141/0.0188 ms; `is_win` 0.0514/0.0633 ms. Các board 12 quân là dữ liệu tổng hợp, không phải profile một ván Renju hoàn chỉnh.
- `ReplayBuffer.add`: 1,000 lần, tổng 1.44 ms, p50/p95 0.0014/0.0015 ms. `GameLogger.log_game` có fsync: 48 lần, tổng 73.3 ms, p50/p95 1.22/1.53 ms. `train.py` ghi log sau self-play, vì vậy các giá trị này không nằm trong wall time `run_games`.
- Worker IPC nằm trong wall time các file `raw-round2-workers*.json`. Với 50 game/iteration, 2 workers đạt median C++ 129.59 games/min, cao hơn 1 worker (86.51) và 4 workers (124.66); RSS process-tree 3.36 GB so với 5.42 GB ở bốn worker. Batch total16/per-worker16 dùng workload khác, có trong raw riêng. Child workers chưa instrument latency từng inference call và batch max; chỉ có mean batch từ GameRecord và tổng NN calls.
- Batch size: 4 game hoạt động trung bình 3.15 sample/lần inference (max 4); 16 game trung bình 7.69–9.60 (max 16) ở matrix. Đây là dynamic batching giữa các game, không phải batch search bên trong một ván.
- CPU/GPU utilization là số lấy mẫu thưa, không phải profiling ở cấp phần cứng. VRAM peak lấy từ `nvidia-smi`; CUDA allocated/reserved của process cha không bao gồm toàn bộ driver/process usage.
- Các timer không tách riêng mọi pybind conversion/allocation hoặc toàn bộ CUDA kernel time; không chạy sanitizer hay hardware GPU profiler. Raw JSON có function counts và cProfile output đầy đủ.
