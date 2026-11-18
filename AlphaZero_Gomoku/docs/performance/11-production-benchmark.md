# 11 — Benchmark phục vụ vận hành thực tế (vòng hai)

## Quy trình đo

Mỗi phép đo dùng `checkpoints/last.pt` SHA-256 `6e29a4e3b8c235449898450de0f44b520829eda097ee74f97ef599238e52ffbc`, CUDA RTX 4060 Laptop GPU, Python 3.10.6/PyTorch 2.14.1+cu126, Normal rules, 200 simulations/move. Mỗi backend/cấu hình có 3 lần đo, xen kẽ thứ tự; warm-up mỗi backend trước matrix. Cấu hình p4 chạy đúng 4 game, p16 đúng 16 game; raw JSON chứa số game, moves/game và NN calls/game nên không lặp lỗi một game giả danh 4-way parallel. Baseline và sau sửa có cùng seeds định sẵn, số moves/NN calls trùng nhau theo từng seed/backend.

Raw data: `raw-round2-baseline.json`, `raw-round2-after.json`, `raw-round2-workers.json`, `raw-round2-workers-saturated.json`, `raw-round2-workers-training-iteration.json`, `raw-round2-profile-before.json`, `raw-round2-profile-after.json`, `raw-round2-auxiliary.json`. Baseline JSON không lưu seed từng dòng; seed khôi phục theo lệnh (`20261118 + repeat*100 + parallel_games`) và thứ tự cấu hình. JSON sau tối ưu lưu seed trực tiếp. Matrix chính và batch 50 game có 3 lần đo/config; saturation 16 game/worker có một lần/config. Các lần đo chạy tuần tự. Median giảm ảnh hưởng biến thiên nhưng không thay thế soak dài.

## Self-play throughput theo số game song song

Số là median của 3 lần đo; simulations/s = moves × 200 / wall (nominal budget, không phải số NN calls). Peak RSS cộng process tree; GPU memory là VRAM sampled từ `nvidia-smi`.

| Backend MCTS | parallel_games | G/min | moves/s | nominal sims/s | NN calls/run | inference batch mean (max) | RSS peak | VRAM peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Python baseline | 4 | 17.23 | 6.39 | 1,278 | 16,288 | 3.40 (4) | 1.14 GB | 1,166 MiB |
| C++ baseline | 4 | 18.66 | 6.78 | 1,357 | 16,288 | 3.40 (4) | 1.17 GB | 1,166 MiB |
| Python sau | 4 | 16.72 | 6.20 | 1,240 | 16,288 | 3.40 (4) | 1.04 GB | 1,131 MiB |
| C++ sau | 4 | 27.13 | 10.06 | 2,012 | 16,288 | 3.40 (4) | 1.03 GB | 1,131 MiB |
| Python baseline | 16 | 21.13 | 8.36 | 1,673 | 71,434 | 8.60 (16) | 1.15 GB | 1,148 MiB |
| C++ baseline | 16 | 24.71 | 9.60 | 1,921 | 71,434 | 8.60 (16) | 1.20 GB | 1,166 MiB |
| Python sau | 16 | 22.60 | 8.85 | 1,771 | 71,434 | 8.60 (16) | 1.05 GB | 1,131 MiB |
| C++ sau | 16 | 60.31 | 23.62 | 4,725 | 71,434 | 8.60 (16) | 1.08 GB | 1,131 MiB |

Inference latency p50/p95 của inference batch (ms): baseline p4 Python 1.47/2.49, C++ 1.56/2.57; sau p4 Python 1.38/2.26, C++ 1.32/2.13. Baseline p16 Python 2.51/5.36, C++ 2.07/3.90; sau p16 Python 2.35/4.83, C++ 1.37/2.10. Latency này là từng `model.predict(batch)` call, không phải từng board.

## So sánh tốc độ theo cặp cùng seed

So cùng checkpoint, game count, seed và trajectory:

- Tối ưu native C++ so với C++ trước đó: median throughput ratio **1.64× ở p4** (3 paired time ratios 1.644×, 1.454×, 1.774×) và **2.44× ở p16** (2.636×, 2.136×, 2.441×). Tương ứng wall giảm khoảng 39% và 59% cho workload paired.
- Sau tối ưu, C++ so với Python: **1.62× ở p4** và **2.67× ở p16** median paired wall-time ratio. C++ p16 đạt 60.31 games/min trên matrix 16 game.
- Python không có code change; p4 baseline/sau gần ngang (median throughput giảm ~3%), p16 tăng ~7%. Dao động 3 mẫu/seed cho thấy không nên gán thay đổi nhỏ cho code optimization.
- NN-call counts không đổi giữa backend và qua tối ưu cho cùng seed; profile cũng ghi cùng visit distribution trong differential tests. Đây không phải playing-strength arena.

## Mở rộng theo số worker (16 game tổng, 200 sims/move)

Ba lần đo/backend/worker count; mỗi worker group chạy độc lập, warm-up trước đo. `record_mean_batch` là batch mean lưu bởi từng GameRecord. Child worker không được instrument từng call inference, vì vậy batch max và inference p50/p95 **không có** cho worker 2/4 (raw cũ dùng 0 làm sentinel cho trường không thu thập; không diễn giải là batch=0). NN calls được cộng từ records.

| MCTS | Workers | G/min median | moves/s | nominal sims/s | record mean batch | RSS peak | VRAM peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| Python | 1 | 22.57 | 8.09 | 1,617 | 14.47 | 1.14 GB | 1,116 MiB |
| C++ | 1 | 60.34 | 21.62 | 4,324 | 14.47 | 1.15 GB | 1,116 MiB |
| Python | 2 | 42.95 | 15.52 | 3,105 | 7.23 | 3.24 GB | 1,379 MiB |
| C++ | 2 | **73.27** | **26.48** | **5,297** | 7.23 | 3.25 GB | 1,379 MiB |
| Python | 4 | 42.82 | 18.29 | 3,659 | 3.60 | 5.22 GB | 1,628 MiB |
| C++ | 4 | 53.02 | 23.24 | 4,648 | 3.60 | 5.23 GB | 1,628 MiB |

Cấu hình 2 workers nhanh nhất trong workload 16 game tổng, nhưng mỗi tiến trình con chỉ xử lý 8 game; đây là benchmark giữ cố định tổng workload, không làm bão hòa worker. Phép đo bổ sung 16 game đồng thời mỗi worker cho thấy throughput tăng theo tổng số worker; batch train 50 game được đo riêng phía dưới.

## Đo bão hòa: 16 game đồng thời mỗi worker

Một lượt/backend/cấu hình; tổng workload lần lượt 16/32/64 game, mỗi worker nhận đúng 16 game. Raw: `raw-round2-workers-saturated.json`. Đây đo throughput ở batch lớn, không phải kích thước một iteration hiện tại.

| MCTS | Workers | Games | G/min | moves/s | nominal sims/s | record mean batch | RSS peak | VRAM peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Python | 1 | 16 | 24.91 | 8.82 | 1,764 | 14.47 | 1.12 GB | 1,104 MiB |
| C++ | 1 | 16 | 66.03 | 23.39 | 4,677 | 14.47 | 1.16 GB | 1,104 MiB |
| Python | 2 | 32 | 46.97 | 18.59 | 3,719 | 14.20 | 3.36 GB | 1,371 MiB |
| C++ | 2 | 32 | 100.68 | 39.85 | 7,970 | 14.20 | 3.37 GB | 1,371 MiB |
| Python | 4 | 64 | 86.68 | 32.69 | 6,537 | 14.16 | 5.59 GB | 1,637 MiB |
| C++ | 4 | 64 | **156.73** | **59.10** | **11,820** | 14.16 | 5.60 GB | 1,637 MiB |

## Một iteration huấn luyện thực tế: đúng 50 game/lần gọi

Chạy đúng game count mặc định mỗi iteration (`games_per_iter=50`), 200 sims/move; 3 lần/config, seed cố định và thứ tự backend luân phiên. `run_games_parallel()` chia 25 game/worker với 2 workers, 13/13/12/12 với 4 workers; tiến trình đơn giữ `parallel_games=16`. Raw: `raw-round2-workers-training-iteration.json`. Mean batch lấy từ GameRecord; chưa instrument percentile batch trong child.

| MCTS | Workers | Active games/proc | G/min median | moves/s | nominal sims/s | NN calls/run | mean batch | RSS peak | VRAM peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Python | 1 | 16 | 22.91 | 8.52 | 1,704 | 211,996 | 15.68 | 1.14 GB | 1,107 MiB |
| C++ | 1 | 16 | 86.51 | 31.21 | 6,241 | 211,996 | 15.68 | 1.17 GB | 1,107 MiB |
| Python | 2 | 25 | 54.08 | 20.87 | 4,175 | 219,880 | 21.87 | 3.34 GB | 1,373 MiB |
| C++ | 2 | 25 | **129.59** | **49.72** | **9,944** | 219,880 | 21.87 | 3.36 GB | 1,373 MiB |
| Python | 4 | 13 max | 80.05 | 29.43 | 5,886 | 211,368 | 11.10 | 5.41 GB | 1,632 MiB |
| C++ | 4 | 13 max | 124.66 | 46.04 | 9,208 | 211,368 | 11.10 | 5.42 GB | 1,632 MiB |

Ở tiến trình đơn, inference p50/p95 C++ là 1.44/2.28 ms so với Python 3.69/6.44 ms; worker child chưa có percentile riêng. Hai workers có median nhanh nhất ở batch 50 game và dùng ít RSS/VRAM hơn bốn workers. Startup + warm-up của hai workers mất 5.31 s, bốn workers 4.91 s; `train.py` tái sử dụng pool qua các iteration nên đây là chi phí ban đầu.

Với 16 game đồng thời/worker, bốn workers đạt 156.73 games/min nhưng dùng 5.60 GB RSS process-tree. Batch train=50 chia thành 25 game/worker ở hai workers, cho batch mean 21.87 và throughput cao hơn bốn workers (batch mean 11.10). Vì vậy cấu hình bão hòa tối đa không đồng nghĩa throughput tốt nhất ở batch huấn luyện hiện tại.

## Giới hạn

Không chạy full tournament round 2, controlled strength arena hay memory soak hàng giờ; các game batches chỉ đại diện checkpoint/seed hiện tại, không bảo đảm game distribution sau nhiều training iterations. `nvidia-smi`/CPU utilization lấy mẫu, không phải profiler hardware; worker inference latency p50/p95 không được thu thập. Thời gian logger/replay microbench không cộng vào self-play wall. Full test suite và GPU train/resume smoke được ghi ở báo cáo 12.

