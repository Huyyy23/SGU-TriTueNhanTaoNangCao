# Báo cáo benchmark thực tế Python/C++ — Gomoku AlphaZero self-play

**Ngày:** 2026-11-21 | **Môi trường:** `.venv/Scripts/python.exe` (Python 3.10.6, torch 2.14.1+cu126, numpy 2.2.6) | **GPU:** NVIDIA GeForce RTX 4060 Laptop GPU (CUDA khả dụng) | **Backend native:** `gomoku_mcts_native.cp310-win_amd64.pyd`

## 1. Mục tiêu

Benchmark hiệu năng thực tế Python vs C++ trong self-play Gomoku AlphaZero, **không sửa mã sản phẩm**, không chạy training, không đổi checkpoint/trọng số/SQLite/dữ liệu hiện có. Xác định bottleneck thật và đề xuất tối ưu có bằng chứng.

## 2. Cấu hình benchmark (giữ nguyên config sản phẩm)

| Tham số | Giá trị | Nguồn |
|---|---|---|
| Checkpoint | `checkpoints/latest.pt` (AZNet, n_blocks=6, n_filters=64) | `configs/normal.yaml` |
| n_simulations | 200 | `configs/normal.yaml` |
| parallel_games | 16 | `configs/normal.yaml` |
| temp_moves | 0 (deterministic selection) | override benchmark |
| rule_set | normal | `configs/normal.yaml` |
| games / run | 8 | workload đại diện ngắn |
| seeds | 1, 2, 3 | nhiều seed |
| repeats | 2 | nhiều lần lặp |
| Backend | python vs cpp | cùng checkpoint, cùng trọng số |

Scripts: `scripts/bench_real_world.py` (end-to-end), `scripts/investigate_phase_timing.py` (phase timing), cả hai throwaway, không commit vào pipeline chính.

## 3. Kết quả end-to-end (median / min / max / std qua 3 seeds × 2 repeats)

| Backend | wall median (s) | games/s median | games/s std | moves/s median | inference calls | mean_batch |
|---|---:|---:|---:|---:|---:|---:|
| Python | 31.408 | 0.255 | 0.029 | 7.6 | 39,495 | 7.63 |
| C++ | 12.353 | 0.648 | 0.108 | 17.3 | 39,495 | 7.63 |

**Speedup C++/Python (games/s, median): 2.54×**

Chi tiết từng run:

| backend | rep | seed | wall (s) | games/s | cpu% | gpu% |
|---|---:|---:|---:|---:|---:|---:|
| python | 0 | 1 | 24.613 | 0.325 | — | — |
| python | 0 | 2 | 32.393 | 0.247 | — | — |
| python | 0 | 3 | 33.110 | 0.242 | — | — |
| python | 1 | 1 | 30.518 | 0.262 | — | — |
| python | 1 | 2 | 32.298 | 0.248 | — | — |
| python | 1 | 3 | 29.053 | 0.275 | — | — |
| cpp | 0 | 1 | 10.407 | 0.769 | — | — |
| cpp | 0 | 2 | 15.679 | 0.510 | — | — |
| cpp | 0 | 3 | 12.324 | 0.649 | — | — |
| cpp | 1 | 1 | 10.132 | 0.790 | — | — |
| cpp | 1 | 2 | 15.447 | 0.518 | — | — |
| cpp | 1 | 3 | 12.382 | 0.646 | — | — |

- Cùng số inference calls (39,495) và mean_batch (7.63) → workload đồng nhất, chỉ khác backend MCTS.
- C++ giảm wall time từ ~31.4s xuống ~12.4s (median), ~61% nhanh hơn.
- GPU utilization đọc được qua `nvidia-smi` subprocess (không cài pynvml mới) trong bản đầu: dao động 11–45%, phụ thuộc seed/repeat (không ổn định trên workload ngắn). Bản phase timing không đọc nvidia-smi để tránh can thiệp thời gian đo.

## 4. Giải thích mâu thuẫn 2.38× vs 92–95%

Bản báo cáo trước profiling một ván đơn lẻ, batch=1 (mỗi nước một lần infer), cho thấy inference chiếm 92–95% thời gian/tick. Điều đó **không đại diện** cho end-to-end batched (`parallel_games=16`, mean_batch≈7.5).

Phase timing instrument `model.predict` với CUDA sync trên toàn bộ 24 ván (3 seed × 2 rep), kết quả:

| Thành phần | Python | C++ |
|---|---:|---:|
| wall (s), median | 31.408 | 12.353 |
| inference (s), median | 16.829 | 9.851 |
| non-inference (s), median | 14.378 | 2.502 |
| inference % of wall | 53.3% | 79.7% |
| non-inference % of wall | 46.7% | 20.3% |
| inference / call (ms) | ~2.1–2.7 | ~1.5–1.6 |
| non-inference / move (ms) | ~59–82 | ~11–12 |

**Mâu thuẫn đã được giải quyết:**
- Profiling cũ (92–95%) đo **một ván đơn lẻ, batch=1**, không có batched inference của `parallel_games=16`.
- End-to-end thực tế: inference chỉ **53.3%** (Python) và **79.7%** (C++); non-inference (MCTS request/receive + Python overhead per tick) chiếm **46.7%** (Python) và **20.3%** (C++).
- C++ giảm inference 1.71× (16.83 → 9.85s) **và** giảm non-inference 5.75× (14.38 → 2.50s). Tổng speedup wall 2.54×.
- "Missing time" trong bản đầu là non-inference Python overhead (per-tick request/receive + play/reset), không được tính trong profiling cũ.

## 5. Profiling riêng theo thành phần (một ván ngắn, 40 nước, n_simulations=200, CUDA sync)

| Thành phần | Python | C++ | Tỷ lệ Python | Tỷ lệ C++ |
|---|---:|---:|---:|---:|
| MCTS (tree) | 1.36 ms | 1.34 ms | 1.9% | 2.1% |
| Feature extraction (`board_to_planes`) | 0.168 ms | 0.137 ms | 0.2% | 0.2% |
| Inference (PyTorch CUDA, 1 leaf, synced) | 67.014 ms | 60.355 ms | 92.0% | 94.9% |
| pybind11 receive | 4.274 ms | 1.744 ms | 5.9% | 2.7% |

- Lần này là **batch=1** (khác end-to-end batched), nên tỷ lệ inference lại cao (92–95%). Đây là chi phí trên **một lần infer đơn lẻ**, không phải trên toàn bộ ván self-play batched.
- MCTS tree ~2% (không phải bottleneck trong cả hai trường hợp).

## 6. Overhead pybind11 (MCTS tree only, 200 sims, cùng board)

| Thực thi | Thời gian | Visits bằng nhau |
|---|---:|---:|
| Python MCTS tree | 464.23 ms | Yes |
| C++ MCTS tree | 274.34 ms | Yes |

**Speedup tree-only: 1.69×**, visit counts giống hệt bitwise. Đo bằng `MCTS.search`/`NativeMCTS.search` cùng evaluator `model.predict` (wrapper unpack `LeafRequest.planes` — sửa lỗi API mismatch của bản thử ban đầu, không ảnh hưởng mã sản phẩm).

## 7. Xác định bottleneck thật (end-to-end, batched)

| Thành phần | Thời gian median (s) | Tỷ lệ wall | Có nên tốiưu? | Lý do |
|---|---:|---:|---:|---|
| NN inference (PyTorch CUDA) | 16.83 (py) / 9.85 (cpp) | 53.3% / 79.7% | Có, nhưng có ràng buộc | Largest single component; SPEC §9.1 không có C++ NN engine |
| Non-inference (MCTS request/receive + Python overhead) | 14.38 (py) / 2.50 (cpp) | 46.7% / 20.3% | Có (đã giảm bằng C++ MCTS) | C++ MCTS đã giảm 5.75×, saved 11.88s (82.6%) |
| MCTS tree (bookkeeping) | ~2% | ~2% | Không | Không phải bottleneck |
| Feature extraction | ~0.2% | ~0.2% | Không | Rất rẻ |

**Kết luận:** Tốc độ self-play tổng thể bị giới hạn bởi **cả inference PyTorch CUDA (~53% Python, ~80% C++) và non-inference overhead (~47% Python, ~20% C++)**. Chuyển MCTS tree sang C++ giúp speedup wall **2.54×** (giảm non-inference 5.75× và inference 1.71×). Không có bằng chứng để kết luận MCTS tree là bottleneck (chỉ ~2%).

## 8. Speedup đáng tin cậy không?

**Có.** Cơ sở:
- Cùng checkpoint `checkpoints/latest.pt`, cùng trọng số, cùng `n_simulations=200`, cùng `parallel_games=16`.
- Cùng số inference calls (39,495) và mean_batch (7.63) → workload đồng nhất.
- `torch.cuda.synchronize()` trước/sau mỗi infer → thời gian GPU chính xác.
- 3 seeds × 2 repeats → median/std ổn định (wall std Python 2.89s, C++ 2.18s).
- C++ non-inference reducción 82.6% (11.88s saved), inference giảm 1.71×.

## 9. Đề xuất tối ưu có bằng chứng

1. **Giữ MCTS C++** (`mcts_backend="cpp"`): đã đồng nhất arena với tournament, giảm non-inference overhead 5.75×, speedup wall 2.54×, không đổi thuật toán/trọng số.
2. **Không chuyển PyTorch inference sang C++**: SPEC §9.1 xác định không có C++ NN engine; inference là thành phần lớn nhất (~53–80% wall), chuyển nó sẽ đổi thuật toán/stack, ngoài phạm vi và rủi ro cao.
3. **Không tốiưu MCTS tree thêm**: chỉ ~2% chi phí, không có bằng chứng bottleneck.
4. **Nếu muốn tăng games/s hơn nữa**: cần xử lý phía inference (batch lớn hơn, nhiều GPU stream, hoặc giảm `n_simulations` — nhưng SPEC §7.1 cấm điều chỉnh ngân sách 200 simulations). Đây là quyết định nghiên cứu, không phải tốiưu code hiện tại.

## 10. Giới hạn

- Chạy trên 1 máy local (Windows), 1 GPU RTX 4060 Laptop, workload ngắn (8 ván/run, 24 ván tổng cộng/Backend).
- GPU utilization đọc được qua `nvidia-smi` nhưng dao động cao trên workload ngắn, không ổn định đủ để kết luận về tầm saturate GPU.
- Không chạy training dài, không thay đổi checkpoint/trọng số/SQLite/dữ liệu, không commit.
- Lỗi API mismatch ban đầu nằm ở script thử tree-only (truyền trực tiếp `req` LeafRequest vào `MCTS.search`), đã sửa bằng wrapper unpack `.planes`; không ảnh hưởng mã sản phẩm.

## 11. Điều không làm

- Không chạy training dài.
- Không xóa checkpoint, SQLite, dữ liệu self-play.
- Không commit.
- Không chuyển PyTorch inference sang C++.
- Không thay đổi thuật toán AlphaZero.
- Không sửa mã sản phẩm; chỉ sửa script benchmark throwaway.
