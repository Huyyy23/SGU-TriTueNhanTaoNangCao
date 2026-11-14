# Mốc hiệu năng

Số đo trên máy RTX 4060 Laptop 8 GB, driver 610.88, torch 2.14.1+cu126, Python 3.10.
Mỗi số là trung bình 100 lượt gọi `AZNet.predict` sau 20 lượt khởi động, `torch.cuda.synchronize()` trước và sau khi đo. Mạng 6 block × 64 filter, đầu vào `(B,4,11,11)` uint8.

| Giai đoạn | Thiết bị | Batch | ms/lượt | Ghi chú |
|---|---|---|---|---|
| GĐ1 | CUDA | 1 | 1.58 | chế độ tất định (GĐ1–GĐ4) |
| GĐ1 | CUDA | 16 | 1.65 | 16 lá gần như miễn phí so với 1 lá |
| GĐ1 | CPU | 1 | 1.62 | |
| GĐ1 | CPU | 16 | 6.66 | |

Đọc bảng này khi làm GĐ5: nếu `mean_batch` tăng mà ms/lượt gần như không tăng (như 1 → 16 ở trên), gom lô là đúng hướng; nếu song song hoá chỉ làm `nn_calls` tăng thì đang eval thừa.

`pytest tests/test_model.py` in ra `2.02 ms/batch` cho batch 16 — số đó gồm cả chi phí khởi tạo numpy/tensor trong test, nên dùng bảng trên làm mốc, không dùng số của test.

Máy không có GPU NVIDIA: dùng các dòng CPU. Một lá tốn ~0.42 ms khi gom 16 (so với ~0.10 ms trên GPU), tức toàn bộ ngân sách 1000 ván tự chơi chạy trong vài giờ — chậm hơn nhưng không phải rào cản.

## GĐ3 — self-play chế độ tất định

Một ván tự chơi trọn vẹn, `n_simulations: 200`, evaluator giả (không torch), board + luật Normal thật, seed 0:

| Số nước | Giây/ván | Giây/20 nước | `nn_calls` | `nn_calls`/nước |
|---|---|---|---|---|
| 56 | 4.08 | 1.46 | 11007 | 196.6 |

`nn_calls`/nước ≈ 200 đúng như kỳ vọng (chênh lệch là các lượt mô phỏng kết thúc ở node thắng, không gọi mạng). Entropy của π ở nước đầu là 4.55, tức π còn mềm, không bị one-hot hoá sớm. Đây là mốc so sánh cho GĐ5: cùng một ván ở chế độ gom lô phải có `nn_calls` tương đương và thời gian thấp hơn.

## GĐ4 — trainer end-to-end (batch = 1)

Mạng 6 block × 64 filter, `configs/normal.yaml` với `--games 10 --simulations 100`, `train_steps_per_iter: 200`, `batch_size: 128`:

| Vòng | Giây tự chơi | Giây train | Mẫu trong buffer | `loss` | `loss_eval_on_fixed_batch` |
|---|---|---|---|---|---|
| 1 | 200.5 | 56.6 | 416 | 4.808 | 4.722 |
| 2 | 176.0 | 35.3 | 878 | 4.726 | 4.759 |
| 3 | 212.7 | 44.2 | 1263 | 4.563 | 4.752 |
| 4 | 273.7 | 34.9 | 1750 | 4.598 | 4.754 |

Đọc bảng này: 200 bước tối ưu mỗi vòng là quá ít để `loss_eval_on_fixed_batch` giảm rõ — nó đi ngang trong khoảng ±0,04 (0,8%). π mục tiêu ở 100 lượt mô phỏng vẫn gần đều (entropy trung bình 3,75 trên tập eval), nên loss còn cách sàn lý thuyết ~1 nat; muốn thấy đường cong đi xuống phải chạy `n_simulations: 200` với đủ `games_per_iter: 50`. `nn_calls`/ván ≈ 4081–4787 cho ván 39–49 nước, tức ≈ 98–105 lượt suy luận mỗi nước ở 100 lượt mô phỏng, khớp kỳ vọng.

## GĐ5 — gom lô

`scripts/bench_batching.py --games 16 --parallel 1 4 16`, RTX 4060, mạng 6×64 random-init (throughput không phụ thuộc trọng số), seed 0, `n_simulations: 200`:

| `parallel_games` | Giây/16 ván | Ván/giờ | `mean_batch` | `nn_calls`/nước |
|---|---|---|---|---|
| 1 | 268,3 | 214,7 | 1,00 | 196,7 |
| 4 | 168,4 | 342,0 | 3,94 | 197,4 |
| 16 | 113,2 | 508,7 | 14,25 | 197,4 |

Đọc bảng: `nn_calls`/nước gần như không đổi (196,7 → 197,4), tức gom lô không eval thừa; `mean_batch` 1 → 14,25; throughput 2,37×. `mean_batch` trong `GameRecord` là trung bình cộng dồn tại thời điểm ván kết thúc, lấy trung bình qua các ván — lô thật thấp hơn G vì các ván kết thúc lệch nhau.

### Spike multiprocessing (đã đo, chưa tích hợp vào train)

`scripts/spike_mp.py`, 4 process × 4 ván so với 1 process × 16 ván:

| Cách chạy | Giây/16 ván | `mean_batch` mỗi worker | `nn_calls`/nước |
|---|---|---|---|
| 1 process, G=16 | 84,3 | 13,4 | 196,3 |
| 4 process × 4 ván | 39,5 | 3,4–3,7 | 195,3–196,9 |

Tăng tốc 2,13×, đổi lại 4 process gọi GPU thường xuyên hơn với lô nhỏ hơn. Chưa nối vào `scripts/train.py`: việc đó cần truyền mẫu về tiến trình chính, gộp buffer và seed theo worker.

### Chạy 20 × 50 (`configs/normal.yaml`, G=16)

`python scripts/train.py --config configs/normal.yaml`, RTX 4060, 1000 ván:

| Vòng | Giây tự chơi | Nước/ván | `nn_calls`/ván | `mean_batch` | `loss_eval_on_fixed_batch` |
|---|---|---|---|---|---|
| 1 | 280,5 | 40,1 | 7 878 | 15,65 | 5,055 |
| 5 | 252,7 | 34,3 | 6 651 | 15,65 | 5,305 |
| 10 | 201,5 | 28,2 | 5 430 | 15,69 | 5,225 |
| 15 | 189,7 | 25,5 | 4 894 | 15,55 | 5,173 |
| 20 | 162,7 | 23,1 | 4 408 | 15,59 | 5,141 |

Tổng thời gian tự chơi 4 370 s (73 phút) + 49 s train cho 1000 ván, tức ~824 ván/giờ — nhanh hơn bảng benchmark vì ván ngắn dần (40 → 23 nước) khi model mạnh lên.

`loss_eval_on_fixed_batch` đạt đỉnh 5,37 ở vòng 4 rồi giảm đều 6 vòng cuối về 5,14, nhưng vẫn cao hơn vòng 1 (5,05). 200 bước/vòng vẫn là quá ít để nói "loss giảm trên tập cố định"; số dùng để kết luận là cột này, không phải `loss` trên batch ngẫu nhiên (5,19 → 4,09, giảm vì model khớp dữ liệu mới chứ không vì giỏi hơn trên tập cố định).

Gate GĐ5: `AZPlayer` (checkpoint cuối, 200 lượt) thắng random **20/20** (`RUN_SLOW=1 pytest tests/test_az_player.py`). Ngưỡng arena ≥ 55% chờ T13 của C.



## Máy đã cài bản CUDA
Máy đo bảng trên đang chạy `torch 2.14.1+cu126`. `requirements.txt` cố ý để `torch` trơn (bản CPU) cho teammate không có GPU, nên **đừng chạy `pip install -r requirements.txt` trên máy này** — pip sẽ hạ cấp về bản CPU và mất CUDA mà không báo lỗi. Kiểm tra trước khi chạy thí nghiệm dài:

```
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Kết quả đúng phải là `2.14.1+cu126 True`. Nếu ra `+cpu False` thì cài lại bản CUDA bằng lệnh ghi trong `requirements.txt`.


## T35 — benchmark giải đấu hoàn tất 2026-11-14

Nguồn: [t35_analysis.json](../evidence/colab_return_20261114T085015Z/t35_analysis.json), tính lại từ JSONL bằng [analyze_t35.py](../scripts/analyze_t35.py), đối chiếu checksum `colab_completed.json`. Chi tiết thắng/thua, màu và provenance: [báo cáo T35](report.md#10-ket-qua-t35-hoan-tat-ngay-2026-11-14).

### Nhóm 1 giây/nước — cùng máy local, 3 cặp song song

Windows, Python 3.10.6, PyTorch 2.14.1+cu126, RTX 4060 Laptop, 16 CPU logic; mỗi process một thread PyTorch. Mạng AZ dùng GPU, tìm kiếm/luật/AlphaBeta dùng CPU. AlphaBeta dùng iterative deepening theo thời gian, không bị khóa depth 1.

Mean/p95 tính trên từng lượt `select_move` của agent trong cặp, không phải trung bình các percentile theo ván. Latency không gồm khởi tạo player hoặc kiểm tra hợp lệ sau chọn nước. Simulations và neural leaves là trung bình mọi nước, kể cả nước override không chạy MCTS. Nodes chỉ có nghĩa cho counter thật của agent, không quy đổi sang simulations.

| Cặp | Agent | Số nước | Mean ms/nước | p95 ms/nước | Simulations/nước | Neural leaves/nước | Override |
|---|---|---:|---:|---:|---:|---:|---:|
| az_v0–alphabeta | az_v0 | 1181 | 998.04 | 1008.79 | 610.15 | 497.04 | 0.00% |
| az_v0–alphabeta | alphabeta | 1181 | 999.74 | 1008.36 | — | — | 0% |
| az_v0–az_v1_threat | az_v0 | 1011 | 999.72 | 1002.37 | 504.28 | 499.56 | 0.00% |
| az_v0–az_v1_threat | az_v1_threat | 1061 | 798.55 | 1003.27 | 396.90 | 396.79 | 20.26% |
| az_v1_threat–alphabeta | az_v1_threat | 1305 | 833.99 | 1002.33 | 412.00 | 410.28 | 16.48% |
| az_v1_threat–alphabeta | alphabeta | 1268 | 1000.25 | 1007.32 | — | — | 0% |

| Cặp | Mean giây/ván | Mean nước/ván |
|---|---:|---:|
| az_v0–alphabeta | 23.59 | 23.62 |
| az_v0–az_v1_threat | 18.58 | 20.72 |
| az_v1_threat–alphabeta | 23.57 | 25.73 |

AlphaBeta đo khoảng 128,26 node/nước khi đấu AZ gốc và 125,92 khi đấu threat. MCTS không có counter node tương đương. Mean latency threat thấp hơn 1 giây vì override: 215/1.061 nước (20,26%) khi đấu AZ gốc và 215/1.305 (16,48%) khi đấu AlphaBeta. Không diễn giải đó là mạng threat suy luận nhanh hơn.

p95 có thể vượt 1.000 ms vì deadline mềm. Ba cặp cùng tranh tài nguyên; số simulations trong 1 giây phụ thuộc cả phần cứng, tiến trình đồng thời và vị trí bàn cờ. Khi một cặp hoàn tất, tải giảm cho các cặp còn lại; đây là benchmark lịch chạy thực, không phải so sánh hiệu năng cô lập giữa các agent.

Wall-clock nhóm 300 ván khoảng **2.372 giây (39,53 phút)**, throughput **455 ván/giờ**. Tổng thời gian từng ván **6.574,36 giây**, không cộng thành thời gian chờ thực. Chưa có lần chạy tuần tự đối chứng nên không công bố mức giảm 50% hoặc speedup 2,77× như kết quả thực nghiệm có kiểm soát.

### Nhóm cố định và giới hạn so sánh depth

Normal 1.500 + Renju 600 ván: AZ/MCTS Pure ngân sách 200 simulations/rollouts, AlphaBeta depth 1. Tám cặp AlphaBeta được chạy mới, 13 cặp không có AlphaBeta giữ lại. Dữ liệu cố định gồm nhiều lần chạy local, Colab T4 và hồ sơ phiên Colab CPU; số worker và tải máy cũng đổi. `t35_analysis.json` giữ mean/p95 và counter của từng cặp để truy xuất, **không dùng chúng làm bảng xếp hạng tốc độ cùng phần cứng**.

Chưa có benchmark depth 1/2 trên cùng vị trí, cùng máy và tải. Các so sánh thời gian ván cũ chỉ là mô tả hai thí nghiệm có nước đi/độ dài ván khác nhau, không chứng minh tăng tốc riêng AlphaBeta 90–98%. Không chạy lại giải chỉ để tạo số benchmark đẹp.
