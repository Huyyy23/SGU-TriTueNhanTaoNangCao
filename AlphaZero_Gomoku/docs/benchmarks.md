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


