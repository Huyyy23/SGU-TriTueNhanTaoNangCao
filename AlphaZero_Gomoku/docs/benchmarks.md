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

## GĐ4 — trainer end-to-end (CPU, batch = 1)

Máy đo không có CUDA dùng được (`torch.cuda.is_available()` = False), mạng 6 block × 64 filter, `configs/normal.yaml` với `--games 10 --simulations 100`, `train_steps_per_iter: 200`, `batch_size: 128`:

| Vòng | Giây tự chơi | Giây train | Mẫu trong buffer | `loss` | `loss_eval_on_fixed_batch` |
|---|---|---|---|---|---|
| 1 | 200.5 | 56.6 | 416 | 4.808 | 4.722 |
| 2 | 176.0 | 35.3 | 878 | 4.726 | 4.759 |
| 3 | 212.7 | 44.2 | 1263 | 4.563 | 4.752 |
| 4 | 273.7 | 34.9 | 1750 | 4.598 | 4.754 |

Đọc bảng này: 200 bước tối ưu mỗi vòng là quá ít để `loss_eval_on_fixed_batch` giảm rõ — nó đi ngang trong khoảng ±0,04 (0,8%). π mục tiêu ở 100 lượt mô phỏng vẫn gần đều (entropy trung bình 3,75 trên tập eval), nên loss còn cách sàn lý thuyết ~1 nat; muốn thấy đường cong đi xuống phải chạy `n_simulations: 200` với đủ `games_per_iter: 50`. `nn_calls`/ván ≈ 4081–4787 cho ván 39–49 nước, tức ≈ 98–105 lượt suy luận mỗi nước ở 100 lượt mô phỏng, khớp kỳ vọng.

## Máy đã cài bản CUDA
Máy đo bảng trên đang chạy `torch 2.14.1+cu126`. `requirements.txt` cố ý để `torch` trơn (bản CPU) cho teammate không có GPU, nên **đừng chạy `pip install -r requirements.txt` trên máy này** — pip sẽ hạ cấp về bản CPU và mất CUDA mà không báo lỗi. Kiểm tra trước khi chạy thí nghiệm dài:

```
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Kết quả đúng phải là `2.14.1+cu126 True`. Nếu ra `+cpu False` thì cài lại bản CUDA bằng lệnh ghi trong `requirements.txt`.


