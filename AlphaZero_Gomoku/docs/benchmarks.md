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

## Máy đã cài bản CUDA

Máy đo bảng trên đang chạy `torch 2.14.1+cu126`. `requirements.txt` cố ý để `torch` trơn (bản CPU) cho teammate không có GPU, nên **đừng chạy `pip install -r requirements.txt` trên máy này** — pip sẽ hạ cấp về bản CPU và mất CUDA mà không báo lỗi. Kiểm tra trước khi chạy thí nghiệm dài:

```
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Kết quả đúng phải là `2.14.1+cu126 True`. Nếu ra `+cpu False` thì cài lại bản CUDA bằng lệnh ghi trong `requirements.txt`.


