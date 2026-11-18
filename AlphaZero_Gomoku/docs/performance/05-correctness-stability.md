# Bằng chứng correctness và stability

## Các kiểm tra đã chạy trong tác vụ này

- Full test suite trong `.venv` của project: **261 passed, 2 skipped trong 37.90 giây**. Bao gồm test Native MCTS hiện có: trạng thái request và legal mask Python/native, fixture luật Normal/Renju (gồm nước cấm, exactly-five và overline), visit counts/value backup, thắng trong một nước, reset cây lặp lại, số lượng/targets self-play và tích hợp multiprocessing spawn. Test chạy trên working tree đã có các thay đổi trước đó; tác vụ này không sửa mã runtime.
- Smoke optimizer CUDA tất định: PyTorch 2.14.1+cu126, RTX 4060 Laptop, `AZNet(n_blocks=1,n_filters=8)`, batch=2, seed CPU/NumPy cố định, một bước Adam qua `Trainer.train_on_batch()`. Thành công với loss 4.8049531, policy loss 4.7965155, value loss 0.00843768; sau đồng bộ CUDA, bộ nhớ cấp phát đỉnh 17,872,384 byte và reserved 23,068,672 byte. Lỗi `CUBLAS_STATUS_EXECUTION_FAILED` được báo cáo trước đó không tái hiện trong đường chạy cô lập tối thiểu này.
- Benchmark self-play seed cố định tại 2/50/100/200 simulations: số nước đi và NN calls ở mỗi mức khớp giữa Python/C++. Raw results từng lượt nằm trong `raw-selfplay-baseline.json`.

## Những điều chưa được xác lập

Không thêm differential suite mới vì quyết định là không sửa code; test differential hiện có đã chạy trong full suite. Điều đó không chứng minh tương đương trên mọi trạng thái có thể xảy ra. Tác vụ này không chạy memory soak dài hạn, sanitizer, arena có kiểm soát, checkpoint/resume hay scaling workers 1/2/4. Smoke optimizer CUDA chỉ cho thấy một bước training nhỏ thành công trong môi trường này; không tái hiện đầy đủ điều kiện gây lỗi pipeline trước đó và không chứng minh training GPU dài hạn ổn định. Không được xem lỗi `CUBLAS_STATUS_EXECUTION_FAILED` là đã khắc phục dựa trên smoke này.