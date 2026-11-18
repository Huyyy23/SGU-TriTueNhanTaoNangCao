# Chất lượng training và mức tương đương giữa backend

## Mức 1 — Correctness thuật toán

Full suite đạt (261 passed, 2 skipped), gồm các kiểm tra correctness Native MCTS hiện có và fixture legal mask Normal/Renju. Trong benchmark self-play seed cố định, các cặp backend ở từng mức simulations có cùng số nước đi và số NN calls. Đây chỉ là quan sát trên workload/trajectory hẹp, không chứng minh semantics giống nhau với mọi seed và trạng thái.

## Mức 2 — Tương đương số học

`tests/test_native_mcts.py` hiện có so sánh leaf planes/legal masks, root visits và root value với Python trên fixture Normal và Renju; các test này đạt trong full suite. Benchmark mới dùng cùng checkpoint, seed, luật và ngân sách search giữa backend; NN calls khớp. Benchmark **không** ghi policy target tensors, phân phối visit riêng trên nhiều seed, value target arrays, model outputs hay sai số cho phép. Vì vậy chưa thiết lập được tương đương số học rộng hơn.

## Mức 3 — Sức chơi

Không chạy arena cân bằng màu và ghép cặp. Không có căn cứ để kết luận C++ giữ nguyên hay cải thiện sức chơi/chất lượng training. Báo cáo trước `docs/bao-cao-dual-backend.md` cũng ghi phần đánh giá chất lượng còn pending. Giữ thay đổi backend ở trạng thái thử nghiệm cho đến khi chạy arena với checkpoint, luật, ngân sách simulations và seed được cố định/ghép cặp.

## Smoke training

Một bước optimizer CUDA nhỏ, tất định đã thành công; điều này chỉ xác nhận các toán tử cơ bản hoạt động. Chưa chạy training CUDA nhiều iteration, checkpoint/resume, xu hướng loss hay so sánh retraining self-play. Không khẳng định lỗi GPU đã được sửa.