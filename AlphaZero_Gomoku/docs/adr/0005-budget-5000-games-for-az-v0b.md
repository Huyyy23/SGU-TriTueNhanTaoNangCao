# Ngân sách 5000 ván cho `az_v0b`, 10000 ván chỉ khi đường cong còn dốc

Sau khi `az_v0` chạy xong 1000 ván, `az_v0b` được huấn luyện tiếp tới 5000 ván (100 vòng × 50 ván), khởi động từ `checkpoints/last.pt` và `checkpoints/buffer_last.npz`, ghi ra `checkpoints/az_v0b/`. `train_steps_per_iter` nâng 200 → 1000. Mốc 10000 ván chỉ mở nếu win rate ở khoảng 3000–5000 ván vẫn còn tăng.

Lý do: chi phí đo được của 1000 ván là 73 phút tự chơi so với 49 giây huấn luyện, tức 0,2 bước huấn luyện mỗi ván. Gấp số ván mà giữ nguyên tỉ lệ đó thì có thêm dữ liệu nhưng mô hình không khớp thêm được bao nhiêu; nâng số bước chỉ tốn thêm khoảng 20 phút trên tổng 6 giờ.

## Considered Options

- **10000 ván ngay**: hai đêm GPU cho một giả thuyết chưa ai đo; nếu đường cong phẳng từ ván 3000 thì phần lớn thời gian là lãng phí.
- **Chỉ tăng ván, giữ 200 bước/vòng**: giữ nguyên tỉ lệ tự chơi : huấn luyện 89 : 1, rẻ hơn 20 phút nhưng khó giải thích nếu mô hình không mạnh lên.
- **Tăng `n_simulations` 200 → 400 hoặc mạng 6×64 → lớn hơn**: đắt hơn nhiều lần và làm mất khả năng so sánh với `az_v0` ở cùng ngân sách tìm kiếm.
- **Huấn luyện lại từ đầu với cấu hình mới**: lịch trình LR sạch, nhưng mất 1000 ván đã học và mất 1,2 giờ; `az_v0` trở thành mốc không tái lập được.

## Consequences

- `--resume` với `n_iter` lớn hơn làm `total_steps` tăng, nên LR nhảy từ ~0,02·1e-3 về ~0,9·1e-3 ở vòng 21. Đây là hệ quả của lịch cosine theo `global_step`, không phải lỗi; báo cáo phải ghi rõ để đường cong không bị đọc là gián đoạn.
- Buffer giữ nguyên 50 000 mẫu, tức cửa sổ trượt khoảng 2200 ván gần nhất. Với 5000 ván, khoảng một nửa dữ liệu cũ bị ghi đè; đó là hành vi mong muốn, không phải giới hạn cần sửa.
- `az_v0` vẫn là mốc cố định cho câu hỏi az_v0 / az_v1. `az_v0b` là thí nghiệm riêng về ngân sách, không thay thế mốc đó.
- Renju không nhận ngân sách lớn: fine-tune `az_renju` giữ nguyên quy mô, chỉ là bằng chứng tính tổng quát.
