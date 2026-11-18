# Nhật ký tối ưu hóa

## Kết quả tác vụ hiện tại: không giữ thay đổi hiệu năng nào

Working tree trước đó đã có tích hợp hai backend. Tôi không sửa các file đó: bằng chứng baseline chưa đủ để tối ưu an toàn hoặc khẳng định speedup self-play đại diện. Đặc biệt, kết quả đo trái chiều giữa các mức simulations và mẫu mới không có batching inference (`mean_batch=1.0`). Giữ thay đổi chưa được chứng minh sẽ trái với yêu cầu chỉ giữ tối ưu có lợi ích đo được.

| Hướng xem xét | Bằng chứng hiện có | Quyết định |
|---|---|---|
| Mở rộng traversal native / port board và rules | Đã xác nhận ranh giới Python cho board, rules, feature, callback và marshaling; chưa đo thời gian từng phần; rủi ro semantics Renju cao | Chưa triển khai; cần profile trước |
| Đổi inference context/allocation | Profile batch 1 cũ cho thấy inference chiếm ưu thế; chưa benchmark inference batch đại diện hiện tại | Chưa triển khai; thiếu bằng chứng mới |
| Tăng số worker self-play | Các báo cáo cũ dùng workload không so sánh trực tiếp; chưa benchmark worker hiện tại theo cặp | Chưa triển khai; đo workers 1/2/4 trên máy đích |
| Tối ưu Patterns cho AZ self-play | Luồng AZ self-play Normal không gọi `find_patterns()` | Không làm vì không giải quyết workload mục tiêu |

Mẫu C++ 200 simulations nhanh hơn về median, nhưng điều kiện đo nhỏ/không batching; kết quả ở 50 simulations lại đảo chiều. Không có tối ưu nào vượt qua tiêu chí giữ thay đổi. Báo cáo này ghi rõ quyết định không thay đổi mã thay vì thêm tối ưu chưa được kiểm chứng.