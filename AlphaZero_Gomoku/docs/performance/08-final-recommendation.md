# Khuyến nghị cho 5.000 ván self-play

## Khuyến nghị: chưa có backend thắng cuộc được đo đủ

Giữ mặc định an toàn hiện tại là Python MCTS cho đến khi có benchmark ghép cặp đại diện. Native MCTS là tùy chọn và chạy được; mẫu một ván 200 simulations nhanh hơn 24.9% theo median, nhưng mẫu 50 simulations chậm hơn 33.3%. Các lượt chạy tuần tự, không xen kẽ, không warm-up, chỉ một ván mỗi lượt (`mean_batch=1`), và không đo RSS/VRAM hay thời gian từng thành phần. Đây chưa đủ căn cứ để khuyến nghị C++ cho workload production 5.000 ván. Patterns nên tiếp tục dùng Python trong AZ self-play: luồng này không gọi patterns. Native Patterns chỉ tác động các agent tournament có dùng nhận diện mẫu.

Cấu hình Normal hiện có gom inference nhiều ván; đường throughput đó không có trong benchmark này. Chưa so sánh worker 1/2/4, nên không thể chọn số worker tối ưu một cách có trách nhiệm. Dùng đường batching một process hiện tại (`workers=1`) làm baseline có thể tái lập, sau đó đo workers 1/2/4 trên máy đích và chọn cấu hình nhanh nhất mà vẫn ổn định; không suy luận rằng process workers sẽ làm nhanh hơn đường batching một process.

## Ước lượng lập kế hoạch 5.000 ván

Không đưa ra dự báo thời lượng production: các trajectory đo được chỉ dài 17–63 nước và chỉ có một seed ở mỗi mức budget, trong khi độ dài ván/chi phí search thay đổi đáng kể. Phép nhân số học đơn thuần từ mẫu 200 simulations là khoảng 10.7 giờ cho Python hoặc 8.0 giờ cho C++ (5.000 × median thời gian một ván tương ứng), **không phải dự báo** và không hợp lệ để ước lượng training 5.000 ván đa dạng. Không dùng các số này để lên lịch production training.

## Điều kiện trước khi chạy training dài

Chạy self-play Normal với checkpoint cố định, batch game như cấu hình thực, seed/backend ghép cặp, warm-up tường minh, ít nhất ba lượt xen kẽ và đủ số ván để ổn định phân phối độ dài. Đo wall time, số nước, simulations thực, NN calls, batch trung bình, inference p50/p95, RSS/VRAM, CPU/GPU utilization, startup riêng và workers 1/2/4. Xác nhận sample-count/target invariants, checkpoint/resume, worker spawn và arena cân bằng màu. Lặp lại với Renju trước khi khuyến nghị cấu hình Renju. Chỉ giữ C++ nếu lợi ích end-to-end vượt nhiễu đo mà không gây regression correctness/stability.

## Rủi ro còn lại trước production

- Native MCTS vẫn dựa vào Python cho board/rules/features và các lần chuyển pybind11; chưa profile thời gian từng phần.
- Chưa có profile bốn cấu hình MCTS/Patterns trên self-play batching đại diện hoặc dữ liệu scaling worker.
- Chưa có memory soak hoặc bằng chứng sanitizer.
- Lỗi optimizer GPU trước đây không tái hiện trong smoke nhỏ, nhưng chưa được sửa/chẩn đoán.
- Chưa đánh giá playing strength có kiểm soát hoặc tương đương số học trên phạm vi rộng.
