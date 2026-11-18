# Vì sao MCTS native không tăng tốc ổn định toàn luồng

## Đã xác nhận từ mã nguồn

C++ sở hữu lưu trữ node và thực hiện PUCT selection, traversal cây, quản lý expansion và backup. C++ không sở hữu trạng thái ván hay luồng inference. Mỗi request trả về leaf node và đường đi nước cờ; Python áp đường đi đó lên `Board`, kiểm tra thắng/kết thúc, tạo legal mask và planes rồi undo. Khi các nước con hòa điểm, việc chọn nước gọi callback Python từ C++. Khi nhận kết quả, Python áp mask/chuẩn hóa policy, sinh root noise và chuyển legal indices/priors thành STL vectors cho pybind11. Vì vậy mỗi leaf được đánh giá vẫn phát sinh công việc Python và trao đổi dữ liệu; đây là ranh giới được xác nhận trong cấu trúc, chưa phải attribution thời gian từng thành phần.

Self-play gom inference qua các ván đang chạy, nhưng mỗi ván chỉ tạo tối đa một leaf đang chờ trong mỗi nhịp scheduler. Benchmark một ván không thể kiểm tra batching: `mean_batch` đo được đúng 1.0. Hai backend có cùng số lần gọi NN và cùng số nước trong từng cặp so sánh seed cố định; mẫu này không cho thấy native giảm lượng inference.

## Bằng chứng định lượng

Trong mẫu một ván, 200 simulations, median ba lần chạy là 7.695 giây/ván Python và 5.782 giây/ván C++, tương ứng C++ nhanh hơn danh nghĩa 24.9%. Ở mức 50 simulations, Python chạy 2.192 giây còn C++ 2.921 giây, tức C++ chậm hơn 33.3%; ở mức 100, median lần lượt là 5.375 và 4.306 giây. Benchmark quá nhỏ và nhiễu để xem là speedup đáng tin cậy. Tournament lịch sử (2 simulations/nước) cho thấy riêng thay đổi MCTS khoảng -0.1% đến +0.6%, trong khi Patterns C++ tăng throughput khoảng 4.8× ở workload AlphaBeta depth 1. Patterns không được gọi trong AZ self-play Normal.

## Kết luận nguyên nhân

Giải thích có căn cứ cho tác động MCTS yếu/trái chiều là phạm vi tối ưu: native kernel chỉ tăng tốc quản lý cây, trong khi board/rules/features và inference vẫn chạy qua Python; ranh giới C++↔Python còn ở request/leaf và receive/result. **Tỷ lệ thời gian thực tế của từng phần chưa được đo** trong self-play nhiều ván đại diện. Vì vậy không thể khẳng định inference hoặc callback chiếm một tỷ lệ cụ thể, cũng không thể nói callback là nguyên nhân duy nhất đã đo được của kết quả tournament.

## Trả lời các giả thuyết được yêu cầu

- **Callback Python có xóa hết lợi ích traversal C++ không?** Có một callback RNG từ C++ sang Python khi xử lý hòa điểm và có công việc adapter Python trên mỗi leaf được đánh giá. Đã xác nhận chúng tồn tại; chưa cô lập thời gian thực thi.
- **Feature/rules có bị tính lặp không?** Feature và legal checks được tạo trên mỗi leaf neural ở cả hai backend; chưa đo thấy tạo trùng không cần thiết. Không nên gọi đây là lãng phí nếu chưa có profile.
- **Copy/allocate board?** MCTS dùng play/undo thay vì copy cả bàn trong cả hai backend. Có Python arrays, path/move containers và pybind conversions; chưa cô lập chi phí allocation.
- **Inference chiếm ưu thế?** Profile batch 1 lịch sử đo thấy inference chiếm phần lớn trong workload đó. Lần này chưa profile latency inference trên self-play batch đại diện.
- **Batching?** `run_games` đã triển khai gom batch; benchmark một ván này không batching. Cần chạy tối thiểu 4/16 ván hoạt động đồng thời để so throughput inference.
- **Workers/serialization/logging?** Đường worker hiện serialize CPU model state và trả records/samples; chưa profile scaling, IPC hay logging cho worker hiện tại.
- **Vì sao Patterns C++ tăng tournament?** Workload tournament có các agent AlphaBeta/heuristic quét pattern; self-play thông thường ghi `patterns: not_used`. Hai kết quả thuộc các call graph khác nhau.

Thí nghiệm có giá trị tiếp theo là đo từng thành phần trên self-play thực sự có batching, seed cố định, các lượt chạy xen kẽ và không có tải cạnh tranh. Chưa nên port toàn bộ board/rules sang native cho đến khi profile cho thấy xử lý board/rules Python chiếm tỷ lệ đáng kể; rủi ro port Renju đặc biệt cao.