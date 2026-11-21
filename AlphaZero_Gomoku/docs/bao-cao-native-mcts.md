# Báo cáo thử nghiệm MCTS C++ gốc

## Kết luận

**Chưa nghiệm thu; không dùng trong huấn luyện hoặc giải đấu.** MCTS hiện vẫn chạy bằng Python. Không có triển khai MCTS C++ khả dụng trong repository.

## Bằng chứng đã ghi nhận

- Bản thử nghiệm đầu tiên không nạp được vì tên hàm khởi tạo extension `PyInit_gomoku_mcts_native` không khớp tên module. Bản tiếp theo không biên dịch do `pybind11::list` không hỗ trợ `rbegin()`/`rend()`; đã sửa cách duyệt danh sách và biên dịch được.
- Sau khi biên dịch, 15 kiểm thử đối chiếu MCTS đạt trên các trường hợp đã thử. Tuy nhiên, phép thử giữ `root` sống sau `del` và `gc.collect()`: node C++ có tham chiếu sở hữu hai chiều qua `parent` và `children`, nhưng Python GC không nhìn thấy các tham chiếu này. Đây là lỗi quyền sở hữu có nguy cơ rò rỉ bộ nhớ qua các lần tìm kiếm.
- Đã loại prototype và binary lỗi; không bật MCTS C++. Báo cáo ghi nhận 27 kiểm thử Python liên quan MCTS/self-play/trainer đạt sau khi khôi phục backend Python. Đây là kết quả lịch sử trong báo cáo trước, không phải lần chạy của phiên kiểm tra hiện tại.

## Phần chưa nghiệm thu

Chưa có MCTS C++ hoàn chỉnh, kiểm thử đối chiếu đầy đủ, benchmark bốn tổ hợp MCTS/Patterns, kiểm tra multiprocessing/batching, luật Normal/Renju, độ ổn định hoặc đánh giá chất lượng mẫu huấn luyện. Không dùng unit test để kết luận chất lượng mô hình; chưa có số liệu speedup self-play.

Thiết kế cần kiểm chứng: sở hữu toàn bộ cây trong arena C++, liên kết cha/con bằng chỉ số không sở hữu, giải phóng arena sau mỗi lượt tìm kiếm. Python tiếp tục điều phối suy luận, luật và RNG. Trước benchmark cần kiểm thử đối chiếu request/receive, PUCT, prior, số lượt thăm, backup và dấu giá trị, nhiễu Dirichlet, temperature, trạng thái kết thúc, nước cấm Renju và batching. Chỉ cân nhắc bật mặc định khi correctness và stability đạt, đồng thời throughput đầu-cuối tăng ít nhất 10% vượt nhiễu đo.

## Khuyến nghị cho 5.000 ván

Chưa chạy và chưa đề xuất cấu hình Native MCTS cho 5.000 ván. Benchmark multiprocessing cũ trong `tasks/benchmark-selfplay-multiprocess-2026-11-13.md` chỉ áp dụng điều kiện thử nghiệm lịch sử, không chứng minh số worker tối ưu hiện tại. `scripts/train.py` chỉ hỗ trợ luật Normal; không suy rộng sang Renju. Trước workload dài cần kiểm tra checkpoint/resume, replay samples, worker startup/shutdown và tài nguyên bằng workload nhỏ.
