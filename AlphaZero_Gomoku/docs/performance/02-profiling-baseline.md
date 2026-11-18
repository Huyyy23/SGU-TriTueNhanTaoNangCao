# Profiling và số đo baseline

## Chi tiết tái lập

Sử dụng `.venv` của project (CPython 3.10.6, PyTorch 2.14.1+cu126), CUDA khả dụng trên NVIDIA GeForce RTX 4060 Laptop GPU, 16 CPU logic và checkpoint `checkpoints/last.pt` (SHA-256 `6e29a4e3b8c235449898450de0f44b520829eda097ee74f97ef599238e52ffbc`). Luật Normal, seed 20261118, Patterns Python, gọi `run_games`, ba lần chạy cho mỗi backend và mức simulations. Giữ model trong bộ nhớ; không tính thời gian load checkpoint. Cấu hình `parallel_games=4` nhưng chỉ yêu cầu một ván, nên `mean_batch=1` ở mọi lượt. Không có bước warm-up model riêng; lượt Python đầu tiên ở mức 2 simulations cho thấy độ nhạy với startup/warm-up. Dữ liệu từng lượt: [`raw-selfplay-baseline.json`](raw-selfplay-baseline.json).

## Kết quả self-play

Median của ba lượt chạy cùng seed/workload một ván. Số nước đi thay đổi theo ngân sách simulations do policy/search làm thay đổi diễn biến ván; chỉ so sánh backend trong cùng một hàng. Throughput được tính là số ván chia cho wall time (một ván). Sims/s là ước lượng danh định `số nước × simulations cấu hình / wall time`; đây chỉ là xấp xỉ vì simulation kết thúc ở terminal có thể không gọi NN.

| Simulations | MCTS | Median giây/ván | Số nước | NN calls | Batch trung bình | Sims/s danh định |
|---:|---|---:|---:|---:|---:|---:|
| 2 | Python | 0.1954 | 63 | 125 | 1.0 | 646 |
| 2 | C++ | 0.1736 | 63 | 125 | 1.0 | 727 |
| 50 | Python | 2.1922 | 25 | 1,201 | 1.0 | 570 |
| 50 | C++ | 2.9214 | 25 | 1,201 | 1.0 | 428 |
| 100 | Python | 5.3750 | 23 | 2,221 | 1.0 | 428 |
| 100 | C++ | 4.3064 | 23 | 2,221 | 1.0 | 534 |
| 200 | Python | 7.6948 | 17 | 3,242 | 1.0 | 442 |
| 200 | C++ | 5.7816 | 17 | 3,242 | 1.0 | 588 |

Ở mức 200 simulations, median C++ thấp hơn 24.9% trong mẫu này; tuy nhiên chỉ có ba lượt chạy tuần tự, không warm-up, không gom nhiều ván thành batch, và thời gian Python mỗi lượt dao động 6.87–8.42 giây. Ở mức 50 simulations, C++ chậm hơn 33.3%. Kết quả ngắn và trái chiều này chưa chứng minh được mức tăng ổn định. Báo cáo tournament lịch sử là workload khác: 2 simulations/nước, AlphaBeta depth 1; Patterns C++ chi phối kết quả ở workload đó trong khi AZ self-play chuẩn không gọi patterns.

## Phân bổ thành phần và giới hạn phép đo

Đã xác minh trực tiếp luồng thực thi trong mã nguồn. Native `Tree.request()` trả đường đi và gọi callback xử lý hòa điểm của Python; `NativeMCTS.request()` thực hiện play/undo bàn cờ bằng Python, kiểm tra luật, legal mask và tạo feature; `receive()` chuẩn bị priors/noise bằng Python rồi chuyển legal moves/priors sang C++. `AZNet.predict()` vẫn thực hiện mọi lần đánh giá mạng. Đây là xác nhận vị trí xử lý, không phải tỷ lệ thời gian đã được đo.

Báo cáo profiling trước đây (`docs/bao-cao-profiling-cpp.md`) đo thấy `predict()` batch 1 chiếm phần lớn thời gian trong các mẫu ngắn có profiler (self-play 3.697/4.731 giây; tournament 3.902/5.179 giây), đồng thời có profile MCTS với evaluator giả (56.1 ms/200 simulations). Đó là workload và cách instrumentation khác; không cộng dồn thời gian hàm lồng nhau hoặc suy rộng attribution batch 1 sang cấu hình `parallel_games=16`.

Lần đo này chưa có: latency inference theo percentile, thời gian từng thành phần, số lượng/thời gian chuyển Python↔C++, CPU utilization, RSS/VRAM đỉnh, scaling nhiều worker, tốc độ tournament toàn ván, soak bộ nhớ hay sanitizer. `mean_batch=1` ở mẫu một ván không đại diện cho batching mặc định. Không đưa ra kết luận cho các chỉ số chưa đo này.
