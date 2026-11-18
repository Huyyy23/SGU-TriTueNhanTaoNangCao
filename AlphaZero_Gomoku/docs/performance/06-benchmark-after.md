# Benchmark sau thay đổi

Không có tối ưu mã nào được triển khai trong tác vụ này để so sánh như trạng thái “sau”. Các phép đo mới chỉ là baseline thăm dò; xem [`02-profiling-baseline.md`](02-profiling-baseline.md) và raw results [`raw-selfplay-baseline.json`](raw-selfplay-baseline.json). Benchmark hai backend trước đây nằm trong [`docs/bao-cao-dual-backend.md`](../bao-cao-dual-backend.md), không phải phép đo self-play “sau”.

| Chỉ số | Python baseline (một ván, 200 sims) | C++ hiện tại (một ván, 200 sims) | C++ đã tối ưu |
|---|---:|---:|---:|
| Thời gian self-play | Median 7.695 giây/ván, 3 lượt | Median 5.782 giây/ván, 3 lượt | Không áp dụng; không giữ tối ưu |
| Self-play simulations/s | ~442 danh định | ~588 danh định | Chưa đo |
| Tournament games/phút | Benchmark bốn cấu hình lịch sử: 7.23 Python/Python | Benchmark bốn cấu hình lịch sử: 7.22 C++/Python | Chưa đo |
| Inference latency p50/p95 | Chưa đo trong mẫu mới | Chưa đo trong mẫu mới | Chưa đo |
| RSS đỉnh | Chưa đo | Chưa đo | Chưa đo |
| VRAM đỉnh | Chưa đo | Chưa đo | Chưa đo |
| Throughput training | Chưa đo | Chưa đo | Chưa đo |

Chênh lệch median 24.9% của mẫu một ván C++ **không được công nhận là speedup production đã xác minh**: chỉ có một diễn tiến ván cố định, không warm-up, không chạy xen kẽ, không batching và độ dao động giữa các lượt lớn. Ở mức 50 simulations, C++ chậm hơn. Không dùng bảng này để lập cam kết thời lượng cho 5.000 ván.