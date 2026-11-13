# Benchmark multiprocessing self-play — 2026-11-13

## Phạm vi và phương pháp

- Windows, RTX 4060 Laptop 8 GB, driver 610.88, PyTorch 2.14.1+cu126; GPU ở chế độ WDDM và có ứng dụng đồ họa chạy nền.
- Checkpoint `checkpoints/last.pt`, iteration 20, SHA-256 `6e29a4e3b8c235449898450de0f44b520829eda097ee74f97ef599238e52ffbc`; luật Normal, `configs/normal.yaml`, 200 simulations/nước. Không train, không sửa checkpoint hoặc implementation.
- 1 worker gọi `run_games`; nhiều worker gọi `run_games_parallel` như `scripts/train.py`. Pool đa tiến trình được khởi động trước khi đo wall time. Startup bao gồm load checkpoint, tạo buffer và khởi động pool; thời gian tải model vào GPU của từng worker vẫn nằm trong wall time. Pool dùng Windows spawn. Thứ tự xác nhận đảo ở lượt giữa; seed 0, 1, 2.
- `mean_batch` là trung bình theo ván của chỉ số tích lũy `GameRecord.mean_batch`, không phải trung bình chính xác theo inference call. GPU utilization và VRAM lấy mẫu bằng `nvidia-smi` khoảng mỗi giây, trên toàn GPU, gồm tiến trình nền; peak là peak **được lấy mẫu**, không bảo đảm peak tuyệt đối. Không đo VRAM riêng từng process vì WDDM.
- Mỗi lượt kiểm tra đủ số ván, ID duy nhất và số mẫu trong buffer bằng tổng số nước. Không chứng minh chỉ bằng đếm rằng không có mẫu trùng hoặc nhãn `z` đúng; các invariant này cần test chuyên biệt trước khi thay đổi code.

## Giai đoạn 1 — Thăm dò: 10 ván/cấu hình

| Worker | Startup (s) | Wall self-play (s) | Tổng nước | Nước/s | Mean batch | VRAM đỉnh mẫu (MiB) | GPU utilization TB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0,245 | 29,59 | 224 | 7,57 | 8,74 | 877 | 28,0% |
| 2 | 1,998 | 18,79 | 210 | 11,17 | 4,42 | 1.140 | 29,6% |
| 4 | 1,996 | 21,11 | 239 | 11,32 | 2,29 | 1.416 | 57,6% |

Chọn 4 worker vì moves/s cao nhất trong lượt thăm dò. Chênh lệch với 2 worker chỉ khoảng 1,3%, một lượt đo không đủ khẳng định 4 nhanh hơn 2. Đo CPU giai đoạn này không hợp lệ do lỗi lấy mẫu, không dùng để kết luận.

## Giai đoạn 3 — Xác nhận: 3 lượt × 50 ván/cấu hình

| Worker | Seed | Startup (s) | Wall (s) | Tổng nước | Nước/s | Mean batch | VRAM đỉnh mẫu (MiB) | GPU utilization TB | CPU TB (% một lõi) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0,179 | 138,93 | 1.129 | 8,13 | 15,67 | 1.183 | 28,1% | 33,1% |
| 1 | 1 | 0,047 | 151,17 | 1.204 | 7,96 | 15,62 | 834 | 30,2% | 33,7% |
| 1 | 2 | 0,066 | 157,00 | 1.181 | 7,52 | 15,58 | 834 | 31,1% | 35,7% |
| 4 | 0 | 2,007 | 43,80 | 1.174 | 26,80 | 11,03 | 1.392 | 41,4% | 210,6% |
| 4 | 1 | 2,039 | 35,76 | 1.078 | 30,14 | 11,27 | 1.364 | 42,5% | 215,5% |
| 4 | 2 | 2,005 | 39,07 | 1.149 | 29,41 | 10,94 | 1.359 | 47,9% | 228,9% |

| Chỉ số | 1 worker | 4 worker |
|---|---:|---:|
| Median moves/s (min–max) | 7,96 (7,52–8,13) | 29,41 (26,80–30,14) |
| Median wall (min–max), giây | 151,17 (138,93–157,00) | 39,07 (35,76–43,80) |
| VRAM peak mẫu lớn nhất, MiB | 1.183 | 1.392 |

Median throughput cao hơn **3,69 lần**; mọi lượt 4 worker cao hơn mọi lượt 1 worker. Median wall giảm khoảng 74,2% cho 50 ván, nhưng tập ván và tổng nước không hoàn toàn giống nhau giữa số worker; đây là benchmark self-play, không phải tốc độ toàn bộ training hay bằng chứng chất lượng model. VRAM của lượt đầu bị ảnh hưởng thứ tự chạy/ứng dụng nền; không coi chênh lệch peak là chi phí bộ nhớ chính xác của worker.

## Giai đoạn 2 — Phân tích

- **Song song CPU và throughput:** 4 worker dùng CPU trung bình khoảng 2,11–2,29 lõi theo mẫu process-tree, so với 0,33–0,36 lõi cho baseline, đồng thời tăng throughput rõ. Số này không phân rã được thời gian CPU, GPU wait và I/O; chưa đủ kết luận CPU-bound hoàn toàn.
- **GPU và batch:** GPU utilization toàn máy trung bình 28–31% với 1 worker và 41–48% với 4; batch trung bình theo ván giảm khoảng 15,6 xuống 10,9–11,3. Không có bằng chứng GPU bão hòa hoặc contention là nút thắt. Utilization lấy mẫu có thể bỏ lỡ các kernel ngắn; không đủ để loại trừ contention.
- **Mất cân bằng worker:** thời gian xử lý chunk đo tại worker ở seed 0 là 39,07; 36,23; 26,70; 43,78 giây (13/13/12/12 ván). Worker nhanh nhất xong sớm hơn worker chậm nhất khoảng 17,08 giây. Seed 1: 35,74; 33,61; 32,91; 25,59 giây; seed 2: 35,02; 35,07; 39,04; 38,46 giây. Đây là bằng chứng mất cân bằng, không chứng minh chia chunk nhỏ hơn sẽ nhanh hơn.
- **CUDA context:** `_play_chunk` tạo model riêng và chuyển lên CUDA trong từng worker process; trên CUDA mỗi process dùng context riêng. Kết quả PID cho thấy 4 worker khác nhau đã xử lý 4 chunk. WDDM không cho số VRAM theo process trong `nvidia-smi`; chưa đo trực tiếp chi phí context.
- **Weights và samples:** code gửi `state_dict` CPU trong payload cho mỗi chunk rồi trả samples về parent. Các tensor weights có tổng kích thước 1.942.040 byte; arrays một sample khoảng 968 byte trước overhead serialization (features uint8 và policy float32). Chưa đo riêng serialization, IPC, copy GPU hay merge buffer; không thể kết luận IPC là bottleneck. Thời gian chunk bao gồm dựng model/load weights và self-play; wall gồm cả gửi/nhận và ghi buffer nên không thể tách chính xác overhead bằng hiệu hai giá trị.
- `parallel_games` của mỗi worker hiện bằng kích thước chunk, không giữ giá trị 16 trong config khi multiprocessing. Vì vậy thí nghiệm phản ánh đúng hành vi hiện tại, không cô lập riêng ảnh hưởng số process.

## Giai đoạn 4 — Đề xuất duyệt

1. **Không sửa code:** dùng `--workers 4` cho workload Normal tương tự 50 ván/vòng trên máy này. Tăng tốc self-play quan sát 3,69× theo median; không cam kết cùng tốc độ cho Renju, checkpoint khác, hoặc toàn bộ train. Pool train thực tế tái sử dụng giữa các vòng; benchmark xác nhận khởi động pool mới mỗi lượt và tách startup khỏi wall.
2. **Thử nghiệm tiếp theo, chưa triển khai:** đo riêng độ lệch chunk, thời gian serialize weights/samples và thời gian parent merge trên workload đại diện. Nếu imbalance tiếp tục chiếm phần đáng kể, thử chia nhiều chunk nhỏ hơn với pool hiện hữu; đo lại trước/sau và so cả số lượt copy weights. **Mức tăng thêm chưa biết**, không ngoại suy từ khoảng chờ worker.
3. Chưa đề xuất inference tập trung: chưa có phép đo chứng minh GPU contention hoặc batch là nút thắt; thiết kế đó tăng độ phức tạp Windows spawn và IPC.

Trước khi chấp nhận bất kỳ thay đổi nào: kiểm thử số ván và ID, không mất/trùng sample, `z` theo lượt, tách rule set, cùng ngân sách MCTS và Windows spawn. So sánh lặp lại cùng checkpoint và khối lượng công việc; giữ thay đổi chỉ khi throughput tăng vượt nhiễu mà correctness vẫn đạt.

Dữ liệu thô: `logs/_selfplay_bench_explore.jsonl`, `logs/_selfplay_bench_confirm.jsonl`. Không đưa các số đo này vào log training.
