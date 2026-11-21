# Báo cáo profiling và kế hoạch tích hợp C++ cho Gomoku

## Kết luận

**Chưa đủ bằng chứng để chọn và triển khai native module.** Bottleneck phụ thuộc workload:

- **AlphaZero, batch 1:** inference chiếm phần lớn thời gian trong hai mẫu đã profile. Chuyển MCTS sang C++ không giải quyết phần inference.
- **AlphaBeta depth 1:** `find_patterns()` chiếm phần lớn thời gian profile. Đây là ứng viên native đáng khảo sát cho tournament dùng AlphaBeta/heuristic.
- **Renju:** tính legal moves khi cache miss tốn nhiều hơn rõ rệt so với cache hit trên bàn thử. Cần đo trong ván thực trước khi quyết định; port luật có nguy cơ regression cao.

Tại thời điểm khảo sát, không sửa mã nguồn, training pipeline, checkpoint hoặc log.

## Môi trường và phương pháp

Windows 10 x64; Python 3.10.6 trong `.venv`; NumPy 2.2.6; PyTorch 2.14.1+cu126; RTX 4060 Laptop GPU. Cấu hình Normal dùng mạng 6 block/64 filter, **200 simulations/nước**, seed 42.

Các số dưới đây là phép đo ngắn trên một máy, chưa đủ mẫu để ước lượng biến thiên. `cProfile` thêm overhead; thời gian cộng dồn của hàm lồng nhau **không được cộng lại**. Inference test dùng model khởi tạo ngẫu nhiên, đầu vào zero, warm-up 3 lần, đo 10 lần; CUDA được đồng bộ. Test MCTS giả lập không chạy mạng.

## 1. Microbenchmark

Bàn gồm 10 nước, lượt Đen; thời gian/call là kết quả tốt nhất trong 3 lượt, trừ Renju cache miss lấy trung bình 10 lần.

| Thao tác | Thời gian |
|---|---:|
| `Board.copy()` | 0,62 µs |
| Normal `legal_moves()` | 1,91 µs |
| Normal `is_win()` | 2,59 µs |
| Renju `legal_moves()`, cache hit | 2,82 µs |
| Renju `forbidden_moves()`, cache hit | 0,17 µs |
| Renju `is_win()` | 2,62 µs |
| Renju `legal_moves()`, cache bị xóa trước mỗi lần | 25,26 ms |

**Giới hạn:** một vị trí không đại diện mọi giai đoạn ván. Cache miss được tạo chủ động, không phản ánh trực tiếp tỷ lệ miss thực tế. `Board.copy()` rẻ trong phép đo này; MCTS hiện dùng `play()`/`undo()` khi duyệt cây, không copy ở mỗi simulation (`az/mcts.py:100–105, 298–306`).

### Inference

| Device | Batch | `predict()` ms/batch |
|---|---:|---:|
| CPU, 10 PyTorch threads | 1 | 1,018 |
| CPU, 10 PyTorch threads | 16 | 5,060 |
| GPU | 1 | 1,204 |
| GPU | 16 | 1,539 |

Phép đo bao gồm `predict()` và truyền dữ liệu theo API hiện tại. GPU batch 16 có throughput inference cao hơn batch 1 trong phép thử này; **không suy ra cùng mức tăng throughput toàn ván**. CPU thread count ảnh hưởng kết quả; tournament mặc định đặt một thread, nên cần đo lại cùng điều kiện trước khi chọn device.

### MCTS không inference

Một lần search Normal, 200 simulations, evaluator cố định: **56,1 ms có profiler**, 200 NN leaf calls giả lập.

| Hàm | Thời gian cộng dồn |
|---|---:|
| `request()` | 33 ms |
| `_select_child()` | 24 ms |
| `receive()` | 22 ms |

`receive()` gồm expansion và backup; phép đo này **chưa tách riêng backup**. `Node.__init__` xuất hiện 21.914 lần, 9 ms cumulative. Selection và cấp phát node đáng khảo sát, nhưng đây không phải profile toàn ván có inference thực.

## 2. Workload thực ngắn

| Workload | Thời gian | Kết quả |
|---|---:|---|
| Self-play, GPU, `parallel_games=1`, 1 ván, có profiler | 4,731 s, chưa tính load checkpoint 0,175 s | 11 nước; 12,68 games/phút; 2,33 nước/s |
| Self-play, GPU, `parallel_games=2`, 2 ván, không profiler | 8,316 s, chưa tính load checkpoint 0,160 s | 37 nước; 14,43 games/phút; 4,45 nước/s; 867 NN leaves/s |
| Tournament `az_v0` vs `random`, 1 ván, có profiler | 5,179 s gồm tạo player; 4,986 s trong ván | 23 nước; 11,59 games/phút; 4,44 nước/s |
| Heuristic, một lượt trên bàn 10 nước, có profiler | 18,8 ms | `find_patterns()` 15 ms cumulative |
| AlphaBeta depth 1, cùng bàn, có profiler | 1,694 s | `find_patterns()` 1,589 s cumulative |

Các mẫu self-play có số ván, độ dài ván và instrumentation khác nhau, **không dùng để tuyên bố speedup batching**. 867 leaves/s không phải simulations/s: simulation kết thúc tại node terminal có thể không gọi mạng.

Trong profile self-play batch 1, `predict()` chiếm 3,697/4,731 s cumulative; tournament là 3,902/5,179 s. Đây là dấu hiệu inference chi phối **hai mẫu đó**, không phải kết luận cho cấu hình mặc định `parallel_games=16`.

### Multiprocessing và tài nguyên

- Chuyển `state_dict` CPU sang pickle: khoảng **1,97 MB**, serialize **4,5 ms**, deserialize **4,7 ms** trong một lần đo. Chưa bao gồm spawn worker, IPC, khởi tạo model/GPU hoặc trả samples.
- Self-play hai ván: PyTorch báo peak GPU allocation **10,4 MiB**. Đây không phải tổng VRAM tiến trình.
- Chưa có số **CPU utilization đáng tin cậy**, peak RAM, GPU utilization, overhead multiprocessing đầu-cuối hay simulations/s thực. Không dùng số `process_time()` hiện có làm utilization Windows.
- Chưa đo build time hoặc Python/C++ transfer: chưa có native module để đo.

## 3. Phân tích tích hợp

- `az/selfplay.py:29–88` xen kẽ nhiều ván trong **một process**, gom leaf vào một lần `model.predict()`. `run_games_parallel()` là đường multiprocessing riêng: pickle `state_dict`, worker tạo model, trả records và samples (`az/selfplay.py:178–239`).
- `az/mcts.py:87–205` thực hiện selection, rule checks, tạo features, expansion và backup; random tie-breaking và đổi dấu value là phần semantics phải giữ.
- Tournament có seed ghép cặp, đổi màu và telemetry; `sims:N` không áp dụng cho AlphaBeta, vốn dùng depth cố định (`scripts/tournament.py:141–180, 409–440`). Vì vậy không so trực tiếp số simulation MCTS với node AlphaBeta.
- `gomoku/rules/renju.py` có cache và kiểm tra forbidden moves; cần giữ đúng overline, double-three, double-four và ngoại lệ exactly-five. Không đưa luật vào MCTS hoặc player.
- Nếu dùng tournament **time budget**, tăng tốc có thể cho phép tìm sâu hơn và đổi nước đi. Không thể đồng thời đảm bảo kết quả tournament giống hệt trong chế độ này; kiểm tra tương đương phải dùng **fixed simulation/depth budget**.

## 4. Xếp hạng ứng viên

Đây là thứ tự **ưu tiên đo tiếp**, không phải dự báo speedup đã chứng minh.

| Ưu tiên | Ứng viên | Lợi ích kỳ vọng | Độ khó / nguy cơ |
|---|---|---|---|
| 1 | `gomoku/patterns.py` cho AlphaBeta/heuristic | Cao với AlphaBeta theo mẫu đo; cần đo tournament hoàn chỉnh | Trung bình/cao: pattern, thứ tự và tie-break phải tương đương |
| 2 | Renju forbidden/legal computation khi cache miss | Có thể cao cho Renju nếu tỷ lệ miss thực cao | Cao: quy tắc đệ quy và nước cấm; cần đối chiếu rộng |
| 3 | MCTS selection + expansion | Có tiềm năng khi inference được batch tốt | Cao: object/array crossing, RNG, float và thứ tự tie-break |
| 4 | Normal rules, board copy | Thấp theo microbenchmark này | Thấp hơn, nhưng crossing Python/C++ có thể xóa lợi ích |

Một hàm native `find_patterns()` gọi riêng cho từng ô vẫn có thể tốn chi phí chuyển đổi đối tượng và trả pattern. Phải đo **toàn bộ call boundary**, không chỉ kernel C++.

## 5. Kế hoạch trước prototype

1. **Đo baseline lặp lại:** ít nhất 3 lần sau warm-up, giữ checkpoint, seed, 200 simulations, device, thread count và cấu hình; báo median và khoảng dao động. Tách startup khỏi steady-state nhưng báo cả tổng.
2. **Mở rộng workload:** self-play Normal và Renju với `parallel_games=1,16`; tournament đủ cặp đổi màu cho AZ, heuristic và AlphaBeta ở fixed budget. Ghi simulations/s thực, games/phút, moves/s, CPU/RAM/GPU và tỷ lệ Renju cache miss.
3. **Tách chi phí:** selection, expansion, backup, rules, inference và IPC bằng instrumentation tạm thời; đo overhead instrumentation. So sánh worker 1 và nhiều worker, tính spawn, truyền weights và samples.
4. **Thử Python trước:** đo ảnh hưởng thread count và batching hiện có; rà soát tính toán lặp trong pattern/rules mà không đổi thứ tự nước, RNG hay semantics. Không giả định NumPy nhanh hơn vòng lặp nhỏ.
5. **Chỉ chọn một module** khi profile xác nhận nó chi phối workload mục tiêu. Đề xuất ngưỡng giữ prototype: cải thiện median **ít nhất 10% throughput đầu-cuối**, lớn hơn biến thiên giữa các lần chạy, không regression correctness; nếu không đạt, giữ Python.
6. **Nếu đạt điều kiện:** pybind11 tùy chọn, fallback Python nguyên vẹn; giữ PyTorch, checkpoint, log và public API. Đối chiếu output trên các bàn cố định và bàn sinh ngẫu nhiên hợp lệ; kiểm tra đầy đủ forbidden moves Renju nếu module chạm luật đó. So sánh chuỗi nước và kết quả ở fixed budget.
7. **Build và đo lại trên Windows và Colab Linux:** kiểm tra compiler, Python ABI và phiên bản NumPy/PyTorch thực tế; đo clean build, import, initialization, crossing và end-to-end. Windows cần MSVC phù hợp Python; Colab cần compiler Linux tương thích. Chưa xác minh build trên hai môi trường.

**Chưa kiểm tra:** throughput Renju thực, tournament nhiều ván, multiprocessing đầu-cuối và Colab. Các số một-ván/một-vị-trí chỉ định hướng, chưa đủ quyết định port C++.
