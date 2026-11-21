# Báo cáo tối ưu Python cho nhận diện mẫu và gom lô suy luận

## Kết luận

Trong workload AlphaBeta của thí nghiệm, `find_patterns()` là nút thắt. Bản tối ưu Python trong `gomoku/patterns.py` chuyển bàn cờ sang danh sách Python một lần mỗi lần gọi, đồng thời kiểm tra hai đầu mút cửa sổ trước khi tạo danh sách cửa sổ. Không đổi API, thứ tự mẫu, luật hay thuật toán tìm kiếm.

Thông lượng median của cặp AlphaBeta–Heuristic tăng **1,62×**, từ 2,604 lên 4,223 nước/giây; khoảng min–max của hai nhóm không chồng nhau. Các ván đổi màu giữ nguyên chuỗi nước và người thắng. Cặp AlphaBeta–AlphaZero tăng 1,64×, nhưng phép đo hai đợt không xen kẽ và inference GPU dao động; không thể quy toàn bộ chênh lệch cho tối ưu `find_patterns()`.

Đây là báo cáo trước khi có Native Patterns. Tại thời điểm đo, chưa có compiler/pybind11 phù hợp để build; chưa đo chi phí chuyển dữ liệu, chạy differential native hay xác nhận ngưỡng tăng 10%. Không diễn giải đây là bằng chứng C++ nhanh hơn hay chậm hơn. Báo cáo Native Patterns về sau ghi nhận kết quả riêng, với workload và phương pháp riêng.

## Phương pháp

- Môi trường ghi nhận khi đo: Windows 10 22631, Python 3.10.6 trong `.venv`, NumPy 2.2.6, PyTorch 2.14.1+cu126, NVIDIA RTX 4060 Laptop GPU, Torch dùng một luồng CPU.
- Luật Normal, `configs/normal.yaml`, checkpoint `checkpoints/last.pt` (SHA-256 `6e29a4e3b8c235449898450de0f44b520829eda097ee74f97ef599238e52ffbc`), seed 42, AZ 200 simulation/nước, AlphaBeta depth 1, không đặt giới hạn thời gian.
- Self-play chạy ba lượt tuần tự cho mỗi giá trị `parallel_games`; model được nạp và warm-up trước khi đo. Số ván và các đường RNG khác nhau giữa cấu hình; seed bằng nhau không tạo cùng chuỗi ván.
- Tournament dùng `schedule()` và `play_game_task()`: ba cặp, mỗi lượt gồm hai ván đổi màu, lặp ba lượt. Thời gian game task gồm tạo player và nạp checkpoint, nhưng không gồm preflight, ghi file hoặc khởi tạo interpreter toàn CLI.
- Độ trễ `predict()` gồm chuyển dữ liệu và đồng bộ ngầm khi lấy kết quả CPU; không phải thời gian kernel GPU. Wrapper MCTS/inference, lấy mẫu bộ nhớ và `nvidia-smi` tạo overhead ở các lượt tournament.
- CPU là CPU time của process chia wall time; 100% tương ứng một lõi. RSS là peak lấy mẫu. GPU utilization/VRAM lấy mẫu trên toàn thiết bị mỗi khoảng 0,5 giây, có thể gồm tiến trình khác và không phải peak chính xác. Đo ngắn trên Windows có nhiễu.
- Ba lượt trước/sau không chạy xen kẽ; chỉ có một checkpoint/seed và các ván ngắn. Min–max là phạm vi ba lượt, không phải khoảng tin cậy.

## Self-play nền theo mức song song

Median (min–max). Simulations/giây được suy ra từ số nước × 200 / wall time, vì `run_games()` hoàn tất ngân sách simulation cố định trước mỗi nước; bộ đếm cũ chỉ tính terminal simulations nên không dùng.

| Số ván song song | Ván/phút | Nước/giây | Simulations/giây | Inference ms/call | CPU % một lõi | RSS MiB | GPU % toàn thiết bị |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 14,66 (14,29–15,15) | 2,69 (2,62–2,78) | 538 (524–556) | 1,53 (1,48–1,58) | 17,5 (13,4–32,8) | 942 (941–942) | 22,9 (15,2–32,4) |
| 2 | 9,28 (8,36–11,38) | 2,86 (2,58–3,51) | 572 (516–702) | 2,48 (2,01–2,79) | 21,2 (20,7–31,4) | 953 (952–953) | 30,8 (30,3–34,2) |
| 8 | 15,33 (15,04–19,19) | 5,30 (5,20–6,64) | 1.060 (1.040–1.327) | 2,78 (2,34–2,82) | 20,7 (10,1–25,5) | 1.011 (1.006–1.012) | 29,3 (26,7–32,1) |
| 16 | 13,43 (12,08–18,22) | 5,90 (5,31–8,01) | 1.181 (1.062–1.602) | 3,36 (2,59–3,66) | 19,8 (18,9–22,6) | 1.080 (1.072–1.087) | 28,3 (28,0–30,9) |

VRAM peak lấy mẫu median (min–max), MiB: 1 ván 1090 (1090–1090); 2 ván 1219 (1140–1299); 8 ván 1171 (1151–1272); 16 ván 1072 (1050–1145). Đây là số toàn GPU, không phải bộ nhớ riêng của mô hình. Batch trung bình lần lượt 1,00; 1,97; 7,30; 13,71. Do workload và RNG khác nhau, không coi đây là phép đo speedup cô lập của batching.

`find_patterns()` không nằm trên luồng AZ self-play cơ bản. Không chạy lại self-play sau tối ưu; không kết luận có tăng tốc self-play.

## Tournament trước và sau tối ưu Python

Median (min–max) qua ba lượt. Mỗi lượt có hai ván đổi màu; throughput gồm thời gian tạo player.

| Cặp đấu | Ván/phút trước | Ván/phút sau | Nước/giây trước | Nước/giây sau | Tỷ lệ median |
|---|---:|---:|---:|---:|---:|
| AlphaBeta–Heuristic | 14,88 (14,85–15,06) | 24,13 (23,96–24,20) | 2,604 (2,598–2,636) | 4,223 (4,193–4,235) | 1,62× |
| AlphaBeta–AlphaZero | 3,55 (2,89–4,22) | 5,83 (5,76–5,83) | 1,360 (1,108–1,618) | 2,233 (2,206–2,236) | 1,64× |
| Heuristic–AlphaZero | 29,81 (29,53–30,36) | 30,71 (30,35–30,71) | 5,216 (5,168–5,313) | 5,374 (5,312–5,375) | 1,03× |

Simulations/giây của các cặp có AlphaZero: AlphaBeta 135,99 (110,83–161,79) → 223,34 (220,63–223,58); Heuristic 496,77 (492,15–506,02) → 511,85 (505,86–511,91). Cặp chỉ gồm agent cổ điển không dùng MCTS. Inference ms/call: AlphaBeta–AlphaZero 2,14 (1,73–2,71) → 1,66 (1,63–1,73); Heuristic–AlphaZero 1,58 (1,55–1,59) → 1,56 (1,52–1,57). Inference không phải mục tiêu của patch; biến động nhấn mạnh nhiễu GPU giữa hai đợt.

| Cặp đấu | CPU % một lõi trước → sau | RSS MiB trước → sau | GPU % toàn thiết bị trước → sau | VRAM MiB trước → sau |
|---|---:|---:|---:|---:|
| AlphaBeta–Heuristic | 18,8 (16,4–19,8) → 13,5 (12,2–20,4) | 506,9 (506,8–506,9) → 507,3 (507,3–507,4) | 4,3 (3,0–8,2) → 7,0 (6,1–7,9) | 884 (884–890) → 905 (903–905) |
| AlphaBeta–AlphaZero | 19,2 (14,6–20,9) → 28,1 (27,1–30,2) | 873,2 (872,8–874,2) → 871,8 (871,6–872,3) | 15,7 (15,6–17,2) → 17,5 (16,3–18,4) | 1154 (1030–1157) → 1042 (1042–1042) |
| Heuristic–AlphaZero | 15,4 (8,5–24,6) → 20,4 (16,8–41,9) | 874,6 (874,2–874,6) → 872,4 (872,0–872,6) | 32,0 (31,7–35,6) → 34,5 (32,9–35,1) | 1089 (1086–1103) → 1042 (1042–1042) |

Không suy luận mức tiết kiệm tài nguyên từ CPU nhiễu hoặc số liệu GPU toàn thiết bị. Trong workload đã đo, từng cặp giữ nguyên chuỗi nước và người thắng giữa các lượt trước/sau.

## Phân tích và giới hạn

- `cProfile` của một ván AlphaBeta–Heuristic ghi nhận `find_patterns()` 7,737/8,084 giây tích lũy trước patch và 3,861/4,183 giây sau patch. `cProfile` tạo overhead; thời gian lồng nhau không cộng trực tiếp.
- Thử nghiệm chỉ thêm kiểm tra biên cửa sổ trước khi tạo list: median giảm từ 2,604 xuống 2,339 nước/giây, khoảng sau 2,204–3,672; nhiễu lớn, không chứng minh lợi ích. Bản cuối kết hợp kiểm tra biên với chuyển grid sang list, đạt 4,223 nước/giây; benchmark không tách riêng đóng góp từng thay đổi.
- Chưa thêm cache theo `(grid bytes, ô, màu)` vì chưa đo hit rate và việc tạo key/lưu cache có chi phí, đồng thời phải xử lý mọi thay đổi bàn cờ. Chưa thêm vectorization NumPy vì cửa sổ ngắn và chưa chứng minh nhanh hơn.
- `run_games()` đã interleave để gom lô inference. Benchmark multiprocessing cũ ghi nhận 4 worker 29,41 so với 1 worker 7,96 nước/giây trên 50 ván, nhưng các tập ván khác nhau. Báo cáo dynamic chunking cũ ghi nhận chậm hơn fixed nhưng không cô lập đủ nguyên nhân. Chưa đo lại multiprocessing trong đợt này.

## Đối chiếu và bước tiếp theo

- So sánh tạm implementation cũ/mới trên **17.790** tổ hợp bàn ngẫu nhiên, ô trống và màu; so sánh `kind`, `direction`, `completion_cells` và thứ tự. Không chứng minh mọi trạng thái.
- Báo cáo ghi nhận lệnh `pytest -q tests/test_patterns.py tests/test_alphabeta.py tests/test_heuristic.py tests/test_threat_space.py tests/test_renju_integration.py`: **30 passed**. Đây là kết quả lịch sử, không chạy lại trong phiên hiện tại.
- Với ngân sách cố định, các lượt tournament được ghi nhận có cùng chuỗi nước. Với deadline, tác tử có thể tìm sâu hơn sau tối ưu và chọn chuỗi khác dù thuật toán/deadline không đổi.
- Tối ưu Python không tự xử lý luật cấm Renju; logic Renju không đổi. Kiểm thử mới khi đó bao gồm broken-three gần biên ở bốn hướng.
- Giữ Python làm reference. Khi đánh giá C++, cần differential test biên bàn, hai màu, thứ tự output, completion cells và tương tác Renju; benchmark phải gồm chi phí chuyển dữ liệu và game task đầu-cuối. Chỉ giữ native nếu cải thiện median ít nhất 10%, vượt nhiễu, không làm sai kết quả và không giảm chất lượng.
