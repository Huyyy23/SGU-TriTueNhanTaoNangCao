# Báo cáo tăng tốc C++ gốc cho `find_patterns()`

## Kết luận và phạm vi

Giữ Native Patterns ở trạng thái tùy chọn. Báo cáo benchmark lịch sử ghi nhận kết quả differential đạt và throughput median cao hơn 10% so với Python đã tối ưu, trong workload tournament cụ thể bên dưới. Ba lượt không chạy xen kẽ; chỉ một checkpoint/seed và các ván ngắn. Kết quả đó không chứng minh lợi ích cho AlphaZero self-play hoặc MCTS C++.

`native/gomoku_native.cpp` chỉ cung cấp `find_patterns()`. Python tiếp tục làm reference, validation public API và chuyển kết quả thành `Pattern`; module native không áp dụng luật Renju. Chọn `auto` có thể dùng native khi import được; ép `native`/`cpp` phải lỗi nếu extension thiếu hoặc không nạp được.

## Kết quả benchmark lịch sử

Median (min–max) qua ba lượt; mỗi lượt gồm hai ván đổi màu.

| Cặp tournament | Python: ván/phút | Native: ván/phút | Python: nước/giây | Native: nước/giây | Tỷ lệ median nước/giây |
|---|---:|---:|---:|---:|---:|
| AlphaBeta–Heuristic | 24,264 (24,259–24,360) | 180,284 (174,608–181,646) | 4,246 (4,245–4,263) | 31,550 (30,556–31,788) | 7,43× |
| AlphaBeta–AlphaZero | 5,871 (5,677–5,905) | 11,263 (11,051–11,698) | 2,251 (2,176–2,264) | 4,317 (4,236–4,484) | 1,92× |

Đây là workload game-task có AlphaBeta gọi `find_patterns()`. Số liệu Python ban đầu trước tối ưu trong báo cáo `bao-cao-toi-uu-patterns.md` thuộc đợt khác, không phải baseline của bảng này.

| Phép đo trên cùng bàn | Python đã tối ưu (µs/lần gọi) | Gọi native trực tiếp (µs/lần gọi) |
|---|---:|---:|
| API `find_patterns()` công khai, gồm validation và tạo `Pattern` | 35,124 (34,604–35,296) | 2,252 (2,215–2,273) |
| Hàm native trực tiếp qua pybind11 | — | 0,957 (0,954–0,999) |

Gọi native trực tiếp vẫn gồm biên Python/C++, kiểm tra buffer và trả tuple; không phải thời gian kernel thuần. Chênh lệch khoảng 1,30 µs/lần gọi so với wrapper gộp validation/chuyển kiểu, không phải phép đo riêng overhead crossing rỗng.

| Cặp | CPU % một lõi Python → native | RSS cuối lượt (MiB) Python → native |
|---|---:|---:|
| AlphaBeta–Heuristic | 19,58 (9,52–23,06) → 25,82 (22,74–28,38) | 486,95 (486,91–486,96) → 486,34 (486,32–486,35) |
| AlphaBeta–AlphaZero | 19,51 (14,76–21,10) → 9,50 (7,31–11,00) | 869,70 (868,90–871,79) → 869,01 (867,42–871,47) |

CPU process/wall có nhiễu do lượt ngắn và độ phân giải Windows. RSS chỉ đo cuối lượt, không phải peak. Không có phép đo GPU riêng đủ tin cậy.

## Môi trường và phương pháp benchmark lịch sử

- Windows x64, CPython 3.10.6 ABI `cp310-win_amd64`, GCC 13.1.0 MinGW, PyTorch 2.14.1+cu126, RTX 4060 Laptop GPU, Torch một luồng CPU.
- Báo cáo ban đầu không cài package `pybind11`; build dùng header đi kèm PyTorch. GCC cần liên kết runtime tĩnh (`-static`) để tránh thiếu `libwinpthread-1.dll`. Khả năng tương thích ngoài tổ hợp đã thử chưa được xác nhận.
- Build lúc đó mất 4,171 giây; subprocess import 0,049 giây, gồm thời gian khởi động Python. Mỗi số chỉ từ một lần đo.
- `scripts/bench_native_patterns.py` dùng seed 42, `checkpoints/last.pt`, Normal, 200 AZ simulations/nước, AlphaBeta depth 1, ba lượt mỗi backend và budget cố định. Game-task bao gồm tạo player/nạp checkpoint mỗi ván; không gồm preflight, ghi log hay toàn bộ CLI.
- Microbenchmark có warm-up 500 lần và 10.000 lần gọi mỗi lượt. Tournament Python/native chạy thành process riêng, tuần tự, không xen kẽ. Các lượt trước đó dùng làm warm-up tournament; không warm-up riêng từng cặp.

## Kiểm tra correctness

- Báo cáo lịch sử ghi nhận 120 bàn ngẫu nhiên, **18.946** tổ hợp bàn/ô trống/màu; đối chiếu toàn bộ `Pattern` gồm thứ tự, `kind`, `direction`, `completion_cells`; xác nhận bàn không bị sửa. Có kiểm tra input sai và một số tình huống Normal/Renju, cùng ca cạnh/góc.
- Báo cáo lịch sử ghi nhận **61 passed** với `GOMOKU_PATTERN_BACKEND=native` cho các test Patterns, AlphaBeta, Heuristic, Threat Space, Renju, self-play, MCTS và trainer. Đây không phải kết quả của phiên hiện tại.
- Trong phiên hiện tại, build extension CPython 3.15/Windows thành công và import subprocess thành công. Chạy `python -m pytest -q tests/test_patterns.py tests/test_native_patterns.py`: **21 passed**. Đây là xác nhận hiện tại cho nhóm test này trong môi trường hiện tại; không thay thế 61 test lịch sử.
- Chưa chạy build/test trên Colab Linux. Chưa thử tất cả tổ hợp compiler, Python ABI và Windows toolchain.
- Native không xử lý luật cấm Renju; tests Renju chỉ chứng minh tương tác với callers đã kiểm tra, không phải native rule engine.

## Build và chọn backend

Chạy từ thư mục `AlphaZero_Gomoku`. Cần compiler C++17, Python headers đúng ABI và pybind11 headers (package `pybind11` hoặc header từ PyTorch).

```powershell
python scripts/build_gomoku_native.py --cxx 'D:\Jetbrain tools\CLion 2025.1.1\bin\mingw\bin\g++.exe'
$env:GOMOKU_PATTERN_BACKEND='native'
python -m pytest -q tests/test_native_patterns.py tests/test_patterns.py
```

```bash
python -m pip install pybind11
python scripts/build_gomoku_native.py --cxx g++
GOMOKU_PATTERN_BACKEND=native python -m pytest -q tests/test_native_patterns.py tests/test_patterns.py
```

`GOMOKU_PATTERN_BACKEND=python` ép Python; `native` hoặc `cpp` yêu cầu extension tương thích và lỗi rõ khi không nạp được; `auto` chọn native nếu import được, nếu không thì dùng Python. Lệnh build tạo extension theo ABI Python đang chạy tại thư mục gốc; không dùng binary khác OS/ABI. Không ghi đè checkpoint/log.

## Giới hạn và bước kế tiếp

Đây không phải benchmark bốn tổ hợp MCTS/Patterns của Dual Backend. Không có đo self-play AZ sau khi đổi Patterns vì luồng AZ cơ bản không gọi `find_patterns()`. Không có số liệu quality model. Muốn đề xuất Native Patterns cho workload khác cần chạy benchmark cùng checkpoint, luật, budget và môi trường, nhiều lượt xen kẽ; ghi CPU/RSS/GPU/wall rõ phạm vi đo và xác nhận mức tăng vượt nhiễu. Native MCTS có báo cáo riêng và vẫn chưa nghiệm thu.
