# Spec: Bộ nhận diện mẫu Gomoku (T27)

## Mục tiêu
Tạo `gomoku/patterns.py` dùng chung để phân loại cấu trúc quân quanh một ô cho cùng màu. Consumers dự kiến: luật Renju, heuristic player và tầng tìm kiếm chiến thuật. Module chỉ nhận diện mẫu, không áp dụng luật hay tự chọn nước.

## Hợp đồng
- Nhận trạng thái bàn cờ, tọa độ ô và màu cần xét; xác thực tọa độ/màu theo quy ước `gomoku.board`.
- Nhận diện theo bốn hướng: ngang, dọc, hai đường chéo.
- Phân loại tối thiểu: tam mở, tứ mở, tứ bị chặn, overline; trả thông tin hướng/ô liên quan đủ để caller kiểm tra nước thắng hoặc nối mẫu.
- Có thể xét ô đang trống như nước giả định cho màu được truyền vào; không sửa `Board`, `grid` hay `history`.
- Không phát hiện 3-3/4-4 như luật Renju hoàn chỉnh; `renju.py` kết hợp kết quả cơ bản để kết luận nước cấm. Không phụ thuộc `az/` hoặc UI.

## Tech stack và cấu trúc
Python hiện có, NumPy chỉ khi cần và theo convention hiện tại.
- `gomoku/patterns.py`: bộ nhận diện.
- `tests/test_patterns.py`: kiểm thử đơn vị.
Không thêm dependency, không sửa consumer trong phạm vi T27.

## Lệnh
- Từ `AlphaZero_Gomoku/`: `python -m pytest -q tests/test_patterns.py`
- Hồi quy: `python -m pytest -q`

## Quy ước mã
Theo style hiện có: hàm thuần khi có thể, kiểu dữ liệu/định danh tiếng Anh, ngoại lệ rõ cho input ngoài miền. Giữ kết quả có cấu trúc đơn giản; không tạo class framework hay cấu hình mở rộng chưa cần.

## Chiến lược kiểm thử
`pytest`, trong `tests/test_patterns.py`:
- Bốn hướng được hỗ trợ; biên bàn không tạo mẫu giả.
- Tam mở, tứ mở, tứ bị chặn và overline nhận đúng loại.
- Ô giả định trống được xét mà bàn không bị thay đổi.
- Bàn/màu/tọa độ không hợp lệ bị từ chối.
- Mẫu không nối liền hoặc khác màu không bị gộp nhầm.

## Ranh giới
- Luôn: giữ module độc lập với luật cụ thể; không mutate board; kiểm thử biên và bốn hướng.
- Hỏi trước: đổi `Board` API, thêm dependency, sửa `renju.py`/players/search, hoặc mở rộng loại mẫu.
- Không: nhúng heuristic chọn nước, luật Renju 3-3/4-4 hoàn chỉnh, hoặc quét toàn bàn trong hot path.

## Tiêu chí hoàn tất
- Các mẫu tối thiểu trên có kết quả ổn định, kèm hướng/ô liên quan.
- `Board` giữ nguyên sau mọi lần phân tích.
- Test riêng và toàn bộ test hiện có chạy đạt.
- Không thêm dependency hoặc thay đổi consumer.

## Giới hạn đã chốt
Mục tiêu là bộ nhận diện mẫu cơ bản dùng chung; phân xử luật và chính sách chọn nước thuộc caller. API chi tiết sẽ được chốt khi triển khai dựa trên test và kiểu dữ liệu thực tế, không mở rộng phạm vi T27.
