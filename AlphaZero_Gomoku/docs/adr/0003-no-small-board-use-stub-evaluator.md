# Bỏ bàn 7x7/9x9, kiểm chứng pipeline bằng evaluator giả

`SIZE = 11` trong `gomoku/board.py` là nguồn sự thật duy nhất; không có bàn nhỏ để validate pipeline, trái với bước 4 trong "Build order" của `AGENTS.md`. Thay vào đó, lỗi pipeline được bắt bằng **evaluator giả** (trả phân phối đều và một giá trị cố định) trong unit test, cộng một smoke test 2 iteration với ngân sách cực nhỏ.

Lý do: phần lớn lỗi của tầng AlphaZero là lỗi dấu giá trị, lỗi nhãn `z`, lỗi đếm số mẫu, và lỗi trộn bộ luật trong buffer — không phải lỗi phụ thuộc kích thước bàn. Evaluator giả bắt được những lỗi đó trong mili-giây, không cần chờ một ván 7x7.

## Considered Options

- **Tham số hoá kích thước bàn cho test**: phá quyết định "một hằng số `SIZE`", phải sửa cả features, model head, và UI.
- **Làm bàn 7x7 thật**: thêm một đường code thứ hai phải bảo trì suốt dự án, và không kiểm chứng được gì thêm về dấu hay nhãn.

## Consequences

- `AGENTS.md` bước 4 phải sửa cho khớp; nếu không, người đọc sau sẽ đi làm lại bàn 7x7.
- Mất mốc "agent học được cái gì đó trên bàn nhỏ": lần chạy 11x11 đầu tiên là lần đầu thấy đường loss thật, nên smoke test 2 iteration trở thành cổng bắt buộc của giai đoạn trainer.
