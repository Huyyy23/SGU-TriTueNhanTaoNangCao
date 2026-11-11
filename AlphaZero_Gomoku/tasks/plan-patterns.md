# Implementation Plan: T27 Bộ nhận diện mẫu

## Phạm vi
Thực hiện riêng T27 theo `SPEC-patterns.md`. Chỉ thêm `gomoku/patterns.py` và `tests/test_patterns.py`; không tích hợp Renju, players hay tactical search.

## Hợp đồng kiến trúc
- Đầu vào bàn cờ, ô cần xét, màu; ô trống được phân tích như nước giả định.
- Phân tích bốn hướng, trả phân loại mẫu và hướng/ô liên quan.
- Không sửa trạng thái bàn; xác thực đầu vào; luật cụ thể thuộc caller.
- Chọn API nhỏ nhất phù hợp convention của `Board`; không thêm abstraction/dependency.

## Task List

### Phase 1: T27 — nhận diện và kiểm thử
- [ ] **Task 1: Chốt API tối thiểu, triển khai nhận diện mẫu**
  - Acceptance: tam mở, tứ mở, tứ bị chặn, overline; bốn hướng; biên bàn chính xác; ô giả định không mutate `Board`; input sai bị từ chối.
  - Verify: kiểm thử tập trung `python -m pytest -q tests/test_patterns.py`.
  - Files: `gomoku/patterns.py`, `tests/test_patterns.py`.
  - Dependencies: spec đã duyệt.
  - Scope: nhỏ.

### Checkpoint: T27
- [ ] Test tập trung và toàn bộ test hiện có pass.
- [ ] Review diff xác nhận chỉ hai file trong phạm vi, không có luật/consumer bị kéo vào.
- [ ] Báo người dùng kết quả; dừng trước tích hợp T29/T30.

## Rủi ro và giảm thiểu
| Rủi ro | Tác động | Giảm thiểu |
|---|---|---|
| Định nghĩa mẫu mơ hồ ở mép bàn hoặc khi có nhiều hướng | False positive/negative cho Renju và chiến thuật | Viết test theo từng hình mẫu và hướng trước khi mở rộng API |
| `pytest` thiếu trong môi trường | Không xác minh được | Cài dependency theo hướng dẫn dự án, không tuyên bố pass nếu chưa chạy |

## Lệnh xác minh
- `python -m pytest -q tests/test_patterns.py`
- `python -m pytest -q`
