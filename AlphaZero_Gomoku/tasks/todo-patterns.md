# T27 Task List: Bộ nhận diện mẫu

Spec: `SPEC-patterns.md` · Plan: `plan-patterns.md`

- [ ] T27.1 — Chốt API nhỏ nhất theo `Board`, triển khai nhận diện tam mở, tứ mở, tứ bị chặn, overline trên bốn hướng; không mutate board; validate input.
  - Acceptance: các mẫu và ca biên trong spec có test; phân biệt hướng/ô liên quan đủ để caller dùng; không thêm dependency hoặc tích hợp consumers.
  - Verify: `python -m pytest -q tests/test_patterns.py`, sau đó `python -m pytest -q`.
  - Files: `gomoku/patterns.py`, `tests/test_patterns.py`.
  - Dependency: T27 spec được duyệt.

## Checkpoint
- [ ] Hai lệnh test pass; diff chỉ gồm hai file implementation/test.
- [ ] Chưa bắt đầu T29/T30 trước khi người dùng duyệt kết quả.
