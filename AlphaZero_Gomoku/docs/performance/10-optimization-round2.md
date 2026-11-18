# 10 — Changelog tối ưu vòng hai

## Ưu tiên sau profiling

| Phương án | Tốc độ kỳ vọng | Độ khó | Rủi ro correctness | Bộ nhớ | Quyết định |
|---|---|---|---|---|---|
| Bỏ reserve exact-size lặp lại trong native tree expansion | Cao theo timer `Tree.receive` | Rất thấp | Thấp; giữ nguyên thứ tự/logic node | Dùng tăng trưởng vector hình học, tránh copy toàn arena mỗi expansion | Triển khai |
| Đổi `torch.no_grad()` sang `inference_mode()` / tối ưu transfer | Trung bình-tiềm năng, inference >50% wall | Thấp-vừa | Numerical/model behavior cần đối chiếu | Không đáng kể | Chưa triển khai; đo benchmark clean trước |
| Native board/rules/features | Thấp trong workload này | Cao | Cao, nhất là Renju | Có thể tăng state/copy | Không làm; board/rules/features đều <2% mỗi vùng đo |
| Tối ưu tie callback/pybind request | Thấp | Vừa | Trung bình | Không đáng kể | Không ưu tiên; callback body ~0.6% wall |
| Tăng worker count | Chưa rõ; có thể contention | Vừa | Stability/RAM/VRAM | Mỗi worker model riêng | Đang benchmark tuần tự 1/2/4 |

## Thay đổi

Trong `native/gomoku_mcts_native.cpp`, xóa `nodes_.reserve(nodes_.size() + moves.size())` khỏi `Tree::receive()`. `std::vector::push_back` đã tăng capacity theo cấp số nhân. Reserve chính xác trước mỗi expansion đã buộc reallocation/copy toàn arena nhiều lần trong một search. Tree vẫn giữ node indices; không có `Node&` sống qua các `push_back` nên relocation không làm mất tham chiếu sử dụng.

Không thay đổi số simulation, PUCT, tie-breaking, root noise, temperature, backup/value sign, board/rules, model, features, targets, RNG hoặc Python mặc định. Python/C++ backend tiếp tục chọn độc lập.

## Kiểm chứng sớm

- Rebuild native module bằng GCC 13.1: thành công; build 5.258 s, import smoke 0.043 s.
- Trước/sau, profile cùng checkpoint, seed 230000, cùng 4 game/111 moves/21,149 neural calls:
  - `pybind.Tree.receive`: 4,822.2 ms → 269.0 ms (−94.4%); p50 0.2492 → 0.0076 ms; p95 0.4657 → 0.0188 ms.
  - Native adapter `receive` inclusive: 5,556.2 → 875.8 ms (−84.2%).
  - Instrumented C++ profile wall: 21.358 → 14.821 s (−30.6%); cProfile/timer overhead khiến đây chỉ là attribution, không phải throughput benchmark.
- Focused tests: `pytest -q tests/test_native_mcts.py tests/test_backend.py` — 13 passed.
- Benchmark self-play không instrumentation, 3 seeds/config: paired C++ throughput tăng median 1.64× (p4) và 2.44× (p16); C++ sau tối ưu nhanh hơn Python 1.62× (p4) và 2.67× (p16). Worker/game-batch configuration cuối cùng xem `11-production-benchmark.md`.

## Kết luận giữ/revert

Giữ thay đổi: profile xác định nguyên nhân trong expansion; timer trực tiếp xác nhận tiết kiệm khoảng 4.56 s trên 21,149 lần expansion; trajectories và NN call counts giữ nguyên. Matrix self-play không instrumentation (3 lần/config) xác nhận lợi ích end-to-end; số liệu cuối ở báo cáo 11. Chưa tối ưu inference vì cần benchmark before/after độc lập; inference hiện là bottleneck kế tiếp.
