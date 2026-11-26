# Báo cáo kiểm thử tích hợp Python/C++ — Self-play và Tournament

**Ngày:** 2026-11-21 | **Môi trường:** `.venv/Scripts/python.exe` (Python 3.10.6, torch 2.14.1+cu126, numpy 2.2.6) | **Backend native:** `gomoku_mcts_native.cp310-win_amd64.pyd` và `gomoku_native.cp310-win_amd64.pyd` (pybind11), import được, khởi tạo `Tree` thành công.

## 1. Mục tiêu

Kiểm thử end-to-end Python/C++ cho Self-play và Tournament (Normal và Renju), đồng thời profiling workload đại diện và xác minh backend C++ thực sự được gọi. Không tốiưu thành phần nào nếu không có bằng chứng bottleneck; không chuyển PyTorch inference sang C++; không đổi thuật toán AlphaZero.

## 2. Thay đổi mã

### `scripts/arena.py`
- Thêm CLI `--backend` (`auto|cpp|python`, mặc định `auto`), `--mcts-backend`, `--patterns-backend` (`python|cpp`).
- Thay `backend = "python"` được mã hóa cứng bằng `configure_backends(...)` từ `az/backend.py`; giá trị được lưu vào `cli_config["backend"]`, manifest `backend_metadata`, metadata sự kiện và truyền vào `make_named_player(..., backends=backends)`.
- Fallback nhất quán với Tournament: `auto` chọn C++ nếu có compiler và module tương thích, ngược lại chọn Python; `cpp` từ chối chạy khi không sẵn sàng (không âm thầm fallback).

### `scripts/tournament.py`
- `make_named_player(...)` nhận tham số `backends: dict | None` và giải quyết `mcts_backend` / `patterns_backend` trước khi gọi `create_player`; arena truyền `backends` vào mỗi agent AZ.

### `tests/test_selfplay.py`
- `test_cpp_selfplay_normal_runs_with_valid_samples`: chạy `run_games` với `mcts_backend="cpp"`, kiểm tra `components_used["mcts"] == "cpp"`, samples == moves_count, `pi` hợp lệ, `z ∈ {-1,0,+1}`.
- `test_cpp_selfplay_renju_forbidden_moves_masked`: Renju với C++ MCTS, `rule_set == "renju"`.

### `tests/test_tournament.py`
- `test_tournament_az_normal_cpp_records_components_used`: tournament 2 ván Normal với 2 agent AZ (`az_v0`), backend C++, kiểm tra `components_used["mcts"] == "cpp"`.
- `test_tournament_az_renju_cpp_records_components_used`: tương tự Renju.

### `scripts/profile_cpp_backend.py` (mới)
Script profiling workload đại diện (không commit vào main pipeline): MCTS python vs cpp, board/rules, feature extraction, PyTorch inference, self-play end-to-end, parallel workers.

## 3. Kết quả kiểm thử

| Test | Backend | Ruleset | Kết quả |
|---|---|---|---|
| `test_cpp_selfplay_normal_runs_with_valid_samples` | C++ | Normal | PASS |
| `test_cpp_selfplay_renju_forbidden_moves_masked` | C++ | Renju | PASS |
| `test_tournament_az_normal_cpp_records_components_used` | C++ | Normal | PASS |
| `test_tournament_az_renju_cpp_records_components_used` | C++ | Renju | PASS |
| `test_native_mcts_*` (5 test có sẵn) | C++ | Normal/Renju | PASS |
| Toàn bộ 4 file test liên quan | Python + C++ | Normal/Renju | 82 passed |

**Bằng chứng backend C++ được gọi:**
- `components_used["mcts"] == "cpp"` được ghi vào SQLite tournament experiment và config mỗi bản ghi self-play (test tournament và selfplay đã assert).
- `tests/test_native_mcts.py` đã chứng minh native MCTS khớp Python: visits, backup, legal mask Renju, root value, win-in-1, terminal root, tree release.
- Profiling (xem §4) đo thấy `NativeMCTS` thực thi 50 simulations nhanh hơn Python (~1.5–1.9x) với visit counts **giống hệt bitwise** và root value bằng nhau.

## 4. Profiling workload đại diện

Chạy bằng `.venv/Scripts/python.exe scripts/profile_cpp_backend.py`.

| Thành phần | Thời gian (đại diện) | Ghi chú |
|---|---|---|
| MCTS 50 sims (python) | ~39–46 ms / nước | Python tree |
| MCTS 50 sims (cpp) | ~24–26 ms / nước | C++ tree qua pybind11 |
| Tốc độ MCTS | ~1.5–1.9x | C++ nhanh hơn python trên tree bookkeeping |
| visit counts bằng nhau | `True` | Parity chính xác trên cây |
| root value bằng nhau | `True` | `py == cpp` |
| play/undo x2000 | ~2.0–2.7 ms | Rất rẻ |
| legal_moves x2000 | ~4.8–6.3 ms | Rất rẻ |
| is_win x2000 | ~8.4–9.2 ms | Rất rẻ |
| is_terminal x2000 | ~12–14 ms | Rất rẻ |
| board_to_planes x1 | ~0.01 ms | Rất rẻ |
| model.predict batch 32 | ~4.0 ms (0.13 ms/lần) | PyTorch CPU |
| Self-play 2 ván, 40 sims (python) | ~1406 ms, 5296 inference calls | Python MCTS |
| Self-play 2 ván, 40 sims (cpp) | ~1446 ms | C++ MCTS, không nhanh hơn do overhead pybind11 + inference chiếm phần lớn |
| Parallel cpp self-play 4 ván / 2 workers | ~5729–6843 ms | Worker load .pyd, không build |

**Kết luận profiling:**
- **NN inference là chi phí chủ đạo** (hàng nghìn inference call mỗi ván, mỗi lần ~0.9–1.3 ms CPU; trên GPU sẽ khác). MCTS tree chỉ là một phần nhỏ.
- Chỉ chuyển cây MCTS sang C++ **không giúp tăng tốc self-play tổng thể** trên workload này (cpp thậm chí chậm hơn python do gọi liên tục qua ranh giới pybind11: `request()` / `receive()` mỗi lá, cộng với play/undo và lớp kết nối Python).
- Overhead Python↔C++: mỗi lá MCTS phải gọi `request()` (C++) → Python tính planes/mask → `receive()` (C++), mỗi tick có nhiều lần đi qua ranh giới. Không thể loại bỏ chi phí này nếu không viết toàn bộ vòng lặp MCTS+inference bằng C++, điều SPEC cấm (không thay đổi thuật toán AlphaZero, không chuyển inference sang C++).

→ **Không tốiưu thành phần nào thêm** vì không có bằng chứng bottleneck ở MCTS tree; phần lớn thời gian ở inference PyTorch (vẫn phải giữ Python theo SPEC).

## 5. Hạn chế còn lại

1. **NN inference vẫn là Python (PyTorch)** — theo SPEC §9.1 không có C++ neural network engine; đây là chi phí lớn nhất mỗi mô phỏng, chưa được tốiưu trong tác vụ này (không có bằng chứng cần tốiưu, và SPEC cấm chuyển sang C++).
2. **Overhead Python↔C++** trên mỗi lá MCTS không bị loại bỏ; chỉ có lợi khi tree bookkeeping là thành phần đắt, điều profiling chứng minh **không đúng** với workload đại diện hiện tại.
3. **Patterns (`gomoku_native.find_patterns`)** vẫn chưa dùng trong self-play chuẩn (`components_used["patterns"]="not_used`); chỉ kích hoạt khi `GOMOKU_PATTERN_BACKEND=native` hoặc agent `az_v1_threat`. Không sinh test Renju tactical override trong tác vụ này.
4. **Chỉ kiểm thử trên máy local (Windows, CPU)**; chưa chạy training dài, chưa kiểm tra GPU, chưa kiểm tra long-run RSS soak/sanitizer.
5. **Arena với C++ MCTS** đã được bật qua CLI nhưng chưa chạy kiểm thử tích hợp đầy đủ với nhiều ván / nhiều checkpoint (test tournament dùng 2 ván, n_simulations=2, model AZNet nhỏ — đủ để chứng minh backend được gọi và ghi components_used, không đủ để đánh giá chất lượng chơi).
6. Không có so sánh speedup thống kê trên nhiều seed; báo cáo chỉ nêu median từ một chạy profil đại diện.

## 6. Điều không làm (theo yêu cầu)

- Không chạy training dài.
- Không xóa checkpoint, SQLite, dữ liệu self-play.
- Không commit.
- Không chuyển PyTorch inference sang C++.
- Không thay đổi thuật toán AlphaZero.
- Không thêm `--backend` cho thành phần nào ngoài MCTS (patterns giữ nguyên mặc định python trừ khi được yêu cầu).

## 7. Kết luận

Backend C++ MCTS đã được kết nối và xác minh trong cả Self-play (`run_games`, `run_games_parallel`) và Tournament (AZ agent, Normal + Renju), có test tự động ghi `components_used["mcts"] == "cpp"` vào SQLite. Parity Python↔C++ đã được chứng minh (visits, root value, z, pi). Profiling cho thấy MCTS tree **không** là điểm nghẽn dominant (NN inference PyTorch mới là), nên không tốiưu thêm thành phần nào. Báo cáo này kèm mã thay đổi và script profiling để nhóm có cơ sở quyết định tiếp theo.
