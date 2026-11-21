# Audit mức độ tích hợp C++ Backend — Gomoku AlphaZero

**Ngày:** 2026-11-20 | **Khoảng làm:** Chi tiết, không sửa code, không commit | **Trạng thái .pyd:** `gomoku_mcts_native.cp315-win_amd64.pyd` và `gomoku_native.cp315-win_amd64.pyd` có sẵn, import được, `Tree` khởi tạo được (simulations_done=0, size=1, root_value=0.0).

## 1. Sơ đồ kiến trúc thực thi

### Self-play (`az/selfplay.py` → `scripts/train.py`)

```
CLI (--backend / --mcts-backend / --workers)
   │
   ▼
scripts/train.py:main()
   ├─ configure_backends() ──┬─→ chọn mcts_backend (python/cpp/auto)
   │                         └─→ fail closed nếu explicit cpp thiếu .pyd
   │
   ├─ run_games() (1 worker) / run_games_parallel() (N workers)
   │     │
   │     ├─ _new_game(): Board() + rules + opening plies  [Python]
   │     │
   │     ├─ MCTS chọn class:
   │     │     ├─ mcts_backend=="cpp" → NativeMCTS (adapter qua pybind11)
   │     │     │     ├─ request(): Tree.request() [C++] +
   │     │     │     │         board.play/undo, rules.is_win/is_terminal,
   │     │     │     │         _get_legal_mask, _get_planes [Python]
   │     │     │     └─ receive(): prior masking + Dirichlet [Python] +
   │     │     │             Tree.receive() [C++]
   │     │     └─ khác → MCTS (pure Python) [Python]
   │     │
   │     ├─ model.predict(batch planes) [PyTorch/Python]
   │     │
   │     └─ _advance() / _finish_game(): pi, z, GameRecord, ReplayBuffer [Python]
   │
   └─ _evaluate_promotion() (Elo gate, mỗi 5000 ván) [Python + sqlite]
```

### Tournament (`scripts/tournament.py` / `scripts/arena.py`)

```
CLI tournament (--backend / --mcts-backend / --agents)
   │
   ▼
scripts/tournament.py:run_tournament()
   ├─ configure_backends() ──┬─→ mcts_backend, patterns_backend
   │                         └─→ ghi components_used vào config + SQLite
   │
   ├─ default_builder() → create_player() → AZPlayer()
   │     └─ AZPlayer.select_move(): MCTS backend theo cfg [Python orchestrator,
   │                                        C++ Tree nếu NativeMCTS]
   │
   ├─ play_game() / evaluate() [pure Python game loop]
   │
   └─ ghi tournament.sqlite (experiment, agent, match, game, move) [Python]

CLI arena (--a / --b / --games / --config ...)
   │
   ▼
scripts/arena.py:main()
   ├─ backend = "python" hardcoded (arena.py:125)
   ├─ KHÔNG có option --backend / --mcts-backend
   ├─ make_player/make_named_player → AZPlayer (cfg từ YAML, mcts_backend mặc định "python")
   └─ az.evaluator.evaluate() [pure Python]
```

## 2. Bảng đối chiếu từng thành phần

| Thành phần | Ngôn ngữ thực thi | Tích hợp C++? | By chứng (file:ligne) |
|---|---|---|---|
| **Board** (state, play, undo, current_player, history) | Python | Không | `gomoku/board.py`; `selfplay.py:150, 227`; `mcts.py:365-366, 381-382` |
| **Rules** (is_win, is_terminal, legal_moves, forbidden_moves) | Python | Không | `gomoku/rules/normal.py`, `renju.py`; `selfplay.py:154, 241`; `mcts.py:368-369` |
| **Legal-move mask** | Python | Không | `mcts.py:324-328` (`_get_legal_mask`) |
| **Feature planes** (`board_to_planes`) | Python | Không | `gomoku/features.py`; `mcts.py:321` |
| **MCTS tree** (selection PUCT, expansion, backup, visit counts) | C++ (pybind11 `Tree`) | Có, qua adapter | `native/gomoku_mcts_native.cpp:29-191`; `az/mcts.py:330` `NativeMCTS` |
| **MCTS orchestrator** (request/receive loop, play/undo, mask, planes, Dirichlet) | Hybrid | C++ tree + Python glue | `az/mcts.py:352-410` (`NativeMCTS.request/receive`) |
| **MCTS纯 Python** (fallback) | Python | Không | `az/mcts.py:41` `class MCTS` |
| **Neural-network inference** (`model.predict`) | Python (PyTorch) | Không | `az/model.py`; `selfplay.py:134`; `az_player.py:61` |
| **Self-play orchestrator** (`run_games`, `run_games_parallel`) | Python | Không | `az/selfplay.py:67`, `283` |
| **Replay buffer** (numpy) | Python | Không | `az/replay_buffer.py` |
| **Pattern detection** (`find_patterns`) | C++ (pybind11) | Có, nhưng chưa dùng trong self-play | `native/gomoku_native.cpp:19-88`; `gomoku/patterns.py:26`; `components_used["patterns"]="not_used"` (`train.py:358`) |
| **Tournament scheduler / pairing** | Python | Không | `scripts/tournament.py:671` `run_tournament` |
| **Arena play_game / evaluate** | Python | Không | `az/evaluator.py:23`, `76` |
| **AZPlayer** (tournament agent) | Python | MCTS backend selectable | `players/az_player.py:14, 24, 40` |
| **create_player** (backend resolution) | Python | Gọi `configure_backends` | `players/factory.py:24, 81-85` |
| **configure_backends** (backend selector) | Python | Điều phối load .pyd | `az/backend.py:95-171` |
| **Elo recalculation** (`recalculate_elo`, `paired_davidson_ci`) | Python + sqlite | Không | `az/elo.py:13-113`, `122-197` |
| **SQLite logging** (selfplay + tournament) | Python | Không | `az/logging_v2.py` |

## 3. Bảng tổng hợp mức độ chuyển đổi C++

| Hệ thống | Thành phần nào chạy C++ | Thành phần nào vẫn Python | Mức độ chuyển đổi |
|---|---|---|---|
| **Self-play** | Chỉ cây tìm kiếm MCTS qua `NativeMCTS` → `gomoku_mcts_native.Tree` (selection/backup/visit counts) | Board, rules, legal moves, features, NN inference, game loop, batching, replay buffer, logging, Elo gate | **Hẹp**: chỉ MCTS tree bookkeeping; không có C++ NN, không có complete self-play engine |
| **Tournament** | Cùng MCTS tree qua `NativeMCTS` (nếu AZ agent được chọn và `mcts_backend="cpp"`) | Toàn bộ scheduler, play_game loop, evaluator, model inference, SQLite logging, Elo math | **Rất hẹp**: chỉ xảy ra khi agent là AZ và backend được chọn cpp; arena mặc định python |

## 4. Backend selection: python / cpp / auto

### `az/backend.py:configure_backends()` (nút điều phối duy nhất)

- Giá trị hợp lệ: `auto`, `python`, `cpp` cho `--backend`; `--mcts-backend` / `--patterns-backend` chỉ nhận `python`/`cpp`.
- **Explicit `cpp`:** fail closed. Nếu `.pyd` thiếu hoặc không tương thích → `ImportError`, **không** fallback âm thầm về Python (`backend.py:123-132`). `scripts/train.py:348` chuyển thành `SystemExit`.
- **`auto`:** chọn `cpp` nếu có compiler (g++/clang++/cl) (`backend.py:28-35`); nếu build/load thất bại → `RuntimeWarning` + dùng Python (`backend.py:114-120`, `134-143`).
- **Worker process** (không phải MainProcess): `_build_forbidden=True` (`backend.py:99-101`); chỉ load, không build; unresolved `auto` → warning + Python (`backend.py:117-118`).

### scripts/train.py (self-play training)

- CLI: `--backend` (default `auto`, train.py:51), `--mcts-backend` (default `None` → kế thừa `--backend`, train.py:52).
- `main()` gọi `configure_backends()` một lần (train.py:346-349), lưu `cfg["mcts_backend"]` (train.py:355), truyền vào `run_games` / `run_games_parallel` qua `runtime_cfg` (train.py:372, 511-521).
- `az/selfplay.py:158`: `NativeMCTS if cfg.get("mcts_backend","python") == "cpp" else MCTS`.

### scripts/tournament.py (tournament CLI)

- `run_tournament(entries, backends=None, ...)` (`tournament.py:688`); nếu `None` → `configure_backends("auto", ...)` (`tournament.py:700-706`).
- `default_builder()` truyền `settings["backend"]["mcts"]` / `["patterns"]` vào `create_player` (`tournament.py:197-199`).
- CLI: `--backend`, `--mcts-backend`, `--patterns-backend` có sẵn (`tournament.py:1075-1077`); `main()` gọi `configure_backends` rồi truyền `backends` vào `run_tournament` (`tournament.py:1118-1124`, `1133`).
- Nếu `auto` và không có AZ-kind agent → MCTS mặc định `python` (`tournament.py:1113-1115`).

### scripts/arena.py (arena CLI)

- `backend = "python"` hardcoded (`arena.py:125`).
- **Không có** option `--backend`, `--mcts-backend`, `--patterns-backend` (`arena.py:37-52`).
- `AZPlayer` đọc `cfg.get("mcts_backend", "python")` (`az_player.py:24`); YAML configs không có key này → Python.

## 5. By chứng kiểm thử

| Test file | Test liên quan C++ | Backend được dùng | Kết quả |
|---|---|---|---|
| `tests/test_native_mcts.py` | `test_native_mcts_matches_python_requests_visits_and_backup` | `mcts_backend="cpp"` | Parity visits/backup với Python |
| | `test_native_mcts_masks_renju_forbidden_moves` | `cpp` | Renju forbidden mask đúng |
| | `test_native_mcts_matches_python_on_renju_forbidden_fixture` | `cpp` | `root_value` bằng Python |
| | `test_native_renju_exact_five_legal_and_overline_forbidden` | `cpp` | Exactly-5 legal, overline forbidden |
| | `test_native_mcts_win_in_one_and_terminal_root` | `cpp` | Terminal root trả về `None` |
| | `test_native_tree_repeated_search_releases_nodes` | `cpp` | `tree_size()==1` sau reset |
| | `test_native_selfplay_keeps_batching_and_sample_targets` | `cpp`, `run_games` | 2 records, samples match move counts, `mean_batch > 1` |
| | `test_native_selfplay_parallel_workers_store_samples` | `cpp`, `run_games_parallel` (2 workers) | `components_used["mcts"] == "cpp"` |
| `tests/test_selfplay.py` | Không có test nào chạy `run_games` với `cpp` | `python` (mặc định qua `small_cfg()`) | Không kiểm thử đường dẫn C++ |
| `tests/test_tournament.py` | Không có test tournament với `cpp` | Tất cả agent là fake (không MCTS) | Không kiểm thử backend C++ |

**Giới hạn thực thi:** Không thể chạy full pytest vì `torch` chưa được cài trong môi trường (`az.model` → `az.mcts` import thất bại, `ModuleNotFoundError`). Smoke test duy nhất thực hiện được: import `gomoku_mcts_native` (.pyd), khởi tạo `Tree` thành công.

## 6. Thành phần còn thiếu hoặc chưa được sử dụng

1. **Model inference chưa có C++.** `model.predict` là PyTorch Python (`az/model.py`), đây là chi phí 주도 mỗi mô phỏng MCTS. Không có C++ neural network engine (SPEC §9.1 xác nhận).
2. **Patterns chưa dùng trong self-play chuẩn.** `components_used["patterns"]="not_used"` (`train.py:358`); `gomoku_native.find_patterns` chỉ được kích hoạt khi `GOMOKU_PATTERN_BACKEND=native` hoặc agent `az_v1_threat` (SPEC §9.1).
3. **Arena không hỗ trợ C++ MCTS qua CLI.** Hardcoded `backend="python"` và không có `--mcts-backend` (`arena.py:125`, `37-52`).
4. **`test_selfplay.py` và `test_tournament.py` không có test C++.** Chỉ `test_native_mcts.py` kiểm thử đường dẫn native.
5. **Không có complete C++ game/rules/board engine.** `gomoku_native` chỉ là pattern detection (SPEC §9.1: "not a native game/rules/board engine").
6. **Worker process không build .pyd.** `_build_forbidden=True` trong worker (`backend.py:99-101`); worker nhận pickled `state_dict` và chỉ load (`selfplay.py:318`, `_play_chunk:343-365`).

## 7. Đề xuất thứ tự ưu tiên hoàn thiện tích hợp C++

| Ưu tiên | Hành động | Lý do | Rủi ro |
|---|---|---|---|
| 1 | Thêm test C++ vào `test_selfplay.py` (chạy `run_games` với `mcts_backend="cpp"` và so sánh `pi`, `z`, winner với Python) | Đóng khoảng kiểm thử đường dẫn native trong self-play thực tế | Cần `torch` cài để chạy |
| 2 | Thêm option `--backend` / `--mcts-backend` vào `scripts/arena.py` (hiện hardcoded python) | Cho phép arena chạy AZ agent với C++ MCTS, đồng nhất với tournament | Thay đổi CLI arena |
| 3 | Xem xét chuyển `board_to_planes` / `legal_moves` sang C++ nếu profil chứng minh chúng là nghẹn sau khi NN inference đã tối ưu | Hiện MCTS tree là phần nhỏ nhất trong chi phí mỗi tick | Có thể không cần thiết nếu NN là chủ đạo |
| 4 | Kích hoạt `patterns` trong self-play nếu Renju tactical override cần (hiện `not_used`) | Hỗ trợ Renju fine-tuning | Cần test Renju forbidden-move parity |

## 8. Kết luận bắt buộc

**Self-play:** Chuyển sang C++ ở mức **rất hẹp**. Một thành phần duy nhất chạy native là cây tìm kiếm MCTS qua `gomoku_mcts_native.Tree`, được gọi thông qua adapter Python `NativeMCTS` (`az/mcts.py:330`), và chỉ khi `cfg["mcts_backend"] == "cpp"` (`az/selfplay.py:158`). Toàn bộ phần còn lại — board, rules, legal-move generation, feature planes, neural-network inference (PyTorch), vòng lặp self-play, batching, replay buffer, logging, Elo gate — vẫn chạy **Python**. Do NN inference được gọi mỗi lá mỗi tick và là chi phí dominant, việc chỉ chuyển MCTS tree sang C++ mang lại lợi ích giới hạn.

**Tournament:** Chuyển sang C++ ở mức **hẹp hơn nữa**. Chỉ xảy ra khi agent là AZ (`az_v0` 등) **và** `mcts_backend` được chọn thành `cpp` qua `create_player` → `AZPlayer` (`players/factory.py:81-85`, `az_player.py:40`). Toàn bộ scheduler, `play_game`/`evaluate` vòng lặp, model inference, SQLite logging, và tính toán Elo vẫn là **Python**. `scripts/arena.py` hardcoded `backend="python"` và không có `--mcts-backend` CLI, nên arena mặc định không bao giờ dùng C++ MCTS.

**Điểm nghẽn cần xử lý tiếp theo:** (1) NN inference là phần chiếm chi phí lớn nhất mỗi mô phỏng và vẫn là Python — nếu mục tiêu là tốc độ self-play, đây là điểm cần profiling trước; (2) thiếu kiểm thử C++ trong `test_selfplay.py` và `test_tournament.py`, donc đường dẫn native chưa được chứng minh trong luồng self-play/tournament thực tế; (3) arena không cho phép chọn C++ MCTS qua CLI. Tất cả ba điểm trên đều nằm ở Python hiện tại, không phải C++ đã tích hợp nhưng chưa dùng.

**Ràng buộc tuân thủ:** Báo cáo chỉ audit, không sửa code, không refactor, không commit. Tất cả bằng chứng trích dẫn trực tiếp từ source code đã đọc và kết quả từ hai subagent song parallel. Không gán tỷ lệ % hoàn thành vì không có tiêu chí đo rõ ràng trong repo.