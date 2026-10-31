# Task List: Gomoku 11x11 + AlphaZero

Plan đầy đủ: `plan.md` (cùng thư mục). Quy ước: mỗi task là lát dọc, code/identifier tiếng Anh, comment/docstring tiếng Anh, trao đổi với user tiếng Việt.

---

## Phân công A/B/C

| Người | Task | Giai đoạn | Sở hữu file |
|---|---|---|---|
| C | T1, T2 | 0 | `requirements.txt`, `config.py`, `configs/` |
| A | T3, T4, T5, T6, T36 | 1 | `gomoku/`, `players/`, `scripts/play.py` |
| A | T7, T27, T28, T33 | 2 | `gomoku/patterns.py`, `players/` |
| B | T8, T9, T10, T11, T12, T20 | 3, 7 | `az/` (trừ `az/evaluator.py`), `logs/`, `checkpoints/` |
| C | T13, T31 | 4 | `az/evaluator.py`, `scripts/arena.py`, `scripts/tournament.py` |
| B | T19 | 6 | `az/game_log.py` |
| C | T21 (gộp vào T31), T22, T26, T32, T34 | 8, 11 | `scripts/`, `docs/` |
| A | T23 | 9 | `ui/` |
| A | T24 | 10 | `gomoku/rules/renju.py` |
| B | T25, T29, T30 | 10, 12 | `az/` |
| B | T35 | 13 | `logs/tournament/` |

**Đã cắt khỏi phạm vi đồ án cuối kỳ:** T14 (`--device`, `--workers`, `--time-budget-hours`,
`--ckpt-every`), T15 (Colab), T16 (`sync_drive`), T17 (selfplay shard), T18 (merge shards).
Lý do: 1000 ván tự chơi chạy xong trong ~1 giờ trên RTX 4060 tại chỗ, và fine-tune Renju chỉ
~1,5 giờ, nên hạ tầng chạy song song không cần cho bất kỳ thí nghiệm nào trong kế hoạch.
`scripts/train.py` hiện đã có `--config`, `--iterations`, `--resume`, `--out-dir`, `--log-dir`,
`--seed`, `--games`, `--simulations`; đủ dùng. Các task đã cắt giữ lại làm hướng phát triển.

Quy ước phối hợp:

- C làm T1+T2 trước, để ba nhánh code song song từ cùng đường giao diện luật.
- Mỗi người sở hữu file của nhánh mình. Task đụng nhiều file do người sở hữu file dẫn dắt, người khác gửi yêu cầu thay đổi.
- T19: B định nghĩa schema `az/game_log.py`, A gắn vào `gomoku/runner.py`, C gắn vào `az/evaluator.py`.
- T25: B dẫn dắt phần tích hợp và fine-tune `checkpoints/renju_last.pt`; A cung cấp `gomoku/rules/renju.py` + `gomoku/features.py`; C đổi `configs/renju.yaml`.
- T29 và T30 do B sở hữu (`az/player_tactical.py`, `az/threat_space.py`), nhưng phụ thuộc `gomoku/patterns.py` của A: nếu T27 chưa qua test thì hai task này chưa được bắt đầu.
- T31 và T32 do C sở hữu; A và B chỉ cung cấp tác tử, không sửa script đánh giá.
- T12 và `scripts/train.py`: C sở hữu file CLI này, B viết `az/trainer.py` và giao diện gọi.
- T20 do B chạy, vì B sở hữu `az/trainer.py`; C không còn giữ task này.
- T22 chia: A và B viết dòng nguồn cho module của mình, C ráp `docs/provenance.md` và viết `docs/README.md`.

---

## Task 1: Skeleton repo + dependency + pytest

**Description:** Tạo cấu trúc thư mục rỗng theo kiến trúc AGENTS.md, file dependency, cấu hình pytest, `.gitignore`, `git init`. Chưa viết logic game.

**Acceptance criteria:**
- [x] `requirements.txt` liệt kê: torch (cu124), numpy, pyyaml, pytest, pygame
- [x] `pip install -r requirements.txt` chạy được, `torch.cuda.is_available()` trả True
- [x] `pytest` chạy trong repo, exit code 0 khi chưa có test
- [x] `.gitignore` loại `checkpoints/`, `logs/`, `__pycache__/`, `*.pt`, `.venv/`
- [x] `git init` xong, commit đầu tiên tạo được, `git status` không thấy file rác
- [x] Chưa cần remote: user tạo repo GitHub và push sau, làm theo `colab/README.md` (T15)

**Verification:**
- [ ] Test: `python -c "import torch; print(torch.cuda.is_available())"` → True
- [ ] Test: `pytest -q` → pass
- [ ] Build: không có (repo Python thuần)

**Dependencies:** None
**Files likely touched:** `requirements.txt`, `.gitignore`, `pyproject.toml` hoặc `pytest.ini`, `.git/`
**Estimated scope:** S (1-2 files)

---

## Task 2: Config loader + file config

**Description:** `config.py` đọc YAML, trả dict có key mặc định. `configs/normal.yaml` và `configs/renju.yaml` chứa hyperparameter theo AGENTS.md.

**Acceptance criteria:**
- [x] `load_config(path)` trả dict, thiếu key dùng giá trị mặc định từ `DEFAULT_CONFIG`
- [x] `configs/normal.yaml` có: `n_iter`, `games_per_iter`, `n_simulations`, `c_puct`, `dirichlet_alpha`, `dirichlet_eps`, `temp_moves`, `buffer_size`, `train_steps_per_iter`, `batch_size`, `arena_win_threshold`, `learning_rate`, `optimizer`, `weight_decay`, `value_loss_weight`, `policy_loss_weight`, `lr_schedule`
- [x] `configs/renju.yaml` kế thừa giá trị normal, đổi `rule_set: renju`
- [x] Giá trị mặc định khớp khối "Default hyperparameters" trong AGENTS.md, cộng thêm: `learning_rate: 1e-3`, `optimizer: adam`, `weight_decay: 1e-4`, `value_loss_weight: 1.0`, `policy_loss_weight: 1.0`, `lr_schedule: cosine`
- [x] Ngân sách ván suy ra từ cấu hình: `n_iter × games_per_iter` (mặc định 20 × 50 = 1000). Đổi sang 24 × 50 = 1200, 30 × 50 = 1500 hay 100 × 50 = 5000 chỉ cần sửa file cấu hình
- [x] Không con số 1000 nào hardcode trong mã; mọi nơi đọc ngân sách từ cấu hình

**Verification:**
- [ ] Test: `tests/test_config.py` — load file thật, assert key và giá trị
- [ ] Test: `pytest -q tests/test_config.py` → pass
- [ ] Build: `python -c "from config import load_config; print(load_config('configs/normal.yaml')['n_iter'])"` → 20

**Dependencies:** T1
**Files likely touched:** `config.py`, `configs/normal.yaml`, `configs/renju.yaml`, `tests/test_config.py`
**Estimated scope:** S (1-2 files)

---

## Task 3: `gomoku/board.py` — state bàn cờ

**Description:** Lớp `Board` giữ state dạng numpy int8 phẳng, có `play`, `undo`, `history`, `current_player`. Không chứa luật chơi.

**Acceptance criteria:**
- [x] Bàn **11x11 cố định**: hằng số `SIZE = 11` trong `gomoku/board.py` là nguồn duy nhất; không có tham số kích thước
- [x] `play(r, c)` đặt đá của người đang đi, tăng `history`, đổi lượt; ngoài bàn hoặc ô đã có đá thì raise
- [x] `undo()` trả về nước vừa nhập và khôi phục state trước đó
- [x] `grid` là numpy int8 1D, index phẳng `r * SIZE + c`; `EMPTY=0, BLACK=1, WHITE=2`
- [x] `gomoku/` không import torch

**Verification:**
- [ ] Test: `tests/test_board.py` — play/undo/history, lỗi out-of-range, lỗi ô bận, đổi lượt
- [ ] Test: `pytest -q tests/test_board.py` → pass
- [ ] Manual: `python -c "from gomoku.board import Board; b=Board(); b.play(5,5); print(b.grid.max())"` → 1

**Dependencies:** T2
**Files likely touched:** `gomoku/board.py`, `gomoku/__init__.py`, `tests/test_board.py`
**Estimated scope:** S (1-2 files)

---

## Task 4: `gomoku/rules/base.py` + `normal.py` — luật Normal

**Description:** Interface `Rules` với `legal_moves`, `is_win`, `is_terminal`, `forbidden_moves` (mặc định rỗng). `NormalRules` kiểm tra 5+ liền nhau (ngang, dọc, 2 đường chéo), 6+ cũng thắng.

**Acceptance criteria:**
- [x] `legal_moves(board)` trả list ô trống, không có ô đã có đá
- [x] `is_win(board)` True khi có 5 liền hoặc nhiều hơn theo 4 hướng
- [x] `is_terminal(board)` True khi thắng hoặc bàn đầy
- [x] Vector hóa: không có vòng lặp Python quét 121 ô cho mỗi lần gọi
- [x] `forbidden_moves(board)` trả list rỗng cho Normal

**Verification:**
- [ ] Test: `tests/test_rules_normal.py` — thắng ngang, dọc, chéo `\`, chéo `/`, thắng 6 quân, hoà khi bàn đầy, legal_moves loại ô bận
- [ ] Test: `pytest -q tests/test_rules_normal.py` → pass
- [ ] Build: `python -c "import gomoku.rules.normal"` → không lỗi

**Dependencies:** T3
**Files likely touched:** `gomoku/rules/base.py`, `gomoku/rules/normal.py`, `gomoku/rules/__init__.py`, `tests/test_rules_normal.py`
**Estimated scope:** M (3-5 files)

---

## Task 5: `gomoku/features.py` — plane + symmetry

**Description:** `board_to_planes(board, rules, player)` trả tensor numpy shape `(4, 11, 11)`. Hàm transform cho 8 phép đối xứng, áp dụng đồng thời lên features và policy.

**Acceptance criteria:**
- [x] 4 plane cố định: `my_stones`, `opp_stones`, `forbidden`, `color_to_move`
- [x] Plane `forbidden` bằng 0 khi rule set không cấm nước nào
- [x] `symmetry_transform(array, k)` trả 8 biến thể; phép tự nghịch áp 2 lần ra identity, rot90 và rot270 nghịch nhau; một hàm dùng chung cho cả features và policy
- [x] Policy one-hot tại `(r,c)` biến đổi ra đúng ô ảnh, khớp với phép biến đổi áp lên plane (cùng `symmetry_transform`)
- [x] Không import torch (trả numpy; torch chuyển đổi ở tầng az/)

**Verification:**
- [ ] Test: `tests/test_features.py` — shape, giá trị plane, 8 biến thể khác nhau, transform∘transform = identity, policy transform khớp feature transform
- [ ] Test: `pytest -q tests/test_features.py` → pass
- [ ] Manual: `python -c "from gomoku.features import board_to_planes"` → OK

**Dependencies:** T4
**Files likely touched:** `gomoku/features.py`, `tests/test_features.py`
**Estimated scope:** S (1-2 files)

---

## Task 6: Player base + random + human + runner + CLI

**Description:** `players/base.py` định nghĩa `select_move(board) -> (r, c)`. `RandomPlayer`, `HumanPlayer`. `gomoku/runner.py` chạy ván tới khi kết thúc, trả kết quả. `scripts/play.py` CLI cho human-vs-AI và AI-vs-AI.

**Acceptance criteria:**
- [x] `Player.select_move(board)` là interface duy nhất mà runner gọi
- [x] `RandomPlayer` chọn ngẫu nhiên trong `rules.legal_moves`
- [ ] `HumanPlayer` nhận tọa độ từ callback, từ chối nước không hợp lệ
- [ ] `Runner.play()` trả `GameResult` (winner, moves, reason: win/draw)
- [ ] `scripts/play.py --mode human-vs-random` chơi được hết ván qua CLI, in bàn cờ mỗi nước
- [ ] `scripts/play.py --mode ai-vs-ai` chạy ván random-vs-random không cần input
- [ ] `gomoku/runner.py` không import torch

**Verification:**
- [ ] Test: `tests/test_runner.py` — ván random-vs-random kết thúc hợp lệ, số nước = len(history), người thắng đúng khi có 5 liền
- [ ] Test: `pytest -q tests/test_runner.py` → pass
- [ ] Manual: `python scripts/play.py --mode human-vs-random` → nhập tọa độ, chơi tới khi thắng/hoà

**Dependencies:** T3, T4
**Files likely touched:** `players/base.py`, `players/random.py`, `players/human.py`, `players/__init__.py`, `gomoku/runner.py`, `scripts/play.py`, `tests/test_runner.py`
**Estimated scope:** M (3-5 files)

---

## Task 7: `players/mcts_pure.py` — baseline rollout

**Description:** MCTS thuần dùng luật + heuristic không có mạng. Làm đối thủ thật để đo chất lượng agent. Chỉ gọi interface rules.

**Acceptance criteria:**
- [ ] `MCTSPure(rules, n_rollouts=...)` chọn nước bằng UCT + rollout ngẫu nhiên
- [ ] Chỉ dùng `rules.legal_moves` / `rules.is_win` / `rules.is_terminal`, không hardcode luật
- [ ] Vị trí thắng trong 1 nước → chọn đúng nước thắng
- [ ] Thắng random >60% trên 11x11 trong 20 ván (cả hai màu)

**Verification:**
- [ ] Test: `tests/test_mcts_pure.py` — case win-in-1; case không chọn nước bị cấm
- [ ] Test: `pytest -q tests/test_mcts_pure.py` → pass
- [ ] Manual: script đấu 20 ván 11x11, in tỉ lệ thắng >0.6

**Dependencies:** T4, T6
**Files likely touched:** `players/mcts_pure.py`, `tests/test_mcts_pure.py`
**Estimated scope:** S (1-2 files)

---

## Task 8: `az/model.py` — ResNet policy/value

**Description:** ResNet nhỏ (5-6 block, 64 filter) theo AGENTS.md, policy head phát ra 121 logits, value head 1 giá trị. API suy luận batch. Save/load checkpoint.

**Acceptance criteria:**
- [x] Input shape `(B, 4, 11, 11)` — số plane cố định 4 cho mọi rule set
- [x] Policy head ra `(B, 121)`, value head ra `(B, 1)`
- [x] `predict(batch)` chạy ở `torch.no_grad()`, trả numpy, ổn định giữa 2 lần gọi ở eval mode
- [x] Tự chọn device: cuda nếu có, không thì cpu
- [x] `save/load_checkpoint` round-trip giữ nguyên trọng số, load được trên máy khác (state_dict + config)

**Verification:**
- [ ] Test: `tests/test_model.py` — shape output, round-trip save/load, batch >1 cho ra kết quả giống batch=1 (tính cùng input)
- [ ] Test: `pytest -q tests/test_model.py` → pass
- [ ] Manual: `python -c "import torch; from az.model import ..."` → chạy được trên cuda

**Dependencies:** T5
**Files likely touched:** `az/model.py`, `az/__init__.py`, `tests/test_model.py`
**Estimated scope:** M (3-5 files)

---

## Task 9: `az/mcts.py` — PUCT, đánh giá lá 1 lượt suy luận, backup đúng dấu

**Description:** Cây tìm kiếu PUCT: Dirichlet noise ở root, temperature theo số nước đã đi. Đánh giá vị trí đúng theo AlphaZero: mỗi lượt mô phỏng đi tới một lá, lá được đánh giá bằng đúng một lượt suy luận; lượt suy luận đó trả về xác suất cho cả 121 ô (làm tiên đoán cho mọi nước đi từ lá) và giá trị của lá. Backup giá trị đúng dấu.

**Acceptance criteria:**
- [x] Root có Dirichlet noise (alpha/eps theo config), chỉ ở root
- [x] Nhiệt độ: `temp_moves` nước đầu dùng temp=1, sau đó giảm dần về 0 (chọn argmax)
- [x] Mỗi lá được đánh giá bằng đúng 1 lượt suy luận; xác suất của cả 121 ô được gán làm tiên đoán cho các nước đi từ lá, không suy luận riêng cho từng nước con
- [x] Backup: giá trị của node con là góc nhìn đối thủ → phủ định khi backup (hard rule 2)
- [x] Nước bị cấm (Renju) không bao giờ được mở rộng (hard rule 4)
- [x] Vị trí thắng trong 1 nước → trả về nước thắng
- [x] `n_simulations` đọc từ config (mặc định 200); số lượt mô phỏng quyết định cả độ mạnh lẫn thời gian chạy nên phải cấu hình được

**Verification:**
- [ ] Test: `tests/test_mcts.py` — win-in-1; test dấu value backup (giá trị con âm ở góc nhìn cha); test nước cấm bị loại khỏi cây
- [ ] Test: `tests/test_mcts.py` — số lượt gọi NN bằng đúng số lá đã mở rộng (không nhân với số nước con)
- [ ] Test: `pytest -q tests/test_mcts.py` → pass
- [ ] Manual: đếm số lần gọi NN cho 1 search, log ra

**Dependencies:** T8
**Files likely touched:** `az/mcts.py`, `tests/test_mcts.py`
**Estimated scope:** M (3-5 files)

---

## Task 10: `az/replay_buffer.py` — lưu mẫu + đối xứng

**Description:** Buffer chứa (features, policy mục tiêu, value z) kèm nhãn rule set. Augmentation 8-fold lúc lấy batch.

**Acceptance criteria:**
- [x] `add_sample(features, pi, z, rule_set)` lưu đủ metadata
- [x] `sample_batch(batch_size)` trả batch numpy/torch đã augment 8-fold ngẫu nhiên
- [x] Augmentation dùng đúng hàm transform của `gomoku/features.py`, policy target đi kèm feature
- [x] Truy vấn theo rule set: không lẫn mẫu Normal và Renju (hard rule 7)
- [x] `__len__` trả số mẫu trước augmentation; giới hạn `buffer_size`

**Verification:**
- [ ] Test: `tests/test_replay_buffer.py` — số mẫu, len đúng, lấy batch đúng shape, filter theo rule set, augment cho 8 biến thể khác nhau
- [ ] Test: `pytest -q tests/test_replay_buffer.py` → pass
- [ ] Manual: không cần

**Dependencies:** T5, T9
**Files likely touched:** `az/replay_buffer.py`, `tests/test_replay_buffer.py`
**Estimated scope:** S (1-2 files)

---

## Task 11: `az/selfplay.py` — sinh ván

**Description:** Sinh ván bằng agent MCTS với nhiệt độ và noise, lưu mẫu vào buffer với z tính sau khi ván kết thúc. Chế độ multiprocessing cho self-play.

**Acceptance criteria:**
- [x] Số mẫu lưu mỗi ván = số nước đã đi
- [x] z lấy theo góc nhìn người đang đi tại mỗi thếp: thắng = +1 cho người thắng, thua = -1, hoà = 0; dấu xen kẽ đúng
- [x] Chế độ 1 tiến trình dùng để test; chế độ multiprocessing sinh đúng số ván yêu cầu
- [x] Worker set `torch.set_num_threads(1)`, có guard `if __name__ == "__main__"` (Windows spawn)
- [x] Mỗi ván lưu rule set hiện tại vào buffer
- [x] Gom lô ở cấp chương trình: lá đang chờ đánh giá từ nhiều ván song song được gửi trong cùng một lệnh suy luận; mỗi lá vẫn tốn đúng một lượt suy luận
- [x] Số ván sinh mỗi vòng đọc từ cấu hình (`games_per_iter`), không hardcode

**Verification:**
- [ ] Test: `tests/test_selfplay.py` — số mẫu = số nước; dấu z xen kẽ trong ván quyết định; multiprocess (2 worker) sinh đủ số ván
- [ ] Test: `pytest -q tests/test_selfplay.py` → pass
- [ ] Manual: không cần

**Dependencies:** T9, T10
**Files likely touched:** `az/selfplay.py`, `tests/test_selfplay.py`
**Estimated scope:** M (3-5 files)

---

## Task 12: `az/trainer.py` + `scripts/train.py`

**Description:** Vòng lặp train: self-play → thêm vào buffer → train N step → lưu checkpoint. Loss = policy CE + value MSE + L2.

**Acceptance criteria:**
- [x] Loss = `policy_CE * c_pi + value_MSE * c_v + L2`, trọng số đọc từ config
- [x] Overfit một batch cố định: loss giảm rõ rệt sau N step (sanity)
- [x] `scripts/train.py --config configs/normal.yaml --iterations 2` chạy xong, ghi checkpoint vào `checkpoints/`, log vào `logs/`
- [x] Có `--resume` để nối tiếp từ checkpoint
- [x] Checkpoint lưu optimizer state + config để train tiếp được

**Verification:**
- [ ] Test: `tests/test_trainer.py` — overfit batch giảm loss; checkpoint load cho train tiếp
- [ ] Test: `pytest -q tests/test_trainer.py` → pass
- [ ] Build: `python scripts/train.py --config configs/normal.yaml --iterations 2` → exit 0, có file checkpoint (smoke test trên 11x11 trước khi chạy full)
- [ ] Manual: xem `logs/` có loss giảm theo iteration

**Dependencies:** T11
**Files likely touched:** `az/trainer.py`, `scripts/train.py`, `tests/test_trainer.py`
**Estimated scope:** M (3-5 files)

> GĐ4 (B): `scripts/train.py` bản tối thiểu đã có với `--config`, `--iterations`, `--resume`, `--out-dir`, `--log-dir`, `--seed`, `--games`, `--simulations`. C tiếp quản file này ở T14 để thêm `--device`, `--workers`, `--time-budget-hours`, `--ckpt-every`.

---

## Task 13: `az/evaluator.py` + `scripts/arena.py` + cổng promote

**Description:** Arena đấu model mới với model tốt nhất (và với MCTSPure). Chỉ thay model tốt nhất khi tỉ lệ thắng vượt ngưỡng.

**Acceptance criteria:**
- [x] `evaluate(model_a, model_b, n_games, rules)` trả tỉ lệ thắng của A, đổi lượt đầu xen kẽ
- [x] `scripts/arena.py --new <ckpt> --best <ckpt> --games 100` in tỉ lệ thắng
- [x] Cổng promote: model mới thành best chỉ khi win rate ≥ `arena_win_threshold` (0.55); ngược lại giữ best cũ
- [x] Test bằng model giả để kiểm tra logic promote, không cần model thật
- [x] Kết quả ghi vào `logs/arena_*.json`

**Verification:**
- [ ] Test: `tests/test_evaluator.py` — logic promote đúng với tỉ lệ giả lập; arena trả tổng số ván đúng
- [ ] Test: `pytest -q tests/test_evaluator.py` → pass
- [ ] Manual: `python scripts/arena.py --new checkpoints/latest.pt --best checkpoints/best.pt --games 20` → in số liệu

**Dependencies:** T12
**Files likely touched:** `az/evaluator.py`, `scripts/arena.py`, `tests/test_evaluator.py`
**Estimated scope:** M (3-5 files)

---

## Task 14: `scripts/train.py` chạy được trên cloud (thiết bị, time budget, resume)

**Description:** Trainer phải chạy được trên Colab: chọn thiết bị, giới hạn số tiến trình self-play theo số core của máy, dừng gọn khi hết time budget (Colab kill session giữa chừng), lưu checkpoint định kỳ để resume, cố định seed, ghi log thời gian để so sánh local với Colab.

**Acceptance criteria:**
- [ ] `--device {auto,cuda,cpu}`: `auto` dùng cuda khi có, `cuda` báo lỗi rõ ràng nếu máy không có GPU
- [ ] `--workers N`: giới hạn số tiến trình self-play (Colab free có 2 core → `--workers 2`)
- [ ] `--time-budget-hours H`: trainer dừng sau H giờ, lưu checkpoint cuối, exit code 0
- [ ] `--ckpt-every N`: lưu `checkpoints/last.pt` sau mỗi N iteration (mặc định 1)
- [ ] `--seed S`: cố định RNG self-play và MCTS để resume cho kết quả tương đương
- [ ] `--out-dir DIR`: ghi checkpoint ra thư mục khác (trỏ thẳng vào Google Drive khi chạy trên Colab)
- [ ] Log mỗi iteration ghi: giây self-play, giây train, số lần gọi NN mỗi ván, win rate arena → `logs/train_*.jsonl`

**Verification:**
- [ ] Test: `tests/test_trainer_cli.py` — `--time-budget-hours` cực nhỏ → trainer thoát sớm, `last.pt` tồn tại, exit 0
- [ ] Test: `pytest -q tests/test_trainer_cli.py` → pass
- [ ] Build: `python scripts/train.py --config configs/normal.yaml --iterations 1 --workers 2 --ckpt-every 1 --out-dir <tmp>` → exit 0, có `last.pt`
- [ ] Manual: file log iteration có đủ 4 trường đo

**Dependencies:** T13
**Files likely touched:** `scripts/train.py`, `az/trainer.py`, `az/selfplay.py`, `tests/test_trainer_cli.py`
**Estimated scope:** M (3-5 files)

---

## Task 15: `colab/train.ipynb` + `configs/colab.yaml`

**Description:** Notebook dựng môi trường Colab: mount Drive, clone code từ GitHub, cài dependency còn thiếu mà không cài lại torch, chạy `scripts/train.py` với time budget theo giới hạn session, đồng bộ checkpoint về Drive. `configs/colab.yaml` để tham số vừa tầng free (2 CPU core, session 12h). Notebook này là instance GPU: nó vừa train vừa tự sinh ván; các instance CPU-only khác sinh shard (T17) để instance GPU này gộp.

**Acceptance criteria:**
- [ ] `configs/colab.yaml`: `workers: 2`, `n_simulations: 100`, `games_per_iter: 20`, `time_budget_hours: 10`, `train_steps_per_iter: 200`, `batch_size: 128`
- [ ] Notebook mount Drive, tạo `MyDrive/gomoku-11x11/{checkpoints,logs,shards}`
- [ ] Code lấy từ GitHub: `git clone $REPO_URL /content/gomoku` với `REPO_URL` khai báo ở cell đầu; repo private thì dùng token từ Colab Secret (`from google.colab import userdata`), không hardcode token trong notebook
- [ ] Clone xong phải có `scripts/train.py`, `configs/colab.yaml`, `az/`; commit SHA được in ra log
- [ ] Không cài lại torch (runtime đã có sẵn bản CUDA); in `torch.cuda.get_device_name()` để xác nhận đúng GPU
- [ ] Cell chạy trainer với `--resume` để nối tiếp khi chạy lại notebook
- [ ] Cell tùy chọn `--buffer-in` gộp shard do các instance CPU-only vừa ghi lên Drive
- [ ] Checkpoint ghi thẳng vào Drive; replay buffer chỉ nằm trong RAM, không ghi ra Drive
- [ ] Cell cuối in tổng thời gian chạy và win rate cuối cùng
- [ ] Đường dự phòng không cần GitHub: `scripts/sync_drive.py` (T16) copy code lên Drive, notebook copy từ Drive — một trong hai đường phải chạy được
- [ ] `colab/README.md`: hướng dẫn để user điền `REPO_URL` và tạo Colab Secret token — tạo repo GitHub, push code, tạo secret `GITHUB_TOKEN` trong Colab, dán URL vào cell đầu notebook, chạy

**Verification:**
- [ ] Test: `tests/test_colab_nb.py` — parse JSON notebook: có cell `git clone`, cell cài đặt, cell chạy `scripts/train.py`, cell mount Drive; `configs/colab.yaml` có `workers: 2`
- [ ] Test: `pytest -q tests/test_colab_nb.py` → pass
- [ ] Manual: chạy notebook trên Colab 1 iteration → có `last.pt` trong Drive
- [ ] Manual: kill session giữa chừng rồi chạy lại notebook → tiếp tục từ `last.pt`, không chạy lại từ iteration 0

**Dependencies:** T14
**Files likely touched:** `colab/train.ipynb`, `colab/README.md`, `configs/colab.yaml`, `tests/test_colab_nb.py`
**Estimated scope:** M (3-5 files)

---

## Task 16: `scripts/sync_drive.py` — đẩy thư mục dự án lên Drive tự động

**Description:** Đường dự phòng cho trường hợp không muốn dùng GitHub (repo private, hoặc chưa có remote). Script copy thư mục dự án từ máy local lên thư mục Google Drive đã mount (Drive for Desktop trên Windows, ví dụ `G:\My Drive\gomoku-11x11`), bỏ qua `checkpoints/`, `logs/`, `.git/`, `__pycache__/`. Ghi manifest để biết trên Drive đang là phiên bản code nào. Không cần Google API key.

**Acceptance criteria:**
- [ ] `python scripts/sync_drive.py --dest "G:\My Drive\gomoku-11x11"` copy mã nguồn, configs, tests, notebook
- [ ] Loại trừ `checkpoints/`, `logs/`, `.git/`, `__pycache__/`, `*.pt`, `*.ipynb_checkpoints`
- [ ] Ghi `MANIFEST.txt` ở đích: số file, timestamp, git commit nếu repo có git
- [ ] Chạy lại không nhân bản: chỉ ghi đè file đã đổi, file không đổi giữ nguyên
- [ ] `--dry-run` in danh sách file sẽ copy rồi thoát, không ghi gì

**Verification:**
- [ ] Test: `tests/test_sync_drive.py` — copy vào thư mục tạm, assert file cần copy có mặt và file loại trừ không có
- [ ] Test: `pytest -q tests/test_sync_drive.py` → pass
- [ ] Build: `python scripts/sync_drive.py --dest <tmp> --dry-run` → in danh sách, không tạo file

**Dependencies:** T1
**Files likely touched:** `scripts/sync_drive.py`, `tests/test_sync_drive.py`
**Estimated scope:** S (1-2 files)

---

## Task 17: `scripts/selfplay_shard.py` — sinh shard self-play cho instance chạy song song

**Description:** Chế độ sinh shard: một instance (GPU hoặc CPU) sinh N ván từ một checkpoint rồi ghi `shard_<model_iter>_<id>.npz` lên Drive. Mỗi instance một id riêng nên không đè lên nhau. Chạy được trên máy không có GPU vì mạng nhỏ, suy luận CPU chấp nhận được. Trên Colab free: instance CPU-only (Runtime type = None) sinh shard, instance GPU (T4) vừa train vừa sinh ván.

**Acceptance criteria:**
- [ ] `python scripts/selfplay_shard.py --model <ckpt> --games 50 --id A --out <dir>` sinh file npz
- [ ] Npz chứa: features (uint8), policy (float32), value (float8), metadata `model_iter`, `games`, `rule_set`, `id`
- [ ] Chạy được khi không có GPU: torch dùng device cpu, không raise khi thiếu CUDA
- [ ] Tên file unique theo `--id`: `shard_<model_iter>_<id>.npz`
- [ ] Features lưu uint8 để shard nhỏ: 1 ván ~60 mẫu × 484 byte ≈ 30 KB
- [ ] Số mẫu trong shard = tổng số nước đã đi của các ván trong shard

**Verification:**
- [ ] Test: `tests/test_selfplay_shard.py` — sinh shard 2 ván, npz load được, shape và metadata đúng, số mẫu = tổng nước đi
- [ ] Test: `pytest -q tests/test_selfplay_shard.py` → pass
- [ ] Manual: chạy shard 1 ván trên CPU, xem kích thước file trên đĩa

**Dependencies:** T11, T14
**Files likely touched:** `az/selfplay.py`, `scripts/selfplay_shard.py`, `tests/test_selfplay_shard.py`
**Estimated scope:** M (3-5 files)

---

## Task 18: `scripts/merge_shards.py` — gộp shard + nạp buffer ngoài

**Description:** Gộp các shard thành một buffer để trainer dùng. Sắp theo `model_iter` tăng dần để dữ liệu của model cũ nằm trước dữ liệu của model mới, tránh trộn policy của hai thế hệ.

**Acceptance criteria:**
- [ ] `python scripts/merge_shards.py --shards <dir> --out <buffer.npz>` gộp mọi shard trong thư mục
- [ ] Sort theo `model_iter` tăng dần; mẫu của model cũ luôn nằm trước mẫu của model mới
- [ ] `scripts/train.py --buffer-in <buffer.npz>` nạp buffer đã gộp trước khi train tiếp
- [ ] Shard hỏng (npz lỗi) bị bỏ qua kèm cảnh báo, không làm hỏng cả lượt merge
- [ ] Metadata rule set được giữ: buffer gộp không lẫn Normal và Renju

**Verification:**
- [ ] Test: `tests/test_merge_shards.py` — 2 shard từ 2 model → merge giữ đúng thứ tự; shard hỏng bị bỏ qua
- [ ] Test: `pytest -q tests/test_merge_shards.py` → pass
- [ ] Manual: merge 2 shard thật rồi `train.py --buffer-in`, loss tiếp tục giảm

**Dependencies:** T10, T17
**Files likely touched:** `az/replay_buffer.py`, `scripts/merge_shards.py`, `tests/test_merge_shards.py`
**Estimated scope:** M (3-5 files)

---

## Task 19: Nhật kí thí nghiệm — schema ván và schema nước đi

**Description:** Mọi ván đều phải để lại dấu vết để trả lời "thua vì sao" và "ai đi trước thắng hay thua". Ghi log dạng JSONL: mỗi ván 1 dòng, kèm schema nước đi tùy chọn. Runner, self-play và script đánh giá dùng chung một bộ ghi log.

**Acceptance criteria:**
- [x] Schema ván (JSONL, 1 dòng/ván): `game_id`, `rule_set`, `model_iter`, `config`, `first_player`, `second_player`, `winner` (`black`/`white`/`draw`/`none`), `end_reason` (`five_in_row`/`forbidden_move`/`board_full`/`timeout`), `moves_count`, `duration_sec`, `nn_calls`
- [x] Schema nước đi (tùy chọn, `--log-moves`): `move_index`, `player`, `row`, `col`, `value_before`, `policy_top_k` (k giá trị + chỉ số), `simulations`
- [x] `end_reason` phân biệt thua vì bị cấm nước (Renju) với thua vì không tạo được 5 — hai nguyên nhân thua khác nhau
- [x] Log ghi vào `logs/games/<ngày>-<run_id>.jsonl`, một dòng = một ván, flush ngay khi ván kết thúc
- [x] Runner, self-play, đánh giá dùng chung đúng một bộ ghi log, không copy logic
- [x] `--log-moves` tắt mặc định: ghi từng nước làm chậm self-play, chỉ bật khi chạy thí nghiệm

**Verification:**
- [ ] Test: `tests/test_logging.py` — chơi 1 ván giả lập, đọc JSONL, assert đủ field và `end_reason` đúng khi thắng/hoà/cấm nước
- [ ] Test: `pytest -q tests/test_logging.py` → pass
- [ ] Manual: đọc 1 file log, thấy được ai thắng, vì sao, đi bao nhiêu nước

**Dependencies:** T6, T11, T12
**Files likely touched:** `az/game_log.py`, `gomoku/runner.py`, `az/selfplay.py`, `tests/test_logging.py`
**Estimated scope:** M (3-5 files)

---

## Task 20: Chạy train 11x11 theo config

**Description:** Chạy pipeline đầy đủ trên bàn 11x11: 20 iteration × 50 ván, 200 lượt mô phỏng mỗi nước. Đo win rate qua từng iteration. Lần đầu chạy local trên RTX 4060; các vòng sau có thể chuyển sang Colab hoặc chia self-play cho nhiều instance.

**Acceptance criteria:**
- [x] Mỗi iteration lưu checkpoint và ghi log loss + arena win rate
- [x] Từ iteration nào đó arena win rate vượt 0.55 so với model iteration 0
- [x] Agent 11x11 thắng random và thắng MCTSPure ở tỉ lệ đáng kể (đặt ngưỡng cụ thể ở log)
- [x] Log trong `logs/`, checkpoint trong `checkpoints/`, không commit
- [x] Cùng một lệnh train chạy được ở local lẫn Colab; log đủ để so sánh tốc độ hai nơi
- [x] Lần chạy đầu: local, đủ 20 iteration × 50 ván, không lỗi giữa chừng
- [x] Mở rộng ngân sách được bằng cách tăng `n_iter` trong cấu hình (24 × 50 = 1200, 30 × 50 = 1500, 100 × 50 = 5000 trong `configs/normal_5000.yaml`): chạy tiếp từ checkpoint, giữ nguyên buffer, không chạy lại từ vòng 0
- [x] Ngân sách và số vòng đọc từ cấu hình, không có con số 1000 hardcode trong mã

**Verification:**
- [ ] Manual: đọc `logs/` dựng được đường cong win rate tăng dần
- [ ] Manual: đấu agent với random 20 ván 11x11, thắng ≥ 18
- [ ] Build: `python scripts/train.py --config configs/normal.yaml` → chạy hết số iteration không lỗi
- [ ] Build: `python scripts/train.py --config configs/normal.yaml --resume` sau khi đổi `n_iter` 20 → 24 → chạy tiếp 4 vòng còn lại, tổng ván tăng đúng 200

**Dependencies:** T13, T15, T18, T19
**Files likely touched:** `configs/normal.yaml` (tinh chỉnh), `logs/`, `checkpoints/`
**Estimated scope:** M (config + chạy, không viết code mới nhiều)

---

## Task 21: `scripts/simulate.py` — 1000 ván đánh giá, đổi lượt đầu xen kẽ

**Description:** Script đánh giá chạy N ván (mặc định 1000) giữa hai bên, xen kẽ ai đi trước, ghi log từng ván, xuất bảng tổng hợp: tỉ lệ thắng theo người đi trước, theo người đi sau, theo màu, và theo nguyên nhân kết thúc ván.

**Acceptance criteria:**
- [ ] `python scripts/simulate.py --a <ckpt> --b <opponent> --games 1000 --out logs/eval_<run_id>.jsonl` chạy xong 1000 ván
- [ ] Ai đi trước xen kẽ: ván chẵn A đi trước, ván lẻ B đi trước → đúng 500/500 với `--games 1000`
- [ ] Đối thủ hỗ trợ: `random`, `mcts_pure`, hoặc path checkpoint khác
- [ ] Bảng tổng hợp in ra: A thắng khi đi trước, A thắng khi đi sau, tổng thắng/hòa/thua, phân bố `end_reason`
- [ ] Tổng hợp ghi cạnh log ván dưới dạng `summary.json`
- [ ] Mọi ván đều nằm trong log để truy vết lại
- [ ] `--games` là tham số: chạy 2000 ván chỉ cần đổi số, không sửa mã; số ván và tỉ lệ đi trước/đi sau luôn khớp
- [ ] Đối thủ MCTS thuần chạy cùng số lượt mô phỏng với agent (200 mỗi nước) để phép so sánh công bằng

**Verification:**
- [ ] Test: `tests/test_simulate.py` — 20 ván với seed cố định, đếm đúng 10 ván A đi trước / 10 ván B đi trước, `summary.json` có đủ số liệu
- [ ] Test: `pytest -q tests/test_simulate.py` → pass
- [ ] Manual: chạy 1000 ván A vs random, đọc `summary.json`, tỉ lệ thắng ≥ 0.9

**Dependencies:** T13, T19, T20
**Files likely touched:** `scripts/simulate.py`, `az/evaluator.py`, `tests/test_simulate.py`
**Estimated scope:** M (3-5 files)

---

## Task 22: `docs/provenance.md` — nguồn code và tài liệu tham khảo

**Description:** Bảng truy xuất nguồn: mỗi phần code lấy từ đâu (tự viết, paper, repo ngoài) và dùng để làm gì. Kèm danh mục tài liệu tham khảo cho proposal. Thêm `docs/README.md` — hướng dẫn cài đặt và chạy, là sản phẩm bàn giao.

**Acceptance criteria:**
- [ ] Bảng: đường dẫn module | nguồn | dùng để làm gì | giấy phép
- [ ] Phần tự viết ghi rõ "tự viết theo AGENTS.md"
- [ ] Mỗi thư viện dùng (torch, numpy, pygame) có dòng nêu phiên bản và vai trò
- [ ] Danh mục tài liệu tham khảo: AlphaZero (Silver et al. 2018), AlphaGo Zero (2017), EfficientZero (2020), luật Renju, và mọi repo tham khảo
- [ ] Nếu có code lấy từ nguồn ngoài: nêu link commit và giấy phép tương ứng
- [ ] `docs/README.md`: cài dependency, cấu hình, chạy train, chạy đánh giá 1000 ván, mở UI; mỗi lệnh nêu đủ để chạy lại từ đầu

**Verification:**
- [ ] Test: `tests/test_provenance.py` — script quét `gomoku/`, `az/`, `players/`, `ui/`, `scripts/` và assert mọi file `.py` đều có dòng tương ứng trong bảng
- [ ] Test: `pytest -q tests/test_provenance.py` → pass
- [ ] Manual: đọc `docs/provenance.md`, không còn module nào chưa rõ nguồn
- [ ] Manual: đọc `docs/README.md`, làm theo từng bước không cần hỏi thêm

**Dependencies:** T1
**Files likely touched:** `docs/provenance.md`, `docs/README.md`, `tests/test_provenance.py`
**Estimated scope:** S (1-2 files)

---

## Task 23: `ui/pygame_app.py`

**Description:** UI pygame: vẽ bàn 11x11, click để đánh, đánh dấu nước cuối, tô ô thắng, chế độ human-vs-AI và AI-vs-AI. Gắn vào runner.

**Acceptance criteria:**
- [ ] Click vào ô trống hợp lệ → đặt đá; click ô bận → phản hồi, không đặt
- [ ] Agent phản hồi trong thời gian chấp nhận được, hiển thị nước vừa đánh
- [ ] Ván kết thúc → hiện kết quả (thắng/hoà) và có thể bấm để chơi ván mới
- [ ] Chế độ AI vs AI tự chạy, có nút tạm dừng
- [ ] Smoke test chạy được headless với `SDL_VIDEODRIVER=dummy` và sự kiện giả lập

**Verification:**
- [ ] Test: `tests/test_ui.py` — smoke test headless, sự kiện giả lập tạo ra nước đi hợp lệ
- [ ] Test: `pytest -q tests/test_ui.py` → pass
- [ ] Manual: `python -m ui.pygame_app` → click thấy đá, AI đáp lại

**Dependencies:** T6, T12
**Files likely touched:** `ui/pygame_app.py`, `ui/__init__.py`, `tests/test_ui.py`
**Estimated scope:** M (3-5 files)

---

## Task 24: `gomoku/rules/renju.py` — luật Renju

**Description:** Luật Renju cho đen: cấm 3-3, cấm 4-4, cấm quá 5 (overline). Nước tạo đúng 5 luôn hợp lệ và thắng. Trắng không bị hạn chế.

**Acceptance criteria:**
- [ ] `forbidden_moves(board)` trả đúng các ô bị cấm cho đen
- [ ] Nước tạo đúng 5 quân hợp lệ và thắng, kể cả khi kèm 3-3 hay 4-4
- [ ] 3-3 bị cấm; 4-4 bị cấm; overline (6+) bị cấm
- [ ] Trắng không bị cấm nước nào
- [ ] Mask được cache theo position, invalidate khi `play()`
- [ ] Luật Renju nằm trong `gomoku/rules/`, không rò logic ra nơi khác (hard rule 1)

**Verification:**
- [ ] Test: `tests/test_rules_renju.py` — case 3-3, 4-4, overline, ngoại lệ đúng 5, trắng không bị cấm, cache invalidate đúng
- [ ] Test: `pytest -q tests/test_rules_renju.py` → pass
- [ ] Manual: không cần

**Dependencies:** T4
**Files likely touched:** `gomoku/rules/renju.py`, `tests/test_rules_renju.py`
**Estimated scope:** M (3-5 files)

---

## Task 25: Tích hợp Renju + fine-tune

**Description:** Features dùng plane `forbidden` thật, MCTS loại nước cấm, buffer tách riêng theo rule set, **fine-tune model Renju từ `checkpoints/last.pt` của Normal** (không train từ đầu). AGENTS.md rule 7 cho phép fine-tune Normal → Renju; fine-tune giữ được tri thức hình học bàn cờ và tiết kiệm ~1 giờ GPU.

**Acceptance criteria:**
- [ ] `features` của Renju có plane `forbidden` khác 0 tại đúng các ô bị cấm
- [ ] MCTS không mở rộng nước cấm, self-play không chọn nước cấm
- [ ] Buffer Renju không lẫn mẫu Normal
- [ ] Fine-tune khởi tạo từ `checkpoints/last.pt`, buffer Renju rỗng lúc bắt đầu, ghi rõ checkpoint gốc trong log
- [ ] Fine-tune 10 vòng × 50 ván chạy xong, `loss_eval_on_fixed_batch` không tăng sau vòng 5
- [ ] Checkpoint kết quả lưu thành `checkpoints/renju_last.pt`, không ghi đè `last.pt` của Normal

**Verification:**
- [ ] Test: `tests/test_renju_integration.py` — plane forbidden đúng; nước cấm không xuất hiện trong self-play
- [ ] Test: `pytest -q` → toàn bộ pass
- [ ] Build: `python scripts/train.py --config configs/renju.yaml --iterations 10 --resume` → exit 0
- [ ] Manual: đọc log, so `loss_eval_on_fixed_batch` với giá trị của model Normal gốc

**Dependencies:** T24, T12, T10
**Files likely touched:** `gomoku/features.py`, `az/mcts.py`, `az/selfplay.py`, `configs/renju.yaml`, `tests/test_renju_integration.py`
**Estimated scope:** M (3-5 files)

---

## Task 26: `docs/report.md` — báo cáo tổng kết

**Description:** Báo cáo cuối mô tả phương pháp đã hiện thực, kết quả thực nghiệm lấy từ log ván và `summary.json`, và phân tích nguyên nhân thua. C viết; A và B cung cấp số liệu và log ván.

**Acceptance criteria:**
- [ ] Mô tả phương pháp khớp với code đã chạy: biểu diễn trạng thái, mô hình, tìm kiếm, tự chơi, huấn luyện, đánh giá
- [ ] Kết quả lấy từ `logs/` và `summary.json`: win rate theo lượt đi trước/đi sau, phân bố `end_reason`, win rate từng mốc 100 ván (E3)
- [ ] Phân tích nguyên nhân thua dựa trên `end_reason` trong log, không suy đoán
- [ ] Nêu lợi thế lượt đầu trước khi trình bày tỉ lệ thắng
- [ ] Bảng truy xuất nguồn và danh mục tài liệu tham khảo lấy từ `docs/provenance.md` (T22), không viết lại
- [ ] Đối chiếu với proposal: mục tiêu, câu hỏi nghiên cứu và các thí nghiệm E1–E5 đều có kết quả tương ứng

**Verification:**
- [ ] Test: `tests/test_report.py` — file có đủ các mục bắt buộc; tỉ lệ thắng nêu trong báo cáo bằng số trong `summary.json`
- [ ] Test: `pytest -q tests/test_report.py` → pass
- [ ] Manual: đọc `docs/report.md`, mọi con số tra được trong log

**Dependencies:** T21, T22, T25
**Files likely touched:** `docs/report.md`, `tests/test_report.py`
**Estimated scope:** M (viết báo cáo, không viết code mới nhiều)

---

## Task 27: `gomoku/patterns.py` — bộ nhận diện mẫu

**Description:** Nhận diện hình dạng quân cho một màu tại một ô theo 4 hướng: năm liền, overline, tứ mở, tứ bị chặn một đầu, tam mở, tam bị chặn, tam gãy. Module dùng chung cho luật Renju (T24), tác tử heuristic (T28) và tầng chiến thuật của `az_v1` (T29, T30). Thuần numpy, không import torch.

**Acceptance criteria:**
- [ ] API: `pattern_at(board, r, c, player) -> set[str]` và `threat_count(board, player) -> int`
- [ ] Nhận diện đúng cả 4 hướng ngang, dọc, chéo `\`, chéo `/`, cho cả Đen và Trắng
- [ ] Tứ mở phân biệt được với tứ bị chặn một đầu (điều kiện quyết định của threat-space)
- [ ] Xử lý biên bàn: mẫu nằm sát cạnh không bị tính là mở
- [ ] Không import torch; không định nghĩa lại logic thắng/thua (vẫn dùng `rules.is_win`)
- [ ] Vector hóa theo hướng, không có vòng lặp Python trên 121 ô mỗi lần gọi

**Verification:**
- [ ] Test: `tests/test_patterns.py` — mỗi mẫu một case, cả 2 màu, cả 4 hướng, case biên
- [ ] Test: `pytest -q tests/test_patterns.py` → pass
- [ ] Manual: in mẫu nhận được cho một thế cờ tay, đối chiếu hình vẽ

**Dependencies:** T4
**Files likely touched:** `gomoku/patterns.py`, `tests/test_patterns.py`
**Estimated scope:** M (3-5 files)

---

## Task 28: `players/heuristic.py` — tác tử cấp 2

**Description:** Tác tử luật/heuristic: ưu tiên thắng ngay, chặn thắng ngay của đối thủ, kiểm soát vị trí trung tâm, ưu tiên nước tạo mẫu mạnh. Dùng `gomoku/patterns.py`, chỉ gọi interface luật, không hardcode luật Renju.

**Acceptance criteria:**
- [ ] Thứ tự ưu tiên bằng văn bản trong docstring: thắng ngay > chặn thắng ngay > tạo tứ mở > chặn tứ mở > tạo tam mở > điểm vị trí
- [ ] Xung đột luật nêu rõ: nếu nhiều nước cùng hạng ưu tiên, chọn nước có điểm vị trí cao hơn; hòa điểm thì theo thứ tự ô
- [ ] Chỉ chọn trong `rules.legal_moves`, không bao giờ chọn nước cấm của Renju
- [ ] Thắng `random` ≥ 90% trong 20 ván 11x11, cả hai màu
- [ ] Không import torch

**Verification:**
- [ ] Test: `tests/test_heuristic.py` — win-in-1, block-in-1, không chọn nước cấm, thứ tự ưu tiên
- [ ] Test: `pytest -q tests/test_heuristic.py` → pass
- [ ] Manual: script đấu 20 ván với `random`, in tỉ lệ thắng

**Dependencies:** T4, T6, T27
**Files likely touched:** `players/heuristic.py`, `tests/test_heuristic.py`
**Estimated scope:** S (1-2 files)

---

## Task 33: `players/alphabeta.py` — tác tử cấp 3

**Description:** Minimax với cắt tỉa Alpha-Beta, giới hạn độ sâu, hàm lượng giá dùng `gomoku/patterns.py`. Cắt tỉa bật/tắt được bằng cờ để phục vụ ablation bắt buộc (T32).

**Acceptance criteria:**
- [ ] `AlphaBetaPlayer(rules, depth=4, pruning=True)`; `pruning=False` chạy minimax đầy đủ cùng độ sâu
- [ ] Hàm lượng giá cộng điểm mẫu của hai bên, có trọng số rõ ràng trong một bảng hằng số
- [ ] Move ordering theo điểm lượng giá để cắt tỉa hiệu quả
- [ ] Đếm được `nodes_visited` và trả về trong kết quả chọn nước (dùng cho ablation)
- [ ] Bật và tắt cắt tỉa ở cùng độ sâu phải cho **cùng một nước đi** (kiểm tra tính đúng đắn của cắt tỉa)
- [ ] Depth 4 trên 11x11 không quá 2 s/nước trên CPU

**Verification:**
- [ ] Test: `tests/test_alphabeta.py` — win-in-1; pruning bật/tắt cùng kết quả; `nodes_visited` giảm khi bật tỉa; không chọn nước cấm
- [ ] Test: `pytest -q tests/test_alphabeta.py` → pass
- [ ] Manual: đo `nodes_visited` và thời gian ở depth 2/3/4, cả bật và tắt tỉa

**Dependencies:** T4, T27
**Files likely touched:** `players/alphabeta.py`, `tests/test_alphabeta.py`
**Estimated scope:** M (3-5 files)

---

## Task 36: `gomoku/game.py` — API game thống nhất theo đề bài

**Description:** Lớp mỏng bọc `Board` + `Rules`, lộ đúng bảy phương thức mà đề bài yêu cầu: `reset()`, `legal_actions()`, `step(action)`, `is_terminal()`, `result()`, `render()`, `clone()`. Không thêm logic luật; chỉ chuyển tiếp.

**Acceptance criteria:**
- [ ] `reset()` đưa về bàn trống, Đen đi trước, `history` rỗng
- [ ] `legal_actions()` trả chỉ số ô phẳng, đã loại nước cấm
- [ ] `step(action)` từ chối nước không hợp lệ bằng exception có thông điệp rõ ràng
- [ ] `result()` trả `(winner_view, end_reason)` theo `az/game_log.py`; `None` khi chưa kết thúc
- [ ] `clone()` tạo bản độc lập: sửa bản clone không ảnh hưởng bản gốc
- [ ] `render()` trả chuỗi ASCII, không cần thư viện đồ họa (UI dùng đường riêng)
- [ ] Không import torch

**Verification:**
- [ ] Test: `tests/test_game.py` — reset, clone độc lập, step bất hợp lệ raise, result đúng ở ván thắng/hoà, render không lỗi ở bàn trống
- [ ] Test: `pytest -q tests/test_game.py` → pass

**Dependencies:** T3, T4
**Files likely touched:** `gomoku/game.py`, `tests/test_game.py`
**Estimated scope:** S (1-2 files)

---

## Task 29: `az_v1` — tactical override dùng chung trọng số `az_v0`

**Description:** Bọc tầng chọn nước của agent: kiểm tra nước thắng ngay và nước chặn bắt buộc trước khi trả kết quả của PUCT. Dùng cờ cấu hình `tactical.enabled`, mặc định tắt. Không train lại, không sửa trọng số.

**Acceptance criteria:**
- [ ] `tactical.enabled: false` cho kết quả **giống hệt** `az_v0` với cùng seed
- [ ] `tactical.enabled: true` đánh nước thắng ngay nếu có; nếu không, chặn nước thắng ngay của đối thủ
- [ ] Không ghi đè lên nước cấm của Renju
- [ ] Log được `override_applied` (bool) và `override_reason` cho mỗi nước
- [ ] Cùng `last.pt`, cùng `n_simulations`, cùng seed → chỉ khác cờ `tactical.enabled`

**Verification:**
- [ ] Test: `tests/test_tactical.py` — win-in-1 chọn đúng; block-in-1 chọn đúng; cờ tắt cho kết quả giống `az_v0`; không chọn nước cấm
- [ ] Test: `pytest -q tests/test_tactical.py` → pass
- [ ] Manual: 20 ván `az_v0` vs `az_v1` với seed cố định, đếm số lần ghi đè

**Dependencies:** T9, T12, T27
**Files likely touched:** `az/player_tactical.py`, `configs/normal.yaml`, `tests/test_tactical.py`
**Estimated scope:** M (3-5 files)

---

## Task 30: Threat-space search

**Description:** Tìm kiếm chọn lọc chỉ trên chuỗi đe dọa bắt buộc, kích hoạt khi bộ nhận diện mẫu báo có tứ mở. Không chạy Alpha-Beta trên toàn bộ cây MCTS.

**Acceptance criteria:**
- [ ] Chỉ kích hoạt khi có tứ mở hoặc chuỗi đe dọa bắt buộc; ngoài điều kiện đó `az_v1` dùng PUCT như cũ
- [ ] Giới hạn cứng 5000 node hoặc 150 ms mỗi nước, cái nào tới trước; ghi lại số lần kích hoạt
- [ ] Giá trị trả về theo góc nhìn người đang đi, khớp quy ước của PUCT
- [ ] Có case test "đối thủ có hai đe dọa cùng lúc" → phải báo thua, không được báo thắng
- [ ] Tắt được bằng cờ cấu hình `threat_space.enabled`, mặc định tắt

**Verification:**
- [ ] Test: `tests/test_threat_space.py` — chuỗi thắng 3 nước tìm ra; hai đe dọa song song báo thua; giới hạn node được tôn trọng
- [ ] Test: `pytest -q tests/test_threat_space.py` → pass
- [ ] Manual: đo thời gian/nước trước và sau khi bật trên 10 ván

**Dependencies:** T27, T29
**Files likely touched:** `az/threat_space.py`, `configs/normal.yaml`, `tests/test_threat_space.py`
**Estimated scope:** M (3-5 files)

---

## Task 31: `scripts/tournament.py` — round-robin + thống kê

**Description:** Chạy round-robin giữa N tác tử, chia đều màu, xuất ma trận đối đầu và các chỉ số đề bài yêu cầu. Thay cho `scripts/simulate.py` (T21) vì T21 chỉ đấu từng cặp.

**Acceptance criteria:**
- [ ] `python scripts/tournament.py --agents random,heuristic,alphabeta,mcts_pure,az_v0,az_v1 --games 100 --budget sims:200` chạy hết mọi cặp
- [ ] Chia đều màu: mỗi cặp đúng 50 ván mỗi bên đi trước với `--games 100`
- [ ] Xuất `logs/tournament/<run_id>/matrix.csv` + `summary.json`: score, thắng/hoà/thua, thời gian trung bình/nước, p95 độ trễ, số node/sims, seed từng cặp
- [ ] Khoảng tin cậy Wilson cho score từng cặp
- [ ] Hỗ trợ hai chế độ ngân sách: `sims:N` (cố định số mô phỏng/độ sâu) và `time:S` (giây mỗi nước)
- [ ] Không hardcode danh sách tác tử; đọc từ tham số

**Verification:**
- [ ] Test: `tests/test_tournament.py` — 2 tác tử giả, 20 ván, assert 10/10 chia lượt, matrix đủ ô, Wilson CI trong [0,1]
- [ ] Test: `pytest -q tests/test_tournament.py` → pass
- [ ] Manual: chạy 3 tác tử × 20 ván, đọc `summary.json`

**Dependencies:** T6, T7, T13, T7
**Files likely touched:** `scripts/tournament.py`, `az/evaluator.py`, `tests/test_tournament.py`
**Estimated scope:** M (3-5 files)

---

## Task 32: `scripts/ablation.py` — hai ablation bắt buộc

**Description:** Chạy hai thí nghiệm độ nhạy mà đề bài yêu cầu: (1) Alpha-Beta bật/tắt cắt tỉa — đo số node và thời gian; (2) quét số mô phỏng của `mcts_pure` ở 50/200/800 — đo score và thời gian.

**Acceptance criteria:**
- [ ] Ablation 1: cùng depth, `pruning=True` và `False` cho cùng nước đi, khác `nodes_visited` và thời gian; xuất bảng
- [ ] Ablation 2: `mcts_pure` với 50/200/800 mô phỏng, mỗi mức đấu 100 ván với `heuristic`; xuất score + thời gian trung bình
- [ ] Kết quả ghi `logs/ablation/<run_id>.json`, có seed và cấu hình
- [ ] Sinh được hình PNG từ JSON mà không sửa mã (notebook hoặc script riêng)

**Verification:**
- [ ] Test: `tests/test_ablation.py` — chạy 2 ván mỗi cấu hình, assert file JSON có đủ trường
- [ ] Test: `pytest -q tests/test_ablation.py` → pass
- [ ] Manual: chạy ablation 1 thật, đối chiếu số node

**Dependencies:** T7, T33
**Files likely touched:** `scripts/ablation.py`, `tests/test_ablation.py`
**Estimated scope:** M (3-5 files)

---

## Task 34: Poster, slide, ZIP nộp bài

**Description:** Đóng gói sản phẩm nộp: poster tóm tắt kết quả, slide thuyết trình, ZIP đúng quy cách tên, README hướng dẫn cài đặt/train/evaluate/demo.

**Acceptance criteria:**
- [ ] ZIP tên `NhomXX_Gomoku_TTNTNC_<HoTen>_<MaSV>.zip`, chứa mã nguồn, configs, tests, `docs/`, log đấu thô, và checkpoint nếu dung lượng cho phép
- [ ] ZIP **không** chứa `checkpoints/` quá lớn, `.venv/`, `__pycache__/`; nếu checkpoint quá nặng thì ghi link ngoài
- [ ] Slide có: câu hỏi nghiên cứu, kiến trúc, ma trận đối đầu, ablation, hạn chế
- [ ] Poster A3 đọc được ở khoảng cách 1 m: một hình chính là ma trận đối đầu
- [ ] `docs/README.md` có đủ 4 lệnh: cài đặt, train, evaluate, demo
- [ ] Bảng phân công và tỉ lệ đóng góp từng thành viên

**Verification:**
- [ ] Manual: giải nén ZIP ở thư mục mới, làm theo `docs/README.md`, chạy được `pytest` và mở được demo
- [ ] Manual: đối chiếu mọi số trên slide/poster với `logs/`

**Dependencies:** T26, T31, T35
**Files likely touched:** `docs/`, `poster/`, `slides/`
**Estimated scope:** M (không viết code mới)

---

## Task 35: Chạy giải đấu và đóng băng số liệu

**Description:** Chạy toàn bộ ma trận đã chốt, đóng băng số liệu trước khi viết báo cáo.

**Acceptance criteria:**
- [ ] Normal chế độ A: 6 tác tử round-robin, 100 ván/cặp = 1.500 ván
- [ ] Normal chế độ B (1 s/nước): `az_v0`, `az_v1`, `alphabeta` = 300 ván
- [ ] Renju: `random`, `heuristic`, `alphabeta`, `az_renju` round-robin, 100 ván/cặp = 600 ván
- [ ] Tổng 2.400 ván, toàn bộ log ván nằm trong `logs/tournament/`, không xoá
- [ ] Số liệu đóng băng: sau mốc này không chạy lại để "làm đẹp" kết quả; nếu phải chạy lại thì ghi rõ lý do
- [ ] Không dùng ván của giải đấu để chọn siêu tham số

**Verification:**
- [ ] Build: chạy 3 lệnh tournament/ablation, exit 0
- [ ] Manual: mọi số trong `summary.json` khớp số ván trong log ván
- [ ] Manual: đối chiếu tổng ván = 2.400 ± sai số do ván bị hủy

**Dependencies:** T31, T32, T25
**Files likely touched:** `logs/tournament/`, `configs/`
**Estimated scope:** M (chạy máy, không viết code mới)

---

## Checkpoints

**Checkpoint A — Sau T2**
- [ ] `pytest` chạy được, torch CUDA available
- [ ] Repo có git, `.gitignore` đúng

**Checkpoint B — Sau T6**
- [ ] Toàn bộ test pass
- [ ] Người chơi đánh hết ván với random qua `scripts/play.py`

**Checkpoint C — Sau T7**
- [ ] MCTSPure thắng random >60% trên 11x11

**Checkpoint D — Sau T12**
- [ ] Smoke test: 2 iteration train trên 11x11 chạy xong, loss giảm
- [ ] Checkpoint sinh ra, load lại được, `train.py --resume` chạy tiếp được
- [ ] Agent 11x11 thắng random

**Checkpoint E — Sau T15 (Colab)**
- [ ] Trainer chạy được với `--time-budget-hours`, `--workers`, `--resume`
- [ ] `git clone $REPO_URL` ra đủ code; `sync_drive.py` chạy được như đường dự phòng
- [ ] Notebook Colab chạy 1 iteration, checkpoint nằm trong Google Drive
- [ ] Kill session rồi chạy lại notebook → nối tiếp đúng chỗ, không mất iteration

**Checkpoint F — Sau T18 (chạy song song)**
- [ ] Instance CPU-only sinh shard npz không cần GPU
- [ ] Gộp 2 shard từ 2 model giữ đúng thứ tự, shard hỏng bị bỏ qua
- [ ] `train.py --buffer-in` nạp buffer đã gộp và train tiếp

**Checkpoint G — Sau T20**
- [ ] Arena win rate tăng qua các iteration trên 11x11
- [ ] Agent 11x11 thắng random và MCTSPure
- [ ] Log ván của lần train đầy đủ, đọc ra được ai thắng và vì sao

**Checkpoint H — Sau T22 (đánh giá + tài liệu)**
- [ ] 1000 ván đánh giá chạy xong, đúng 500 ván mỗi bên đi trước
- [ ] `summary.json` có tỉ lệ thắng theo đi trước/đi sau và phân bố `end_reason`
- [ ] `docs/provenance.md` phủ hết mọi module, không còn file nào chưa rõ nguồn

**Checkpoint I — Sau T23**
- [ ] UI chơi được human-vs-AI và AI-vs-AI

**Checkpoint J — Sau T25**
- [ ] Cả Normal và Renju đều train được, buffer tách biệt
- [ ] Toàn bộ test pass

**Checkpoint K — Sau T26**
- [ ] Báo cáo đủ mục, mọi con số tra được trong log
- [ ] `docs/README.md` cho người mới cài và chạy được không cần hỏi thêm

**Checkpoint L — Sau T7/T28/T33 (đủ tác tử cấp 1–4)**
- [ ] `random`, `heuristic`, `alphabeta`, `mcts_pure` chọn được nước trên cả Normal và Renju
- [ ] `heuristic` thắng `random` ≥ 90%; `mcts_pure` thắng `random` > 60%
- [ ] `gomoku/game.py` (T36) có đủ 7 phương thức theo đề bài

**Checkpoint M — Sau T31 + T29 (đo được sức mạnh, có `az_v1`)**
- [ ] Có score của `az_v0` trước ít nhất 3 tác tử cổ điển, chia đều màu
- [ ] Cờ `tactical.enabled=false` cho kết quả trùng `az_v0` với cùng seed
- [ ] Ma trận round-robin chạy được với 6 tác tử

**Checkpoint N — Sau T35 (số liệu đóng băng)**
- [ ] 2.400 ván chạy xong, log đầy đủ trong `logs/tournament/`
- [ ] Hai ablation có kết quả
- [ ] Không ai sửa test hoặc cấu hình sau mốc này để đổi kết quả
