# Thiết kế tầng AlphaZero (`az/`)

**Phạm vi:** `az/model.py`, `az/mcts.py`, `az/replay_buffer.py`, `az/selfplay.py`, `az/trainer.py`, `az/game_log.py`, và `players/az_player.py`. **Người sở hữu:** B (Huy).

**Không thuộc phạm vi:** `az/evaluator.py` (C), `gomoku/runner.py` + `ui/` (A), luật Renju (A). Các phần đó chỉ xuất hiện ở đây dưới dạng **hợp đồng giao diện**.

**Từ vựng:** `GLOSSARY.md`. **Quyết định nền:** `docs/adr/0001` (gom lô cấp chương trình + API từng bước), `0002` (buffer cạnh checkpoint), `0003` (không bàn nhỏ).

---

## 1. Nguyên tắc bất biến

1. Mọi giá trị theo **góc nhìn người đang đi**. Giá trị node con là góc nhìn đối thủ nên **đảo dấu** khi truyền ngược (`AGENTS.md` hard rule 2).
2. `az/` không chứa luật: chỉ gọi `rules.legal_moves` / `rules.forbidden_moves` / `rules.is_terminal` / `rules.is_win` và `gomoku.features.board_to_planes`. Không đọc `board.grid` để tự suy ra luật.
3. Ba bất biến kiểm chứng được bằng test: `nn_calls` = số lá đã mở rộng **không tính node kết thúc**; số mẫu mỗi ván = số nước; `z` đổi dấu xen kẽ trong ván quyết định.
4. **Đúng trước, nhanh sau.** GĐ1–GĐ4 chạy chế độ tất định (mỗi lượt suy luận một lá); gom lô chỉ xuất hiện ở GĐ5.
5. Cố ý **không** làm: dùng lại cây giữa các nước, virtual loss, π thưa, resign, nhiều GPU.

---

## 2. Hợp đồng giao diện

### 2.1 Nhận từ `gomoku/`

- `board_to_planes(board, rules, player) -> np.uint8 (4, 11, 11)`, thứ tự plane cố định: 0 quân bên đang đi, 1 quân đối thủ, 2 nước cấm, 3 màu bên đang đi.
- `rules.legal_moves(board)`, `rules.forbidden_moves(board)`, `rules.is_terminal(board)`, `rules.is_win(board)`; `Board.play` / `Board.undo` / `Board.current_player`.
- `gomoku/` không import torch. `az/` không import `gomoku.rules.*` (chỉ nhận instance `rules`).

### 2.2 `az/model.py`

- `AZNet(n_blocks=6, n_filters=64)`: `forward(x: (B,4,11,11) float32) -> (logits (B,121), value (B,1))`, value qua `tanh`.
- `predict(features: np.uint8 (B,4,11,11)) -> (probs (B,121) float32, values (B,) float32)`: chạy trong `no_grad`, mode `eval`, tự chuyển device.
- Trả **xác suất thô trên cả 121 ô**, không tự mask. Mask hợp lệ do MCTS áp khi gán prior rồi chuẩn hoá lại — nhờ vậy hard rule 4 thoả: nước cấm không bao giờ có prior > 0.
- `save_checkpoint(path, meta)` / `load_checkpoint(path) -> (AZNet, meta)`; `meta` gồm `format_version`, `n_blocks`, `n_filters`, `n_iter`, `rule_set`, `config`.
- `predict` **phải** cho kết quả giống nhau giữa `batch=1` và `batch=k` trên cùng input (test bắt buộc, vì cả hệ đứng trên giả định này).

### 2.3 `az/mcts.py` — API từng bước

```python
@dataclass
class LeafRequest:
    node: Node                      # lá cần đánh giá
    planes: np.ndarray              # (4, 11, 11) uint8
    legal_mask: np.ndarray          # (121,) bool

class MCTS:
    def __init__(self, rules, evaluator, n_simulations, c_puct,
                 dirichlet_alpha, dirichlet_eps, root_noise: bool)
    def request(self) -> LeafRequest | None   # None: đã đủ ngân sách mô phỏng
    def receive(self, policy: np.ndarray, value: float) -> None
    def visit_counts(self) -> np.ndarray      # (121,) int
    def select_move(self, temperature: float) -> int
    def root_value(self) -> float
    def reset(self) -> None                   # dựng cây mới cho nước tiếp theo
```

Ngữ nghĩa:

- `request()` tự lặp: chọn nhánh theo PUCT xuống tới lá. Nếu lá là node **kết thúc** (thắng/hoà/hết ô), gán giá trị chính xác ±1/0, truyền ngược, rồi **tiếp tục** lượt mô phỏng kế — không trả `LeafRequest`, không gọi mạng. Chỉ khi gặp lá cần đánh giá mới trả về; khi đủ `n_simulations` thì trả `None`.
- `receive(policy, value)` biến lá thành node đã mở rộng: tạo **tất cả** con hợp lệ ngay, prior = `policy[move]` đã mask và chuẩn hoá; lưu `value` theo góc nhìn người đang đi tại lá; đảo dấu dần khi truyền ngược.
- Nhiễu Dirichlet chỉ áp ở gốc, chỉ khi `root_noise=True` (tự chơi), không áp khi đấu người hay đánh giá.
- `select_move(temperature)`: `temperature == 0` → argmax; ngược lại lấy mẫu theo `visits^(1/τ)`.
- Bộ đếm `nn_calls` tăng trong `receive()`. Bộ đếm là một phần hợp đồng: T9 kiểm tra nó.

Chế độ tất định (dùng cho test và cho `az_player`):

```python
for _ in range(n_simulations):
    req = mcts.request()
    if req is None:
        break
    mcts.receive(*evaluator.predict_batch([req]))
```

### 2.4 `az/selfplay.py` — driver gom lô

- `run_games(model, rules, cfg, n_games, seed) -> Iterator[GameRecord]` chạy `parallel_games` ván xen kẽ trong **một tiến trình**.
- Mỗi tick: mỗi ván chưa có lá chờ tiến **đúng một** lượt mô phỏng; gom mọi `LeafRequest` đang chờ thành một lô; một lượt `predict`; phân kết quả về từng ván bằng `receive`. Ván đủ `n_simulations` → chọn nước theo τ bậc thang → lưu mẫu `(planes, π, )` → `board.play`. Ván kết thúc → gán `z`, đẩy mẫu vào buffer, ghi JSONL, nạp ván mới vào chỗ trống.
- Lô thật dao động 1…`parallel_games`, nên phải **log kích thước lô trung bình**, không giả định bằng `parallel_games`.
- Chế độ tất định (`parallel_games: 1`) phải cho cùng kết quả với seed cố định — đó là mốc so sánh của GĐ5.

### 2.5 `az/replay_buffer.py`

- Cấp phát trước `buffer_size` (50 000) slot: `features` uint8 `(N,4,11,11)`, `pi` float16 `(N,121)`, `z` float32 `(N,)`, `rule_set` uint8 `(N,)`. Đầy thì **ghi đè mẫu cũ nhất**; `__len__` trả số mẫu trước tăng cường.
- `add(features, pi, z, rule_set)`; `sample(batch_size, rule_set, rng)` → mỗi mẫu một phép đối xứng ngẫu nhiên trong 8 phép, áp **đồng thời** lên feature và π; trả tensor torch float32.
- Trộn hai bộ luật trong một buffer → `raise` (hard rule 7).
- `save_npz` / `load_npz` ghi `buffer_last.npz` cạnh checkpoint, chỉ trên đĩa local (ADR-0002).

### 2.6 `az/trainer.py`

- Loss = `policy_loss_weight × CE(π, log p)` + `value_loss_weight × MSE(z, v)` + `weight_decay × L2`, mọi trọng số đọc từ config.
- Adam lr 1e-3, lịch cosin theo **số bước tối ưu** đã chạy (tổng = `n_iter × train_steps_per_iter`).
- Mỗi iteration: `train_steps_per_iter` (200) bước × batch 128. Log cả `buffer_size` và **số mẫu khác nhau thực sự được dùng**, để báo cáo không nói sai về oversample giai đoạn đầu.
- `loss_eval_on_fixed_batch`: 512 mẫu cố định (seed cố định), **không** train trên đó — đây là thước đo "loss giảm" cho báo cáo, vì loss trên batch ngẫu nhiên dao động quá mạnh để kết luận.
- Checkpoint: `last.pt` (model + optimizer + meta) + `buffer_last.npz`; `--resume` nạp cả hai. Thiếu `buffer_last.npz` thì chạy tiếp với buffer rỗng **và in cảnh báo**.

### 2.7 `az/game_log.py`

- Một dòng JSONL mỗi ván, field theo T19, cộng `nn_calls` và `mean_batch`.
- `end_reason`: `five_in_row` | `board_full` | `forbidden_move` | `timeout`.
  - `forbidden_move` = một bên **đánh** nước cấm và thua ngay. Chỉ xảy ra từ người chơi/UI. Trong self-play, sự kiện này là bug → ghi log rồi `raise`.
  - `timeout` chỉ tồn tại khi bật `move_time_budget_sec`; self-play không bao giờ sinh ra giá trị này.
- Một writer duy nhất dùng chung cho runner, self-play và đánh giá; flush ngay sau mỗi ván.

### 2.8 `players/az_player.py`

- `AZPlayer(checkpoint, rules, cfg, time_budget_sec=None, seed=None)` với `select_move(board) -> (r, c)` đúng interface `Player` của A.
- Chạy chế độ tất định (lô = 1). Dừng khi đạt `n_simulations` **hoặc** hết `time_budget_sec`, cái nào tới trước.
- Không tự ghi log ván — runner/UI là bên ghi.

---

## 3. Cấu hình bổ sung

Vào `DEFAULT_CONFIG` + `configs/*.yaml` (C sở hữu `config.py`):

| Khoá | Mặc định | Vì sao |
|---|---|---|
| `parallel_games` | 16 | số ván xen kẽ, quyết định kích thước lô thật |
| `pi_target_temp` | 1.0 | cố ý: π luôn mềm, tách khỏi τ dùng để chọn nước |
| `move_time_budget_sec` | `null` | chỉ dùng khi đấu người / arena |
| `eval_fixed_batch_size` | 512 | thước đo loss ổn định |
| `checkpoint_buffer` | `true` | ghi `buffer_last.npz` cạnh checkpoint |
| `n_blocks` / `n_filters` | 6 / 64 | để `load_checkpoint` kiểm tra được kiến trúc |
| `log_moves` | `false` | bật khi chạy thí nghiệm cần phân tích từng nước |

---

## 4. Năm giai đoạn

### GĐ1 — Mạng và hợp đồng tensor (T8)

**Mục tiêu:** một mạng 6×64 đọc `(B,4,11,11)` và trả policy 121 + value trong [-1,1], lưu ra và đọc lại được.

**Việc:** `AZNet`; `predict`; `save/load_checkpoint` + meta; chọn device; đo giây/lượt ở batch 16.

**Cổng:** `pytest tests/test_model.py` pass; một lượt `predict` batch 16 chạy trên CUDA; có số giây/lượt làm mốc cho GĐ5.

**Test:** shape đầu ra · `batch>1 == batch=1` · round-trip checkpoint (kể cả load trên CPU sau khi lưu từ GPU) · `predict` ổn định qua hai lần gọi · fallback CPU khi không có CUDA.

### GĐ2 — PUCT tuần tự, đúng dấu (T9)

**Mục tiêu:** từ một vị trí bất kỳ, chọn được nước đi bằng PUCT với evaluator giả — chưa cần học, nhưng phải **đúng**.

**Việc:** `Node`; cặp `request/receive`; prior từ policy; PUCT `c_puct=1.5`; nhiễu Dirichlet ở gốc; τ bậc thang; mask nước cấm; giá trị chính xác cho node kết thúc; `visit_counts`/`select_move`/`root_value`; bộ đếm `nn_calls`.

**Cổng:** 5 test pass — win-in-1 · `test_backup_negates_child_value` · `nn_calls` = số lá đã mở rộng (trừ node kết thúc) · nước cấm có prior 0 và không được mở rộng · `request()` trả `None` khi hết ngân sách. Đo giây/nước với 200 lượt mô phỏng trên evaluator giả để có mốc CPU.

**Test dùng evaluator giả**, không cần mạng thật — đây là thứ thay bàn 7x7 (ADR-0003).

### GĐ3 — Self-play một ván + buffer (T10, T11a)

**Mục tiêu:** một ván tự chơi sinh ra dữ liệu đúng định dạng.

**Việc:** driver một ván; π = `visits^1` chuẩn hoá; `z` gán sau khi ván kết thúc theo góc nhìn người đang đi; buffer cấp phát trước + 8-fold; nhãn `rule_set`.

**Cổng:** số mẫu = số nước · `z` đổi dấu xen kẽ trong ván quyết định · buffer lọc đúng `rule_set` và `raise` khi trộn · phép đối xứng map π đúng (policy one-hot tại `(r,c)` biến đổi ra đúng ô ảnh) · một ván 20 nước với evaluator giả chạy xong dưới 5 giây.

Ở giai đoạn này **chưa có gom lô**: lô = 1.

### GĐ4 — Trainer + end-to-end + `az_player` (T12)

**Mục tiêu:** đường chạy khép kín tự chơi → buffer → train → checkpoint → agent đánh được một ván.

**Việc:** `az/trainer.py`; loss + cosin; `last.pt` + `buffer_last.npz`; `scripts/train.py --iterations 2`; `players/az_player.py`.

**Cổng:** overfit một batch cố định (loss giảm rõ rệt sau ≤300 bước) · 2 iteration 11x11 exit 0 và `loss_eval_on_fixed_batch` không tăng · `--resume` nối tiếp đúng chỗ (dừng giữa chừng rồi chạy lại) · `AZPlayer` chơi hết ván hợp lệ với `RandomPlayer` qua runner, 20 ván đều ghi log.

Chưa đặt ngưỡng thắng random ở đây — 2 iteration là quá ít để có ngưỡng có nghĩa. Ngưỡng 18/20 thuộc GĐ5.

### GĐ5 — Gom lô, nhật kí, chạy 1000 ván (T11b, T19, T20)

**Mục tiêu:** chạy hết ngân sách 1000 ván với chi phí chấp nhận được, và mọi ván để lại dấu vết.

**Việc:** driver `parallel_games` ván xen kẽ + gom lô; `az/game_log.py`; log iteration; chạy 20 × 50.

**Cổng:** `nn_calls_per_game` xấp xỉ `200 × số nước` và **không đổi** khi tăng `parallel_games` (nếu đổi → đang eval thừa) · `mean_batch` tăng theo `parallel_games`, đo ở G = 1, 4, 16 và ghi thành bảng số · thông lượng (ván/giờ) tăng rõ so với chế độ tất định (nếu không tăng, gom lô là vô nghĩa và phải xem lại) · 20 × 50 chạy xong, có checkpoint + log · `loss_eval_on_fixed_batch` giảm qua các iteration · `AZPlayer` thắng random ≥ 18/20 · arena của C báo thắng `best` ≥ 55% ở ít nhất một mốc.

---

## 5. Rủi ro riêng của tầng AZ và cách phát hiện sớm

| Rủi ro | Phát hiện bằng |
|---|---|
| Sai dấu giá trị khi truyền ngược | `test_backup_negates_child_value` + test `z` đổi dấu xen kẽ |
| π bị one-hot hoá sớm (mất tín hiệu học) | assert entropy π > 0 ở các nước đầu ván trong self-play |
| `nn_calls` bùng nổ (eval node kết thúc hoặc eval trùng lá) | đếm phải bằng số lá mở rộng; chênh gấp đôi là dấu hiệu |
| Buffer trộn hai bộ luật | `raise` + test riêng |
| Quên mask nước cấm ở Renju | test prior của nước cấm == 0, và nước cấm không xuất hiện trong self-play |
| Lô luôn bằng 1 do bug điều phối | log `mean_batch`; khi `parallel_games > 1` mà `mean_batch` ≈ 1 thì scheduler sai |
| `move_time_budget_sec` không cắt được lượt đang chờ | test với budget 0.01 s |
| Loss nhiễu, báo cáo kết luận sai | `loss_eval_on_fixed_batch` là số duy nhất được dùng để nói "loss giảm" |

---

## 6. Ngoài phạm vi 5 giai đoạn này

Dùng lại cây giữa các nước · virtual loss · π lưu thưa · resign · huấn luyện nhiều GPU · `az/evaluator.py` (C, T13) · giao diện (A, T23) · logic luật Renju (A, T24) · gom shard song song (C, T17–T18).

---

## 7. Liên kết

`tasks/plan.md` (kiến trúc, 27 task) · `tasks/todo.md` (T8–T12, T19, T20) · `GLOSSARY.md` (từ vựng) · `docs/adr/0001`–`0003` (ba quyết định nền).
