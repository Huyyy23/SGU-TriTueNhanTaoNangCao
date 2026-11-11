# Phần việc của B (không đụng A và C)

Tài liệu này liệt kê toàn bộ những việc thuộc quyền sở hữu của **B** theo phân công `tasks/todo.md`. Chỉ tập trung vào những thứ không đụng tới nhánh của **A** (`gomoku/`, `players/`, `ui/`) hoặc **C** (`az/evaluator.py`, `scripts/arena.py`, `scripts/tournament.py`, `scripts/ablation.py`, `configs/`, `docs/`).

## 0. Trạng thái đã kiểm tra (2026-11-01)

Đo trên repo, không phải kế hoạch:

| Hạng mục | Kết quả |
|---|---|
| `pytest -q` | 79 passed, 2 skipped |
| `gomoku/patterns.py` (T27, A) | **Chưa tồn tại** → T29, T30, T28 vẫn bị block |
| `az/` | `model.py`, `mcts.py`, `selfplay.py`, `replay_buffer.py`, `trainer.py`, `game_log.py`, `evaluator.py` đều có |
| `players/` | mới có `random.py`, `az_player.py` (A còn thiếu heuristic, alphabeta, mcts_pure) |
| `az_v0` | xong 20 vòng × 50 ván; `checkpoints/last.pt` (10/28) |
| `az_v0b` 5000 ván | có ADR 0005 + `configs/normal_5000.yaml`, **chưa chạy ván nào** |
| Việc B đã làm | assert giá trị lá trong `az/mcts.py`, export API trong `az/__init__.py` |

## 1. Danh sách task trực tiếp của B

Theo bảng phân công, các task sau đây đều là của B:

| Task | Tên | File sở hữu | Trạng thái | Ghi chú |
|---|---|---|---|---|
| **T8** | `az/model.py` - ResNet policy/value | `az/model.py`, `az/__init__.py` | Đã có / Cần rà soát | Kiểm tra shape, `predict()`, `save/load_checkpoint`. Thường chỉ cần review, ít sửa. |
| **T9** | `az/mcts.py` - PUCT, backup đúng dấu | `az/mcts.py` | Đã có / Cần củng cố | Quan trọng nhất. Cần đảm bảo backup phủ định giá trị con (hard rule 2), chỉ 1 lượt suy luận/lá, Dirichlet noise chỉ ở root, `temp_moves`. |
| **T10** | `az/replay_buffer.py` - Buffer + 8-fold augmentation | `az/replay_buffer.py` | Đã có / Cần rà soát | Kiểm tra lọc theo `rule_set` (không lẫn Normal/Renju), `buffer_size`, augmentation dùng đúng `gomoku/features.py` (chỉ gọi, không sửa). |
| **T11** | `az/selfplay.py` - Sinh ván tự chơi | `az/selfplay.py` | Đã có / Cần rà soát | Kiểm tra multiprocessing trên Windows (`if __name__ == "__main__"`, `torch.set_num_threads(1)`), `z` theo góc nhìn người đang đi, gom lô ở cấp chương trình. |
| **T12** | `az/trainer.py` - Vòng lặp huấn luyện | `az/trainer.py` | Đã có / Cần củng cố | B sở hữu `az/trainer.py`. **KHÔNG** sửa `scripts/train.py` (thuộc C). Đảm bảo loss = `policy_CE*c_pi + value_MSE*c_v + L2`, resume checkpoint đúng. |
| **T19** | `az/game_log.py` - Schema JSONL ván/nước đi | `az/game_log.py` | Có thể cần hoàn thiện | Schema bắt buộc: `game_id`, `rule_set`, `model_iter`, `first_player`, `second_player`, `winner`, `end_reason`, `moves_count`, `duration_sec`, `nn_calls`. |
| **T20** | Chạy train 11x11 theo config | `logs/`, `checkpoints/` | Chạy thực tế | Chạy local/Colab theo `configs/normal.yaml`. Không sửa script của C, chỉ sử dụng `scripts/train.py` như hiện tại. |
| **T25** | Fine-tune Renju từ Normal | `az/` (tích hợp) | Sau cải tiến | Chỉ bắt đầu sau khi hoàn thành phần cải tiến AlphaZero nếu còn thời gian. Phụ thuộc vào `gomoku/rules/renju.py` (của A) và config (của C) theo quy ước phối hợp. |
| **T29** | `az_v1` Tactical Override | `az/player_tactical.py` | **BỊ BLOCK** | Chỉ bắt đầu **sau khi** `gomoku/patterns.py` (T27 của A) tồn tại và `tests/test_patterns.py` pass. |
| **T30** | `az_v1` Threat-space search | `az/threat_space.py` | **BỊ BLOCK** | Chỉ bắt đầu sau T29. Cũng phụ thuộc `gomoku/patterns.py`. |
| **T35** | Chạy giải đấu & đóng băng số liệu | `logs/tournament/` | Sau T31/T32 | C phụ trách script, B phụ trách chạy và đóng băng log. Không sửa script của C. |

## 2. Những việc có thể làm NGAY (an toàn, không đụng A/C)

Vì `gomoku/patterns.py` chưa có, đây là những việc bạn nên ưu tiên nhất lúc này.

| Ưu tiên | Việc | Mục tiêu | Ghi chú |
|---|---|---|---|
| **1** | **Hoàn thiện `az/game_log.py` (T19)** | Đảm bảo schema đầy đủ theo `todo.md` T19 | 100% thuộc B. Rất hữu ích cho self-play, evaluate sau này. |
| **2** | **Rà soát `az/mcts.py`** | Kiểm tra backup dấu, số lượt inference/lá, Dirichlet noise, `temp_moves` | Dễ phát hiện bug cốt lõi. Có thể thêm assert ngắn gọn để bảo vệ logic. |
| **3** | **Củng cố `az/selfplay.py`** | Kiểm tra multiprocessing (Windows spawn), `z` xen kẽ đúng, số mẫu = số nước đi | Quan trọng cho việc sinh dữ liệu đúng. |
| **4** | **Rà soát `az/replay_buffer.py`** | Kiểm tra lọc `rule_set`, giới hạn buffer, augmentation 8-fold | Tránh trộn dữ liệu giữa Normal/Renju (hard rule 7). |
| **5** | **Rà soát `az/trainer.py`** | Kiểm tra công thức loss, resume từ checkpoint, optimizer state | Chỉ sửa file này, không chạm `scripts/train.py`. |
| **6** | **Review `az/model.py`** | Kiểm tra shape I/O, `save/load_checkpoint` round-trip | Thường ổn, nhưng tốt để xác nhận trước khi dùng cho `az_v1`. |
| **7** | **Bổ sung unit test cho `az/`** | Thêm/sửa `tests/test_mcts.py`, `tests/test_selfplay.py`, `tests/test_replay_buffer.py`, `tests/test_trainer.py` | An toàn tuyệt đối. Chỉ nằm trong `tests/` và test module `az/`. |

## 3. Những việc KHÔNG được làm lúc này

| Việc | Lý do |
|---|---|
| **Tạo/sửa `gomoku/patterns.py`** | Thuộc A (T27). Đừng tự tạo workaround. Hãy chờ A hoàn thành và pass test. |
| **Sửa bất kỳ file trong `gomoku/`, `players/`, `ui/`** | Thuộc A. |
| **Sửa `az/evaluator.py`, `scripts/arena.py`, `scripts/tournament.py`, `scripts/ablation.py`** | Thuộc C. Quy ước: *"A và B chỉ cung cấp tác tử, không sửa script đánh giá."* |
| **Sửa `scripts/train.py`** | C sở hữu CLI này. B chỉ viết `az/trainer.py` và giao diện gọi. |
| **Sửa `configs/*`** | Thuộc C. Chỉ đề xuất nếu cần flag mới cho `az_v1`, không tự sửa. |
| **Sửa `docs/*`** | Chủ yếu thuộc C. |

## 4. Hành động đề xuất ngay

1. **Ngay lập tức:** Mở và rà soát `az/game_log.py` theo schema T19. Hoàn thiện những field còn thiếu.
2. **Tiếp theo:** Chạy rà soát lần lượt `az/mcts.py` -> `az/selfplay.py` -> `az/replay_buffer.py` -> `az/trainer.py`. Ưu tiên `az/mcts.py` vì dễ gây bug nghiêm trọng nhất.
3. **Song song:** Bổ sung unit test cho các module trên nếu thấy chưa đủ coverage.
4. **Chờ tín hiệu:** Khi A báo `gomoku/patterns.py` đã xong và `pytest tests/test_patterns.py` pass, **mới** bắt đầu `T29` (`az/player_tactical.py`).

**Quy tắc vàng:** Nếu file không nằm trong `az/` và không nằm trong danh sách sở hữu của B, thì **không sửa**. Tập trung củng cố lõi `az/` trong lúc chờ là cách đúng và an toàn nhất.
