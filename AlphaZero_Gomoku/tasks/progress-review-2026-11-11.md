# Rà soát tiến độ AlphaZero Gomoku — 2026-11-11

Phạm vi: `AlphaZero_Gomoku/` (phần Gomoku 11x11 + AlphaZero), đối chiếu với bảng phân công
`tasks/todo.md` và lộ trình `tasks/Lo_trinh_cai_tien_AlphaZero_Gomoku.md`.

Trạng thái repo lúc rà soát: `HEAD = d0e8528`.

## 0. Cách kiểm và giới hạn

1. `graphify update .` — graph trước khi cập nhật chỉ có **679 node** và thiếu hẳn
   `gomoku/patterns.py`, `az/player_tactical.py`, `az/threat_space.py`, `players/heuristic.py`,
   `players/alphabeta.py`, `ui/`. Sau cập nhật: **904 node, 2264 cạnh, 41 community**.
2. Truy vấn Graphify đã dùng:
   - `graphify query "find_patterns consumers heuristic tactical override threat space integration"`
   - `graphify path "find_patterns" "TacticalAZPlayer" --undirected` → 2 hop,
     `EXTRACTED`: `az/threat_space.py` import `find_patterns()`
   - `graphify explain "HeuristicPlayer"` → chỉ có test và nội bộ class, không có consumer ngoài test.
3. Đối chiếu source và chạy test thật bằng `.venv` (Python 3.10.6, torch 2.14.1):

   ```
   146 passed, 2 skipped in 282.89s
   ```

   2 test bị skip là hai cổng `RUN_SLOW` trong `tests/test_az_player.py:69` và `:105`.
4. **Giới hạn:** git chỉ có 2 tài khoản (`Huy`, `WhaleeYHN`) nên không phân biệt được A/B/C
   theo authorship. Tiến độ dưới đây suy ra từ **file sở hữu + test + log**, đúng theo bảng
   phân công `tasks/todo.md`, không suy ra từ người commit.

## 1. Tổng quan

| Người | Số task theo phân công | Xong | Chưa xong |
|---|---|---|---|
| A — Khang | 11 (T3–T7, T23, T24, T27, T28, T33, T36) | 10 | T36 |
| B — Huy | 10 (T8–T12, T19, T20, T25, T29, T30, T35) | 8 | T25, T35 (T20 mở rộng chưa chạy; T29/T30 chưa nối) |
| C — Hường | 8 (T1, T2, T13, T22, T26, T31, T32, T34) | 3 | T22, T26, T31, T32, T34 |

## 2. A — Khang (engine, luật, tác tử cổ điển, giao diện)

| Task | Bằng chứng | Trạng thái |
|---|---|---|
| T3 `gomoku/board.py` | 8 test | Xong |
| T4 `gomoku/rules/base.py` + `normal.py` | 11 test | Xong |
| T5 `gomoku/features.py` | 11 test, có test mặt phẳng nước cấm và 8 đối xứng | Xong |
| T6 player base + random + human + runner + CLI | `players/{base,random,human}.py`, `gomoku/runner.py`, `scripts/play.py`, `main.py`, 7 test | Xong, nhưng CLI chỉ có `random` / `mcts-pure` / checkpoint |
| T7 `players/mcts_pure.py` | 3 test | Xong |
| T23 `ui/pygame_app.py` | 4 test | Xong (UI cũng chỉ chọn random / mcts-pure / checkpoint) |
| T24 `gomoku/rules/renju.py` | 13 test, gồm 3-3, 4-4, overline (quá 5) và ngoại lệ đúng 5 | Xong |
| T27 `gomoku/patterns.py` | 12 test, `find_patterns()` tại L17 (commit `430f8d0`, `1de3d8a`) | Xong |
| T28 `players/heuristic.py` | 3 test (commit `c6fe228`) | Xong |
| T33 `players/alphabeta.py` | 2 test (commit `c6fe228`) | Xong |
| **T36 `gomoku/game.py`** | `gomoku/game.py` không tồn tại, `tests/test_game.py` không tồn tại | **Chưa làm** |

Kết luận: engine, luật, tác tử cấp 1–4 đều có code và test. Hai lỗ hổng của A:

1. T36 — API 7 phương thức (`reset`, `legal_actions`, `step`, `is_terminal`, `result`, `render`,
   `clone`) là yêu cầu ghi thẳng trong đề bài, chưa có file.
2. `players/heuristic.py` và `players/alphabeta.py` **không nối được vào đường chạy nào**:
   `scripts/play.py:53` chỉ cho `("random", "mcts-pure")`, `ui/pygame_app.py:337` cũng vậy.
   Đề bài yêu cầu "chọn được tác tử theo cặp" nên phần này chưa thoả.

## 3. B — Huy (lõi AlphaZero, self-play, huấn luyện, log)

| Task | Bằng chứng | Trạng thái |
|---|---|---|
| T8–T12 `az/model.py`, `az/mcts.py`, `az/replay_buffer.py`, `az/selfplay.py`, `az/trainer.py` | 44 test (9 + 8 + 9 + 12 + 6) | Xong |
| T19 `az/game_log.py` | Đủ 10 field bắt buộc: `game_id`, `rule_set`, `model_iter`, `first_player`, `second_player`, `winner`, `end_reason`, `moves_count`, `duration_sec`, `nn_calls`; có cả `first_player_won`, `moves`, `config`; 6 test | Xong |
| T20 train 11x11 (`az_v0`) | `logs/train_20261028-063106.jsonl`: 20 vòng × 50 ván = 1000 ván. Vòng cuối: `loss 4,091`, `policy_loss 3,770`, `value_loss 0,321`, 23,1 nước/ván, 4.408 lượt suy luận/ván, ~163 s self-play + 2,45 s train mỗi vòng, RTX 4060 Laptop | Xong |
| T20 mở rộng 5000 ván (`az_v0b`) | `configs/normal_5000.yaml` + ADR 0005 có; log duy nhất của `az_v0b` là **1 vòng 2 ván** (`logs/train_20261101-163215.jsonl`, `n_iter: 1`, `games_per_iter: 2`, `n_simulations: 20`). `checkpoints/az_v0b/last.pt` trùng mtime và size với `checkpoints/last.pt` | **Chưa chạy** |
| T29 `az/player_tactical.py` | 6 test; ctor `(base_player, enabled=False, cfg=None)` | **Code xong, chưa nối** |
| T30 `az/threat_space.py` | 6 test; `ThreatSpaceSearcher(max_nodes=5000, time_limit_ms=150)` | **Code xong, chưa nối** |
| T25 fine-tune Renju | Không có checkpoint hay log Renju nào | Chưa bắt đầu |
| T35 chạy giải đấu | `logs/tournament/` không tồn tại | Chưa chạy |

Kết luận: lõi `az/` mạnh nhất trong ba nhánh, nhưng **T29/T30 hiện là đảo không có cầu**.
`grep -r "TacticalAZPlayer|ThreatSpaceAZPlayer"` cho 13 kết quả: toàn bộ nằm trong
`tests/test_tactical.py`, `tests/test_threat_space.py`, và hai file định nghĩa. Không có
`config.py` key, không có `configs/*.yaml` key, không script nào dựng chúng.

## 4. C — Hường (hạ tầng, đánh giá, tài liệu)

| Task | Bằng chứng | Trạng thái |
|---|---|---|
| T1 skeleton/config | `requirements.txt`, `config.py`, 4 test | Xong |
| T2 config loader | `configs/{normal,renju,normal_5000}.yaml`, 4 test | Xong |
| T13 `az/evaluator.py` + `scripts/arena.py` | 4 test + 1 smoke log `logs/arena/smoke_az_v0_vs_random` (20 ván) | Xong, nhưng chỉ đấu được 2 tác tử |
| **T31 `scripts/tournament.py`** | Không có file, không có `tests/test_tournament.py` | **Chưa làm** |
| **T32 `scripts/ablation.py`** | Không có file, không có `tests/test_ablation.py` | **Chưa làm** |
| **T22 `docs/provenance.md`** | Không có | Chưa làm |
| **T26 `docs/report.md`** | Không có | Chưa làm |
| **T34 poster, slide, ZIP nộp bài** | Không có trong repo | Chưa làm |

`docs/` hiện chỉ có `benchmarks.md`, `docs/adr/0001–0005`, `docs/agents/*`.

## 5. Chỗ đang kẹt, xếp theo mức chặn

1. **T31 `scripts/tournament.py` (C) — nút thắt số 1.**
   `scripts/arena.py:25-38` chỉ nhận `"random"` hoặc một checkpoint, nên **không thể cho
   `az_v0` đấu `heuristic`, `alphabeta` hay `mcts_pure`**. Không có T31 thì T35 không chạy
   được, Chương 4 không có số, và câu hỏi nghiên cứu (`az_v1` có tốt hơn `az_v0`?) không
   trả lời được. `Lo_trinh` §7 xếp mục này vào loại **"không cắt"**.

2. **Cầu nối `az_v1` chưa tồn tại.**
   T29/T30 xong code nhưng thiếu ba thứ nhỏ: key `tactical.enabled` / `threat_space.enabled`
   trong `config.py` + `configs/*.yaml` (C), một agent factory dựng đúng wrapper (C, trong
   arena/tournament), và một lựa chọn trong `scripts/play.py` (A). Thiếu cả ba thì không ai đo
   được mốc 2.1 và 2.2.

3. **T36 `gomoku/game.py` (A).**
   Thiếu file và test; đây là API thống nhất mà đề bài yêu cầu, không phải hạng mục phụ.

4. **Số đo sức mạnh `az_v0` gần như trống.**
   Thứ duy nhất có là 20 ván với random: thắng 19/20 (95%), Wilson 95% cho tỉ lệ thắng của
   random là [0,9%, 23,6%], ván trung bình 9,0 s, 38,85 nước/ván. Hai test cổng thật
   (`tests/test_az_player.py:69` và `:105`, 20 ván) **bị skip vì cần `RUN_SLOW=1`**, nên chưa
   từng chạy trong suite. `az_v0` **chưa đấu một tác tử cổ điển nào** — đúng như cảnh báo ở
   `Lo_trinh` §0.

5. **Tài liệu nộp (T22, T26, T34) trống hoàn toàn.** Đây là hạng mục tính điểm.

6. **T32 ablation (C) chưa có.** Đề bài yêu cầu hai ablation: cắt tỉa Alpha-Beta bật/tắt, và
   quét số mô phỏng của `mcts_pure` ở 50/200/800.

7. **T20 mở rộng 5000 ván (B) chưa chạy ván nào** — mới có config và ADR.

8. **T25 fine-tune Renju (B) chưa bắt đầu.**

## 6. Ưu tiên đề xuất

| Mức | Việc | Người | Ước lượng | Điều kiện xong |
|---|---|---|---|---|
| **P1** | T31 `scripts/tournament.py` + agent factory (heuristic, alphabeta, mcts_pure, az_v0, az_v1), `tests/test_tournament.py` | C | ~1 ngày | `python scripts/tournament.py --agents random,heuristic,alphabeta,mcts_pure,az_v0 --games 20` chạy hết mọi cặp, ra `matrix.csv` + `summary.json` có Wilson CI |
| **P2** | Đo `az_v0` bằng số thật, 100 ván/cặp, chia đều màu, lưu `logs/arena/` | C vận hành, B cấp checkpoint | ~2–4 giờ chạy (≈9 s/ván ở 200 sims) | Có bảng score theo màu cho từng cặp, không nhập tay số |
| **P3** | Nối `az_v1`: cờ config + factory + `--agent az-v1-tactical` trong `play.py` | C + A | ~0,5 ngày | Ba mốc `az_v0` → `+tactical` → `+threat` dựng được từ CLI, cùng trọng số, cùng sims |
| **P4** | T36 `gomoku/game.py` + `tests/test_game.py`; đưa `heuristic`/`alphabeta` vào danh sách agent của CLI và UI | A | ~0,5 ngày | `pytest -q tests/test_game.py` pass; chọn được 4 tác tử theo cặp |
| **P5** | T35: chạy 2.400 ván, đóng băng `logs/tournament/<run_id>/`; bật `RUN_SLOW=1` cho 2 gate test | B | ~6 giờ máy | Có log đóng băng + gate test pass trong CI |
| **P6** | T22 `docs/provenance.md`, T26 `docs/report.md` (Chương 2–3 do B, Chương 4 do C), T34 poster/slide/ZIP | C | 2–3 ngày | Đủ file nộp, số liệu lấy từ `logs/` |
| **P7** | Cắt được nếu trễ: T32 ablation (giữ ablation cắt tỉa), T20 mở rộng 5000 ván, T25 fine-tune Renju | C, B | — | Ghi rõ trong báo cáo là mục đã cắt và lý do |

## 7. Rủi ro cần nói trước

- `az_v0` 1000 ván trên 11×11 rất có thể thua heuristic hoặc Alpha-Beta độ sâu 4. Đó là kết quả
  hợp lệ (cấp không tương ứng sức mạnh) miễn là có số và ghi rõ, không đổi luật chơi để cứu.
- Lợi thế đi trước của Gomoku Normal: mọi bảng score phải tách Đen/Trắng.
- Tactical override có thể che lấp điểm yếu của mạng: cần ghi tỉ lệ số nước bị ghi đè.
- Threat-space có giới hạn cứng 150 ms/nước; nếu vượt thì tắt và ghi nhận.
- Graphify chỉ là công cụ định vị. Mọi kết luận trong tài liệu này đã đối chiếu với source,
  test và log thật.
