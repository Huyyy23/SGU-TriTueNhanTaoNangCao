# Hướng dẫn chạy Ablation (T32)

Script: `scripts/ablation.py`. Mỗi lần chạy ghi một file `logs/ablation/<run_id>.json` (seed, config, kết quả); nếu file đã tồn tại thì script dừng, không ghi đè. Không có resume.

## 1. Alpha-Beta: bật/tắt pruning cùng độ sâu

```bash
python scripts/ablation.py alphabeta --depths 1,2 --positions 20 --position-plies 100 --seed 0 \
    --run-id T32_alphabeta_d12_p20_s0 --plot
```

- Cùng một thế cờ, cùng độ sâu, chạy `AlphaBetaPlayer(pruning=True)` rồi `pruning=False`; ghi nước đi, `nodes_visited`, thời gian, `same_move`, tỷ lệ node và thời gian (tắt/bật).
- Thế cờ lấy từ `--position-plies` nước ngẫu nhiên hợp lệ với seed cố định (danh sách nước lưu trong JSON để tái lập).
- Ở độ sâu 1 không có nhánh nào bị cắt, nên node bằng nhau; khác biệt chỉ thấy từ độ sâu 2.
- Bàn cờ càng trống càng nặng. Trên bàn 100 quân, độ sâu 2 mất khoảng 1,4–1,8 giây khi bật và 2,8–3,6 giây khi tắt (smoke test). Huy chọn `--position-plies` và `--depths` theo cấu hình nhóm đã chốt; nên thử 1 thế trước để đo thời gian.

## 2. MCTS Pure: 50/200/800 simulations

```bash
python scripts/ablation.py mcts --sims 50,200,800 --games 100 --opponent heuristic --seed 0 \
    --run-id T32_mcts_50_200_800_g100_s0 --plot
```

- Mỗi mức chơi `--games` ván với `--opponent` (chẵn, chia đều màu), dùng lại lịch thi đấu và cách tính điểm của `scripts/tournament.py`.
- Kết quả mỗi mức: thắng/hòa/thua, điểm, Wilson 95%, điểm khi cầm Đen/Trắng, thời gian trung bình và p95 mỗi nước, số rollout mỗi nước; có cả log từng ván.
- `--opponent` nhận cùng cú pháp với `--agents` của tournament (ví dụ `alphabeta@depth=1`), trừ agent cần checkpoint.

## 3. Vẽ biểu đồ từ JSON

```bash
python scripts/ablation.py plot logs/ablation/T32_mcts_50_200_800_g100_s0.json
```

Cần `matplotlib` (không có trong `requirements.txt`: `pip install matplotlib`). PNG nằm cạnh file JSON.

## 4. Kiểm tra đủ dữ liệu sau khi chạy

- Alpha-Beta: số dòng `results` = số thế x số độ sâu; `same_move` nên là true; ở độ sâu 1 `nodes_on` = `nodes_off`.
- MCTS: mỗi mức `games` = `--games`, `games_black` = `games_white`, `illegal_moves` = 0, `wins + draws + losses` = `games`; `rollouts_per_move` đúng bằng mức.

Mẫu thật từ smoke test: `docs/samples/ablation/`.
