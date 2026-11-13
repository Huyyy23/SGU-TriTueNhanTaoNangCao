# Hướng dẫn chạy Tournament (T31) cho chạy dài

Tài liệu này chỉ dùng lệnh và tham số có thật trong `scripts/tournament.py`. Không cần sửa source để đổi agent, số ván, luật, seed hay ngân sách.

## 1. Chuẩn bị

```bash
cd AlphaZero_Gomoku
pip install -r requirements.txt
pytest -q tests/test_tournament.py tests/test_agent_factory.py     # phải pass trước khi chạy dài
```

- Checkpoint đặt trong `checkpoints/` (thư mục này bị `.gitignore`, không commit). Ví dụ: `checkpoints/last.pt` cho Normal, checkpoint Renju riêng cho `az_renju`.
- Script chỉ đọc checkpoint. SHA-256 của mỗi checkpoint được ghi vào `config.json` trước khi chạy và kiểm tra lại khi xong (`checkpoint_unchanged` trong `summary.json`).
- Xem trước cấu hình, không chơi ván nào:

```bash
python scripts/tournament.py --agents random,heuristic,alphabeta,mcts_pure,az_v0,az_v1 \
    --games 100 --budget sims:200 --checkpoint checkpoints/last.pt --dry-run
```

## 2. Chọn agent

`--agents` là danh sách cách nhau bằng dấu phẩy, tối thiểu 2 agent, không trùng tên.

| Tên | Agent | Ghi chú |
|---|---|---|
| `random` | RandomPlayer | không tìm kiếm |
| `heuristic` | HeuristicPlayer | không tìm kiếm |
| `alphabeta` | AlphaBetaPlayer | độ sâu cố định `--alphabeta-depth` (mặc định 1) |
| `mcts_pure` (hoặc `mcts-pure`) | MCTSPurePlayer | số rollout lấy từ `--budget sims:N` |
| `az_v0` (hoặc `az`) | AZPlayer | cần `--checkpoint` |
| `az_v1_tactical` (bí danh `az_v1`) | TacticalAZPlayer bọc AZPlayer | tactical bật |
| `az_v1_threat` | ThreatSpaceAZPlayer bọc AZPlayer | tactical + threat-space bật |

Tùy chọn riêng cho từng agent, nối bằng `@`:

- `alphabeta@depth=2`, `alphabeta@pruning=false`
- `mcts_pure@sims=50` (ghi đè `--budget` cho riêng agent này)
- `az_v0@checkpoint=checkpoints/renju_last.pt` (checkpoint khác cho riêng agent này)

Tên đầy đủ (kể cả phần `@...`) là nhãn của agent trong `matrix.csv`, nên có thể đặt hai biến thể cùng loại trong một giải, ví dụ `mcts_pure@sims=50,mcts_pure@sims=800`.

Cả ba mốc AlphaZero dùng chung một checkpoint, cùng số simulations và cùng quy tắc seed. Chúng được tạo qua `players/factory.py`, cùng factory với `scripts/play.py` và giao diện Pygame. Cờ `tactical.enabled` và `threat_space.enabled` trong `configs/*.yaml` mặc định `false`; factory tự bật theo tên agent.

## 3. Ngân sách `--budget`

| Agent | `sims:N` | `time:S` |
|---|---|---|
| `az_*` | N mô phỏng PUCT mỗi nước | dừng tìm sau S giây mỗi nước (kiểm tra sau mỗi mô phỏng, nên có thể vượt S vài mili giây); số mô phỏng tối đa `--time-sims-cap` (mặc định 100000) |
| `mcts_pure` | N rollout mỗi nước | **không có điều khiển thời gian** trong `players/mcts_pure.py`: chạy cố định `--mcts-rollouts-time-mode` rollout (mặc định 200) và đếm số nước vượt S |
| `alphabeta` | độ sâu cố định, N không áp dụng | **không có điều khiển thời gian**: độ sâu cố định, đếm số nước vượt S |
| `random`, `heuristic` | không áp dụng | không áp dụng |

Cách hiểu ngân sách của từng agent được in ra đầu mỗi lần chạy và ghi vào `config.json` (`agent_settings.<agent>.budget_note`).

## 4. Các tham số khác

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--games` | 100 | số ván mỗi cặp, phải chẵn; nửa ván mỗi bên cầm Đen |
| `--ruleset` | `normal` | `normal` hoặc `renju` |
| `--config` | `configs/<ruleset>.yaml` | YAML AlphaZero (c_puct, ...); không đổi trọng số |
| `--seed` | 0 | seed gốc; seed từng ván = crc32(seed, cặp, khối 2 ván, A/B) |
| `--random-opening` | 0 | số nước khai cuộc ngẫu nhiên dùng chung cho 2 ván đổi màu; 0 = tắt |
| `--workers` | 1 | số tiến trình chơi song song |
| `--torch-threads` | 1 | số luồng torch mỗi tiến trình; 0 = mặc định của torch |
| `--out-root` | `logs/tournament` | thư mục gốc của log |
| `--run-id` | tự sinh `<UTC>_<ruleset>_<budget>_s<seed>` | tên thư mục log; nếu đã tồn tại thì script dừng, không ghi đè |

Ván 2k và 2k+1 của mỗi cặp dùng cùng seed và cùng khai cuộc, chỉ đổi màu, nên kết quả theo màu so sánh được. Nước đi trái luật làm bên đó thua ván và được đếm trong `illegal_moves_*`.

**Không có resume.** Nếu chạy bị ngắt, `games.jsonl` vẫn giữ các ván đã xong nhưng phải chạy lại cả giải với `--run-id` mới. Với cùng lệnh và cùng seed, các ván được tạo lại giống hệt (trừ thời gian đo).

## 5. Đầu ra `logs/tournament/<run_id>/`

| File | Nội dung |
|---|---|
| `config.json` | lệnh, agent, ngân sách, luật, seed, SHA-256 checkpoint, commit git, phiên bản thư viện; ghi trước ván đầu tiên |
| `games.jsonl` | mỗi dòng một ván: người đi trước/sau, thắng, lý do kết thúc, toàn bộ nước đi, seed, thời gian từng nước, node/sims, override, threat-space |
| `matrix.csv` | điểm của agent hàng gặp agent cột (thắng 1, hòa 0,5, thua 0) và điểm tổng |
| `pairs.csv` | mỗi cặp: thắng/hòa/thua, điểm, Wilson 95%, điểm khi cầm Đen/Trắng, thời gian trung bình và p95 mỗi nước, node, số lần gọi mạng |
| `summary.json` | tất cả ở trên, cộng thống kê từng agent: override (số lần, tỷ lệ, lý do, thời gian thêm), threat-space (số lần tìm, kích hoạt, chạm giới hạn, node, ms), `warnings` |

Mã thoát: 0 nếu không có nước trái luật và checkpoint không đổi; 2 nếu có.

Mẫu thật từ smoke test nằm trong `docs/samples/tournament/` (xem README trong đó).

## 6. Lệnh cho các giải chính thức (T35)

Thay đường dẫn checkpoint bằng checkpoint đã chốt. Nên chạy `--dry-run` trước.

```bash
# Normal chế độ A: 6 agent, 15 cặp x 100 ván = 1.500 ván
python scripts/tournament.py --agents random,heuristic,alphabeta,mcts_pure,az_v0,az_v1 \
    --games 100 --budget sims:200 --checkpoint checkpoints/last.pt --seed 0 --workers 4 \
    --run-id T35_normal_A_sims200_s0

# Normal chế độ B (1 s/nước): 3 cặp x 100 ván = 300 ván
python scripts/tournament.py --agents az_v0,az_v1,alphabeta \
    --games 100 --budget time:1 --checkpoint checkpoints/last.pt --seed 0 --workers 4 \
    --run-id T35_normal_B_time1_s0

# Renju: 4 agent, 6 cặp x 100 ván = 600 ván
python scripts/tournament.py --agents random,heuristic,alphabeta,az_v0 --ruleset renju \
    --games 100 --budget sims:200 --checkpoint checkpoints/<renju>.pt --seed 0 --workers 4 \
    --run-id T35_renju_sims200_s0
```

Thêm mốc threat-space vào giải: thêm `az_v1_threat` vào `--agents`.

## 7. Ước tính thời gian (đo trên máy Linux CPU, chỉ để lập kế hoạch)

Từ log smoke test, 1 tiến trình:

| Agent | Thời gian trung bình mỗi nước |
|---|---|
| `random` | khoảng 0,1 ms |
| `heuristic` | khoảng 18 ms |
| `alphabeta` độ sâu 1 | khoảng 2,2 giây (p95 khoảng 2,8 giây) |
| `mcts_pure` 200 rollout | khoảng 0,14–0,16 giây |
| `az_v0`, mạng 6x64, 32 mô phỏng, `--torch-threads 1` | khoảng 0,13 giây (Normal), 0,28 giây (Renju, 16 mô phỏng) |

Ví dụ đo thật: `random,heuristic,alphabeta`, 20 ván/cặp, `sims:200` mất 1.004 giây với 1 tiến trình (khoảng 176 giây khi thử `--workers 6`); hai lần chạy cùng seed cho ra 60 ván giống hệt nhau. Cặp `random` vs `alphabeta` chiếm phần lớn thời gian (trung bình 35 nước/ván). Ước lượng cho giải chính thức: (số ván) x (số nước trung bình mỗi ván) x (thời gian mỗi nước) / số worker. Huy nên chạy thử 2 ván/cặp trên máy mình để đo lại.

Lưu ý: torch mặc định dùng nhiều luồng; với một thế cờ mỗi lần suy luận thì chậm hơn khoảng 10 lần trên CPU trong smoke test, vì vậy mặc định `--torch-threads 1` và dùng `--workers` để song song.

## 8. Checklist trước khi chạy dài

- [ ] `pytest -q tests/test_tournament.py tests/test_agent_factory.py` pass trên máy chạy.
- [ ] `--dry-run` in đúng danh sách agent, số cặp, tổng số ván và cách hiểu ngân sách.
- [ ] Checkpoint đúng file, đúng luật (Normal/Renju), đúng kiến trúc trong YAML; `az_v0`, `az_v1_tactical`, `az_v1_threat` tạo được (script tạo thử mọi agent trước ván đầu, lỗi checkpoint sẽ dừng ngay).
- [ ] Chạy thử 2 ván/cặp với `--run-id` thử: `illegal_moves_total` = 0, `checkpoint_unchanged` = true, `games_a_black` = `games_a_white` mọi cặp.
- [ ] Ước tính thời gian theo mục 7 từ lần chạy thử.
- [ ] Đặt `--run-id` mới cho mỗi giải chính thức; không xóa thư mục log cũ.
- [ ] Đọc `warnings` trong `summary.json`. Nếu báo hai agent tất định cho ra ván giống hệt nhau (ví dụ `heuristic` vs `alphabeta`), cân nhắc `--random-opening 2` và ghi lý do.
- [ ] Sau khi chạy: số dòng `games.jsonl` = `total_games` trong `config.json`; tổng thắng+hòa+thua mỗi cặp = `--games`.

## 9. Những điều script không làm

- Không resume giải bị ngắt.
- `alphabeta` và `mcts_pure` không tuân thủ `time:S` (chỉ đếm số nước vượt).
- Không train, không sửa trọng số, không đổi luật.
