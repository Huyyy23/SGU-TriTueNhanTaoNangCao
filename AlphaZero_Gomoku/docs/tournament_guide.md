# Hướng dẫn chạy Tournament (T31) cho chạy dài

Tài liệu này chỉ dùng lệnh và tham số có thật trong `scripts/tournament.py`. Không cần sửa source để đổi agent, số ván, luật, seed hay ngân sách.

## 1. Chuẩn bị

```bash
cd AlphaZero_Gomoku
pip install -r requirements.txt
pytest -q tests/test_tournament.py tests/test_agent_factory.py     # phải pass trước khi chạy dài
```

 - Checkpoint đặt trong `checkpoints/` (không commit); script chỉ đọc file và ghi SHA-256 cùng provenance vào SQLite.
 - `--dry-run` kiểm tra cấu hình mà không chạy ván hoặc tạo run.
- Xem trước cấu hình, không chơi ván nào:

```bash
python scripts/tournament.py --agents random,heuristic,alphabeta,mcts_pure,az_v0,az_v1_tactical \
    --games 100 --budget sims:200 --checkpoint checkpoints/best.pt --dry-run
```

## Backend Python/C++

Mặc định `--backend auto`: tournament chỉ chọn các native component mà agent thực sự dùng. Nếu `g++`, `clang++` hoặc `cl` có trong `PATH`, module thiếu/cũ sẽ được build một lần trước khi chạy; nếu không có compiler, build lỗi hoặc module không nạp được, chương trình cảnh báo ngắn và dùng Python. Worker chỉ nạp module đã sẵn sàng, không build.

- `--backend cpp`: bắt buộc native; không khả dụng thì dừng với lỗi rõ ràng.
- `--backend python`: không dò compiler, build hay dùng native.
- `--mcts-backend` và `--patterns-backend`: ghi đè độc lập preset.

 Backend thực tế và component được dùng được lưu trong event SQLite cùng game/run; không ghi thêm file `backend_runs.sqlite3` hay JSONL.

## 2. Chọn agent

`--agents` là danh sách cách nhau bằng dấu phẩy, tối thiểu 2 agent, không trùng tên.

| Tên | Agent | Ghi chú |
|---|---|---|
| `random` | RandomPlayer | không tìm kiếm |
| `heuristic` | HeuristicPlayer | không tìm kiếm |
| `alphabeta` | AlphaBetaPlayer | độ sâu cố định `--alphabeta-depth` (mặc định 1) |
| `mcts_pure` (hoặc `mcts-pure`) | MCTSPurePlayer | số rollout lấy từ `--budget sims:N` |
| `az_v0` (hoặc `az`) | AZPlayer | Normal mặc định dùng `checkpoints/normal/latest.json`; có thể chỉ định checkpoint riêng |
| `az_v1_tactical` | TacticalAZPlayer bọc AZPlayer | tactical bật; `az_v1` bị từ chối vì mơ hồ |
| `az_v1_threat` | ThreatSpaceAZPlayer bọc AZPlayer | tactical + threat-space bật |

Tùy chọn riêng cho từng agent, nối bằng `@`:

- `alphabeta@depth=2`, `alphabeta@pruning=false`
- `mcts_pure@sims=50` (ghi đè `--budget` cho riêng agent này)
- `az_v0@checkpoint=checkpoints/renju_last.pt` (checkpoint khác cho riêng agent này)

 Tên đầy đủ (kể cả phần `@...`) là nhãn agent trong SQLite; có thể đặt hai biến thể cùng loại trong một giải.

Cả ba mốc AlphaZero dùng chung một checkpoint, cùng số simulations và cùng quy tắc seed. Chúng được tạo qua `players/factory.py`, cùng factory với `scripts/play.py` và giao diện Pygame. Cờ `tactical.enabled` và `threat_space.enabled` trong `configs/*.yaml` mặc định `false`; factory tự bật theo tên agent.

## 3. Ngân sách `--budget`

| Agent | `sims:N` | `time:S` |
|---|---|---|
| `az_*` | N mô phỏng PUCT mỗi nước | dừng tìm sau S giây mỗi nước (kiểm tra sau mỗi mô phỏng, nên có thể vượt S vài mili giây); số mô phỏng tối đa `--time-sims-cap` (mặc định 100000) |
| `mcts_pure` | N rollout mỗi nước | deadline mềm S giây, tối đa `--mcts-rollouts-time-mode` rollout; tác vụ đang chạy có thể vượt deadline |
| `alphabeta` | độ sâu cố định, N không áp dụng | iterative deepening tới `--depth`, deadline mềm S giây |
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
| `--db` | `data/tournament.sqlite` | database local dùng chung; không ghi đè database khác |
| `--run-id` | tự sinh | ID ổn định của experiment; resume yêu cầu ID đã tồn tại |
Ván 2k và 2k+1 của mỗi cặp dùng cùng seed và cùng khai cuộc, chỉ đổi màu, nên kết quả theo màu so sánh được. Nước đi trái luật làm bên đó thua ván và được đếm trong `illegal_moves_*`.

**Resume:** chạy lại cùng cấu hình với `--resume --run-id RUN_ID`. Runner đọc dữ liệu đã commit trong SQLite, kiểm tra cấu hình/checkpoint/seed, rồi chỉ tiếp tục các ván còn thiếu. Dùng `--pair NAME,NAME` để giới hạn một cặp. Không có shard merge/file-based resume mới; historical JSONL vẫn giữ nguyên và chỉ importer được phép đọc chúng.

## 5. Dữ liệu và truy vấn

Kết quả được ghi vào `data/tournament.sqlite`; không tạo thư mục output riêng, JSONL, manifest, CSV hay summary file. Schema lưu experiment/config, agents, matches, games, moves, events, provenance/backend metadata và lịch sử Elo. Cùng database dùng cho resume, `scripts/analyze_t35.py`, và `scripts/recalculate_elo.py`.

```sql
SELECT experiment_id, rule_set, config_json FROM experiments;
SELECT agent_id, display_name, checkpoint_hash, backend FROM agents;
SELECT experiment_id, winner, count(*) FROM games
GROUP BY experiment_id, winner;
PRAGMA foreign_key_check;
PRAGMA integrity_check;
```

`scripts/analyze_t35.py --db data/tournament.sqlite` chỉ truy vấn SQLite; `--out PATH` tạo JSON export tường minh. Không xóa hoặc thay database để đồng bộ; trao đổi thí nghiệm qua `scripts/exchange_experiment.py`.

Mẫu trong `docs/samples/tournament/` là evidence lịch sử, không phải output mặc định mới.

## 6. Lệnh cho các giải chính thức (T35)

Thay đường dẫn checkpoint bằng checkpoint đã chốt. Nên chạy `--dry-run` trước.

```bash
# Normal chế độ A: 6 agent, 15 cặp x 100 ván = 1.500 ván
python scripts/tournament.py --agents random,heuristic,alphabeta,mcts_pure,az_v0,az_v1_tactical \
    --games 100 --budget sims:200 --checkpoint checkpoints/best.pt --seed 0 --workers 4 \
    --run-id T35_normal_A_sims200_s0

# Normal chế độ B (1 s/nước): 3 cặp x 100 ván = 300 ván
python scripts/tournament.py --agents az_v0,az_v1_tactical,alphabeta \
    --games 100 --budget time:1 --checkpoint checkpoints/best.pt --seed 0 --workers 4 \
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
 - Đặt `--run-id` duy nhất cho mỗi giải; resume chỉ với cùng ID/cấu hình.
 - Sau khi chạy, đối chiếu số game và kết quả trực tiếp trong SQLite; chạy `PRAGMA integrity_check` và `PRAGMA foreign_key_check`.
 - Không dọn file evidence cũ trong khuôn khổ Logging V2.

## 9. Những điều script không làm

 - `--logging-level none` chỉ dùng làm baseline đo, không lưu dữ liệu và không thể resume; không dùng cho nghiên cứu.
 - Tournament không train, không sửa checkpoint, không đổi luật hoặc thuật toán AI.
