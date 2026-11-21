# Gomoku 11x11 + AlphaZero

Agent Gomoku trên bàn 11x11, tự học bằng self-play theo AlphaZero: mạng ResNet nhỏ, tìm kiếm cây PUCT, huấn luyện từ những ván agent tự đấu. Hỗ trợ hai bộ luật: **Normal** (thắng khi có 5 quân liền hoặc hơn) và **Renju** (cấm nước 3-3, 4-4, quá 5 quân cho bên đen; nước tạo đúng 5 quân luôn hợp lệ).

## Trạng thái

Dự án đã có lõi game 11x11, agent AlphaZero và pipeline huấn luyện/đánh giá.
CLI và giao diện Pygame hỗ trợ Normal/Renju, người đấu máy và máy đấu máy.
Checkpoint Renju và log fine-tune đã có; nguồn gốc, số liệu huấn luyện và giới hạn được đối chiếu trong [báo cáo](docs/report.md).

Các tài liệu kế hoạch:

| Tài liệu | Nội dung |
|---|---|
| [tasks/plan.md](tasks/plan.md) | Kế hoạch triển khai: 26 task, 11 phase, quyết định kiến trúc, bảng rủi ro |
| [tasks/todo.md](tasks/todo.md) | Chi tiết từng task: tiêu chí nghiệm thu, cách kiểm chứng, file dự kiến đụng |
| [tasks/proposal.md](tasks/proposal.md) | Đề tài trình bày: mục tiêu, phương pháp, thiết kế thí nghiệm E1–E5 |
| [tasks/proposal-requirements.md](tasks/proposal-requirements.md) | Yêu cầu đề tài theo từng khối |
| [docs/report.md](docs/report.md) | Thuật toán, huấn luyện, nguồn gốc checkpoint và kết quả đánh giá có bằng chứng |

Đối chiếu tiêu chí và tiến độ theo `tasks/todo.md`.

## Mục tiêu

| # | Mục tiêu | Tiêu chí nghiệm thu |
|---|---|---|
| M1 | Game Gomoku 11x11 chơi được | Đấu người với máy, hoặc hai máy đấu nhau qua giao diện đồ họa |
| M2 | Agent tự ra quyết định | Mọi nước đi đến từ mạng + tìm kiếm cây, không có bảng tra cứu hay heuristic hardcode |
| M3 | Hai bộ luật Normal và Renju | Một kiến trúc mạng, hai bộ luật, agent chơi được cả hai |
| M4 | Đo lường được | 1000 ván đánh giá có log đầy đủ, tổng hợp theo lượt đi trước và nguyên nhân kết thúc ván |

Câu hỏi nghiên cứu:

> Với ngân sách 1000 ván tự chơi, agent Gomoku 11x11 đạt được trình độ nào, và win rate tăng thế nào theo từng 100 ván?

## Cấu trúc dự án (dự kiến)

```
gomoku/        Lõi game, không phụ thuộc AI hay UI
  board.py       Bàn 11x11, play/undo, history
  rules/         base.py (interface), normal.py, renju.py
  features.py    Bàn -> tensor 4 plane, đối xứng 8 phép
  runner.py      Chạy một ván, gọi player qua interface
az/            Tầng AlphaZero
  model.py       ResNet nhỏ (6 block, 64 filter), đầu chính sách (121), đầu giá trị
  mcts.py        PUCT, nhiễu Dirichlet ở gốc, nhiệt độ
  selfplay.py    Sinh ván, multiprocessing, gom lô suy luận
  replay_buffer.py  Buffer + tăng cường đối xứng 8 phép
  trainer.py     Loss = policy CE + value MSE + L2
  evaluator.py   Arena, cổng thay thế mô hình tốt nhất
  logging_v2.py  Nguồn log có cấu trúc SQLite cho self-play, training và tournament
players/       base.py (select_move), random, human, mcts_pure, az_player
ui/            Pygame: human vs AI, AI vs AI
scripts/       train.py, play.py, arena.py, simulate.py, sync_drive.py, selfplay_shard.py, merge_shards.py
colab/         train.ipynb + README (chạy dài trên Google Colab)
configs/       normal.yaml, renju.yaml, colab.yaml
tests/         Test luật (kỹ Renju), MCTS, self-play, huấn luyện, arena
docs/          provenance.md, report.md, README.md
```


Bốn quy tắc kiến trúc giữ xuyên suốt: luật chơi nằm sau một interface duy nhất (`legal_moves`, `is_win`, `is_terminal`), giá trị của node con là góc nhìn đối thủ nên phủ định khi backup, mỗi bộ luật một model riêng, và `gomoku/` không import torch.

## Siêu tham số mặc định

| Nhóm | Tham số | Giá trị |
|---|---|---|
| Tìm kiếm | số lượt mô phỏng mỗi nước | 200 |
| Tìm kiếm | hệ số khám phá `c_puct` | 1.5 |
| Tìm kiếm | nhiễu Dirichlet alpha / epsilon | 0.15 / 0.25 |
| Tìm kiếm | số nước dùng nhiệt độ = 1 | 15 |
| Mô hình | số block / số filter | 6 / 64 |
| Mô hình | tối ưu hoá / tốc độ học | Adam / 1e-4, hằng số cho checkpoint Normal hiện tại |
| Mô hình | hệ số chuẩn hoá L2 | 1e-4 |
| Mô hình | trọng số mất mát chính sách / giá trị | 1.0 / 1.0 |
| Huấn luyện | số bước mỗi vòng / batch size | 100 / 32 cho checkpoint Normal hiện tại |
| Huấn luyện | dung lượng buffer (trước tăng cường) | 50000 |
| Lịch Normal | ván mỗi vòng | 50; tiếp tục từ generation chính thức |
| Định danh | mô hình Normal chính thức | AZ Normal |
| Đánh giá | điều kiện chọn best | cận dưới CI 95% của ΔElo so với best hiện tại > 0 |
| Đánh giá | cỡ mẫu mỗi gate | 100 ván tổng cộng, không mở rộng |
| Đánh giá | opening | 4 nước ngẫu nhiên, đổi màu theo cặp |

Mỗi 5000 ván self-play mới, candidate đấu best; nếu chưa có best thì đấu Heuristic. Khi đã có best, gate không đo Heuristic.
Renju giữ lịch fine-tune riêng trong `configs/renju.yaml`.

## Môi trường

| Mục | Yêu cầu |
|---|---|
| Python | 3.12 (`../.venv` dùng chung toàn repo) |
| Thư viện | Cài từ `../requirements.txt` |
| GPU | NVIDIA RTX 4060 8 GB (sm_89) cho lần train đầu; CPU vẫn chạy được để test |
| Mạng | pip OK |

Colab free chỉ cho 2 vCPU, nên một instance Colab chậm hơn máy local. Colab dùng cho vòng chạy dài và cho việc sinh ván song song, không phải để thay thế local.

## Lệnh dự kiến

Chạy lệnh từ thư mục `AlphaZero_Gomoku`, dùng `.venv` ở root:

```powershell
..\.venv\Scripts\python scripts/play.py --mode human-vs-ai --agent mcts-pure --ruleset normal
..\.venv\Scripts\python scripts/play.py --mode ai-vs-ai --agent random --ruleset renju
..\.venv\Scripts\python scripts/play.py --mode ai-vs-ai --black-agent heuristic --white-agent alphabeta
..\.venv\Scripts\python scripts/play.py --mode human-vs-ai --human-color white --agent heuristic
..\.venv\Scripts\python scripts/play.py --mode ai-vs-ai --black-agent az --white-agent mcts-pure
..\.venv\Scripts\python -m ui.pygame_app --mode human-vs-ai --agent mcts-pure
..\.venv\Scripts\python -m ui.pygame_app --mode ai-vs-ai
..\.venv\Scripts\python scripts/train.py --iterations 701
..\.venv\Scripts\python scripts/arena.py --a random --b az --games 100
..\.venv\Scripts\python scripts/arena.py --a heuristic --b az_v1_tactical --games 100 --budget sims:200
..\.venv\Scripts\python scripts/tournament.py --agents random,heuristic,alphabeta,mcts-pure,az,az_v1_threat --games 100 --budget sims:200
..\.venv\Scripts\python scripts/tournament.py --agents az,az_v1_threat,alphabeta --games 100 --budget time:1
..\.venv\Scripts\python scripts/tournament.py --agents random,heuristic,alphabeta,az_renju --rules renju --games 100 --budget sims:200
..\.venv\Scripts\python -m pytest -q
```


`--agent` đặt cùng một agent cho cả hai phía để giữ tương thích; dùng
`--black-agent` và `--white-agent` để chọn riêng. `--human-color` đổi màu
quân của người chơi. Agent hỗ trợ: `random`, `mcts-pure`, `heuristic`,
`alphabeta`, `az`, `az_v0`, `az_v1_tactical`, and `az_v1_threat`. `az` dùng biến thể
`az_v1_threat` với weights chính thức hiện tại; chọn `az_v0` tường minh để dùng v0.
`--depth`/`--no-pruning` cấu hình Alpha-Beta, `--rollouts` cấu hình MCTS Pure.
Trong chế độ AI-vs-AI, nhấn Space để tạm dừng/tiếp tục và New game để chơi lại.
Checkpoint Normal chính thức: `checkpoints/normal/latest.json`; generation được xác minh
bằng hash và bao gồm weights cùng replay. Training mặc định nạp generation, optimizer và
trạng thái phục hồi; thiếu checkpoint là lỗi, không tự tạo model mới. `--fresh` là
chế độ khởi tạo mới tường minh. Lịch sử thí nghiệm và checkpoint cũ được giữ riêng.
Log Normal: `data/normal.sqlite`; tournament: `data/tournament.sqlite`.
Mốc lịch sử 60k là danh nghĩa, không cộng 20k ván từ run độc lập đã loại bỏ.
Tournament ghi games, moves, agent/config metadata và events vào SQLite; không sinh
JSONL/CSV/JSON output mặc định. Mỗi cặp đổi màu, vì vậy `--games` phải chẵn.
`--checkpoint NAME=PATH` chọn weights riêng cho agent; Normal mặc định dùng
`checkpoints/normal/latest.json`, Renju dùng checkpoint riêng.
`--db` chọn database tournament; mặc định là `data/tournament.sqlite`.
`sims:N` đặt số mô phỏng cho AZ/MCTS Pure, Alpha-Beta dùng `--depth` (mặc định 2);
`time:S` đặt ngân sách giây mỗi nước cho tác tử tìm kiếm. Random/heuristic không
bị làm chậm để dùng hết ngân sách. Báo cáo độ trễ và lượng tìm kiếm thực tế,
không coi số node Alpha-Beta bằng số simulation MCTS.
`--pair NAME,NAME` chạy một cặp với seed theo vị trí trong danh sách đầy đủ.
Để tiếp tục run đã có, dùng cùng cấu hình với `--resume --run-id RUN_ID`; resume
đọc các ván đã commit trong database và từ chối nếu run/config không khớp.

Dry-run inventory migration mặc định quét `logs/` và `evidence/`, không ghi dữ liệu:

```powershell
.venv\Scripts\python.exe scripts/migrate_logs_to_sqlite.py
.venv\Scripts\python.exe scripts/migrate_logs_to_sqlite.py --import
```

Import bổ sung vào hai database chính, idempotent theo source identity/hash, không
xóa hoặc sửa file nguồn. Dùng `scripts/analyze_t35.py --db data/tournament.sqlite`
để truy vấn tournament; JSON export chỉ được tạo khi truyền `--out`.

## Backend Python/C++ (thử nghiệm)

Mặc định là `python`. `scripts/train.py` và `scripts/tournament.py` nhận preset `--backend python|cpp`, cùng override độc lập `--mcts-backend` và `--patterns-backend`; override thắng preset. Build Patterns bằng `scripts/build_gomoku_native.py --module patterns`; build MCTS bằng `scripts/build_gomoku_native.py --module mcts`.

Native MCTS dùng arena C++ sở hữu node theo index; Python vẫn xử lý rules, board, RNG và inference. Có thể chọn `--mcts-backend cpp` sau khi build extension đúng Python ABI. Không fallback im lặng khi extension thiếu/không tương thích. Python vẫn là mặc định: test native mới đạt, nhưng chưa chạy integration với PyTorch/checkpoint, multiprocessing/resume hoặc benchmark end-to-end; microbenchmark search/fake evaluator mới tăng 7,9%, chưa đạt ngưỡng 10%. Không xem backend C++ là production-ready hoặc dùng cho 5.000 ván.

Tournament log ghi cả backend đã chọn và `components_used` thực tế. Patterns C++ không tăng tốc AlphaZero self-play cơ bản, vốn không gọi `find_patterns()`. Xem [báo cáo Dual Backend](docs/bao-cao-dual-backend.md) để biết lệnh, số liệu và các gate còn thiếu.
