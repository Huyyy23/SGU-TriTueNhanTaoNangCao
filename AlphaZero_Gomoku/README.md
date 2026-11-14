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
  game_log.py    Schema ván JSONL
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
| Mô hình | tối ưu hoá / tốc độ học | Adam / 1e-3, giảm dần theo cosin |
| Mô hình | hệ số chuẩn hoá L2 | 1e-4 |
| Mô hình | trọng số mất mát chính sách / giá trị | 1.0 / 1.0 |
| Huấn luyện | số bước mỗi vòng / batch size | 200 / 128 |
| Huấn luyện | dung lượng buffer (trước tăng cường) | 50000 |
| Ngân sách | số vòng × số ván mỗi vòng | 20 × 50 = 1000 ván mỗi bộ luật |
| Ngân sách mở rộng | lần chạy `az_v0b`, `configs/normal_5000.yaml` | 100 × 50 = 5000 ván, 1000 bước huấn luyện mỗi vòng |
| Đánh giá | ngưỡng thay thế mô hình tốt nhất | tỉ lệ thắng ≥ 0.55 |
| Đánh giá | số ván đánh giá | 1000, xen kẽ lượt đầu |

Ngân sách ván nằm trong `configs/*.yaml`, tăng lên 1200, 1500 hay 5000 ván chỉ cần sửa YAML rồi chạy tiếp từ checkpoint. Lý do chọn mốc 5000 và điều kiện lên 10000: `docs/adr/0005-budget-5000-games-for-az-v0b.md`.

## Môi trường

| Mục | Yêu cầu |
|---|---|
| Python | 3.10 |
| Thư viện | torch (cu124), numpy, pyyaml, pytest, pygame |
| GPU | NVIDIA RTX 4060 8 GB (sm_89) cho lần train đầu; CPU vẫn chạy được để test |
| Mạng | pip OK |

Colab free chỉ cho 2 vCPU, nên một instance Colab chậm hơn máy local. Colab dùng cho vòng chạy dài và cho việc sinh ván song song, không phải để thay thế local.

## Lệnh dự kiến

Các lệnh dưới đây là hợp đồng với `tasks/todo.md`, sẽ chạy được sau khi có mã nguồn:

```bash
pip install -r requirements.txt

python scripts/play.py --mode human-vs-ai --agent mcts-pure --ruleset normal
python scripts/play.py --mode ai-vs-ai --agent random --ruleset renju
python scripts/play.py --mode ai-vs-ai --black-agent heuristic --white-agent alphabeta
python scripts/play.py --mode human-vs-ai --human-color white --agent heuristic
python scripts/play.py --mode ai-vs-ai --black-agent az_v1_tactical --white-agent mcts-pure --black-checkpoint checkpoints/last.pt
python -m ui.pygame_app --mode human-vs-ai --agent mcts-pure
python -m ui.pygame_app --mode ai-vs-ai --checkpoint checkpoints/last.pt
python scripts/train.py --config configs/normal.yaml
python scripts/arena.py --a random --b checkpoints/last.pt --games 100
python scripts/arena.py --a heuristic --b az_v1_tactical --games 100 --budget sims:200
python scripts/tournament.py --agents random,heuristic,alphabeta,mcts-pure,az_v0,az_v1_threat --games 100 --budget sims:200
python scripts/tournament.py --agents az_v0,az_v1_threat,alphabeta --games 100 --budget time:1
python scripts/tournament.py --agents random,heuristic,alphabeta,az_renju --rules renju --games 100 --budget sims:200
pytest -q
```

`--agent` đặt cùng một agent cho cả hai phía để giữ tương thích; dùng
`--black-agent` và `--white-agent` để chọn riêng. `--human-color` đổi màu
quân của người chơi. Agent hỗ trợ: `random`, `mcts-pure`, `heuristic`,
`alphabeta`, `az`/`az_v0`, `az_v1_tactical`, và `az_v1_threat`. Các biến thể
AlphaZero cần checkpoint; có thể truyền checkpoint riêng cho mỗi bên.
`--depth`/`--no-pruning` cấu hình Alpha-Beta, `--rollouts` cấu hình MCTS Pure.
Trong chế độ AI-vs-AI, nhấn Space để tạm dừng/tiếp tục và New game để chơi lại.
Checkpoint ghi vào `checkpoints/`, log vào `logs/`. Cả hai nằm ngoài git.

Giải đấu ghi `games.jsonl` (nước đi, độ trễ, lượng tìm kiếm), `manifest.json`
(seed, cấu hình, SHA-256 checkpoint), `matrix.csv` và `summary.json` vào
`logs/tournament/<run_id>/`. Mỗi cặp đổi màu, vì vậy `--games` phải chẵn.
`--checkpoint NAME=PATH` chọn trọng số từng agent; mặc định các biến thể Normal
dùng `checkpoints/last.pt`, Renju dùng `checkpoints/renju_last.pt`.
Thư mục `--out-dir` phải rỗng để không trộn số liệu giữa các lần chạy.
`sims:N` đặt số mô phỏng cho AZ/MCTS Pure, Alpha-Beta dùng `--depth` (mặc định 2);
`time:S` đặt ngân sách giây mỗi nước cho tác tử tìm kiếm. Random/heuristic không
bị làm chậm để dùng hết ngân sách. Báo cáo độ trễ và lượng tìm kiếm thực tế,
không coi số node Alpha-Beta bằng số simulation MCTS.
`--pair NAME,NAME` chạy một cặp với seed theo vị trí trong danh sách đầy đủ.
Sau khi đủ mọi cặp, dùng `--merge-root SHARD_ROOT --out-dir FROZEN_DIR` để
kiểm tra và gộp log mà không đấu lại. Merge từ chối thiếu/trùng cặp, lệch
cấu hình, checkpoint, seed, số ván hoặc màu; không sửa log nguồn.
Tiếp tục một cặp bị gián đoạn bằng cùng lệnh, thêm `--resume --pair NAME,NAME`
và giữ nguyên `--out-dir`. Resume kiểm tra cấu hình, checkpoint và seed/màu
từng ván đã ghi, chỉ chạy các chỉ số còn thiếu; giữ nguyên manifest và tiền tố
log cũ. Log hỏng hoặc cấu hình khác bị từ chối, không tự xóa bằng chứng.
