# Gomoku 11x11 + AlphaZero

Agent Gomoku trên bàn 11x11, tự học bằng self-play theo AlphaZero: mạng ResNet nhỏ, tìm kiếm cây PUCT, huấn luyện từ những ván agent tự đấu. Hỗ trợ hai bộ luật: **Normal** (thắng khi có 5 quân liền hoặc hơn) và **Renju** (cấm nước 3-3, 4-4, quá 5 quân cho bên đen; nước tạo đúng 5 quân luôn hợp lệ).

## Trạng thái

Dự án đang ở giai đoạn lập kế hoạch, chưa có mã nguồn. Các tài liệu đã chốt:

| Tài liệu | Nội dung |
|---|---|
| [tasks/plan.md](tasks/plan.md) | Kế hoạch triển khai: 26 task, 11 phase, quyết định kiến trúc, bảng rủi ro |
| [tasks/todo.md](tasks/todo.md) | Chi tiết từng task: tiêu chí nghiệm thu, cách kiểm chứng, file dự kiến đụng |
| [tasks/proposal.md](tasks/proposal.md) | Đề tài trình bày: mục tiêu, phương pháp, thiết kế thí nghiệm E1–E5 |
| [tasks/proposal-requirements.md](tasks/proposal-requirements.md) | Yêu cầu đề tài theo từng khối |

Việc triển khai bắt đầu từ `tasks/todo.md`, task T1 (skeleton repo + dependency + pytest).

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
| Đánh giá | ngưỡng thay thế mô hình tốt nhất | tỉ lệ thắng ≥ 0.55 |
| Đánh giá | số ván đánh giá | 1000, xen kẽ lượt đầu |

Ngân sách ván nằm trong `configs/*.yaml`, tăng lên 1200 hay 1500 ván chỉ cần sửa YAML rồi chạy tiếp từ checkpoint.

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

python scripts/play.py --mode human-vs-random     # chơi với agent ở CLI
python scripts/train.py --config configs/normal.yaml   # huấn luyện
python scripts/arena.py --new checkpoints/latest.pt --best checkpoints/best.pt --games 100
python scripts/simulate.py --a checkpoints/best.pt --b random --games 1000
python -m ui.pygame_app                          # giao diện đồ họa
pytest -q                                        # toàn bộ test
```

Checkpoint ghi vào `checkpoints/`, log vào `logs/`. Cả hai nằm ngoài git.
