# Output mẫu từ smoke test (KHÔNG phải kết quả nghiên cứu)

Các thư mục này là output thật của `scripts/tournament.py` và `scripts/ablation.py`, chạy ngày 14/11/2026 trên máy Linux CPU (Python 3, torch CPU). Chúng chỉ để xem định dạng file và kiểm tra luồng chạy. Số ván rất ít (2–20 ván/cặp), **không dùng làm số liệu báo cáo**.

| Thư mục | Lệnh (rút gọn) | Mục đích |
|---|---|---|
| `tournament/20261113-203643_normal_sims200_s0/` | `--agents random,heuristic,alphabeta --games 20 --budget sims:200` (đúng lệnh nghiệm thu, run id tự sinh) | lệnh nghiệm thu T31, 60 ván, 1.004 giây với 1 tiến trình |
| `tournament/smoke_light_rha_g4_open2_s0/` | 3 agent nhẹ, 4 ván/cặp, `--random-opening 2` | smoke test agent thật, có khai cuộc ngẫu nhiên |
| `tournament/smoke_az_normal_g2_sims32_s0/` | `az_v0,az_v1_tactical,az_v1_threat,random`, 2 ván/cặp | load checkpoint, đủ ba mốc AlphaZero, metric override/threat-space |
| `tournament/smoke_az_renju_g2_sims16_s0/` | ba mốc AlphaZero, `--ruleset renju` | Renju: 0 nước trái luật |
| `tournament/smoke_time_g2_t0.5_s0/` | `az_v0,mcts_pure,heuristic`, `--budget time:0.5` | chế độ ngân sách thời gian |
| `ablation/smoke_ab_d12_p2_s0.*` | `alphabeta --depths 1,2 --positions 2` | ablation pruning, JSON + PNG |
| `ablation/smoke_mcts_50_200_800_g2_s0.*` | `mcts --sims 50,200,800 --games 2` | ablation MCTS, JSON + PNG |

**Checkpoint trong các smoke test AlphaZero là mạng 6x64 khởi tạo ngẫu nhiên, chưa train** (`untrained_init_6x64.pt`, tạo bằng `az.model.save_checkpoint` với `torch.manual_seed(0)`), vì lúc kiểm tra repo và thư mục Drive không có file `.pt` nào. Nó chỉ chứng minh việc load checkpoint, nối wrapper và ghi metric chạy được; thắng thua giữa các mốc AlphaZero ở đây không nói gì về sức mạnh.

`git.commit` trong `config.json` là commit trên máy chạy thử, sẽ khác hash sau khi các commit được áp vào repo.
