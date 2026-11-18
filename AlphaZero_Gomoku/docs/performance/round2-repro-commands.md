# Lệnh tái lập vòng hai

Chạy từ thư mục `AlphaZero_Gomoku` trên Windows PowerShell/CMD hoặc shell có Python venv tương ứng; GPU benchmark phải chạy tuần tự, không chạy matrix khác cạnh tranh CUDA.

```powershell
# Baseline — trước thay đổi native, build cũ đã cài trong venv
.\.venv\Scripts\python.exe scripts\bench_selfplay_round2.py --mode matrix --simulations 200 --repeats 3 --out docs\performance\raw-round2-baseline.json

# Build native module sau khi sửa tree receive
.\.venv\Scripts\python.exe scripts\build_gomoku_native.py --module mcts --cxx g++

# Paired post-change matrix: Python/C++, parallel_games 4/16, 3 runs/config
.\.venv\Scripts\python.exe scripts\bench_selfplay_round2.py --mode matrix --simulations 200 --repeats 3 --out docs\performance\raw-round2-after.json

# Worker scaling lịch sử: tổng 16 game cho mỗi cấu hình; 3 runs/backend/worker count.
.\.venv\Scripts\python.exe scripts\bench_selfplay_round2.py --mode workers --worker-total-games 16 --simulations 200 --repeats 3 --out docs\performance\raw-round2-workers.json

# Saturation production-like: 16 active games/worker; 1 full repeat/config
.\.venv\Scripts\python.exe scripts\bench_selfplay_round2.py --mode workers --worker-games-per-worker 16 --simulations 200 --repeats 1 --out docs\performance\raw-round2-workers-saturated.json

# Production iteration: games_per_iter=50; workers receive 50 total games
.\.venv\Scripts\python.exe scripts\bench_selfplay_round2.py --mode workers --worker-total-games 50 --simulations 200 --repeats 3 --out docs\performance\raw-round2-workers-training-iteration.json

# Instrumented component profile; run separately from throughput measurements
.\.venv\Scripts\python.exe scripts\bench_selfplay_round2.py --mode profile --simulations 200 --repeats 1 --out docs\performance\raw-round2-profile-after.json
```

`matrix` warm-up chạy 4 game thật với 8 simulations trước mỗi backend; mỗi cấu hình chạy đúng 4 hoặc 16 game. Seed cố định theo repetition/parallel count; backend được luân phiên để giảm bias thứ tự. `workers` warm-up từng nhóm rồi luân phiên backend; mặc định chạy 16 game/worker (`--worker-games-per-worker 16`), còn `--worker-total-games 16|50` giữ tổng số game cố định. `run_games_parallel()` chia toàn bộ chunk cho từng process; số game active mỗi process có thể lớn hơn `parallel_games` của config, và được ghi trong `worker_groups`. Warm-up/startup không tính vào steady-state throughput. `--checkpoint` mặc định `checkpoints/last.pt`, `--device` chọn CUDA khi khả dụng; đổi tham số để chạy checkpoint/device khác.

JSON chứa SHA-256 checkpoint, môi trường, seed, số moves/NN calls từng game, wall time, throughput, batch size, RSS process-tree, CUDA allocation và VRAM/GPU/CPU sampled. Với workers >1, inference chạy trong child; p50/p95 latency và max batch per-call không thu thập, được để trống trong harness hiện tại. File lịch sử `raw-round2-workers.json` được tạo trước khi sửa telemetry nên dùng 0 làm sentinel cho các trường chưa đo; không diễn giải thành batch bằng 0. Profile có instrumentation overhead, không so wall với matrix.
