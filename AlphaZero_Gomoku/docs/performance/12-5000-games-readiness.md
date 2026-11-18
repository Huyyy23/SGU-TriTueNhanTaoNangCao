# 12 — Readiness chạy 5.000 ván

## Khuyến nghị cấu hình hiện tại

Khuyến nghị: **Normal, 200 simulations/move, C++ MCTS, `workers=2`, `games_per_iter=50`**. Python backend mặc định không đổi; chọn C++ tường minh bằng `--backend cpp`. Ba lượt production-sized (mỗi lượt đúng 50 game) cho median 129.59 games/min với 2 workers, so với 86.51 với một worker và 124.66 với bốn. RSS peak median 3.36 GB/2 workers so với 5.42 GB/4; VRAM 1,373 so với 1,632 MiB.

## Dự kiến 5.000 game

Ở mức 129.59 games/min, phép chia tuyến tính 5.000/129.59 cho **38.6 phút self-play**. Tốc độ đo trên đúng 50-game iteration; pool startup + warm-up ~5.31 s ở hai workers và được tái sử dụng giữa iteration. Không cộng optimizer, evaluation, checkpoint/replay, logging; throughput thay đổi khi độ dài ván, checkpoint và game distribution đổi.

## Validation đã thực hiện

- Full suite: `261 passed, 2 skipped`.
- Focused native/backend tests sau C++ thay đổi: `13 passed`.
- Differential search với cùng RNG seed, root noise bật: Python/C++ có cùng request planes/mask, visit counts, root value và simulations (`tests/test_native_mcts.py`, full suite).
- Self-play paired 4 game, seed `20261122`, 200 sims/move: 83 samples mỗi backend; exact equality cho mọi feature plane, policy target và value target; game lengths `[19,21,20,23]`, NN calls `[3710,3823,3840,4386]`, winner/end reason trùng.
- Worker scaling spawn trên Windows: đủ game hoàn thành cho 1/2/4 workers; worker 2 và 4 chạy child process, không có crash/worker fallback.
- GPU pipeline thật (`docs/performance/round2-gpu-smoke.yaml`): checkpoint ban đầu → self-play C++ CUDA + Adam train + eval + checkpoint/buffer save 2 iterations × 10 optimizer steps → `--resume` restore iteration 2/global_step 20 → iteration 3; global step 30. Không tái hiện `CUBLAS_STATUS_EXECUTION_FAILED`. Peak sampled process RSS 1.40 GB, VRAM 1,238 MiB. Đã xác minh checkpoint và replay buffer tồn tại trước resume.
- Training smoke giữ nguyên ResNet 6×64, 200 simulations, batch size và loss/target logic; chỉ giảm game count và optimizer steps cho thời lượng kiểm chứng.
- Lệnh smoke: `python scripts/train.py --config docs/performance/round2-gpu-smoke.yaml --iterations 2 --games 4 --simulations 200 --seed 922 --backend cpp --out-dir <temp>/checkpoints --log-dir <temp>/logs`; sau đó gọi lại cùng lệnh với `--iterations 3 --resume`. Hai lệnh dùng cùng thư mục tạm; checkpoint/buffer bị xóa sau kiểm tra.
- Resume smoke đổi tổng iteration budget từ 2 sang 3, nên cosine LR horizon trong test cũng đổi; phép thử chứng minh load optimizer/state và tiếp tục bước train, không chứng minh schedule không đổi khi resume ở config production cố định.

## Chưa được chứng minh

- Một lượt GPU train/resume 30 optimizer steps không phải soak dài, không chứng minh lỗi CUBLAS đã được root-cause/fix cho các phiên train nhiều giờ.
- Không chạy 5.000 game soak, không đo thermal throttling laptop, không chạy controlled arena hoặc đánh giá playing strength. Exact training targets ở một self-play seed và differential search tests chứng minh semantics của patch giữ nguyên, không phải đánh giá chất lượng học sau nhiều iteration.
- RSS worker benchmark là tổng RSS process-tree (shared pages có thể bị tính trùng); VRAM/GPU utilization được lấy mẫu. Cần giữ headroom cho desktop/CUDA cache và tiến trình khác.

## Quyết định chạy

Tối ưu hiện tại cải thiện C++ p16 self-play paired 2.44× so với trước và nhanh hơn Python 2.67× ở workload 16 game. Với batch `games_per_iter=50`, benchmark lặp chọn 2 workers; 4 workers gần tốc độ nhưng tốn thêm khoảng 2 GB RSS. 5.000 game **có thể chạy trong controlled run**, chưa được chứng nhận production-safe vì chưa có memory soak/thermal profiling dài hạn hay quality arena. Theo dõi RSS/VRAM/nhiệt độ, dừng nếu memory tăng theo iteration, lưu checkpoint/buffer/log ra vị trí đủ dung lượng.
