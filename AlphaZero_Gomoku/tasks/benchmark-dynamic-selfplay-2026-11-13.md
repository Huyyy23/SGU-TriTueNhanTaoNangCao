# Benchmark dynamic scheduling self-play — 2026-11-13

## Kết luận

Không giữ thay đổi code: các phương án dynamic chunk 3, 5, 10 đều chậm hơn baseline 4 worker cũ. Dùng `--workers 4` cho workload Normal tương tự trên máy này. Đây không phải phép so sánh ngẫu nhiên xen kẽ baseline và dynamic, nên con số tuyệt đối có thể chịu ảnh hưởng điều kiện máy; khoảng kết quả không chồng nhau nhưng chưa khẳng định nguyên nhân duy nhất.

## Phương pháp

Windows spawn, RTX 4060 Laptop 8 GB, torch 2.14.1+cu126, `configs/normal.yaml`, 50 ván, 200 simulations, luật Normal, `checkpoints/last.pt` iteration 20 SHA-256 `6e29a4e3b8c235449898450de0f44b520829eda097ee74f97ef599238e52ffbc`. Mỗi phương án 3 lượt seed 0, 1, 2. Cấu hình cũ 1 và 4 worker lấy từ `benchmark-selfplay-multiprocess-2026-11-13.md`: cùng checkpoint, config, số ván, thiết bị, công cụ đo và worker preload; đo ở lượt riêng, không xen kẽ. Baseline 2 worker và dynamic đo mới. Cùng seed không tạo cùng ván giữa số worker/chunk.

Startup gồm load checkpoint, cấp buffer và chạy worker; wall bắt đầu sau startup, gồm model load vào GPU, self-play, IPC và merge buffer. `mean_batch` là trung bình theo ván của thống kê tích lũy trong `GameRecord`, không phải trung bình theo lượt inference. GPU utilization và VRAM là mẫu toàn GPU khoảng mỗi giây, gồm ứng dụng nền; VRAM không phải peak tuyệt đối. CPU là trung bình process tree, 100% tương ứng một lõi, có thể bỏ sót các khoảng mẫu.

## Baseline và profile

| Phương án | Median nước/s | Min–max nước/s | Median wall (s) | Nước từng lượt | Mean batch từng lượt | GPU TB (%) | CPU TB (% một lõi) | VRAM đỉnh mẫu (MiB) |
|---|---:|---:|---:|---|---|---|---|---|
| 1 worker, fixed | 7,96 | 7,52–8,13 | 151,17 | 1129, 1204, 1181 | 15,67; 15,62; 15,58 | 28,1–31,1 | 33,1–35,7 | 834–1183 |
| 2 worker, fixed | 20,44 | 18,81–21,12 | 56,26 | 1188, 1041, 1163 | 21,63; 22,41; 21,68 | 29,1–31,3 | 47,5–61,6 | 1121–1257 |
| 4 worker, fixed | 29,41 | 26,80–30,14 | 39,07 | 1174, 1078, 1149 | 11,03; 11,27; 10,94 | 41,4–47,9 | 210,6–228,9 | 1359–1392 |

Startup median hai worker 1,912 s; startup baseline một/bốn worker trong báo cáo cũ. Một iteration đại diện với checkpoint 20 và buffer mới: self-play 56,262 s, train 200 bước 3,466 s, fixed-batch evaluation 0,022 s, tạo fixed batch 0,004 s. Tỷ lệ trên tổng bốn phần đo: self-play 94,2%, train 5,8%, evaluation 0,04%; chưa tính checkpoint, game logging, khởi tạo trainer hay thời gian khác. Đây là lượt đo riêng, không thay đổi checkpoint huấn luyện và không đo arena evaluation.

`run_games_parallel` chia 50 ván thành 13/13/12/12; `ProcessPoolExecutor` được tái sử dụng bởi `scripts/train.py`; mỗi chunk truyền CPU state dict, tạo model mới, load weights và đưa model vào CUDA, sau đó trả samples về parent để thêm vào buffer. Benchmark cũ đo một lượt có chunk 26,70–43,78 s, chênh 17,08 s: imbalance quan sát được, chưa chứng minh là bottleneck chính. State dict 1.942.040 byte tensor; sample khoảng 968 byte dữ liệu mảng trước serialization. Không đo riêng IPC hay thời gian load model, không gán thời gian chênh cho chúng.

## A/B dynamic scheduling

Prototype chỉ tồn tại trong script đo tạm, không sửa production. Bốn worker lấy các chunk kế tiếp từ executor; CPU weights đặt vào shared memory một lần mỗi lần đo; mỗi worker cache model với token duy nhất cho lần đo, không tạo lại model mỗi chunk. `parallel_games` bằng số ván chunk, như đường fixed hiện tại. Mỗi chunk có seed riêng; kiểm tra đủ 50 ID duy nhất và sample count bằng tổng moves. Prototype không được đưa vào production; token và cập nhật weights giữa các iteration không được kiểm chứng như hợp đồng production.

| Chunk | Median nước/s | Min–max nước/s | Median wall (s) | Nước từng lượt | Mean batch từng lượt | GPU TB (%) | CPU TB (% một lõi) | VRAM đỉnh mẫu (MiB) |
|---:|---:|---:|---:|---|---|---|---|---|
| 3 | 14,94 | 13,89–15,10 | 77,13 | 1231, 1165, 1139 | 2,71; 2,72; 2,72 | 58,8–64,2 | 151,3–227,6 | 1335–1509 |
| 5 | 18,82 | 18,14–19,58 | 57,69 | 1086, 1070, 1116 | 4,56; 4,56; 4,51 | 50,5–53,0 | 90,9–198,3 | 1335–1435 |
| 10 | 19,38 | 19,17–19,86 | 57,76 | 1147, 1179, 1109 | 8,88; 8,88; 8,98 | 37,2–40,1 | 76,7–159,5 | 1391–1537 |

Median chunk 10 đạt 65,9% throughput baseline fixed 4, tức chậm khoảng 34,1%; chunk 5 đạt 64,0%, chunk 3 đạt 50,8%. Mean batch giảm mạnh, phù hợp giả thuyết giảm hiệu quả gom batch, nhưng phép đo không cô lập batching khỏi số ván từng worker, IPC, scheduling và điều kiện máy. Chunk 10 có 5 chunk nên worker cuối có thể phải chơi chunk thứ hai khi các worker khác rảnh; không kết luận nó là cấu trúc dynamic tối ưu. Không có phép đo IPC riêng hoặc GPU contention, không đề xuất centralized inference.

## Correctness và giới hạn

`.venv/Scripts/python.exe -m pytest -q tests/test_selfplay.py tests/test_mcts.py tests/test_replay_buffer.py tests/test_trainer.py tests/test_renju_integration.py`: **38 passed**. Benchmark đủ game, unique ID và số sample khớp số nước; đếm không chứng minh không trùng sample hoặc `z` chính xác. Suite hiện có kiểm tra `z`, không trộn rule set, MCTS, buffer, resume/checkpoint và multiprocessing; không khẳng định đã kiểm thử cache weights qua iteration vì không giữ cache production. Các process benchmark kết thúc qua context manager, không quan sát deadlock/OOM; không có phép đo độc lập chứng minh không còn worker leak. Các phương án dynamic không đạt performance gate nên không đưa vào training và không làm thay đổi API, logic MCTS hay giá trị target.

Dữ liệu thô: `logs/_bench_fixed2.jsonl`, `logs/_bench_dynamic.jsonl`; dữ liệu baseline cũ: `logs/_selfplay_bench_confirm.jsonl`. Không dùng benchmark làm dữ liệu train.
