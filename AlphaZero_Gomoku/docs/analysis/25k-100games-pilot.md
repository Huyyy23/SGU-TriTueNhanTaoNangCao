# Pilot continuation 100 ván từ checkpoint 25k

## Phạm vi và thiết lập

Run mới `az-25k-pilot100-seed314159`, artifact tại `az_100pilot_botfozch/`, parent `checkpoints/latest.pt` SHA-256 `f7f521c1cc37a5054ed854201b087e484e9fcef3d4d6b1877ab81dff2e51d7e4`. Generation đầu có tensor **trùng parent**, replay rỗng, optimizer Adam reset và global step 0. Không dùng legacy replay; không ghi đè checkpoint/SQLite cũ.

Normal 11×11, ResNet 6 block/64 filter, C++ MCTS **200 simulations mỗi nước**, `c_puct=1.5`, Dirichlet α=0,15, ε=0,25, uniform replay sampling, no tactical/threat override. Adam LR gốc `1e-4`, cosine, batch 32; một worker, `parallel_games=2`, seed 314159. Chọn **50 iteration × 2 ván × 4 updates** thay vì 25 × 4 × 8 để có generation đúng mốc 50 ván; tổng ngân sách vẫn 100 ván, 200 updates và cosine horizon 200. Đây không phải so sánh matched với run trước có batch game/iteration khác.

Lệnh và gate được lưu trong `az_100pilot_botfozch/run_pilot.py`, config `pilot.yaml`, log `train.log`; evaluation cuối dùng `evaluate_paired.py` và `paired.json`. Run trên Windows 10 22631, NVIDIA RTX 4060 Laptop GPU. Không có promotion: `best.json` và tournament SQLite của training không được tạo; ngưỡng promotion hiện hành 5.000 game không được chạm tới.

## Preflight và integrity

- Parent SHA/kiến trúc/ruleset khớp; CUDA khả dụng. Filesystem thực tế còn **375.328.980.992 bytes** trước chạy; tạo file tạm, ghi, fsync và đọc lại thành công.
- Ba test trọng điểm process exit sau SQLite, retry rollback và giữ nhánh cũ: **3 passed**. Đây không phải kiểm thử mất điện hoặc concurrent writers.
- Các gate 20, 40, 60, 80, 100 kiểm tra SHA checkpoint/replay và toàn bộ ancestor/outbox đã commit; source checkpoint file SHA đối chiếu với tensor digest thực tế của worker, run/game ID và ruleset. Replay sample count bằng số nước, IDs nối SQLite, π hữu hạn/không âm/tổng 1/không mass trên ô đã chiếm, z khớp winner theo màu người đi. Các search log đều ghi 200 simulations. Các gate đều **pass**.
- Sau run: SQLite `PRAGMA quick_check=ok`, 50 training metrics, step 4–200. Nạp checkpoint/replay cuối bằng `Trainer.load` cho iteration 50, step 200, 986 sample và đúng run ID. 51 generation gồm generation ban đầu và 50 iteration. Parent SHA sau run vẫn khớp.

| Gate ván | Updates | Replay positions | Đen thắng | Trung bình ply | SHA checkpoint |
|---|---:|---:|---:|---:|---|
| 20 | 40 | 196 | 20/20 | 9,80 | `88c4d7458b82bd9f9c883e27008a8d5c02fd4372295ced55cedb1c05922682c3` |
| 40 | 80 | 396 | 40/40 | 9,90 | `9bf5ad28281d4cb24a308345edc94e199b8c75ab7a059b005e69f1826ce402fc` |
| 60 | 120 | 600 | 60/60 | 10,00 | `b1ad4f3a418ef5f4e1601b4730beedc62789ac2341b69677264867ac513452ac` |
| 80 | 160 | 788 | 80/80 | 9,85 | `a926dfcba03df135bba1ceb3bcd12636b710e012060c6a111b1ae15310cea779` |
| 100 | 200 | 986 | 100/100 | 9,86 | `ece41a2074037b6025fe1807905572e8ef91955534aa5ca5834d6e482f28a40b` |

Tất cả 100 game mở đầu ô tâm 60; độ dài: **65 ván 9 ply, 27 ván 11 ply, 8 ván 13 ply**. Đây là dấu hiệu phân phối tự chơi rất tập trung trong run một seed, không chứng minh lỗi luật hay tự nó đo sức chơi. Không ép tỷ lệ Đen/Trắng về 50/50.

## Learning curve

| Ván | Updates | Replay | Training loss | Policy loss | Value loss | Fixed-batch loss |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 4 | 22 | 1,97030 | 1,96960 | 0,00070 | 2,03518 |
| 20 | 40 | 196 | 2,23469 | 2,23467 | 0,00002 | 2,04235 |
| 40 | 80 | 396 | 2,19389 | 2,19378 | 0,00011 | 2,06343 |
| 50 | 100 | 504 | 2,12274 | 2,12272 | 0,00002 | 2,07274 |
| 60 | 120 | 600 | 2,30929 | 2,30918 | 0,00011 | 2,08022 |
| 80 | 160 | 788 | 2,12866 | 2,12863 | 0,00004 | 2,08223 |
| 100 | 200 | 986 | 2,31270 | 2,31268 | 0,00001 | 2,09005 |

Loss từng iteration đo trên batch khác nhau; fixed batch chỉ bốn vị trí đóng băng từ replay, không phải holdout theo game và không chứng minh cải thiện sức chơi/calibration. Training driver mất **175,172 s** gồm gate và tactical 50/100; tổng self-play time theo log 146,75 s, tổng train time 1,92 s. Parent tactical baseline thêm 7,02 s ngoài driver. Không diễn giải chênh lệch này như benchmark throughput.

## Tactical ở 50 và 100 ván

Tập holdout cố định 24 state, 12 mỗi màu, seed 420, C++ MCTS 200 simulations, không noise. Không có state trùng D4 replay mới. Các biến thể trong cùng family tương quan; không biết historical 25k từng thấy các state tương tự.

| Mốc | Tổng | Đen / Trắng | Win | Block | Fork | Forced defense | Trung bình prior / visit mass đáp án |
|---|---:|---:|---:|---:|---:|---:|---:|
| Parent | 16/24 | 9/12, 7/12 | 0/6 | 4/6 | 6/6 | 6/6 | 0,4451 / 0,4087 |
| 50 ván | 15/24 | 9/12, 6/12 | 0/6 | 3/6 | 6/6 | 6/6 | 0,4424 / 0,4493 |
| 100 ván | 15/24 | 9/12, 6/12 | 0/6 | 3/6 | 6/6 | 6/6 | 0,4360 / 0,4462 |

Tactical giảm một case block so parent, không tăng accuracy; không đủ kết luận hồi quy sức chơi từ 24 state tương quan. Trước chạy, pilot đặt ngưỡng dừng nghiêm trọng là giảm ≥8/24 so parent, forced-defense ≤1/6 nếu parent ≥5/6, hoặc block 0/6 nếu parent ≥4/6. Cả hai mốc không chạm ngưỡng đó. Các điều kiện integrity được kiểm tra riêng và đều pass.

## Đối kháng cuối pilot

Đã chạy đúng **40 ván**: 20 với Heuristic và 20 với parent 25k; mỗi đối thủ có 10 opening seed cố định `40100–40109`, mỗi opening bốn nước hợp lệ, không terminal và không có thắng ngay, đổi màu theo cặp. Hai AZ dùng C++ MCTS, 200 simulations mỗi search, temperature 0, no root noise; Heuristic dùng C++ patterns. Không có tournament tại các mốc 20 ván. Raw openings, history, kết quả và telemetry ở `az_100pilot_botfozch/paired.json`, SHA-256 `76d1516e22d4874bb89a80a66cd1068d4a0e4c4b021563f9258c84b0bc1575f7`.

| Đối thủ | Pilot cầm Đen | Pilot cầm Trắng | Tổng W/D/L | Score | Bootstrap 95% theo 10 opening pairs |
|---|---:|---:|---:|---:|---:|
| Heuristic | 1/10 thắng | 0/10 thắng | 1/0/19 | 5% | [0%, 15%] |
| Parent 25k | 7/10 thắng | 5/10 thắng | 12/0/8 | 60% | [40%, 80%] |

Bootstrap percentile 10.000 lần, seed 420, resample mỗi opening cùng cả hai màu. Chỉ 10 cụm và một training seed: khoảng rất thô, không phải bằng chứng tổng quát hoặc Elo đáng tin. Đặc biệt khoảng với Heuristic dựa trên duy nhất một chiến thắng và có thể đánh giá thấp bất định ngoài mẫu. Kết quả với parent **không đủ chứng minh** pilot mạnh hơn; kết quả với Heuristic không so trực tiếp với 46/400 lịch sử vì checkpoint byte identity và protocol không được chứng minh tương đương.

Value được đo ở lượt đầu của pilot, chuyển `p=(v+1)/2` rồi so với outcome trận thực tế. Với Heuristic, Brier Đen/Trắng lần lượt **0,8968 / 0,00023**, mỗi màu 10 ván; với parent là **0,2975 / 0,4970**. Outcome phụ thuộc đối thủ và search, mẫu nhỏ; đây không phải minimax calibration. Đối kháng mất **154,71 s** wall: 194 lượt search pilot trước Heuristic; trước parent, 189 lượt pilot và 187 lượt parent. Mỗi search thực sự chạy 200 simulations.

## Kết luận và bước tiếp theo

**Không nên tiếp tục ngay 5.000 ván với recipe này.** Integrity đạt, không có regression nghiêm trọng theo ngưỡng tactical đã đặt trước, nhưng 100/100 self-play Đen thắng, 100/100 opening ở tâm, 92/100 ván dài 9–11 ply; tactical win vẫn 0/6 và block giảm 4/6 xuống 3/6. Pilot đạt 1/20 trước Heuristic; 12/20 trước parent có CI rộng. Đây là bằng chứng cần điều chỉnh và kiểm chứng phân phối self-play/phòng thủ, không phải bằng chứng chắc chắn rằng thêm game sẽ hoặc sẽ không giúp.

Checkpoint cuối có global step **200/200** của cosine horizon; LR lưu của bước cuối `6,1684e-9`, và LR bước kế tiếp với cùng horizon là **0**. Không tiếp tục bằng cách chỉ gọi resume hoặc âm thầm đổi horizon. Giữ nguyên generation, replay, outbox và SQLite; thiết kế thí nghiệm có đối chứng cho đa dạng opening/target phòng thủ, định nghĩa lại lịch LR và đánh giá bằng game/seed độc lập trước khi quyết định scale. Không tự động promote hoặc xóa artifact.
