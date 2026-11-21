# Báo cáo screening: AlphaZero sau 10.000 ván tiếp tục vs Heuristic

## Kết luận

Trong 100 ván Normal, AlphaZero thắng **55**, hòa **1**, thua **44**; score **55,5%**. Ước lượng chênh lệch là **+38,4 Elo**, nhưng khoảng tin cậy 95% theo cặp khai cuộc là **−27,6 đến +106,4 Elo**. Khoảng này chứa 0: kết quả screening **chưa đủ bằng chứng kết luận AlphaZero mạnh hơn Heuristic**.

## Thiết lập và nguồn

- Ngày 20/11/2026; experiment `az-35k-vs-heuristic-screen-20261120`.
- Checkpoint cuối `checkpoints/az_25k_continuation_v1/generations/ba897320ab8d406c881de90496d7f3a0/checkpoint.pt`; SHA-256 `73c7d292a1588255cbcfe9a2d1fbc6dd471cb3e2e6cedb9850d0b47f05e899e9`.
- Checkpoint thuộc iteration 700: 35.000 ván trong chuỗi continuation, gồm 25.000 ván đã có và 10.000 ván mới; chuỗi này được bootstrap từ checkpoint trước đó. Không xem 35.000 là tổng lịch sử huấn luyện của model.
- Luật Normal 11×11; 50 cặp khai cuộc, mỗi cặp hai ván đổi màu và cùng bốn nước khai cuộc ngẫu nhiên. Seed `20261120`, seed scheme `paired-crc32`.
- `az_v0`: 200 PUCT simulations mỗi nước, không tactical/threat override; MCTS Python. Heuristic dùng patterns Python. Hai worker, một torch thread/worker; loader mặc định chọn CUDA nếu khả dụng, nhưng device thực tế không được lưu trong record tournament. Đây là screening, không phải protocol nghiên cứu 400 ván bắt đầu từ bàn trống.
- Config `configs/normal_continuation.yaml`, SHA-256 `da17ddd8ce9461f62b2b1dd07e6043c6c11d43db0f62b3d2c3cc6afb357d99f0`. File `players/heuristic.py` SHA-256 `9650d5dc07b9e06933956d56526f50aac887e7721305e4093f87804543bba1c1`; hash này không bao gồm mọi dependency của Heuristic.
- Môi trường Windows 10 22631, PyTorch `2.14.1+cu126`, NVIDIA RTX 4060 Laptop GPU. Git commit và engine version không được ghi trong experiment; working tree không sạch, vì vậy chưa thể tái lập đầy đủ code identity từ record này.

## Kết quả

| AlphaZero cầm | Thắng | Hòa | Thua | Score |
|---|---:|---:|---:|---:|
| Đen | 32 | 0 | 18 | 64% |
| Trắng | 23 | 1 | 26 | 47% |
| Tổng | 55 | 1 | 44 | 55,5% |

Score = `(thắng + 0,5 × hòa) / số ván`. Trong 50 cặp đổi màu, AlphaZero thắng cả hai ván ở 14 cặp, thua cả hai ở 9 cặp, chia 1–1 ở 26 cặp và đạt 1,5–0,5 ở một cặp. Chênh lệch theo màu đáng theo dõi nhưng 50 ván mỗi màu chưa đủ để tách riêng ảnh hưởng màu khỏi khai cuộc và độ mạnh agent.

Ước lượng dùng `az.elo.paired_davidson_ci`: pairwise Davidson với hiệu chỉnh 0,5 cho từng loại kết quả; bootstrap **10.000** lần, seed `20261120`, lấy mẫu lại theo **50 cặp**, không coi hai ván đổi màu là độc lập. Vì đây là screening một checkpoint và một lịch khai cuộc, khoảng tin cậy không bao quát biến thiên do train lại model, thay seed huấn luyện hoặc thay protocol.

## Vận hành và kiểm tra

- SQLite có đúng 100 game; `PRAGMA integrity_check = ok`, `foreign_key_check` không trả lỗi.
- 99 ván kết thúc bằng năm quân liên tiếp, 1 ván hòa do đầy bàn; không có game ghi nước trái luật hoặc lỗi.
- Trung bình 36,59 nước/ván, khoảng 10–121 nước; thời lượng trung bình 5,68 giây/ván. Hai worker chạy song song nên không cộng thời lượng từng ván để suy ra wall time; log ghi tổng **291,1 giây**.
- 1.632 lượt AZ có ghi `simulations=200`; latency trung bình **342,2 ms**, p95 **408,4 ms**. Các lượt khác có `simulations=NULL`, không suy diễn chúng là search AZ 0 simulations.
- SQLite `data/az_35k_heuristic_screen_20261120.sqlite`, SHA-256 tại thời điểm phân tích `fb0e96f1da6d401488e64675218207a66447d24e02925d67b9749772a42e925f`. Log `data/az_35k_heuristic_screen_20261120.log`, SHA-256 `e8d5e9a3e6c590f1cb350df264bd7bbb2ef971c630a9a15c6ee5802e51f68edd`. Các hash này áp dụng cho bytes tại thời điểm đo; SQLite có thể đổi khi được ghi thêm.

## Diễn giải và bước tiếp theo

Kết quả 55,5% là tín hiệu đáng tiếp tục kiểm tra, không chứng minh model đã đạt mục tiêu nghiên cứu. Không so trực tiếp với các báo cáo lịch sử khác checkpoint, backend, opening hoặc seed rồi gán chênh lệch cho 10.000 ván vừa train. Nếu cần kết luận sức chơi, dùng protocol đã khóa và 400 ván chính theo SPEC, báo W/D/L theo màu cùng khoảng tin cậy theo cặp; so checkpoint trước và sau với cùng điều kiện để kiểm tra tác động của phần training tiếp tục.
