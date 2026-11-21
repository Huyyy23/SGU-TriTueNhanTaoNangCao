# Điều tra nguyên nhân AlphaZero thua Heuristic tại mốc 25k

## Kết luận điều hành

**Chưa xác định được một nguyên nhân gốc duy nhất.** Có bằng chứng trực tiếp về hạn chế chiến thuật trên trajectory, nhưng chưa có control cô lập để quy thất bại cho lỗi dấu MCTS, lỗi target, số updates, backend hay collapse. Các phát hiện về độ tin cậy dữ liệu và protocol là rủi ro hoặc sai lệch contract, không mặc nhiên là nguyên nhân của 46/400.

Trong `data/tournament.sqlite`, experiment `az_v0-15000-continuation-20261119-promotion-500-heuristic-0` ghi AZ thắng/hòa/thua **46/0/354**, Normal, Python MCTS, 200 simulations, 4 opening plies; cầm Đen thắng 35/200, cầm Trắng 11/200. Có 200 cặp dùng cùng opening và đổi màu/seed. Đây là mô tả mẫu, không coi 400 ván là độc lập. Candidate có SHA `2384043ae275dfe5eef24cb9ec1f87d771ae0238ccfff01ffa14591baffeb731`; `checkpoints/latest.pt` hiện tại SHA `f7f521c1cc37a5054ed854201b087e484e9fcef3d4d6b1877ab81dff2e51d7e4`. Hai byte hash khác nhau; hiện chưa chứng minh weights khác hay giống. Chi tiết và SQL tái hiện: [heuristic-loss-analysis.md](heuristic-loss-analysis.md).

Experiment 15k có 6/100, không phải so sánh matched: backend C++ thay vì Python, seed khác, code historical dirty và mẫu nhỏ hơn. Không chứng minh được training từ 15k đến 25k gây mức cải thiện 6% → 11,5%. Không có chứng cứ trong cuộc điều tra này về tournament riêng `az_v1` tại đúng mốc 25k.

## Phát hiện và mức ưu tiên

| Mức / trạng thái / tin cậy | Bằng chứng, tác động và tái hiện | Đề xuất |
|---|---|---|
| **P1 — xác nhận quan sát, cao** | Trong 20 ván thua đầu theo ID, replay một-ply thấy 3 ván bỏ chặn khi đối phương có đúng một nước thắng và AZ không có thắng ngay; 17 ván gặp ít nhất một vị trí đối thủ có nhiều nước thắng ngay, 0 ván bỏ lỡ thắng ngay. Mẫu không ngẫu nhiên và phân loại không xác định sai lầm đầu tiên hoặc khả năng cứu ván. Lệnh, IDs và phương pháp ở [phân tích ván](heuristic-loss-analysis.md). | Đóng băng các vị trí và kiểm tra oracle 1–2 ply, candidate priors, visits và value; tìm sai lầm trước khi hình thành đa đe dọa. Không coi 3/20 là tỷ lệ toàn tập. |
| **P1 — xác nhận thiếu provenance, cao** | SHA candidate và `latest.pt` khác; run có dirty code, không đủ để chứng minh identity weights. `scripts/train.py` có thể lưu lại checkpoint sau evaluation, nhưng chưa chứng minh đó là lý do của chênh SHA. Xem [training](training-data-audit.md) và [ván thua](heuristic-loss-analysis.md) để tái hiện. | Tìm candidate đúng SHA, lưu manifest bất biến và so tensor nếu tìm được; không tự đánh đồng hai file. |
| **P1 — nguy cơ integrity, cao cho cơ chế / chưa xác nhận sự cố** | `Trainer.save` ghi checkpoint và replay riêng, không cùng transaction; interruption có thể để hai thế hệ không khớp. Chưa chứng minh artifacts hiện tại lệch thế hệ hoặc nhãn hỏng. Chi tiết dòng code và lệnh: [training](training-data-audit.md). | Sau phê duyệt, thêm liên kết generation/hash và test interruption; bảo toàn artifacts gốc. |
| **P1 — mất đa dạng dữ liệu là giả thuyết mạnh, cao cho số đo / thấp cho nhân quả** | Run continuation có 15.000 game records, 14.931 Đen thắng, 69 Trắng thắng, trung bình 10,13 nước/ván, 14.960 ván bắt đầu ở tâm. Replay Normal 50.000 mẫu có 28.583 feature bytes khác nhau; 4.906 mẫu là bàn trống và median policy support là 3. Đây là tập trung dữ liệu thực đo, không chứng minh nguyên nhân hay collapse toàn cục. [Training/data](training-data-audit.md) có truy vấn, giới hạn lineage và đề xuất. | Đo trajectory/π entropy theo giai đoạn và seed, so holdout tách game; kiểm tra trước khi thay đổi self-play. |
| **P1 — đo lường training không độc lập, cao** | Fixed batch lấy từ replay theo slot, cùng ván có thể lọt vào training, ring overwrite làm slot không định danh mẫu cũ. Từ iteration 201 đến 500 train loss giảm trong khi fixed loss tăng, nhưng điều này không chứng minh overfit hay sức chơi. [Training](training-data-audit.md) có số liệu và lệnh. | Holdout tách theo game/seed; báo calibration/value và playing strength, không chọn model dựa trên loss này. |
| **P1/P2 — sai contract / đầu vào bất thường, cao cho code, chưa liên hệ nhân quả** | Các entry point có thể cho budget khác 200; MCTS consumer chưa xác thực đầy đủ policy. Tournament đang xét ghi `sims:200`, chưa có chứng cứ model tạo policy lỗi. [MCTS](mcts-correctness-audit.md) có code paths, tái hiện và phân loại. | Chặn budget ngoài contract và xác thực policy ở boundary sau phê duyệt, kèm regression tests; không suy rằng đây gây 46/400. |

**P0:** chưa có bug sai sign hay target huấn luyện nào được xác nhận là đã xảy ra trong run 25k. Không có nghĩa toàn bộ implementation đã được chứng minh đúng.

## Kiểm chứng thực nghiệm đã chạy

- Bốn unit test MCTS về backup sign, forbidden mask, root noise và win-in-1: **4 passed in 1.21s**; lệnh chính xác trong [MCTS audit](mcts-correctness-audit.md). Các fixture không chứng minh đúng trên toàn bộ cây và lịch sử.
- Python/C++: sáu cặp tìm kiếm trên bàn Normal trống, evaluator cố định, seeds 11–13, noise bật/tắt, mỗi search 200 simulations; visits, root statistics và root value khớp trong các fixture này. Không chứng minh Renju, trajectory đầy đủ, phân bố targets hoặc strength tương đương. Chi tiết và lệnh: [MCTS audit](mcts-correctness-audit.md).
- Checkpoint hiện tại `f7f521…`: CPU một thread, Normal, Python, seed 0, không noise, 200 simulations. Thế thắng ngay chọn (5,5), 199/199 child visits, prior 0,841874, value 0,998483; thế cần chặn duy nhất chọn (5,5), 68/199 visits, prior 0,242531, value −0,939954. Hai thế tổng hợp có danh sách nước trong [MCTS audit](mcts-correctness-audit.md); không thể suy từ đây ra tính đúng của mọi thế hay checkpoint lịch sử.
- Checkpoint hiện tại khai báo iteration 500, `global_step=484000`; optimizer state và BatchNorm counters cũng ghi 484000. 500 × 50 = 25000 là số ván danh nghĩa từ config, không phải phép đếm độc lập. Không thể kết luận 484000 updates đủ hay thiếu nếu chưa có learning curve và dữ liệu holdout độc lập.

## Giới hạn và bước tiếp theo

Working tree có thay đổi sẵn và historical code được ghi `dirty=true`: đọc source hiện tại không chứng minh implementation nào đã tạo dữ liệu cũ. Không chạy self-play quy mô lớn, tournament mới, training hay thay đổi thuật toán; không xóa checkpoint, replay hoặc log. Không thể suy luận value/policy collapse, overfitting hoặc tác động nhân quả chỉ từ loss và vài thế. Kiểm tra tập held-out theo game, provenance, replay trajectory và canary trước khi đề nghị can thiệp; thứ tự chi tiết tại [recovery-plan.md](recovery-plan.md).
