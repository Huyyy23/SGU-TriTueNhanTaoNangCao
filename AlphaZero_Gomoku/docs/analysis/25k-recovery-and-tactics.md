# Recovery và tactical thí điểm sau 25k

## Kết luận

- **Đã sửa lỗi crash/resume:** game commit trong SQLite trước generation không còn buộc phải tái tạo và va chạm với record cũ. Recovery dùng outbox bất biến chứa replay targets và payload gốc; collision không khớp vẫn bị từ chối.
- **Can thiệp chỉ là thí nghiệm:** sau khi hoàn thành đủ 200 simulations, nếu có nước thắng ngay thì chọn nước đó bằng bộ quét độc lập. Trên sáu canary broken-four cũ tại parent, baseline đúng 0/6 và can thiệp đúng 6/6. Trên 12 thế mới, cả hai đúng 11/12; đối kháng mẫu nhỏ đều 0/4, và can thiệp không kích hoạt. **Chưa có bằng chứng tăng sức chơi hoặc khả năng phòng thủ. Không tích hợp can thiệp vào sản phẩm.**
- **Chưa cho phép continuation dài.** Cần kiểm chứng recovery trên môi trường chạy dài, mở rộng tập tactical và đối kháng theo seed/opening với uncertainty đủ sức phát hiện khác biệt, và xác nhận replay/lineage qua các điểm khôi phục. Không dùng replay legacy như generation đã xác minh.

## 1. Crash, resume và tính toàn vẹn

### Cơ chế lỗi và cách sửa

Trước sửa, `scripts/train.py` ghi game vào SQLite rồi mới train và commit generation. Nếu process dừng ở giữa, resume tạo lại game cùng ID nhưng timestamp/duration khác; `az.logging_v2.record_selfplay_game` phát hiện payload khác và dừng do collision. Regression `test_resume_after_games_commit_preserves_original_targets` đã thất bại trước sửa ở đúng điểm này với exception injection qua CLI.

Thay đổi trong `az/training_recovery.py`, `scripts/train.py`, `az/trainer.py` và `SPEC.md`:

1. Với STANDARD logging, ghi outbox mới theo run ID, iteration và SHA checkpoint nguồn **trước** khi commit game SQLite. Outbox chứa replay snapshot, sample targets kể cả sample đã bị ring buffer ghi đè, record/move/root telemetry gốc và manifest hash. Không thay thế record hiện hữu bằng bản tạo lại.
2. Resume kiểm tra identity/hash, nạp lại outbox và ghi SQLite idempotent bằng đúng payload gốc; từ chối game ID thuộc run khác hoặc dữ liệu không khớp. Nếu có game cũ nhưng không có outbox tương ứng, dừng an toàn thay vì đoán hay tái tạo targets.
3. Generation gắn hash manifest và metrics đã lưu trước commit pointer. Nếu dừng sau pointer nhưng trước metrics SQLite, resume kiểm tra và hoàn tất commit metrics idempotent.
4. Rollback dùng run ID mới và không dùng outbox của nhánh khác; manifest giữ model parent thực và lịch sử commit. Generation, SQLite record và artifact lịch sử không bị xóa hoặc ghi đè.

Trước khi outbox được commit, các game chưa được bảo đảm khôi phục; các tệp tạm không được tự động xóa. Hash phát hiện thay đổi ngoài ý muốn khi reference đáng tin cậy, không chống tác nhân sửa đồng thời manifest và dữ liệu. Không khẳng định hỗ trợ concurrent writers.

### Bằng chứng runtime

Ngoài test exception injection, đã chạy self-play thực từ parent 25k, ε=0,25, 200 simulations, seed 712, bốn ván. Hook gọi `os._exit(73)` ngay trước generation commit: process thoát mã 73 sau khi SQLite đã commit đủ **4 game**. Outbox có **36 positions**, sample NPZ SHA-256 `39e5b860bf01f355d62356be0ee98c523732fbbbc2fb88d13133c8f3c3f1e633`. Hai lần resume cùng run ID đều thoát thành công; payload SQLite trước/sau không đổi, replay features/policy/value/game IDs khớp từng phần tử với samples trong outbox, và có đúng một training metrics record. Artifact riêng: `az_crashproof_mkxtuk9c/`.

`tests/test_train_resume_logging.py::test_process_exit_after_sqlite_commit_resumes_without_regenerating` kiểm tra thêm process exit thực qua subprocess và khẳng định resume không gọi lại self-play. Test tham số hóa khác kiểm tra crash sau SQLite và sau pointer, resume hai lần; các test kiểm tra rollback, game ID thuộc owner khác, source collision, tamper và sample bị ring ghi đè. Bộ kiểm tra:

```text
pytest -q -p no:cacheprovider tests/test_train_resume_logging.py tests/test_trainer.py tests/test_selfplay.py tests/test_logging_v2.py tests/test_native_mcts.py tests/test_tactical_benchmark.py
78 passed
```

Test subprocess là termination process, **không phải mô phỏng mất điện**. Đồng bộ directory entry trên Windows và durability khi mất điện chưa được chứng minh. Kết quả Python/C++ parity chỉ áp dụng các trạng thái và test đã chạy.

## 2. Vì sao bỏ lỡ broken-four?

Parent `checkpoints/latest.pt` SHA-256 `f7f521c1cc37a5054ed854201b087e484e9fcef3d4d6b1877ab81dff2e51d7e4`. Ở `holdout_broken_four_black`, oracle độc lập quét bốn hướng cho đáp án duy nhất ô 60; đặt ô đó trực tiếp làm `NormalRules.is_win` và `is_terminal` đều đúng, lượt chuyển sang đối phương. Trạng thái trước nước đi chưa terminal.

Với C++ MCTS, CUDA, seed 420, noise tắt và đúng 200 simulations, root expansion chiếm một simulation; tổng child visits 199. Python/C++ chọn cùng nước và có cùng visits/prior trong phép so sánh này:

| Nước root | Prior | Visits | Q theo root | PUCT sau budget |
|---|---:|---:|---:|---:|
| 60 — thắng ngay | 0,02975 | 0 | 0 | 0,631 |
| 63 — chọn | 0,66344 | 147 | 0,99904 | 1,094 |
| 57 | 0,23795 | 52 | 0,99769 | 1,093 |

Sau pilot 8 updates, cùng canary cơ sở có prior ô 60 là 0,38834 và nhận 96 visits; cả Python và C++ chọn 60, dùng 200 simulations nhưng chỉ 104 NN calls vì các nhánh terminal không gọi network. Đây là thay đổi dự đoán thực tế theo checkpoint, không tách riêng được tác động prior với value.

Ô thắng chưa được thăm, nên terminal detection đúng nhưng không được gọi trên nhánh thắng trong search. Prior nhỏ cùng Q của các nhánh cạnh tranh gần +1 làm PUCT tiếp tục ưu tiên chúng. Đây là giải thích cho **thế và checkpoint đã đo**, không kết luận sign flip sai hoặc mọi leaf value sai. Báo cáo `25k-continuation-pilot.md` ghi ablation trước đó: uniform prior nhưng giữ value vẫn sai; giữ prior và đặt leaf value 0 chọn được ô thắng. Ablation là chẩn đoán, không chứng minh value calibration hay biện minh thay thuật toán.

## 3. Can thiệp và protocol matched

Can thiệp là wrapper thử nghiệm trong `tactical_experiment_20261120/experiment.py`, **không sửa luật, ResNet hay MCTS sản phẩm**: chạy MCTS đủ 200 simulations, sau đó quét các nước hợp lệ bằng contiguous-ray oracle độc lập; nếu có nước thắng ngay, chọn nước đó, nếu không giữ lựa chọn MCTS. Cả baseline và can thiệp dùng cùng checkpoint, seed, opening, backend và budget MCTS; can thiệp có thêm chi phí quét, nên không khẳng định FLOPs hoặc wall time bằng nhau tuyệt đối.

Một run bootstrap mới dùng parent 25k, replay rỗng có provenance, seed 111, ε=0,25, C++ MCTS, một worker, 4 ván, 8 updates, 200 simulations. Replay cuối có 40 positions, game IDs đối chiếu bốn record SQLite; tensor generation đầu khớp parent. Checkpoint sau pilot SHA-256 `56e97e5c94f079e6065cebe38a892c7a8b26629d1b2b6e69f63c52aedd60f973`. Cả hai arm evaluation dùng **cùng chính checkpoint** ở mỗi mốc; đây là đối chứng inference, không phải hai nhánh training khác nhau. Không dùng replay legacy. Config, lệnh, hash và raw output nằm trong `tactical_experiment_20261120/commands.txt`, `hashes.txt`, `parent.json`, `continued.json`.

| Tập / checkpoint | Baseline | Thắng-ngay sau MCTS | Diễn giải |
|---|---:|---:|---|
| Sáu canary broken-four cũ / parent | 0/6 | 6/6 | Can thiệp được thiết kế trực tiếp cho tình huống này; không phải holdout độc lập |
| 12 thế mới / parent | 11/12 | 11/12 | Không tăng accuracy |
| 12 thế mới / sau 8 updates | 11/12 | 11/12 | Không tăng accuracy |

Tập mới gồm ba family win và ba family block, cả hai màu, edge/vertical/diagonal/gap khác nhau. Các history được replay hợp lệ, không terminal trước root; đáp án oracle contiguous-ray đối chiếu với oracle cửa sổ 5 ô. Không thấy trùng D4 với fixture cũ hay replay pilot mới; không biết exposure trong dữ liệu 25k lịch sử. Các màu/case trong cùng family không độc lập. Case duy nhất sai ở cả hai arm sau pilot là block-edge Trắng: nước phòng thủ, không phải nước thắng ngay; can thiệp không có cơ chế sửa nó.

## 4. Đối kháng và chi phí

Hai opening bốn nước hợp lệ, không terminal và không có thắng tức thì, đổi màu theo cặp. Từng checkpoint được đánh giá bốn ván baseline và bốn ván can thiệp; dùng cùng opening/seed và Heuristic. Đây là hai cặp opening mỗi checkpoint, không phải tournament đủ lớn.

| Checkpoint | Baseline với Heuristic | Can thiệp với Heuristic | Số lượt can thiệp kích hoạt |
|---|---:|---:|---:|
| Parent 25k | 0/4 | 0/4 | 0 |
| Sau pilot 8 updates | 0/4 | 0/4 | 0 |

Với 0/4, khoảng Clopper–Pearson hai phía minh họa là 0–60,2% nếu coi bốn ván là Bernoulli độc lập; hai màu cùng opening không độc lập, nên khoảng đó **không phải CI paired hợp lệ**. Chỉ hai cặp không đủ ước lượng tin cậy chênh lệch hay Elo; quan sát cả hai arm cùng thua không chứng minh tương đương. Không suy ra cải thiện strength từ 6/6 canary cũ.

Training pilot mất 6,83 s wall. Hai lượt đánh giá toàn bộ tactical và paired games mất lần lượt 25,46 s và 22,84 s. Riêng bốn trận parent: mỗi arm 38 MCTS searches, phần quét thêm khoảng 0,0063 s tổng; sau pilot: mỗi arm 34 searches, phần quét thêm khoảng 0,0059 s. Thời gian là phép đo nhỏ trên NVIDIA GeForce RTX 4060 Laptop GPU, Python 3.10.6, PyTorch 2.14.1+cu126 và NumPy 2.2.6, không phải benchmark hiệu năng ổn định.

## 5. Điều kiện continuation dài

1. Chạy crash/restart tại các seam trên máy và filesystem triển khai thực tế, kể cả sau pointer/trước metrics, rollback sang ancestor, outbox hỏng/collision và nhiều lần resume; xác nhận game IDs, hashes, replay targets và số metrics. Chứng minh hoặc ghi rõ giới hạn durability khi mất điện; không chạy concurrent writers nếu chưa có khóa/kiểm thử thích hợp.
2. Phát triển tập tactical holdout đa family và opening/seed độc lập hơn, gồm **forced defense** và các trạng thái trước khi thua; xác minh oracle và rò rỉ đối với replay có provenance. Exposure của 25k lịch sử vẫn unknown.
3. Chọn can thiệp dựa trên nhiều seed, cặp opening và CI theo cụm, theo dõi cả phòng thủ, calibration và sức chơi so với parent/Heuristic. Không chọn chỉ theo training loss hoặc canary. Can thiệp thắng-ngay ở đây chưa có bằng chứng tổng quát hóa, không được tích hợp chỉ dựa trên kết quả 0/6→6/6 trên tập đã biết.
4. Dùng parent 25k đã xác thực SHA, run/SQLite mới và replay mới có provenance; không nhập replay legacy chưa xác minh. Giữ 200 simulations, không ép tỷ lệ màu và không mở rộng lên 5.000 ván trước khi các gate trên đạt.
