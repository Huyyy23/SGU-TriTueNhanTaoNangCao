# AlphaZero 25k: chuẩn bị continuation

## Kết luận và trạng thái bằng chứng

**Chưa sẵn sàng chạy thêm 5.000 ván.** Pipeline bootstrap mới đã có kiểm thử và pilot, nhưng hai ván mỗi nhánh không đủ xác nhận dữ liệu đa dạng, khả năng học chiến thuật hoặc sức chơi. Chưa có bằng chứng chọn `dirichlet_eps=0.50` thay `0.25`, cũng chưa có bằng chứng chọn reset hay restore optimizer cho continuation dài. Checkpoint 25k chỉ nên dùng làm **parent weights** cho run mới; không coi replay legacy là dữ liệu đã xác minh, không ghi đè tài sản cũ.

Các kết quả trong báo cáo này là thử nghiệm nhỏ, không phải bằng chứng cải thiện Elo. Benchmark cố định có các biến thể tương quan; chưa biết liệu các thế tương tự từng xuất hiện trong 25k ván lịch sử.

## Pipeline và các lỗi đã sửa

- `scripts/train.py --bootstrap CHECKPOINT --bootstrap-optimizer reset|restore` tạo output directory và SQLite mới, generation ban đầu bất biến và replay rỗng. Bootstrap/continuation yêu cầu STANDARD logging. Không nhập replay legacy. `SPEC.md` mô tả contract mới.
- `az/trainer.py` lưu parent checkpoint SHA-256, cấu hình parent, lựa chọn optimizer, optimizer state, global step và replay hash. Reset tạo Adam mới và global step/iteration 0. Restore kiểm tra optimizer, tham số LR/weight decay, schedule và horizon; khôi phục global step, bắt đầu từ iteration tương ứng, từ chối checkpoint giữa iteration. Resume xác thực generation và hash checkpoint/replay; rollback chỉ chọn ancestor đã commit.
- **Lỗi đã xác nhận, đã sửa:** trước sửa, restore đặt iteration về 0 dù global step đã ở giữa cosine schedule. Với parent thật, step 484.000/500.000 sẽ lặp thêm 500 iteration thay vì 16; sau horizon LR bị kẹp về 0. Regression `test_bootstrap_restore_starts_at_schedule_iteration` thất bại trước sửa (`0 != 1`), sau sửa pass; thêm test từ chối step không nằm ở ranh giới iteration. Kiểm tra parent thật sau sửa: iteration 484, global step 484.000, còn 16 iteration theo horizon cũ, LR lưu trong optimizer `2.524806814795777e-06`. Đây không phải khuyến nghị dùng restore cho run 5.000 ván: horizon cũ chỉ còn 16.000 update và cách đổi horizon cần thí nghiệm riêng.
- **Lỗi đã xác nhận, đã sửa:** `az.logging_v2.initialize` dùng `Path.as_uri()` trên đường dẫn SQLite tương đối đã tồn tại; bootstrap thực tế dừng với `ValueError: relative path can't be expressed as a file URI`. `test_initialize_existing_database_with_relative_path` tái hiện thất bại trước sửa; sửa tối thiểu bằng `resolve().as_uri()` và test pass. Các artifact của lần khởi động thất bại được giữ lại.
- `az/selfplay.py`, `az/replay_buffer.py` và `az/logging_v2.py` gắn game ID, seed, ruleset, run/iteration lineage, SHA file checkpoint nguồn và digest tensor thực tế sau worker load; replay lưu game ID theo sample để nối với SQLite. Hai loại hash có ý nghĩa khác nhau. Không gán provenance giả cho sample legacy.
- Trên Windows, `test_generation_survives_process_termination_at_commit` thực sự kill subprocess tại hai phía điểm commit pointer, xác nhận generation trước/sau tương ứng, rollback ancestor, và hash artifact lịch sử giả lập không đổi. Test tamper và replay sai checkpoint cũng pass. **Chưa kiểm chứng mất điện hay độ bền directory entry của Windows filesystem**; kill process không chứng minh được điều đó.
- Kiểm thử sau sửa: `pytest -q -p no:cacheprovider tests/test_trainer.py tests/test_train_resume_logging.py tests/test_selfplay.py tests/test_replay_buffer.py tests/test_logging_v2.py tests/test_tactical_benchmark.py tests/test_native_mcts.py`: **85 passed** trên Windows. Có các test Python/C++ MCTS trong tập này; không suy diễn tương đương tuyệt đối cho mọi trạng thái. `graphify update .` đã chạy; phiên bản package cũ hơn phiên bản skill và phát cảnh báo.

## Thiết kế pilot

Cả hai nhánh bootstrap từ `checkpoints/latest.pt`, SHA-256 `f7f521c1cc37a5054ed854201b087e484e9fcef3d4d6b1877ab81dff2e51d7e4`, reset Adam, Normal, C++ MCTS, **200 simulations/nước**, seed 123, một worker, RTX 4060 Laptop GPU. Mỗi nhánh: 2 ván, 4 update, batch 32, `parallel_games=2`, `eval_fixed_batch_size=1`, `n_iter=1`. Các budget này giảm để pilot, không phải cấu hình sản xuất hay toàn bộ budget AZ gốc. Baseline giữ Dirichlet epsilon gốc `0.25`; phương án duy nhất đổi epsilon lên `0.50` để kiểm tra giả thuyết đa dạng exploration. Alpha và các hyperparameter không nêu giữ từ config mặc định. Cùng seed và budget **không** nghĩa là hai nhánh nhận cùng trajectory hay cùng số phép tính thực tế.

| Đo đạc | Baseline ε=0.25 | Exploration ε=0.50 |
|---|---:|---:|
| Ván / vị trí replay | 2 / 20 | 2 / 20 |
| Số nước hai ván | 11, 9 | 9, 11 |
| Kết quả | Đen thắng 2 | Đen thắng 2 |
| Update / global step sau pilot | 4 / 4 | 4 / 4 |
| Self-play wall time | 3,05 s | 3,14 s |
| Train wall time | 0,17 s | 0,17 s |
| NN calls trung bình/ván | 1.722 | 1.696 |
| Train loss trung bình | 2,48175 | 2,97434 |

Training loss đo trên replay khác nhau, không đo chất lượng ngoài mẫu; không dùng để xếp hạng hai nhánh. Fixed-batch loss của trainer lấy sample theo vị trí từ replay, không phải holdout theo game và không so sánh độc lập; kích thước pilot chỉ 1 mẫu. Hai ván Đen thắng và thời gian ngắn là quan sát, không đủ kết luận epsilon không thể cải thiện đa dạng.

## Kiểm tra provenance thực tế

Artifact pilot mới nằm dưới `az_readiness_native_wo5zl7ml/`: `baseline_run2/`, `baseline2.sqlite`, `exploration_run/`, `exploration.sqlite`. Đây là run thử riêng, không phải checkpoint/replay lịch sử. SQLite được đối chiếu ở chế độ read-only: mỗi replay có 20 sample, 2 game ID khác nhau, tập ID khớp chính xác hai record game. Seed thực tế lần lượt `5494761258405613189` và `5494761258405613190` ở cả hai nhánh; ruleset là Normal.

SHA file generation khởi đầu là `62808431cec2ce7b746a4b20df2354d3f54a5f77071367c7bfa5d889f1171067` cho baseline và `0f39d49c40d76f1222b4fa92d7a016b384c59dc334c71b8c28a4f16737651cde` cho exploration. Hai file khác nhau vì metadata/config khác nhau; digest tensor của cả hai và digest được worker ghi đều là `cb20ab991a05ae60a9d45b0d65a6a3a6759cc5c9659badd9da820be1de9f73b9`. Metadata của mỗi generation ban đầu đều trỏ tới SHA parent 25k ở trên. Đây là bằng chứng hai pilot xuất phát từ cùng weights và game record ghi đúng identity generation đã dùng, không phải bằng chứng replay legacy khớp parent.

Checkpoint sau 4 update: baseline SHA `ab84494b19af5621cfb2c27044c6909258618fafe780423318651eb38b5775a0`; exploration SHA `661fbcc07ef80df025834ac2c914eaaf0c09781a79c624b3c289ec6da832b4fe`. Chỉ dùng các file này cho pilot/evaluation, không gắn nhãn best production.

## Tactical benchmark và chất lượng policy/value

`scripts/tactical_benchmark.py` có 72 thế Normal dựng cố định: train/dev/holdout mỗi tập 24, 12 mỗi màu và cân bằng bốn nhóm win, block, fork, forced defense. Oracle cửa sổ 5 ô độc lập với MCTS kiểm tra chiến thuật nông; forced defense chỉ bao phủ horizon được định nghĩa trong script, không chứng minh kết quả toàn ván. `--check-integrity` báo không trùng tên, không trùng family giữa split, không có state tương đương D4. Variants trong cùng family chỉ thêm quân và tương quan mạnh; 24 thế không phải 24 game độc lập. Không thể chứng minh tách khỏi dữ liệu 25k lịch sử. Đối chiếu cả tám phép D4 với replay mới: **0/20 state trùng holdout** ở mỗi nhánh.

Đánh giá toàn bộ 24 holdout trên CUDA, C++ MCTS, 200 simulations, seed 420, không root noise, temperature 0:

| Metric | Baseline | Exploration |
|---|---:|---:|
| Đúng tổng | 15/24 | 17/24 |
| Đúng Đen | 9/12 | 9/12 |
| Đúng Trắng | 6/12 | 8/12 |
| Win | 0/6 | 0/6 |
| Block | 3/6 | 5/6 |
| Fork | 6/6 | 6/6 |
| Forced defense | 6/6 | 6/6 |
| Trung bình policy prior trên tập đáp án oracle | 0,35149 | 0,28701 |
| Trung bình tỷ phần root visits trên đáp án | 0,43865 | 0,43698 |
| Thời gian benchmark | 6,90 s | 7,12 s |

Hai case tăng điểm là cùng family block broken-four Trắng (một bản cơ sở và một biến thể), nên không coi +2/24 là bằng chứng tổng quát hóa. Cả hai vẫn bỏ lỡ toàn bộ 6 win-in-one kiểu broken-four. Ví dụ baseline `holdout_block_broken_four_white`: prior đáp án 0,00203, root visit fraction 0,00503, root NN value -0,99649 và chọn sai; exploration cùng case prior 0,00137, visits 0,43719 và chọn đúng. Prior thấp không đủ giải thích thay đổi visits: backup/value và đường tìm kiếm cũng tác động. Root value gần -1 không tự nó chứng minh sai calibration vì oracle nông không xác định kết quả tối ưu cả ván. Giữ đây là giả thuyết cần phân rã bằng canary và nhãn kết quả độc lập, không sửa thuật toán theo một case.

## Đối kháng nhỏ và value theo màu

Sau pilot, mỗi checkpoint đấu Heuristic trên hai opening ngẫu nhiên seed 202 và 203, đổi màu theo cặp; lần lượt là `[22,42,52,55]` và `[38,94,114,41]`. Hai nhánh dùng cùng opening/seed; không train trên các ván evaluation này. Mỗi nước AZ chạy đúng 200 simulations, C++ MCTS; suy luận CUDA, pattern backend C++. Test `tests/test_native_patterns.py`: **8 passed**. Đối thủ là Heuristic, không có tactical override cho AZ.

| Nhánh | Kết quả AZ theo bốn ván | Điểm / ván | Điểm theo từng opening pair |
|---|---|---:|---|
| Baseline | thua Đen, thua Trắng; thua Đen, thua Trắng | 0/4 | 0/2, 0/2 |
| Exploration | thua Đen, thua Trắng; thua Đen, thua Trắng | 0/4 | 0/2, 0/2 |

Tổng thời gian GPU của hai batch lần lượt 11,626 s và 9,752 s, gồm cả hai nhánh. Có thêm một lần thử CPU/Python patterns bị dừng tại giới hạn 300 s sau một ván baseline thua; không gộp ván này vào bảng vì backend và tài nguyên khác. Hai cặp opening quá ít để tính khoảng tin cậy theo pair đáng tin cậy, và hai ván cùng opening không độc lập. Khoảng Clopper–Pearson minh họa 0–60,2% cho 0/4 chỉ hợp lệ nếu giả định bốn ván Bernoulli độc lập; giả định đó không đúng đầy đủ ở đây. **Không kết luận chênh lệch sức chơi hoặc Elo.**

Tại lượt đầu tiên của AZ trong mỗi ván, chuyển value `v` thành dự báo thắng `(v+1)/2`; nhãn là kết quả thực tế của chính ván với Heuristic. Tất cả 8 ván AZ thua:

| Nhánh / màu AZ | Dự báo thắng trung bình | Brier trên 2 ván |
|---|---:|---:|
| Baseline / Đen | 0,992396 | 0,984906 |
| Baseline / Trắng | 0,001156 | 0,00000143 |
| Exploration / Đen | 0,961394 | 0,925689 |
| Exploration / Trắng | 0,001060 | 0,00000113 |

Đây là dấu hiệu overconfidence theo màu trên **hai opening**, không phải đường calibration đáng tin cậy; kết quả phụ thuộc Heuristic và search thực tế, không phải ground truth minimax. Chưa kiểm chứng toàn bộ D4-overlap giữa trajectory evaluation và lịch sử 25k. Thử nghiệm cho thấy có thể đo theo game/màu ngoài replay pilot, nhưng chưa đủ chọn epsilon hay reset/restore.

## Điều kiện trước khi mở rộng

1. Tạo và lưu tập evaluation theo **game/seed**, riêng hẳn replay train và dev, cân bằng màu và opening, kiểm tra trùng theo D4; xác lập provenance cho game mới, không khẳng định biết lịch sử 25k. Đo policy cross-entropy/accuracy và value Brier/calibration theo màu với target có nguồn gốc, không lấy training/fixed-batch loss làm holdout.
2. Lặp pilot với nhiều seed và đủ game để kiểm tra game length, openings, màu, replay age và cả hash worker/checkpoint; so sánh epsilon và lựa chọn optimizer/LR với cùng ngân sách. Giữ giới hạn 200 simulations. Kiểm tra canary win/block trên họ vị trí mới và dùng paired openings đối kháng với khoảng bất định theo game/pair, không dùng Elo từ vài ván để tuyên bố cải thiện.
3. Thực hành resume/rollback từ generation thật của run mới, xác thực lineage SQLite–replay sau khi tiếp tục, kiểm tra hành vi crash và bảo toàn artifact trên môi trường Windows mục tiêu. Power-loss durability vẫn là giới hạn riêng.
4. Chỉ khi không có suy giảm nghiêm trọng và có tín hiệu có thể lặp lại trên holdout cùng đối kháng mới chọn cấu hình continuation. Không dùng tactical training từng thất bại; không ép tỷ lệ thắng hai màu. Nếu bắt đầu pilot mới, dùng replay rỗng có provenance, thư mục/SQLite mới, parent SHA được xác nhận; không chuyển replay legacy vào generation đã xác minh.

Tái kiểm tra integrity: `.venv/Scripts/python.exe scripts/tactical_benchmark.py --check-integrity`. CLI tactical có `--split holdout --backend cpp --seed 420`; CLI này dùng CPU, còn số trong bảng đo trên CUDA, nên sai khác số học/timing có thể xuất hiện. Ví dụ bootstrap phải chỉ định rõ `--bootstrap checkpoints/latest.pt --bootstrap-optimizer reset` hoặc `restore`, cùng `--out-dir`, `--db`, `--tournament-db`, `--run-id` mới; các lựa chọn optimizer và horizon cho run lớn **chưa được phê duyệt bởi bằng chứng ở đây**.
