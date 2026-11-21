# Kiểm chứng continuation AlphaZero 25k

## Kết luận

**Chưa có bằng chứng model mạnh hơn trong chơi thực tế, chưa đủ điều kiện mở rộng self-play.** Có tín hiệu học trên một họ tactical cố định ở một số seed của ε=0,25, nhưng không lặp lại ở tất cả seed và chưa đủ để suy rộng ra kỹ năng phòng thủ hoặc sức chơi. Giữ ε=0,25 làm baseline; ε=0,50 có thể tiếp tục là nhánh nghiên cứu, không nên đưa vào continuation chính từ kết quả này. Không dùng Elo từ pilot nhỏ để tuyên bố tiến bộ.

## Kiểm chứng 6 lỗi win-in-one

Tại parent `checkpoints/latest.pt` SHA-256 `f7f521c1cc37a5054ed854201b087e484e9fcef3d4d6b1877ab81dff2e51d7e4`, sáu thế `holdout_broken_four_*` đều có một đáp án thắng là ô 60. Một phép kiểm tra độc lập quét quân liên tiếp theo bốn hướng, replay history xen kẽ hợp lệ và xác nhận không terminal trước root; đối thủ không có thắng ngay. Feature planes tương ứng đúng quân mình/đối thủ, màu đến lượt và forbidden plane rỗng.

Python và C++ MCTS, CPU, seed 420, 200 simulations, noise tắt: cả hai đều chọn 63 ở cả sáu thế; cùng prior và visits, ô thắng nhận **0/199 child visits**. Root network value 0,9988–0,99996; prior ô thắng 0,00744–0,04938. Root expansion chiếm một simulation nên số child visits là 199; không phải thiếu một simulation. Đây là điểm yếu tìm kiếm/model đã tái hiện, **không phải bằng chứng sai sign flip hay bug encoding**.

Phân rã trên case Đen cơ sở: thay prior thành uniform nhưng giữ value vẫn chọn sai; giữ prior gốc và thay leaf value bằng 0 thì chọn ô 60 với 170 visits; thay cả hai chọn ô 60 với 149 visits. Đây là can thiệp chẩn đoán, không phải đề xuất thay thuật toán hay chứng minh value sai trên toàn bộ phân bố. Root value gần +1 có thể đúng trên thế có thắng ngay; tác động của value ở các leaf cạnh tranh mới là điều cần kiểm tra thêm. Thử nghiệm tái hiện: `.venv/Scripts/python.exe scripts/tactical_benchmark.py --checkpoint checkpoints/latest.pt --split holdout --backend python --seed 420`, thay `python` bằng `cpp` để đối chiếu. Benchmark mặc định CPU; pilot holdout dưới đây dùng CUDA.

## Thiết kế pilot

Ba seed huấn luyện 111, 222, 333; mỗi seed hai nhánh ε=0,25 và ε=0,50, cùng parent 25k, Normal, cùng optimizer Adam reset, 200 MCTS simulations, C++ MCTS, một worker, `parallel_games=4`, batch 32, 2 iteration × 4 ván và 8 updates/iteration. Mỗi nhánh tổng **24 ván, 48 updates** trên ba model độc lập; mỗi model nhận 8 ván, 16 updates. LR mặc định 0,001 với cosine horizon 16 steps; `eval_fixed_batch_size=4`. Đây là **pilot ngắn**, không phải lịch continuation được xác nhận. Hai nhánh chỉ khác epsilon trong YAML; cùng seed không bảo đảm cùng trajectory hoặc cùng số NN calls.

Run, config, SQLite, immutable generations và log mới nằm trong `az_contpilot_wzdps5fu/`; không dùng replay legacy. Training chạy trên NVIDIA RTX 4060 Laptop GPU. Thời gian process tổng cho cả sáu run là 68,18 s; tổng thời gian self-play + training được từng iteration báo cáo là 22,47 s ở ε=0,25 và 26,67 s ở ε=0,50, không cộng startup, kiểm chứng hay evaluation.

| Kết quả self-play | ε=0,25 | ε=0,50 |
|---|---:|---:|
| Ván / Đen thắng | 24 / 24 | 24 / 24 |
| Tổng positions | 238 | 282 |
| Số nước trung bình | 9,92 | 11,75 |
| Updates tổng trên ba model | 48 | 48 |

ε=0,50 tạo nhiều vị trí hơn trong mẫu này nhưng chưa chứng minh diversity hữu ích, giảm lệch màu, khả năng học chiến thuật hay tăng strength. Quan sát 48/48 Đen thắng không phải yêu cầu ép tỷ lệ 50/50.

## Learning curve tactical holdout

Đánh giá cùng 24 thế holdout, 200 simulations, C++ MCTS, CUDA, seed 420, không noise; tổng thời gian 87,63 s cho parent và 12 checkpoint. Hai nhóm ba biến thể trong mỗi family có tương quan; số thế không phải số game độc lập. Không biết các thế tương tự có xuất hiện trong dữ liệu 25k lịch sử hay không. Đối chiếu D4 với replay mới: 0 state trùng trong cả sáu run.

| Nhánh | Seed | Parent: đúng/24, win/6 | Sau 8 updates | Sau 16 updates |
|---|---:|---:|---:|---:|
| ε=0,25 | 111 | 16, 0 | 21, 5 | 22, 6 |
| ε=0,25 | 222 | 16, 0 | 15, 0 | 19, 4 |
| ε=0,25 | 333 | 16, 0 | 15, 0 | 15, 0 |
| ε=0,50 | 111 | 16, 0 | 15, 0 | 15, 0 |
| ε=0,50 | 222 | 16, 0 | 15, 0 | 14, 0 |
| ε=0,50 | 333 | 16, 0 | 15, 0 | 15, 0 |

Chỉ hai seed ε=0,25 tăng trên họ win-in-one; ở seed 111 prior ô thắng Đen cơ sở tăng từ khoảng 0,030 ở parent lên 0,915 sau 16 updates và visits đạt 100%. Sự thay đổi model là quan sát thật trên benchmark này, không chứng minh tổng quát hóa vì chỉ một family win và biến thể rất gần nhau; seed 333 không cải thiện, còn nhánh ε=0,50 không giải được case win nào. Accuracy block sau 16 updates lần lượt ε=0,25: 4/6, 3/6, 3/6; ε=0,50: 3/6, 2/6, 3/6; parent 4/6. Không có bằng chứng phòng thủ cải thiện.

Train losses theo iteration/seed: ε=0,25 lần lượt 2,534→2,596; 2,664→2,408; 2,142→2,212. ε=0,50: 3,149→3,100; 2,902→3,087; 3,035→2,738. Replay mỗi nhánh khác nhau; loss và fixed-batch bốn vị trí không phải holdout theo game và không dùng để kết luận sức mạnh.

## Đối kháng paired với Heuristic

Mỗi checkpoint cuối và parent đấu trên sáu opening seed 1222, 1223, 1444, 1445, 1666, 1667: hai opening độc lập cho mỗi training seed, đổi màu theo cặp, tổng 12 ván/model group và 36 ván. Mỗi opening gồm bốn nước hợp lệ, không terminal và không có thắng ngay cho cả hai màu; không chọn thế gần terminal. Cả ba group dùng đúng các opening tương ứng, Heuristic như nhau, MCTS C++ 200 simulations/nước, CUDA; pattern C++. Mỗi trận và history nằm trong `az_contpilot_wzdps5fu/paired.log`; tổng wall time 97,27 s.

| Checkpoint | AZ Đen | AZ Trắng | Tổng | Kết quả theo training seed 111 / 222 / 333 |
|---|---:|---:|---:|---|
| Parent 25k | 2/6 | 0/6 | 2/12 | 0/4, 0/4, 2/4 |
| ε=0,25 sau 16 updates | 0/6 | 0/6 | 0/12 | 0/4, 0/4, 0/4 |
| ε=0,50 sau 16 updates | 0/6 | 0/6 | 0/12 | 0/4, 0/4, 0/4 |

Chênh lệch điểm quan sát so với parent là -2/12 cho mỗi nhánh; không đủ mẫu để kết luận suy giảm hay tiến bộ. Bootstrap percentile 10.000 lần theo training seed rồi opening trong seed cho ε=0,25−parent và ε=0,50−parent đều cho khoảng **[-0,50; 0,00]** theo score/game. Chỉ có ba training seed, nên khoảng này rất không ổn định; hai nhánh đều thua tất cả nên bootstrap nhánh−nhánh suy biến thành [0,0], **không phải bằng chứng hai epsilon tương đương**. Không báo Elo.

Value tại lượt đầu của AZ được chuyển thành xác suất `(v+1)/2` và so với kết quả trận gặp Heuristic. Brier theo Đen/Trắng: parent 0,658/0,149; ε=0,25 0,782/0,045; ε=0,50 0,786/0,141, mỗi màu chỉ sáu trận. Các số này mô tả kết quả với một đối thủ/search cụ thể, không phải calibration minimax hay bằng chứng thống kê đáng tin. Trong các opening này không có bằng chứng từ đối kháng rằng cải thiện một family tactical đã chuyển thành sức chơi.

## Provenance, resume và sửa lỗi

Sáu SQLite được đọc read-only. Mỗi run có 8 game IDs duy nhất; replay ID khớp record SQLite, sample count bằng tổng số nước, cursor bằng số sample. Mỗi record có ruleset Normal, game seed riêng và checkpoint file SHA nguồn; digest tensor thực tế ghi trong game được đối chiếu với digest tính lại từ checkpoint nguồn của từng iteration. Cả sáu khớp; nạp checkpoint/replay cuối trong `Trainer.load` trả iteration 2, global step 16. Parent SHA của mỗi generation khớp 25k. Những run so sánh dùng **một process**; kiểm chứng đa process được tách khỏi so sánh tài nguyên.

**Bug rollback đã xác nhận và sửa:** trước sửa, load ancestor rồi `Trainer.save` ghi `manifest.previous` là tip cũ chứ không phải checkpoint thực sự được load. Regression `test_rollback_save_records_actual_parent_and_preserves_abandoned_generation` thất bại trước sửa, sau sửa pass. `previous` nay ghi model/replay parent thực; `commit_previous` chỉ xuất hiện khi phân nhánh và giữ đường truy cập tới tip cũ trong lịch sử commit. `resolve_generation` chấp nhận cả manifest cũ và mới; `SPEC.md` ghi rõ hai quan hệ. Không xóa hay sửa generation cũ.

Trên bản sao `recovery_111`, chạy CLI `--resume --generation 63d4eb270cee4277bb166a25f44f2a71 --workers 2` với run ID mới: khôi phục iteration 1/step 8, chơi thêm 4 ván và lưu iteration 2/step 16. Manifest mới ghi `previous=63d4eb270cee4277bb166a25f44f2a71`, `commit_previous=0c7d48ace7a141838ff53ec2b424ff79`; cả ancestor lẫn tip cũ vẫn resolve được. Cả bốn game của hai worker có SHA checkpoint nguồn `2b125e7ca6b74d20338679583545be65f1fafda024f6fd1d3d1dda88b653c4ab`, digest tensor `adc8c622f02d5564b140cee7bdd5a3b0c81815dd9bf1f9404c47af62f1296685` khớp checkpoint, và game ID có trong replay. Bốn ván recovery là kiểm thử riêng, không gộp vào so sánh epsilon; tổng phát sinh ε=0,25 là 28 ván, dưới giới hạn 100.

`pytest -q -p no:cacheprovider tests/test_trainer.py tests/test_train_resume_logging.py tests/test_native_mcts.py tests/test_tactical_benchmark.py tests/test_selfplay.py tests/test_logging_v2.py`: **72 passed**. `graphify update .` đã chạy; Graphify cảnh báo package/skill khác phiên bản. Các test parity không chứng minh tương đương trên mọi trạng thái.

Một rủi ro recovery khác cần xử lý trước run dài: game SQLite commit trước generation; nếu crash ở giữa, chạy lại cùng game ID có thể gặp collision vì record chứa timestamp/duration. Trên bản sao SQLite **trong bộ nhớ**, phát lại một record pilot cùng game ID nhưng duration khác đã thực sự báo `game identity collision with different data`. Source `scripts/train.py` xác nhận cửa sổ giữa hai commit; thí nghiệm này chưa tái hiện crash toàn pipeline. Không xóa hoặc tự động thay log để vượt lỗi.

## Điều kiện trước khi mở rộng

- Cần thử nghiệm độc lập nhiều seed/game hơn, holdout theo game và family chiến thuật mới, đo policy/value theo màu với nhãn có nguồn gốc; phải tách rõ holdout với replay mới và không khẳng định biết exposure của 25k lịch sử.
- Kiểm chứng crash giữa SQLite commit và generation commit; thiết kế recovery hoặc fail-closed có quy trình vận hành rõ ràng trước khi chạy dài. Test process-kill tại pointer không chứng minh durability khi mất điện trên Windows.
- Chọn cấu hình bằng so sánh paired openings, uncertainty theo opening/training seed, kiểm tra không suy giảm phòng thủ và lặp lại được; không chọn ε=0,50 chỉ vì ván dài hơn. Không ép cân bằng màu, không tăng simulations và không dùng replay legacy chưa xác minh.
