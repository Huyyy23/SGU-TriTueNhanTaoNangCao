# Tóm tắt phiên phục hồi AlphaZero Gomoku 25k

## Bối cảnh

Mục tiêu là tìm nguyên nhân AlphaZero Gomoku 11×11 còn yếu trước Heuristic và chuẩn bị continuation an toàn. Kết quả lịch sử được báo cáo là **46/400 ván thắng Heuristic (11,5%)**; candidate tournament lịch sử và `checkpoints/latest.pt` có SHA file khác nhau, nên chưa khẳng định hai artifact có cùng tensor. Trong suốt các thử nghiệm, giữ luật Normal, ResNet và **200 MCTS simulations mỗi lần search**; không chạy thêm 5.000 ván, không commit hoặc xóa dữ liệu lịch sử.

Checkpoint parent được dùng cho các pilot có SHA-256:

```text
f7f521c1cc37a5054ed854201b087e484e9fcef3d4d6b1877ab81dff2e51d7e4
```

## 1. Điều tra dữ liệu và chiến thuật

- Run SQLite continuation có 15.000 record; **14.931 ván Đen thắng**, 13.657 ván kết thúc sau 9 hoặc 11 nước. Trong 5.000 ván ở iteration 400–499, Đen thắng 4.976 và 4.982 ván mở đầu ở tâm. Đây là lệch phân phối tự chơi mạnh; chưa xác định được bao nhiêu do lợi thế tự nhiên của Đen và chưa chứng minh quan hệ nhân quả với kết quả trước Heuristic.
- Replay legacy có 50.000 mẫu nhưng thiếu game ID/cursor và không chứng minh được ghép cùng checkpoint. Không dùng replay đó như generation hoặc holdout đã xác minh.
- Trên sáu canary broken-four, oracle xác nhận nước 60 thắng ngay, nhưng parent bỏ qua với **0/199 child visits**; Python/C++ khớp trên các case đã đo. Terminal detection đúng khi trực tiếp chơi nước thắng. Prior thấp và Q cao của nhánh cạnh tranh giải thích hành vi PUCT ở case đã đo; chưa xác nhận lỗi sign flip.
- Wrapper thử nghiệm chọn thắng-ngay sau đủ 200 simulations đạt 6/6 trên canary cũ, nhưng tập 12 thế mới không cải thiện (**11/12** ở cả hai arm); mẫu đối kháng đều 0/4. Không tích hợp tactical override.

## 2. Pipeline continuation và recovery

- Bổ sung bootstrap Normal từ checkpoint sang run mới, replay rỗng có provenance; lựa chọn rõ reset hoặc restore optimizer/LR. Các generation checkpoint/replay bất biến, có manifest/hash; game records có game ID, seed, ruleset, SHA checkpoint nguồn và digest tensor worker.
- Sửa các lỗi đã tái hiện: SQLite khởi tạo lại bằng đường dẫn tương đối; restore optimizer bắt đầu sai iteration của cosine schedule; lineage generation sai sau rollback; resume nhầm run ID; retry rollback sau crash trước generation commit bị chặn.
- Bổ sung outbox phục hồi cho crash giữa SQLite game commit và generation commit: resume dùng lại payload và policy/value targets đã lưu, kiểm tra hash/identity và từ chối collision khác dữ liệu. Có kiểm thử process exit, resume lặp, rollback, collision, tamper và ring-buffer overwrite.
- Kiểm thử thực trên Windows: sau `os._exit(73)` ở cửa sổ sau SQLite commit, hai lần resume giữ nguyên bốn game, 36 target và một metrics record. Một phép rollback trên bản sao dùng hai worker xác nhận source hash, game IDs và giữ được nhánh cũ.
- Bộ kiểm thử recovery/MCTS liên quan đã chạy ở giai đoạn final readiness: **79 passed**. Preflight trước pilot 100 ván chạy ba test recovery trọng điểm: **3 passed**. Đây không phải bằng chứng về mọi trạng thái Python/C++ hoặc mất điện thực.

## 3. Các thí nghiệm training nhỏ

- Pilot nhiều seed so ε=0,25 và ε=0,50: mỗi nhánh 24 ván, ba seed, 48 updates tổng. Cả 48 ván Đen thắng. Một số seed ε=0,25 cải thiện họ tactical cố định nhưng không ổn định; kết quả không đủ chọn ε=0,50 hay chứng minh tăng strength.
- Thí nghiệm offline dùng cùng replay self-play mới có provenance, cùng parent, optimizer và compute: uniform sampling so với tăng tỷ lệ sample thế muộn lên 50%. Trên tactical holdout cũ hai nhánh đều **15/24**, tập 12 thế mới đều **11/12**. Paired nhỏ với Heuristic: uniform **2/8**, late-sampling **1/8**; mẫu quá nhỏ để kết luận sức chơi hoặc tương đương. Không tích hợp sampling thử nghiệm.

## 4. Pilot continuation 100 ván

Run `az-25k-pilot100-seed314159`, artifact tại `az_100pilot_botfozch/`:

- Bootstrap từ parent, replay ban đầu rỗng, Adam reset, LR `1e-4`, ε=0,25, uniform sampling, một worker. Chạy **50 iteration × 2 ván × 4 updates = 100 ván, 200 updates**, checkpoint bất biến mỗi iteration.
- Các gate 20/40/60/80/100 ván kiểm tra generation/replay/outbox hash, lineage, worker tensor, game IDs, policy legality và z; tất cả pass. SQLite `quick_check=ok`; nạp lại checkpoint cuối thành công. Không có auto-promotion.
- Self-play: **100/100 Đen thắng**, 100/100 opening ở tâm; 65 ván dài 9 ply, 27 ván 11 ply và 8 ván 13 ply. Replay cuối có **986 positions**.
- Tactical parent **16/24**; mốc 50 và 100 đều **15/24**. Win-in-one ở họ canary cũ vẫn 0/6; block giảm từ 4/6 xuống 3/6. Mẫu tactical có các biến thể tương quan, không đủ kết luận hồi quy sức chơi.
- Đối kháng cuối gồm 40 ván paired trên opening/seed cố định: pilot thắng **1/20 trước Heuristic**; trước parent thắng **12/20**, bootstrap theo opening pairs cho score **[40%, 80%]**. Chỉ 10 opening pairs mỗi đối thủ và một training seed; không kết luận tăng Elo hoặc so trực tiếp với kết quả lịch sử khác protocol/checkpoint.
- Checkpoint cuối SHA-256 `ece41a2074037b6025fe1807905572e8ef91955534aa5ca5834d6e482f28a40b`. Cosine horizon đã hết ở step 200/200; LR bước kế tiếp theo lịch cũ là 0.

## 5. Kết luận hiện tại

**Không nên tiếp tục ngay 5.000 ván với recipe pilot.** Integrity đã vượt các gate được chạy, nhưng dữ liệu tự chơi vẫn tập trung, tactical chưa cải thiện và kết quả trước Heuristic rất yếu. Cần thử có đối chứng các cách cải thiện đa dạng opening hoặc tín hiệu phòng thủ, xác định lại lịch LR và đánh giá theo game/seed độc lập trước khi scale. Không suy ra một thay đổi cụ thể đã cải thiện sức chơi từ các mẫu hiện có.

Giới hạn còn lại: chưa kiểm chứng durability khi mất điện Windows hoặc concurrent writers; exposure của các tactical holdout đối với dữ liệu 25k lịch sử chưa biết; bootstrap CI và Brier từ các mẫu nhỏ chỉ mang tính mô tả.

## Báo cáo và artifact chi tiết

- [Phục hồi và tactical sau 25k](25k-recovery-and-tactics.md)
- [Chuẩn bị continuation](25k-continuation-readiness.md)
- [Pilot nhiều seed](25k-continuation-pilot.md)
- [Final readiness và thí nghiệm training](25k-final-readiness.md)
- [Báo cáo pilot 100 ván](25k-100games-pilot.md)
- Artifact runtime: `az_100pilot_botfozch/`, gồm config, script, logs, SQLite, generations, outbox, tactical gates và kết quả paired.
