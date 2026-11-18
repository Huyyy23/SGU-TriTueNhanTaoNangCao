# Danh sách công việc đang thực hiện: Gomoku AlphaZero — theo SPEC.md

`SPEC.md` là hợp đồng kỹ thuật và nghiên cứu; `plan.md` là lộ trình theo thứ tự phụ thuộc. Danh sách này thay thế checklist tiền triển khai cũ và là backlog hiện hành. Các ô chưa đánh dấu là công việc dự kiến, không phản ánh trạng thái đã triển khai. Dùng các nhãn bằng chứng **Verified / Implemented-Unverified / Planned / Unknown** như trong SPEC.

## Trạng thái công việc cũ (không suy luận từ checkbox cũ)

| Mã cũ | Bằng chứng/trạng thái hiện biết |
|---|---|
| T1–T13, T19–T20 | Source chính của game/AZ/training đã tồn tại. Tiêu chí và checkbox lịch sử không phải kết quả test hiện tại. Báo cáo các lần training nằm riêng trong `docs/report.md`. |
| T14–T18 | Kế hoạch Colab/shard ban đầu đã bị cắt; khả năng worker/resume/shard hiện tại khác với hợp đồng task cũ. Không khôi phục thiết kế cũ nếu chưa sửa SPEC. |
| T21 | Được thay bằng `scripts/arena.py` và `scripts/tournament.py`; `scripts/simulate.py` không còn là đường đánh giá chính. |
| T22, T23, T24–T33, T36 | Có tài liệu, UI, luật, agents, search và tournament liên quan; việc có code không đồng nghĩa parity luật chuyên sâu hay tiêu chí nghiên cứu đã đạt. |
| T34 | Tài liệu nộp môn là deliverable riêng; chỉ theo dõi nếu môn học còn yêu cầu, không xem là cổng chất lượng AlphaZero. |
| T35 | `docs/report.md` ghi nhận 2.400 ván và tập kết quả được chọn; Heuristic thắng AZ v0/AZ threat trong các cặp Normal cố định được báo cáo và thắng AZ Renju trong cặp Renju được báo cáo. Đây là bằng chứng lịch sử, không phải Elo hay chứng minh thắng theo mọi protocol. |

Tiêu chí task cũ dựa trên loss, win-rate giả định, throughput Colab hoặc mốc 1.000 ván đã được thay bằng Elo và quality gates trong SPEC. Không coi checkbox cũ chưa đánh dấu là backlog hiện hành, cũng không coi checkbox cũ đã đánh dấu là bằng chứng đạt tiêu chí SPEC.

## Sơ đồ phụ thuộc

`P1` sẵn sàng baseline/protocol → `P2` kiểm chứng luật/search → `P3` chẩn đoán AZ.  
`P1` → `P4` provenance lần chạy; `P2` → `P5` Python/C++ parity.  
`P1` + `P4` → `P6` Elo analysis → `P7` pilot/screening → `P8` đánh giá baseline chính.  
`P3` + `P4` + `P8` → `P9` rà soát checkpoint 1.000 ván → `P10` 2.500 → `P11` 5.000 → `P12` 10.000+.  
`P5` + `P8` → `P13` ablation. Kết quả được chấp nhận và manifests → `P14` báo cáo nghiên cứu.

## Giai đoạn 0 — Protocol và baseline hiện tại

### P1: Chốt đầu vào nghiên cứu Elo v1.0
- **Mô tả:** Xác định implementation/config Heuristic đã đóng băng, ruleset/phiên bản chuẩn, model IDs, evaluation config, cách sinh opening/seed và lựa chọn fit BT/Davidson trước khi thu thập dữ liệu chính thức.
- **Tiêu chí nghiệm thu:**
  - [ ] Heuristic được đặt mốc 1200 trong hai rating pool Normal và Renju riêng.
  - [ ] Protocol ghi rõ screening 100 ván, main 400 ván (200 cặp opening đổi màu), AZ cố định 200 simulations, temperature 0, tắt Dirichlet noise, SHA-256/config/rules/seed/opening identity.
  - [ ] Đăng ký trước paired-cluster bootstrap, xử lý hòa, tiêu chí (ΔElo > 0 và cận dưới CI 95% > 0; mục tiêu mạnh ΔElo ≥ +100) và mở rộng 600/800.
  - [ ] Lựa chọn chưa giải quyết được ghi Unknown; mọi thay đổi phương pháp trước main run phải tăng protocol version.
- **Cách kiểm chứng:** Đối chiếu protocol đã chốt với `SPEC.md` §10.2 và CLI/log fields hiện có của `scripts/tournament.py`; synthetic fixtures sẽ được kiểm tra ở P6/P7. Không chạy trận ở task này.
- **Phụ thuộc:** Không có.
- **File dự kiến:** `tasks/plan.md`, `tasks/todo.md`; nếu quyết định làm thay đổi SPEC thì cập nhật `SPEC.md`.
- **Phạm vi:** S.

### P2: Kiểm chứng lại luật và tính đúng của MCTS
- **Mô tả:** Kiểm tra các hợp đồng hiện tại trước khi diễn giải kết quả thua hoặc thay đổi training.
- **Tiêu chí nghiệm thu:**
  - [ ] Normal win/draw và Renju exact-five, overline, double-three/double-four, tính hợp lệ của quân Trắng có test hiện hành hoặc báo cáo thiếu hụt cụ thể.
  - [ ] MCTS win-in-1, dấu khi backup, terminal handling, legal masking, trường hợp zero-visit và lựa chọn temperature-0 đều đạt.
  - [ ] Sai khác giữa source và test được ghi nhận trước khi sửa code; chưa khẳng định tương đương luật Renju quốc tế nếu thiếu nguồn có thẩm quyền.
- **Cách kiểm chứng:** Chạy lệnh tập trung `python -m pytest -q tests/test_rules_normal.py tests/test_rules_renju.py tests/test_mcts.py tests/test_native_mcts.py` (chỉ chạy native tests khi extension tồn tại). Ghi lại interpreter và kết quả.
- **Phụ thuộc:** P1.
- **File dự kiến:** `tests/test_rules_normal.py`, `tests/test_rules_renju.py`, `tests/test_mcts.py`; chỉ bổ sung coverage tối thiểu cho gap đã xác định.
- **Phạm vi:** M.

### Cổng A — Tính toàn vẹn baseline
- [ ] Đối chiếu dữ liệu T35 với raw games/manifests; báo cáo sai khác, không ghi đè logs gốc.
- [ ] Đầu vào protocol đã được chốt hoặc ghi rõ Unknown.
- [ ] Kết quả kiểm chứng tập trung được lưu lại; chưa bắt đầu tác vụ tốn tài nguyên.

## Giai đoạn 1 — Chẩn đoán khoảng cách sức chơi

### P3: Chẩn đoán vì sao AlphaZero thua Heuristic
- **Mô tả:** Theo SPEC §10.4: tái dựng các trận thua, kiểm tra search, targets, lỗi chiến thuật, độ nhạy theo budget và chất lượng checkpoint theo từng đối chiếu cô lập.
- **Tiêu chí nghiệm thu:**
  - [ ] Mỗi game/position được xem xét có ID bất biến, hash checkpoint/config/rules, màu quân, seed và kết quả.
  - [ ] Audit metrics gồm policy targets hợp lệ/được chuẩn hóa, visit distribution/entropy, góc nhìn `z`, value calibration trên holdout tách theo game và các loại bỏ lỡ win/block/fork/endgame.
  - [ ] Đối chiếu search budget 50/100/200/400 giữ nguyên checkpoint/positions/openings/seeds, báo cáo Elo/score, tactical error rate và latency; ghi đây là ablation, không phải Elo chính với 200 simulations.
  - [ ] Nguyên nhân không có đối chứng cô lập vẫn để Unknown; không kết luận chỉ từ loss giảm.
- **Cách kiểm chứng:** Bổ sung/kiểm tra deterministic diagnostic fixtures và tests; lập bảng bằng chứng gồm baseline/control, metrics và giới hạn. Chạy arena cần được cho phép riêng.
- **Phụ thuộc:** P2.
- **File dự kiến:** `tests/test_mcts.py`, chỉ thêm diagnostic tooling nếu thật sự cần (tối đa 5 file).
- **Phạm vi:** M.

## Giai đoạn 2 — Độ tin cậy và parity backend

### P4: Bổ sung provenance cho lần chạy và checkpoint
- **Mô tả:** Xác định và triển khai experiment manifest tối thiểu theo SPEC §12, không đổi hyperparameters training.
- **Tiêu chí nghiệm thu:**
  - [ ] Lưu resolved config, git/dirty identity, ruleset, seed streams độc lập/cách sinh, parent/current checkpoint hashes, backend/ABI, môi trường, command và artifact hashes.
  - [ ] Resume từ chối parent/config/rules/architecture không khớp và ghi nhận tình huống buffer thiếu/hỏng.
  - [ ] Tests xác nhận manifest round-trip và từ chối artifact mismatch.
- **Cách kiểm chứng:** Chạy manifest/resume tests tập trung, sau đó một CPU smoke nhỏ trong thư mục output tạm; không training dài.
- **Phụ thuộc:** P1.
- **File dự kiến:** `scripts/train.py`, `az/trainer.py`, `az/model.py`, `tests/test_trainer.py` (tối đa 4 file).
- **Phạm vi:** M.

### P5: Đặt cổng parity Python/C++
- **Mô tả:** Mở rộng differential tests trước khi xem native search là backend nghiên cứu được hỗ trợ.
- **Tiêu chí nghiệm thu:**
  - [ ] Cùng state cho legal moves, terminal results và forbidden masks giống hệt nhau.
  - [ ] Với evaluator/tie policy cố định, requests, visits và số simulations tương ứng; công bố tolerance số học cho values.
  - [ ] Self-play ghép cặp nhiều seed so sánh feature planes, `pi`, `z`, winner và end reason; kiểm tra lifecycle, spawn/resume.
  - [ ] Mismatch chặn promote backend; báo cáo tốc độ tách biệt với chất lượng.
- **Cách kiểm chứng:** `python -m pytest -q tests/test_backend.py tests/test_native_mcts.py tests/test_native_patterns.py`; chỉ build nếu toolchain có sẵn và được cho phép. Full suite và sanitizer/soak là các cổng riêng.
- **Phụ thuộc:** P2.
- **File dự kiến:** `tests/test_native_mcts.py`, `tests/test_backend.py`, `tests/test_native_patterns.py` và chỉ sửa adapter trực tiếp liên quan nếu cần.
- **Phạm vi:** M.

### Cổng B — Độ tin cậy
- [ ] Provenance training đủ để định danh lần chạy/checkpoint bất biến.
- [ ] Kết quả parity native được phân loại Exact / Numerical / Statistical; liệt kê Unknown còn lại.

## Giai đoạn 3 — Elo và đánh giá baseline

### P6: Triển khai Elo v1.0
- **Mô tả:** Bổ sung đường phân tích rating/log tối thiểu; giữ raw tournament output làm dữ liệu nguồn. Bradley–Terry/Davidson là chính; Elo tuần tự K=16 chỉ tham khảo.
- **Tiêu chí nghiệm thu:**
  - [ ] Tách pool Normal/Renju; Heuristic neo ở 1200; model identity không phụ thuộc backend, còn execution identity có ghi backend.
  - [ ] Kiểm tra game logs phát hiện ID trùng/thiếu, hash/config mismatch, illegal forfeits và game chưa hoàn tất.
  - [ ] Analysis báo cáo BT Elo, ΔElo, paired-opening bootstrap CI 95%, W/D/L, score, phân tầng theo màu/opening và hiệu năng.
  - [ ] Elo tuần tự K=16 giữ cố định mốc Heuristic và cập nhật đối xứng hai đối thủ không phải mốc.
  - [ ] Tests đạt `AT-ELO-01..06` trong SPEC §10.2; không hardcode rating thực nghiệm.
- **Cách kiểm chứng:** Synthetic fixtures cân bằng/mạnh hơn/yếu hơn/chỉ hòa/không liên thông; replay log xác định và bootstrap lặp lại với cùng seed. Chỉ thêm CLI command vào tài liệu sau khi command được triển khai.
- **Phụ thuộc:** P1, P4.
- **File dự kiến:** `scripts/tournament.py` hoặc một rating module nhỏ, cùng tests rating/tournament tập trung (tối đa 4 file).
- **Phạm vi:** M.

### P7: Thẩm định mô hình Elo và screening
- **Mô tả:** Kiểm tra fit diagnostics, paired-cluster bootstrap và giả định cỡ mẫu bằng dữ liệu tổng hợp; sau đó mới chạy 100 ván/cặp nếu được cho phép rõ ràng.
- **Tiêu chí nghiệm thu:**
  - [ ] Synthetic effects khôi phục đúng chiều và uncertainty hợp lý; dữ liệu không liên thông/chỉ hòa/thoái hóa phải cho kết quả inconclusive.
  - [ ] Pilot dùng 50 cặp opening có đổi màu, identity/config cố định, không dùng training data để tuning.
  - [ ] Báo cáo cluster variance/power và quyết định protocol v1.0 có phù hợp trước nghiên cứu chính hay không.
- **Cách kiểm chứng:** Unit tests cùng pilot logs/manifest đã được cho phép; không gọi pilot là kết quả nghiên cứu cuối cùng.
- **Phụ thuộc:** P6.
- **File dự kiến:** Elo analysis tests và run manifest; pilot logs là artifacts bất biến, không phải source.
- **Phạm vi:** M; chạy tournament cần được cho phép riêng.

### P8: Đánh giá chính ghép cặp các checkpoint hiện có
- **Mô tả:** So sánh các AZ checkpoints đã đóng băng với Heuristic và champion theo Elo v1.0.
- **Tiêu chí nghiệm thu:**
  - [ ] 400 ván/cặp (200 cặp opening), cùng opening cho hai ván đổi màu; AZ 200 simulations/move, temperature 0, tắt root noise.
  - [ ] Hashes, seeds, rules, resolved config, backend execution identity và W/D/L logs đầy đủ được xác thực trước analysis.
  - [ ] Báo cáo ΔElo/CI, W/D/L, score và hiệu năng; chỉ mở rộng 600/800 theo quy tắc đăng ký trước; chưa kết luận ở 800 thì ghi inconclusive.
  - [ ] Chỉ tuyên bố đạt mục tiêu khi ΔElo > 0 và cận dưới CI 95% > 0; mục tiêu mạnh còn yêu cầu ΔElo ≥ +100.
- **Cách kiểm chứng:** Tính lại summary/rating từ raw game logs và đối chiếu analysis hash đã lưu. Không gộp Normal/Renju hoặc pool cố định/time-budget khác nhau.
- **Phụ thuộc:** P6, P7.
- **File dự kiến:** Không sửa source trừ khi phát hiện lỗi correctness đã được duyệt; output phải ở thư mục mới, rỗng.
- **Phạm vi:** L, chạy thí nghiệm cần được cho phép rõ ràng.

### Cổng C — Sẵn sàng đánh giá
- [ ] Elo implementation đạt synthetic tests và replay tests.
- [ ] Screening/main-run decisions được đăng ký trước; chưa có rating claim trước khi logs được xác thực.

## Giai đoạn 4 — Training dựa trên chất lượng

### P9: Đánh giá lại checkpoint Normal lịch sử 1.000 ván
- **Mô tả:** Gắn checkpoint thực tế với SHA-256/config/lineage nếu có thể; đánh giá theo v1.0 và ghi Unknown nếu thiếu provenance lịch sử.
- **Tiêu chí nghiệm thu:**
  - [ ] So sánh chính 400 ván hoặc phần mở rộng inconclusive theo v1.0.
  - [ ] Không suy ra lineage checkpoint chỉ từ filename.
  - [ ] Tái tạo được baseline từ raw logs.
- **Cách kiểm chứng:** Tournament và replay logs đã được cho phép; bảo toàn artifacts lịch sử.
- **Phụ thuộc:** P4, P8.
- **File dự kiến:** Chỉ output đánh giá mới.
- **Phạm vi:** L, chạy thí nghiệm cần được cho phép rõ ràng.

### P10: Huấn luyện/đánh giá đến tổng cộng 2.500 ván
- **Tiêu chí nghiệm thu:**
  - [ ] Checkpoint có parent link bất biến; manifest, sample count và optimizer steps hợp lệ.
  - [ ] Có Elo v1.0 so với Heuristic và champion.
  - [ ] Chỉ promote nếu cận dưới CI so với champion > 0; regression được xác nhận thì chặn recipe; kết quả inconclusive theo quy tắc mở rộng rồi tạm dừng để chẩn đoán.
- **Cách kiểm chứng:** Training smoke và resume tests trước run theo giai đoạn đã được duyệt; tính lại Elo từ raw logs.
- **Phụ thuộc:** P3, P4, P9.
- **File dự kiến:** Chỉ thay config nếu được duyệt riêng; tạo artifact directories mới.
- **Phạm vi:** L, chạy thí nghiệm cần được cho phép rõ ràng.

### P11: Huấn luyện/đánh giá đến tổng cộng 5.000 ván
- **Tiêu chí nghiệm thu:** Giữ nguyên integrity/Elo gates của P10; bổ sung RSS/VRAM/worker health, replay diversity và phân rã runtime. Config hoặc ước tính tuyến tính không chứng minh đã chạy xong hay chất lượng tăng.
- **Cách kiểm chứng:** Run theo giai đoạn đã được duyệt, xác thực checkpoint/hash, replay rating v1.0.
- **Phụ thuộc:** P10.
- **File dự kiến:** Chỉ artifacts mới.
- **Phạm vi:** L, chạy thí nghiệm cần được cho phép rõ ràng.

### P12: Huấn luyện/đánh giá đến tổng cộng 10.000+ ván
- **Tiêu chí nghiệm thu:** Mỗi checkpoint có Elo/CI, provenance và resource data. Chỉ tiếp tục sau 10k cho câu hỏi đã đăng ký trước hoặc bằng chứng Elo hữu ích; dừng khi có regression xác nhận hoặc lỗi toàn vẹn.
- **Cách kiểm chứng:** Run theo giai đoạn đã được duyệt và phân tích logs tái lập được; không ngoại suy từ games/min.
- **Phụ thuộc:** P11.
- **File dự kiến:** Chỉ artifacts mới.
- **Phạm vi:** L, chạy thí nghiệm cần được cho phép rõ ràng.

## Giai đoạn 5 — Ablation và báo cáo nghiên cứu

### P13: Chạy ablation AZ có kiểm soát
- **Tiêu chí nghiệm thu:**
  - [ ] So sánh riêng MCTS 50/100/200/400, search/training hyperparameters, mạng 3×32/6×64/9×96, self-play 1k/2.5k/5k/10k và Python/C++.
  - [ ] Mỗi contrast chỉ đổi một yếu tố; ghép cặp seeds/openings, giữ evaluation pool/protocol cố định, tắt evaluation noise và giữ checkpoint nếu phù hợp.
  - [ ] Báo cáo Elo/ΔElo/CI, W/D/L/score, compute, latency, memory và data counts; xử lý multiple comparisons hoặc ghi rõ là exploratory.
  - [ ] Đăng ký trước số training seeds và backend non-inferiority margin từ pilot trước confirmatory tests; không chạy full Cartesian sweep.
- **Cách kiểm chứng:** Config/manifests chứng minh các biến được kiểm soát; analysis tests đạt; kết quả cần Elo runs đã được cho phép.
- **Phụ thuộc:** P5, P8.
- **File dự kiến:** `scripts/ablation.py`, tests tập trung và/hoặc một analysis module (tối đa 5 file).
- **Phạm vi:** M cho thiết kế, L cho chạy thí nghiệm; cần được cho phép.

### P14: Công bố kết luận có thể tái lập
- **Tiêu chí nghiệm thu:**
  - [ ] Mọi claim liên kết với run ID, checkpoint/log hash, protocol và CI.
  - [ ] Báo cáo phân biệt mục tiêu với kết quả, Verified với Planned/Unknown, và Normal với Renju.
  - [ ] Ghi cả thất bại, exclusions và giới hạn; learning curve chỉ dùng checkpoint thực tế.
- **Cách kiểm chứng:** Tái dựng độc lập từ raw logs và manifest; không tạo kết quả giả hoặc nội suy.
- **Phụ thuộc:** P8 và các kết quả training/ablation được đưa vào báo cáo.
- **File dự kiến:** `docs/report.md` và tests báo cáo nếu có (tối đa 3 file).
- **Phạm vi:** M.

## Cổng kết thúc

- [ ] Không đánh dấu task hoàn thành nếu thiếu bằng chứng nghiệm thu.
- [ ] Claim về AZ đạt Elo v1.0; claim về backend đạt parity và quality gates.
- [ ] Công việc dài đã được cho phép rõ ràng và artifacts được bảo toàn.
- [ ] Báo cáo cuối phân loại chính xác Verified / Implemented-Unverified / Planned / Unknown.
