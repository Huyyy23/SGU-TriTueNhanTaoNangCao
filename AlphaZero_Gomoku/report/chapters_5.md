# Chương 5. Quá trình thực nghiệm và cải tiến hệ thống

Chương này tái dựng quá trình thử nghiệm theo các mốc được ghi trong báo cáo, log, SQLite và kiểm toán artifact của dự án. Trọng tâm không phải liệt kê mọi thay đổi mã, mà là chỉ ra vì sao từng phép thử được thực hiện, giả thuyết nào được đặt ra, điều kiện đo cụ thể, kết quả quan sát và quyết định tiếp theo. Các con số “ngày” dưới đây là ngày chạy hoặc ngày ghi nhận của thí nghiệm trong tài liệu; chúng **không đồng nghĩa ngày commit**. Tên checkpoint, SHA-256 của checkpoint, generation, database và Git commit là những định danh khác nhau. Git history được kiểm tra ở cấp dự án nhưng không cung cấp cơ sở để gán SHA commit cho từng thí nghiệm; vì vậy chương không biến SHA artifact thành SHA commit.

Kết quả huấn luyện, sức chơi và tốc độ được xem là ba loại bằng chứng riêng. Loss giảm không chứng minh agent thắng nhiều hơn; số ván self-play không chứng minh các ván thuộc cùng lineage; tốc độ cao không chứng minh chất lượng Python/C++ tương đương. Phạm vi kết luận ở mỗi mục được giới hạn theo luật, checkpoint, opening, backend, ngân sách tìm kiếm và workload thực sự được ghi nhận. Nền tảng kỹ thuật và nhãn trạng thái bằng chứng được trình bày trong [`SPEC.md`](../SPEC.md), đặc biệt các mục 1–2, 8–11.

## 5.1. Cách đọc timeline và phép đo

Hệ thống Normal dùng bàn 11×11, mạng policy–value ResNet 6 block × 64 filters và MCTS 200 simulations cho mỗi lần tìm kiếm. Self-play sinh policy target từ phân bố visit và value target từ kết quả cuối ván; replay buffer chỉ chứa một ruleset. Do đó, các lần fine-tuning Normal và Renju không thể cộng chung thành một chuỗi dữ liệu huấn luyện duy nhất. Cấu hình trong tệp YAML cho biết điều dự định chạy, trong khi log/DB mới cho biết phần nào thực sự đã chạy. Đây là khác biệt quan trọng giữa mục tiêu nghiên cứu và bằng chứng vận hành (`SPEC.md`, §§6–8; `docs/report.md`, §§4–8).

Một số mốc mang nhãn “25k”, “40k”, “50k”, “60k” là tên gọi của checkpoint hoặc lịch sử continuation được đối chiếu. Chúng không phải lúc nào cũng là tổng số ván có thể xác minh độc lập. Chẳng hạn, kiểm toán 60k mô tả mốc danh nghĩa gồm parent lịch sử 25k và 35k ván continuation; riêng lịch sử 25k không được đếm độc lập đầy đủ. Run 20k riêng có checkpoint khởi đầu khác hẳn checkpoint 60k, nên không thể cộng nó thành “80k liên tục” (`docs/az-80k-checkpoint-forensics.md`, §§1–2). DB `data/normal.sqlite` ghi 75.000 game records và 1.500 metric rows qua nhiều run; DB `data/az_25k_continuation_v1.sqlite` ghi 35.000 game records. Đây là hai snapshot không được cộng/gộp nếu chưa chứng minh identity và lineage; trường `git_commit` và `engine_version` trong metadata liên quan để trống (`report/bang_doi_chieu.md`, E19; `data/normal.sqlite`; `data/az_25k_continuation_v1.sqlite`; `docs/az-80k-checkpoint-forensics.md`, §§2–3).

## 5.2. Mốc khởi đầu: Normal 1.000 ván

### Bối cảnh và giả thuyết

Thử nghiệm Normal đầu tiên có mục tiêu chứng minh toàn bộ vòng lặp AlphaZero có thể chạy: tự chơi, lưu mẫu, tối ưu policy–value và ghi checkpoint/log. Giả thuyết vận hành là pipeline tạo được dữ liệu và loss train sẽ giảm khi tối ưu trên các mẫu self-play. Đây là kiểm tra khả năng chạy và tín hiệu học trong train set, không phải giả thuyết đã định lượng rằng checkpoint sẽ mạnh hơn Heuristic.

### Thiết kế

Log ngày 2026-10-28 ghi 20 iterations × 50 games = 1.000 games. Mỗi nước dùng 200 simulations; cấu hình ghi 16 ván hoạt động để gom lô inference, mạng 6×64, batch size 128 và 200 bước train/iteration. Thiết bị được ghi là NVIDIA GeForce RTX 4060 Laptop GPU. Run có 4.000 global steps ở iteration 20. Thông tin này đến từ log huấn luyện và log từng game, không phải suy ra từ YAML (`docs/report.md`, §§8.1, 8.3, lines 129–148).

### Kết quả, nhận xét và quyết định

| Metric | Iteration 1 | Iteration 20 |
|---|---:|---:|
| Replay samples | 2.005 | 30.452 |
| Global step | 200 | 4.000 |
| Train loss | 5,19370 | 4,09124 |
| Policy loss | 4,79342 | 3,76979 |
| Value loss | 0,40028 | 0,32145 |
| Fixed-batch loss | 5,05494 | 5,14099 |
| Nước trung bình mỗi ván | 40,10 | 23,10 |
| Network leaf evaluations trung bình mỗi ván | 7.877,7 | 4.408,2 |

Loss train giảm từ 5,19370 xuống 4,09124, nhưng loss trên batch cố định tăng từ 5,05494 lên 5,14099 và dao động giữa các iteration. Số nước trung bình giảm cũng không chỉ ra nguyên nhân hay sức chơi: độ dài ván còn phụ thuộc opening, màu và cách kết thúc. Vì fixed batch lấy từ stream self-play chứ không phải holdout theo game, ngay cả metric này cũng không phải ước lượng generalization độc lập (`docs/report.md`, §§6, 8.1, lines 105–115, 133–147; `SPEC.md`, §6.2).

Kết luận hợp lý của mốc 1.000 là pipeline đã tạo được checkpoint và ghi log huấn luyện, đồng thời train objective có thay đổi theo tối ưu. Kết quả **không** xác lập rằng Normal đã hội tụ hoặc tốt hơn đối thủ. Một log ngày 2026-11-01 chỉ ghi một iteration, hai ván và 95 mẫu; nó không phải continuation quy mô lớn. Cấu hình lịch sử đặt mục tiêu 5.000 ván cũng không chứng minh đã thực thi đủ ngân sách (`docs/report.md`, §8.1, lines 148–149).

## 5.3. T35: lần đánh giá đối kháng có quy mô và protocol rõ

### Bối cảnh và thiết kế

Sau các thử nghiệm chạy pipeline, nhu cầu chuyển từ “mô hình có train được không?” sang “agent chơi ra sao trước các baseline?”. T35 hoàn tất ngày 2026-11-14, tổng 2.400 ván: 1.500 Normal fixed-budget, 600 Renju fixed-budget và 300 Normal time-budget. Mỗi matchup có 100 ván, mỗi tác tử cầm Đen 50 ván; seed gốc `20261113`. Sau khi đổi baseline Alpha-Beta fixed từ depth 2 sang depth 1, tám matchup có Alpha-Beta được chạy lại đủ 800 ván; 13 matchup không có Alpha-Beta giữ nguyên 1.300 ván. Các log depth 2, smoke và run gián đoạn được giữ riêng, không cộng vào tập selection cuối (`docs/report.md`, §§10.1–10.2, lines 225–239).

Trong nhóm fixed, AZ và MCTS Pure dùng 200 simulations, còn Alpha-Beta dùng depth 1; đây không phải ngân sách tính toán tương đương. Nhóm time-budget dùng một giây mỗi nước và Alpha-Beta iterative deepening, do đó không khóa ở depth 1. Normal và Renju được phân tích trong pool riêng. Mỗi cặp cân bằng màu, nhưng các ván trong một pool không nhất thiết độc lập; Wilson interval với hòa tính nửa điểm chỉ là xấp xỉ, không phải bằng chứng hai agent tương đương (`docs/report.md`, §§9.3, 10.1–10.4, lines 203–209, 229–233; `SPEC.md`, §10.1).

### Kết quả và quyết định

Trong Normal fixed, Heuristic thắng `az_v0` 100–0; Heuristic cũng thắng `az_v1_threat` 100–0. `az_v0` đạt 149/500 = 29,8% score trong đội hình này; `az_v1_threat` đạt 369/500 = 73,8%, nhưng đây là biến thể khác có threat search và không thể gán kết quả cho trọng số mạng. Trong Renju fixed, Heuristic thắng `az_renju` 100–0; AZ Renju hòa 50–50 với Alpha-Beta depth 1, nhưng bảng phân màu cho thấy Alpha-Beta thắng toàn bộ ván khi cầm Đen. Trong Normal time, `az_v1_threat` đạt 87/100 trước Alpha-Beta; fixed và time là hai protocol khác nhau (`docs/report.md`, §§10.2–10.3, lines 241–298).

Kết quả cho thấy ở các pool đã chạy, AZ v0 yếu hơn Heuristic và biến thể threat có kết quả tốt hơn trong một số matchup Normal. Chúng không chứng minh AZ Renju ngang Alpha-Beta hoặc threat search luôn mạnh hơn: ngân sách không đồng nhất, một tập opening cụ thể được dùng, và biến thể không phải ablation mạng matched. Quyết định nghiên cứu sau T35 là cần phân tích lỗi chiến thuật và dữ liệu, thay vì dùng loss hoặc số ván huấn luyện như đại diện cho strength. T35 là điểm tham chiếu lịch sử, không được cộng với các screen sau thành một tournament duy nhất.

## 5.4. Chẩn đoán 25k: dữ liệu tự chơi, tactical và continuation

### Vì sao điều tra mốc 25k

Báo cáo phiên 25k nêu kết quả lịch sử 46/400 trước Heuristic (11,5%), đồng thời tìm thấy self-play mất cân bằng màu mạnh: trong 15.000 record được kiểm tra, 14.931 ván do Đen thắng; trong một đoạn 5.000 ván, 4.976 ván Đen thắng và 4.982 mở đầu ở tâm. Đây là mô tả phân bố quan sát; không đủ dữ liệu để tách lợi thế tiên thủ Normal khỏi thiên lệch model/search hoặc khẳng định thiên lệch màu là nguyên nhân duy nhất khiến agent yếu (`docs/analysis/25k-session-summary.md`, §§1, 5).

Một giả thuyết cụ thể hơn là MCTS bỏ lỡ nước thắng tức thì vì PUCT phân bổ visit theo prior/value, dù luật và terminal check đúng. Trên sáu canary broken-four cũ, oracle xác nhận nước 60 thắng ngay nhưng parent bỏ lỡ cả sáu; tại thế gốc, ô thắng nhận 0/199 child visits. Python và C++ chọn cùng nước trong phép đối chiếu. Phân tích chỉ giải thích các fixture này, không chứng minh lỗi sign flip hoặc lỗi chung ở MCTS (`docs/analysis/25k-session-summary.md`, §1; `docs/analysis/25k-continuation-pilot.md`, §§1–2).

### Pilot exploration ε=0,25 so với ε=0,50

Giả thuyết là tăng Dirichlet epsilon từ 0,25 lên 0,50 sẽ khuyến khích khám phá, tăng đa dạng self-play và có thể cải thiện tactical. Ba seed (111, 222, 333) chạy mỗi nhánh, cùng parent 25k, Normal, Adam reset, 200 MCTS simulations, C++ MCTS, một worker, `parallel_games=4`, batch 32; mỗi seed/nhánh chỉ 8 ván và 16 updates, tổng 24 ván/nhánh. RTX 4060 Laptop GPU được dùng. Đây là pilot nhỏ, hai nhánh không bảo đảm cùng trajectory hay cùng số inference calls (`docs/analysis/25k-continuation-pilot.md`, §§3–4, lines 15–28).

Self-play kết thúc với Đen thắng 24/24 ở cả hai nhánh. ε=0,50 tạo 282 positions so với 238 của ε=0,25, nhưng số vị trí cao hơn không đồng nghĩa dữ liệu hữu ích hay màu cân bằng hơn. Trên 24 tactical positions cố định, parent đạt 16/24; sau 16 updates, ε=0,25 lần lượt 22/24, 19/24, 15/24 theo seed, còn ε=0,50 đạt 15/24, 14/24, 15/24. Tín hiệu của ε=0,25 không tái lập trên cả ba seed và ε=0,50 không giải được các ca win-in-one trong đánh giá đó. Accuracy phòng thủ không tăng rõ. Đối kháng ghép cặp với Heuristic chỉ có 12 ván mỗi nhóm: parent 2/12, cả hai nhánh sau 16 updates 0/12. Khoảng bootstrap theo ba seed quá bất ổn để kết luận suy giảm hoặc cải thiện (`docs/analysis/25k-continuation-pilot.md`, §§4–7, lines 30–59).

Quyết định: giữ ε=0,25 làm baseline; không nâng ε=0,50 thành recipe continuation chính, không báo Elo từ pilot và không scale thêm chỉ vì loss hoặc positions. Báo cáo readiness xác nhận chưa có bằng chứng chọn epsilon, cũng chưa có bằng chứng chọn reset hay restore optimizer cho continuation dài (`docs/analysis/25k-continuation-readiness.md`, §§1–2).

### Thử wrapper thắng-ngay sau MCTS

Giả thuyết của can thiệp tactical là một lượt quét oracle độc lập sau khi MCTS hoàn tất có thể sửa lỗi không chọn nước thắng ngay mà không đổi luật, ResNet hay MCTS sản phẩm. Thiết kế matched dùng cùng checkpoint/seed/opening/backend/budget 200 simulations; wrapper chỉ thêm quét thắng ngay sau MCTS. Trên sáu canary cũ, baseline đạt 0/6 và wrapper 6/6 — tập này chính là tình huống can thiệp được thiết kế xử lý. Trên 12 thế mới, baseline và wrapper đều 11/12, cả trước và sau tám updates. Trong screen Heuristic chỉ có bốn ván mỗi arm/checkpoint; cả hai arms đều 0/4 và wrapper không kích hoạt (`docs/analysis/25k-recovery-and-tactics.md`, §§2–4, lines 53–78).

Nhận xét: có bằng chứng wrapper làm đúng hành động hẹp “chọn thắng ngay” trên canary đã biết; không có bằng chứng cải thiện holdout mới hoặc full-game strength. Không tích hợp wrapper vào sản phẩm. Phòng thủ forced-block vẫn là khoảng trống, và bốn ván không đủ để ước lượng chênh lệch sức chơi. Báo cáo cũng ghi can thiệp có thêm chi phí quét nên không khẳng định FLOPs/wall time bằng baseline (`docs/analysis/25k-recovery-and-tactics.md`, §§3–5).

### Recovery và an toàn continuation

Một lỗi pipeline được tái hiện: game đã commit vào SQLite trước khi generation checkpoint hoàn tất; resume tái tạo game cùng ID với timestamp/duration khác và bị từ chối do collision. Cách sửa ghi outbox bất biến trước khi commit game, để resume dùng đúng payload/replay targets thay vì tái tạo. Trong một thử nghiệm process exit thực trên parent 25k, seed 712, ε=0,25, 200 simulations, bốn ván đã commit, outbox có 36 positions; hai lần resume giữ nguyên payload và replay, chỉ một metrics record. Đây là bằng chứng phục hồi sau process termination, không phải mất điện (`docs/analysis/25k-recovery-and-tactics.md`, §§1, 2; `SPEC.md`, §8.3).

Kết quả integrity cho phép tiếp tục nghiên cứu với generation có provenance, nhưng không tự xác nhận chất lượng mô hình. Giới hạn Windows directory-entry/power-loss durability, dữ liệu legacy thiếu provenance, và holdout có thể tương quan vẫn cần giữ trong mọi quyết định sau đó (`SPEC.md`, §§8.2–8.3; `docs/analysis/25k-final-readiness.md`, mục 1).

## 5.5. Continuation Normal: 40k, 50k và mốc danh nghĩa 60k

### 40k/50k: so sánh hai checkpoint

Kiểm định chất lượng sau khi huấn luyện thêm không sinh thêm training hay self-play; nó dùng các checkpoint candidate tại gate 40k và 50k. Mốc 40k là parent 25k cộng 15.000 ván mới; mốc 50k là parent cộng 25.000 ván mới. Checkpoint 40k có SHA-256 `89fddabc…3f909`; checkpoint 50k có SHA-256 `28ea9001…35c5cd`. Manifest, replay và event SQLite được xác thực theo báo cáo. Hai so sánh độc lập dùng Normal, Python MCTS 200 simulations, không root noise, temperature 0, bốn nước opening ngẫu nhiên, seed 508001; mỗi so sánh 400 ván/200 opening pairs đổi màu, CI bootstrap 10.000 lần theo opening pair (`docs/analysis/50k-quality-validation.md`, §§1–2, lines 3–19).

| So sánh | W–D–L của 50k | Score | Ước lượng/CI | Kết luận |
|---|---:|---:|---:|---|
| 50k vs 40k | 278–5–117 | 70,125% | ΔElo +149,91; 95% CI [+114,88; +188,51] | 50k trội hơn 40k trong protocol này |
| 50k vs Heuristic | 213–3–184 | 53,625% | ΔElo +25,36; 95% CI [−7,86; +58,00] | Chưa chứng minh vượt Heuristic |

Kết quả phân màu cho thấy chênh lệch: trước Heuristic, 50k thắng 126/200 khi cầm Đen nhưng chỉ 87/200 khi cầm Trắng. Dù ván đã đổi màu trong opening pairs, chênh lệch này nhắc rằng score chung không diễn giải được riêng hiệu ứng màu hay khả năng tổng quát hóa. Kết luận có thể bảo vệ là checkpoint 50k tốt hơn 40k ở đúng protocol được đo; không thể kết luận mục tiêu nghiên cứu “AZ mạnh hơn Heuristic” đã đạt (`docs/analysis/50k-quality-validation.md`, §§2, 4, lines 21–41, 98–104; `SPEC.md`, §§1–2).

So sánh checkpoint không cô lập riêng số ván khỏi các khác biệt khác trong quá trình continuation; vì vậy không quy toàn bộ chênh lệch strength cho lượng dữ liệu mới.

### Kiểm tra phòng thủ và thí nghiệm value 50k thất bại

Sau kết quả 50k, một tập audit lấy mẫu 1.000 ván self-play tìm thấy 161 lượt Trắng không chặn cũng không chọn thắng ngay trong 281 cơ hội mà Đen chỉ có một ô thắng. Đây là cơ hội cụ thể sau khi lọc theo oracle, không phải tỷ lệ lỗi trên mọi nước. Tại gate 50k, bốn biến thể gần nhau của tình huống block duy nhất đều chọn action 63 thay vì action 60; prior của nước sai cao hơn nước chặn. Đây là bằng chứng chẩn đoán, không chứng minh prior/value riêng lẻ là nguyên nhân nhân quả (`docs/analysis/50k-quality-validation.md`, §§3–4, lines 43–77).

Giả thuyết tiếp theo là tăng trọng số value loss ở state sâu có thể cải thiện dự báo value/phòng thủ mà không làm giảm policy. Protocol preregistered dùng parent/replay đúng generation, seed 314159/314160/314161, 100 updates mỗi nhánh, batch 32, Adam reset, LR cố định `1e-4`, weight decay `1e-4`; tổng 600 updates. Candidate đổi trọng số value MSE theo số quân, chuẩn hóa mean weight trong minibatch; control giữ loss gốc. Ba seed được giữ riêng, và seed 314159 được chọn trước để làm candidate strength evaluation. Tập holdout cuối gồm 99 games và 25 state sâu, SHA-256 được khóa trước pilot; do review phát hiện thiếu loại trừ artifact tactical cũ, bản tạm bị loại và builder chạy lại. Tuy vậy, counts đã thấy trước khi bản cuối tạo, nên đây không phải holdout khóa trước khi xem counts (`docs/analysis/50k-final-submission.md`, §§1–2, lines 3–20).

Kết quả không qua gate value: parent deep-state MSE 0,35155; cải thiện MSE candidate−control theo quy ước “control trừ candidate” lần lượt −0,00879, −0,00158, +0,01596; trung bình +0,00186, dưới ngưỡng preregistered +0,01, và chỉ một trong ba seed cải thiện. Tactical gate inconclusive do chỉ một state pre-fork Trắng và không đủ số mẫu mỗi loại/màu; parent và candidate đều bỏ lỡ oracle answer đó. Hai tournament candidate 400 ván mỗi đối thủ cũng không đạt gate strength đã định. Quyết định cuối là **B: giữ parent 50k, không promote candidate**. Đây là thí nghiệm thất bại theo tiêu chí đã đặt trước, không phải bằng chứng mọi thay đổi value weighting đều vô ích (`docs/analysis/50k-final-submission.md`, §§2–4, lines 11–33).

### Mốc 60k và cách không cộng nhầm các nhánh

Báo cáo checkpoint forensic ngày 2026-11-21 xác nhận checkpoint continuation AZ 60k có generation, optimizer 70.000 steps và một chuỗi lịch sử được đặt tên 25k continuation; trong đó DB snapshot `az_25k_continuation_v1.sqlite` ghi 35.000 games/700 iterations. Báo cáo nói rõ 60k là mốc danh nghĩa: lịch sử parent 25k không được đếm độc lập đầy đủ. Vì vậy chương chỉ dùng “60k” như nhãn checkpoint lineage theo kiểm toán, không tuyên bố 60.000 game records đã được tái đếm hoặc độc lập xác minh (`docs/az-80k-checkpoint-forensics.md`, §§1–3; `report/bang_doi_chieu.md`, E12, E19).

Một run khác `selfplay_20k_cpp_2w` có 20.000 game records, 40 iterations × 500 ván và 40.000 optimizer steps. Tuy nhiên checkpoint khởi đầu run này khác 96/96 tensor so với checkpoint AZ 60k và checkpoint cuối 20k; optimizer ban đầu rỗng, global step 0, metadata thiếu bootstrap/parent. Forensic report kết luận không xác nhận 20k bắt đầu từ AZ 60k; bằng chứng trực tiếp bác bỏ việc bắt đầu từ checkpoint 60k đã định danh. Bởi vậy **không cộng 20k vào 60k thành AZ 80k**, cũng không suy diễn mô hình khởi tạo ngẫu nhiên nếu không biết nguồn bên ngoài (`docs/az-80k-checkpoint-forensics.md`, §§1–2, lines 5–28).

Quyết định nghiên cứu là giữ các nhánh như artifacts/lineages riêng, không dùng tổng số games trên nhiều nhánh làm “độ dài huấn luyện” của một model. Báo cáo đề xuất có thể khai thác offline dữ liệu run 20k trong thí nghiệm mới, nhưng đó sẽ là dữ liệu off-policy và chưa có importer được xác minh; không phải continuation on-policy từ AZ 60k (`docs/az-80k-checkpoint-forensics.md`, §§4–5).

## 5.6. Renju: 500 ván, 1.500 ván và fine-tuning

### Bối cảnh, thiết kế và lineage

Fine-tuning Renju nhằm chuyển trọng số Normal sang bộ luật có forbidden moves nhưng giữ replay tách luật. Báo cáo ghi log ngày 2026-11-12 về run 500 ván (10 × 50), buffer mới; loss train tăng 3,16399 → 3,57626, value loss tăng 0,15362 → 0,45555, fixed-batch loss tăng 3,33481 → 3,45334. Nguồn checkpoint trong log chỉ là đường dẫn `checkpoints/last.pt`, không có hash nguồn, vì vậy không chứng minh được bằng mật mã artifact Normal chính xác được dùng (`docs/report.md`, §8.2–8.3, lines 150–176).

Audit Renju ngày 2026-11-23 đối chiếu checkpoint 500-game SHA `42bdd1c9…55870` với checkpoint 1.500-game SHA `5e2e45d5…af9b5`. Audit ghi nguồn của checkpoint 1.500-game là Normal generation `379bac…`, SHA `17ab2889…`; checkpoint 500-game thiếu source identity. Chưa đối chiếu đầy đủ manifest chain giữa các generation, nên không kết luận SHA được ghi là parent trực tiếp hay nguồn bootstrap ban đầu. Không có checkpoint 10.000-game trong worktree đã khảo sát; 1.500-game không được gọi là model 10k (`.audit/renju_diagnostic_20261123.md`, §1).

### Kết quả data và đánh giá

DB 1.500-game ghi Black thắng cả 1.500 ván, không có hòa. Replay có 14.308 mẫu; 7.904 mẫu Black-to-move có `z=+1`, 6.404 mẫu White-to-move có `z=-1`. Điều này xác nhận outcome distribution một phía, dù từng nhãn theo góc nhìn bên đi được ghi đúng theo audit. Loss/value loss nhỏ không chứng minh value head tổng quát khi tập outcome suy biến (`.audit/renju_diagnostic_20261123.md`, §2; `SPEC.md`, §§8.3, 17).

Screen đối kháng cân bằng 100 ván dùng bốn-ply randomized openings, 50 cặp đổi màu, 200 C++ MCTS simulations, không tactical override. Checkpoint 500-game thua Heuristic 100/100; ở mười tình huống White có đúng một block hợp lệ, checkpoint 500-game bỏ lỡ cả mười. Checkpoint 1.500-game chọn block cả 10/10 tình huống fixed này, với prior/visit tăng rõ. Tuy nhiên full tournament của checkpoint 1.500-game chỉ đạt 32 thắng, 68 thua, không hòa (score 32%); Black 13–37, White 19–31. Mục tiêu 70% không đạt. Đây là một screen cho mỗi checkpoint, không phải nhiều tournament độc lập; fixed-set tactical improvement không chứng minh strength tổng quát (`.audit/renju_diagnostic_20261123.md`, §§2–3, lines 13–32).

Không có đối chứng matched trước/sau fine-tuning với cùng luật, checkpoint source, seeds/openings và protocol. T35 cũng ghi Heuristic thắng AZ Renju 100/100. Do đó chưa thể kết luận fine-tuning gây ra cải thiện sức chơi tổng thể, dù có tín hiệu giải quyết một nhóm forced-block trên tập cố định. Quyết định vận hành được ghi là dừng mở rộng run chẩn đoán; outcome guard được thiết kế để không tiếp tục đốt games khi phân phối kết quả một phía. Tách riêng cải thiện chiến thuật cụ thể khỏi mục tiêu thắng Heuristic là cần thiết (`.audit/renju_diagnostic_20261123.md`, §§3–4; `docs/report.md`, §10.2; `SPEC.md`, §8.3).

## 5.7. Hiệu năng: CPU/GPU, batching, native và số worker

### Cấu hình nền và giả thuyết

Vấn đề hiệu năng được đặt ra khi inference/model không phải chi phí duy nhất: Python MCTS request/receive và overhead mỗi tick có thể hạn chế throughput. Cần đo end-to-end cùng checkpoint, luật, simulations và workload, không suy luận từ microbenchmark một nước. Mã hiện tại cho phép Python hoặc C++ native MCTS, đồng thời gom inference từ nhiều game; GPU chạy model, CPU điều khiển board/rules/search. Native extension không phải engine chơi/huấn luyện hoàn toàn bằng C++: Python vẫn sở hữu Board, Rules, features, RNG và inference (`SPEC.md`, §§4, 9.1; `docs/report.md`, §4).

### Benchmark Python/C++: workload cụ thể của hệ số 2,54×

Báo cáo ngày 2026-11-21 dùng Windows, RTX 4060 Laptop GPU, Python 3.10.6, PyTorch 2.14.1+cu126, NumPy 2.2.6; cùng `checkpoints/latest.pt`, Normal, 200 simulations, `parallel_games=16`, temperature 0, 8 games/run, ba seeds × hai repeats. Python đạt median 0,255 games/s (31,408 s), C++ đạt 0,648 games/s (12,353 s), tỷ lệ throughput **2,54×**. Hai backend ghi cùng 39.495 inference calls và mean batch 7,63; đây là hai chỉ dấu tổng hợp giống nhau, chưa chứng minh mọi request và phân bố batch đều giống nhau. C++ giảm non-inference time từ median 14,378 xuống 2,502 giây, đồng thời inference time giảm 16,829 xuống 9,851 giây; report quy mức giảm này vào toàn bộ execution path, không riêng tree-search. GPU utilization trên workload ngắn dao động 11–45% (`docs/cpp-real-world-benchmark.md`, §§1–4, 7–10, lines 1–75, 98–129).

Một profiling khác một ván, batch=1, cho inference chiếm 92–95% wall time; đây không mâu thuẫn với 53,3% (Python) và 79,7% (C++) trong workload batched nhiều ván. Batch inference thay đổi tỷ trọng và overhead. Phép đo tree-only riêng báo 464,23 ms ở Python và 274,34 ms ở C++, tương ứng 1,69×; các đơn vị và phạm vi đo khác với profiling một ván nên không so trực tiếp tỷ phần giữa chúng. Không có căn cứ áp dụng 2,54× cho mọi workload: số này chỉ mô tả self-play Normal ngắn, một GPU/máy và cấu hình cụ thể, không đo playing strength, thời gian train toàn iteration hay soak dài (`docs/cpp-real-world-benchmark.md`, §§4–8, 10).

### Benchmark vận hành: batching và worker count

Vòng benchmark vận hành thứ hai dùng `checkpoints/last.pt` SHA `6e29a4e3…ffbc`, RTX 4060, 200 simulations, warm-up và ba lần đo mỗi cấu hình; các lần đo tuần tự, xen kẽ backend. Ở matrix active-game count, sau tối ưu C++ đạt 27,13 games/min tại `parallel_games=4` và 60,31 tại `parallel_games=16`; Python lần lượt 16,72 và 22,60. C++ p16 dùng batch inference mean 8,60, peak RSS 1,08 GB và VRAM 1.131 MiB. Đây là một benchmark khác với 2,54×, checkpoint và quy trình đo khác (`docs/performance/11-production-benchmark.md`, §§1–2, 9–10, lines 3–24).

Số process workers cũng không tuyến tính. Với workload tổng cố định 16 games, C++ 1/2/4 workers đạt lần lượt 60,34 / 73,27 / 53,02 games/min; peak RSS 1,15 / 3,25 / 5,23 GB. Hai worker là nhanh nhất trong workload này, không phải vì hai worker luôn tối ưu mà vì game count tổng được giữ nguyên. Trong workload bão hòa 16 games cho mỗi worker (16/32/64 games tổng), C++ đạt 66,03 / 100,68 / 156,73 games/min, song bốn workers dùng peak RSS 5,60 GB. Các phép đo saturation chỉ một lần/config, nên không có độ lặp của matrix chính (`docs/performance/11-production-benchmark.md`, §§3–5, lines 35–61).

Phép gần nhất với một iteration train thực tế chạy đúng 50 games, 200 simulations, ba lần/config. `run_games_parallel()` chia 25 games/process với hai workers hoặc tối đa 13 games/process với bốn workers; một process dùng `parallel_games=16`. Median C++: một worker 86,51 games/min, hai worker 129,59 games/min, bốn worker 124,66 games/min. Python cùng cấu hình đạt 22,91 / 54,08 / 80,05 games/min. Hai workers cho throughput cao nhất ở batch 50 và RSS 3,36 GB, trong khi bốn workers dùng 5,42 GB. Warm-up/startup hai worker 5,31 giây; pool được tái sử dụng qua iteration (`docs/performance/11-production-benchmark.md`, §§6–7, lines 63–78).

Đây là workload cụ thể và nguồn trực tiếp cho khuyến nghị tạm thời hai worker khi gọi batch 50 trên máy đo, không phải khuyến nghị chung cho mọi cấu hình. `games/min` phụ thuộc độ dài game; CPU/GPU utilization là snapshot/sample chứ không phải profiler đầy đủ. Báo cáo ghi rõ không chạy full tournament round 2, controlled strength arena hoặc memory soak hàng giờ; không ngoại suy 38,6 phút cho 5k games như thời gian chạy đã chứng minh (`docs/performance/11-production-benchmark.md`, §8; `SPEC.md`, §11).

### Quyết định hiệu năng và điều chưa chứng minh

Kết quả ủng hộ việc giữ native MCTS như lựa chọn tăng throughput trong workloads đã đo; đồng thời batching nhiều games giúp amortize inference overhead. Nhưng C++ không chuyển toàn bộ game/rules/NN sang native; PyTorch inference vẫn là phần lớn wall time, nhất là khi chạy C++. Không giảm budget 200 simulations để tăng tốc, vì đó là biến chất lượng cần protocol nghiên cứu riêng (`SPEC.md`, §§7.1, 9.1–9.3; `docs/cpp-real-world-benchmark.md`, §§7–10).

Các báo cáo readiness ghi một paired 4-game self-play cho samples giống nhau ở seed cụ thể và smoke GPU train/resume nhỏ; đó là evidence về đường thực thi hẹp, không phải stochastic parity dài hạn hoặc sức chơi tương đương. Chưa chứng minh C++/Python parity trên nhiều seed/trajectories, chất lượng sau training kéo dài, GPU utilization ổn định, RAM soak hàng giờ, sức chơi qua arena matched, hay hiệu năng phần cứng khác. Các mục tiêu ấy vẫn là việc cần đánh giá chứ không phải kết quả đã hoàn thành (`SPEC.md`, §§9.2–9.3; `docs/performance/12-5000-games-readiness.md`).

## 5.8. Tổng hợp quyết định và giới hạn bằng chứng

| Giai đoạn | Giả thuyết/động lực | Quan sát quan trọng | Quyết định có thể bảo vệ |
|---|---|---|---|
| Normal 1.000 | Pipeline chạy và train objective thay đổi | Train loss giảm; fixed-batch loss tăng | Xác nhận run, không tuyên bố strength |
| T35 | Định lượng agent trước baseline | Heuristic thắng AZ v0 và AZ Renju 100/100 trong pool tương ứng | Chẩn đoán khoảng cách; giữ protocol và pool riêng |
| 25k epsilon pilot | Tăng khám phá có thể cải thiện dữ liệu/tactical | tín hiệu không ổn định theo seed; đối kháng quá nhỏ | ε=0,25 làm baseline, không promote ε=0,50 |
| Tactical wrapper | Quét thắng-ngay sửa lỗi MCTS | 6/6 canary cũ; 11/12 bằng baseline trên thế mới; 0/4 game cả hai arms | Không tích hợp vì chưa có gain holdout/strength |
| 40k–50k | Continuation thêm games tăng sức chơi | 50k hơn 40k trong paired protocol; chưa chứng minh hơn Heuristic | Ghi nhận gain có phạm vi, không tuyên bố đạt mục tiêu nghiên cứu |
| Value pilot 50k | Tăng value weight cải thiện dự báo/phòng thủ | MSE gate fail; tactical inconclusive; strength gates không đạt | Giữ parent 50k, không promote candidate |
| Renju 500/1.500 | Fine-tuning luật Renju chuyển giao hữu ích | 1.500-game sửa 10/10 forced-block cố định nhưng screen chỉ 32/100; data Black-win toàn bộ | Không kết luận fine-tuning tăng strength; dừng mở rộng không kiểm soát |
| Native/performance | C++/batching/workers tăng throughput | 2,54× trong workload 8 games/run; 129,59 games/min ở 50-game batch, 2 workers | Chỉ tuyên bố hiệu năng trên workload/máy đo; không suy ra quality |

Nhìn tổng thể, tiến trình cải tiến chuyển từ hoàn thiện pipeline, sang phát hiện lỗi chiến thuật và provenance, rồi sang kiểm định candidate bằng tournament và gate định trước. Kết quả tích cực có phạm vi hẹp: 50k thắng 40k trong protocol đã chạy; tactical block trên tập Renju cố định tốt hơn ở 1.500 so với 500 games; native C++ tăng throughput trong workloads đo. Các kết quả chưa đạt cũng quan trọng: T35 ghi Heuristic vượt AZ trong các pool đã công bố; 50k không chứng minh vượt Heuristic; value candidate không qua gate; Renju 1.500-game chưa đạt target; run 20k không thuộc lineage 60k; benchmark không phải kiểm chứng soak hay sức chơi.

Do đó, kết luận học thuật không nên quy một thay đổi đơn lẻ thành nguyên nhân của tiến bộ hoặc suy giảm nếu thiếu đối chứng matched, nhiều seed/opening, holdout độc lập và identity checkpoint đầy đủ. Các database hiện có còn thiếu Git commit và engine version trong một số run; một số lineage historical không khép kín. Mục tiêu “AlphaZero thắng Heuristic với bằng chứng đủ tin cậy” vẫn chưa được xác nhận theo tiêu chí của SPEC (`SPEC.md`, §§1–2, 10, 12; `report/bang_doi_chieu.md`, E19, dòng 32 về mốc Git log; `data/normal.sqlite`; `data/az_25k_continuation_v1.sqlite`).
