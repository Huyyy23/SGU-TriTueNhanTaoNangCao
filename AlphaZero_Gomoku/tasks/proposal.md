# Đề xuất dự án: Gomoku 11x11 với agent AlphaZero tự học

**Người thực hiện:** Nhóm 3 người — A: Khang, B: Huy, C: Hường
**Lĩnh vực:** Trí tuệ nhân tạo, học tăng cường, tìm kiếm cây
**Trạng thái:** Bản đề xuất để thảo luận

---

## 1. Tóm tắt

Dự án xây dựng một bản game Gomoku trên bàn 11x11 chơi được qua giao diện đồ họa, kèm một bộ tác tử nhiều cấp độ để so sánh công bằng. Ở cấp thấp nhất là tác tử ngẫu nhiên, tác tử luật/heuristic, Minimax có cắt tỉa Alpha-Beta và MCTS thuần. Ở cấp cao nhất là agent tự quyết định nước đi bằng self-play: mạng ResNet nhỏ làm chính sách và đánh giá vị trí, tìm kiếm cây bằng PUCT với đánh giá theo lô, tự chơi 1000 ván rồi học từ chính dữ liệu đó. Hệ thống hỗ trợ hai bộ luật: Normal (5 quân liền nhau thắng) và Renju (đen bị hạn chế 3-3, 4-4, quá 5 quân).

Agent `az_v0` không dùng bảng tra cứu nước đi và không hardcode chiến thuật: mọi quyết định của nó đến từ mô hình đã huấn luyện kết hợp tìm kiếm cây. Tác tử heuristic là một tác tử riêng ở cấp 2, không phải thành phần của `az_v0`. Biến thể `az_v1` dùng thêm tri thức chiến thuật và được ghi rõ là biến thể thực nghiệm trong báo cáo.

## 2. Bối cảnh và động lực

Gomoku là bài toán thí nghiệm thuận tiện cho AlphaZero: trạng thái là mảng rời rạc kích thước cố định, không cần xử lý ảnh hay ngôn ngữ, nên toàn bộ công sức kỹ thuật dồn vào thuật toán tìm kiếm và huấn luyện. Ở khổ 9x9 và 11x11, số nước mỗi ván tăng rõ rệt, thuật toán phải hiệu quả về tính toán mới đủ tự chơi trong phạm vi tài nguyên của nhóm.

Biến thể Renju bổ sung điều kiện khó cho agent: mỗi nước đi phải vượt qua kiểm tra luật trước khi được đánh giá, và nước bị cấm phải bị loại khỏi cây tìm kiếm. Bài toán trở thành: một agent chơi được Gomoku 11x11 trên hai bộ luật khác nhau, mà không cần bản dịch thủ công hay bảng tra cứu nước đi.

Dự án gồm hai phần: sản phẩm chơi được với agent đủ mạnh, và hệ thống đo lường được để kiểm chứng chất lượng agent từ gốc đến ngọn (mạng, tìm kiếm, tự chơi, đánh giá).

## 3. Mục tiêu

### 3.1 Mục tiêu sản phẩm

| # | Mục tiêu | Tiêu chí nghiệm thu |
|---|---|---|
| M1 | Game Gomoku 11x11 chơi được | Người chơi đấu với máy hoặc hai máy đấu nhau qua giao diện đồ họa; hiển thị nước cuối, ván thắng, hòa |
| M2 | `az_v0` tự ra quyết định | Không bảng tra cứu nước đi, không heuristic hardcode: mọi nước đi của `az_v0` đến từ mô hình + tìm kiếm cây |
| M3 | Hai bộ luật Normal và Renju | Một kiến trúc mạng, hai bộ luật, agent chơi được cả hai |
| M4 | Đo lường được | Round-robin có log đầy đủ, chia đều màu, score kèm khoảng tin cậy Wilson; ngân sách ván tăng được bằng cấu hình mà không phải sửa mã |
| M5 | Đủ bốn tác tử cổ điển cấp 1–4 | Random, heuristic, Alpha-Beta, MCTS-rollout; tất cả đi qua đúng một interface luật |
| M6 | Demo chọn được cấp độ | UI chọn tác tử theo tên, chơi người–máy và máy–máy |
| M7 | Câu hỏi nghiên cứu trả lời được bằng số | So sánh `az_v0` với `az_v1` ở cùng 200 lượt mô phỏng, cộng hai ablation, có khoảng tin cậy |

### 3.2 Câu hỏi nghiên cứu

**Câu hỏi chính:**

> **Trong cùng ngân sách 200 lượt mô phỏng mỗi nước, việc bổ sung tri thức chiến thuật Gomoku (bộ nhận diện mẫu, tactical override, threat-space search) có làm tăng score của AlphaZero so với chính nó khi không có các thành phần đó hay không?**

Ba điều kiện để câu hỏi này trả lời được: `az_v1` dùng đúng trọng số của `az_v0`; hai bên chạy cùng số lượt mô phỏng; mỗi mốc cải tiến bật thêm đúng một thành phần. Chi tiết ở `tasks/Lo_trinh_cai_tien_AlphaZero_Gomoku.md`.

**Câu hỏi phụ (trả lời bằng số liệu đã có sẵn):**

> **Với ngân sách 1000 ván tự chơi, `az_v0` đạt trình độ nào so với bốn tác tử cổ điển, và loss trên tập cố định biến đổi ra sao theo từng vòng?**

Số liệu cho câu hỏi phụ đã tồn tại trong `logs/train_20261028-063106.jsonl`; phần còn thiếu là score của `az_v0` trước các tác tử cổ điển, do round-robin cung cấp.

Ba hướng mở rộng sau khi đạt mục tiêu: so sánh một mô hình dùng chung cho Normal và Renju với hai mô hình riêng; mở rộng ngân sách tự chơi lên 1200, 1500 hay 5000 ván; và chạy lại `az_v1` với self-play có hướng dẫn heuristic.

## 4. Phạm vi

**Trong phạm vi**

- Bàn 11x11, hai bộ luật Normal và Renju.
- Lõi game tách riêng khỏi tầng AI, mọi thành phần đi qua một giao diện luật duy nhất.
- API game thống nhất bảy phương thức: `reset`, `legal_actions`, `step`, `is_terminal`, `result`, `render`, `clone`.
- Bốn tác tử cổ điển cấp 1–4: random, heuristic, Alpha-Beta, MCTS-rollout.
- Agent AlphaZero: mạng ResNet, tìm kiếm PUCT, tự chơi, huấn luyện (`az_v0`); fine-tune cho Renju (`az_renju`).
- Biến thể `az_v1`: `az_v0` cộng tactical override và threat-space search, **dùng chung trọng số**.
- Giao diện đồ họa cho người chơi, chọn được tác tử theo cấp.
- Round-robin giữa các tác tử, hai ablation bắt buộc, nhật kí thí nghiệm và báo cáo.

**Ngoài phạm vi**

- Bản 9x9, bản 13x13 hoặc bàn không vuông.
- Tác tử cấp 5 (học có giám sát từ chuyên gia): cấp 6 đã thoả yêu cầu "cấp 5 hoặc 6".
- PPO, A2C, DQN, prioritized replay, curriculum learning.
- Hạ tầng chạy self-play phân tán: Colab, shard, gộp shard, đồng bộ Drive. Ngân sách hiện tại chạy xong trong ~1 giờ trên một GPU nên không cần.
- Xử lý ảnh, nhận dạng nét viết.
- Bản phân phối cho người dùng cuối, đăng ký người chơi, cơ sở dữ liệu người chơi.
- Huấn luyện mô hình ngôn ngữ hay đa phương tiện.

## 5. Phương pháp

### 5.1 Biểu diễn trạng thái

Mỗi vị trí là tensor 4 kênh kích thước 11x11: quân của bên đang đi, quân đối phương, mask nước bị cấm theo luật, và màu quân của bên đang đi. Số kênh cố định ở cả hai bộ luật nên một kiến trúc mạng dùng chung, không cần sửa lại mô hình khi đổi luật.

Mô hình Renju (`az_renju`) được **fine-tune từ checkpoint của Normal**, không train từ đầu: `AGENTS.md` rule 7 cho phép đường này, và fine-tune giữ được tri thức hình học bàn cờ mà chỉ cần ~1,5 giờ GPU. Buffer vẫn tách riêng theo bộ luật.

### 5.2 Mô hình

ResNet 6 block, 64 filter. Đầu chính sách phát ra 121 giá trị (một giá trị cho mỗi ô). Đầu giá trị phát ra 1 số trong khoảng [-1, 1]. Hàm mất mát: cross entropy trên chính sách (hệ số 1.0), sai số bình phương trên giá trị (hệ số 1.0), cộng chuẩn hoá L2 trọng số (hệ số 1e-4). Tối ưu bằng Adam, tốc độ học 1e-3, giảm dần theo hàm cosin theo số bước đã huấn luyện.

### 5.3 Tìm kiếm cây

Tìm kiếm theo PUCT với hệ số khám phá c_puct = 1.5, chạy 200 lượt mô phỏng cho mỗi nước đi. Nhiễu Dirichlet (alpha = 0.15, epsilon = 0.25) áp dụng ở gốc cây. Nhiệt độ = 1 trong 15 nước đầu, sau đó giảm dần về 0 để chọn nước tốt nhất.

Cách đánh giá vị trí đúng theo AlphaZero: mỗi lượt mô phỏng đi từ gốc xuống một lá, lá đó được đánh giá bằng đúng một lượt suy luận mạng. Lượt suy luận đó trả về hai thứ cùng lúc: đầu chính sách cho xác suất của cả 121 ô, dùng làm tiên đoán cho mọi nước đi có thể đi từ lá; đầu giá trị cho giá trị của chính lá đó. Giá trị này là góc nhìn của đối thủ nên phủ định khi truyền ngược lên gốc. Như vậy một lượt suy luận phục vụ toàn bộ nhánh con của lá, không cần suy luận riêng cho từng nước đi.

Về hiệu năng: để tận dụng GPU, các lượt suy luận được gom lại thành lô thật. Cụ thể là gom nhiều lá đang chờ đánh giá, đến từ nhiều ván self-play chạy song song, rồi gửi một lệnh suy luận duy nhất. Đây là lô ở cấp chương trình, không phải lô ở cấp cây: không gom các nước đi con của một node để suy luận riêng cho từng con, vì cách đó tốn gấp nhiều lần số lượt tính toán mạng và không phải cách làm chuẩn của AlphaZero.

### 5.4 Tự chơi và huấn luyện

Ngân sách mặc định: 20 vòng, mỗi vòng 50 ván, tổng 1000 ván cho mỗi bộ luật. Ngân sách nằm trong tệp cấu hình (số vòng × số ván mỗi vòng) nên tăng lên 1200, 1500 hay 5000 ván chỉ cần đổi cấu hình, không cần sửa mã. Mỗi vòng dùng 200 bước huấn luyện với batch 128 mẫu, lấy mẫu ngẫu nhiên từ buffer tối đa 50000 mẫu. Mỗi nước đi lưu lại trạng thái, phân phối xác suất mục tiêu từ cây tìm kiếm, và giá trị z tính sau khi ván kết thúc theo góc nhìn người đang đi. Dữ liệu được tăng cường bằng 8 phép đối xứng của bàn cờ, áp dụng đồng thời lên trạng thái và chính sách. Buffer tách riêng cho từng bộ luật, không trộn dữ liệu Normal với Renju. Sau mỗi vòng, mô hình mới được đấu với mô hình tốt nhất đang có, và chỉ thay thế mô hình tốt nhất khi tỉ lệ thắng đạt từ 55% trở lên.

### 5.5 Siêu tham số

| Nhóm | Tham số | Giá trị |
|---|---|---|
| Tìm kiếm | số lượt mô phỏng mỗi nước | 200 |
| Tìm kiếm | hệ số khám phá c_puct | 1.5 |
| Tìm kiếm | nhiễu Dirichlet alpha / epsilon | 0.15 / 0.25 |
| Tìm kiếm | số nước dùng nhiệt độ = 1 | 15 |
| Mô hình | số block / số filter | 6 / 64 |
| Mô hình | tối ưu hoá | Adam |
| Mô hình | tốc độ học | 1e-3, giảm dần theo cosin |
| Mô hình | hệ số chuẩn hoá L2 | 1e-4 |
| Mô hình | trọng số mất mát chính sách / giá trị | 1.0 / 1.0 |
| Huấn luyện | số bước mỗi vòng | 200 |
| Huấn luyện | batch size | 128 |
| Huấn luyện | dung lượng buffer (trước tăng cường) | 50000 |
| Ngân sách | số vòng × số ván mỗi vòng | 20 × 50 = 1000 ván mỗi bộ luật |
| Đánh giá | ngưỡng thay thế mô hình tốt nhất | tỉ lệ thắng ≥ 55% |
| Đánh giá | số ván mỗi cặp trong round-robin | 100, xen kẽ lượt đầu |

### 5.6 Nhật kí và đánh giá

Mọi ván đều ghi vào tệp JSONL, mỗi ván một dòng, gồm: mã ván, bộ luật, vòng huấn luyện, bên đi trước, bên đi sau, người thắng, nguyên nhân kết thúc ván (thắng 5 quân / bị cấm nước / hòa / hết giờ), số nước, thời gian, số lượt gọi mạng. Ghi từng nước (giá trị ước lượng trước nước đi, vài xác suất hành động cao nhất, số lượt mô phỏng) là tùy chọn, bật khi chạy thí nghiệm cần phân tích.

Bộ đánh giá chạy 1000 ván giữa agent và đối thủ (ngẫu nhiên, MCTS thuần, hoặc mô hình cũ), xen kẽ lượt đầu: ván chẵn bên A đi trước, ván lẻ bên B đi trước, cho đúng 500 ván mỗi bên. Tổng hợp lưu thành summary.json: tỉ lệ thắng khi đi trước, tỉ lệ thắng khi đi sau, tổng thắng/hòa/thua, phân bố nguyên nhân kết thúc ván. Số ván đánh giá là tham số cấu hình, mặc định 1000.

Với đồ án cuối kỳ, phần so sánh nhiều tác tử dùng `scripts/tournament.py` (T31) chạy round-robin một lần cho cả ma trận, thay vì gọi script đấu từng cặp nhiều lần; cách chia đều lượt đi và định dạng log giữ nguyên.

## 6. Thiết kế thí nghiệm

| Nhóm thí nghiệm | Nội dung | Số ván | Đầu ra |
|---|---|---|---|
| E1 | Round-robin 6 tác tử Normal (`random`, `heuristic`, `alphabeta`, `mcts_pure`, `az_v0`, `az_v1`), ngân sách cố định 200 sims / depth 4 | 1.500 (100/cặp) | ma trận score + Wilson CI, tách theo màu |
| E2 | Chế độ ngân sách thời gian 1 s/nước cho `az_v0`, `az_v1`, `alphabeta` | 300 | score trong ngân sách thời gian thật |
| E3 | Đường cong học của `az_v0` qua 20 vòng (lấy từ log đã có) | 1.000 (đã chạy) | `loss_eval_on_fixed_batch` theo vòng; đối chiếu score cuối |
| E4 | Thí nghiệm chính của câu hỏi nghiên cứu: `az_v0` vs `az_v1` (cùng trọng số, cùng sims) | 100 mỗi mốc | score + khoảng tin cậy, số lần tactical override |
| E5 | Ablation: Alpha-Beta bật/tắt cắt tỉa; quét sims `mcts_pure` 50/200/800 | theo cấu hình | bảng node/thời gian/score |
| E6 | Renju: round-robin `random`, `heuristic`, `alphabeta`, `az_renju` | 600 (100/cặp) | score, phân bố `forbidden_move` |

E4 và E5 là hai phần bắt buộc của đề bài: ít nhất hai ablation/độ nhạy có ý nghĩa. E6 là bằng chứng tính tổng quát (điểm cộng), không phải nơi trả lời câu hỏi nghiên cứu.

Số ván của mỗi nhóm là tham số cấu hình, nên có thể tăng gấp đôi để giảm sai số thống kê mà không sửa mã.

Chỉ số theo dõi: score = (thắng + 0,5 × hoà)/tổng, khoảng tin cậy Wilson, score tách theo màu quân, phân bố `end_reason`, thời gian trung bình/nước, độ trễ p95, số node và số lượt gọi mạng mỗi ván, `loss_eval_on_fixed_batch` trong huấn luyện.

Về lợi thế lượt đầu: trong Gomoku, người đi trước thường thắng nhiều hơn. Báo cáo cần nêu điều này trước khi trình bày số liệu, để tỉ lệ thắng cao khi đi trước không bị hiểu nhầm là agent mạnh hơn đối thủ.

## 7. Kế hoạch triển khai

### 7.1 Các giai đoạn

Ba tuần, ba người. Mỗi tuần có một cổng chặn: không qua cổng thì cắt phạm vi, không dời hạn.

| Tuần | Nội dung | Task | Cổng cuối tuần |
|---|---|---|---|
| 1 | Tác tử cấp 1–4, bộ nhận diện mẫu, API game, arena | T27, T6, T7, T28, T33, T36, T13 | Bốn tác tử cổ điển chơi được cả Normal và Renju; `heuristic` thắng `random` ≥ 90%; arena in được score của `az_v0` |
| 2 | Tactical override, threat-space, Renju, UI, thống kê giải đấu | T29, T30, T24, T25, T23, T31, T32 | `az_v1` chạy được với cùng trọng số `az_v0`; luật Renju qua test 3-3/4-4/overline; `az_renju` fine-tune xong; round-robin chạy được |
| 3 | Chạy giải đấu, ablation, đóng gói | T35, T22, T26, T34 | 2.400 ván xong và số liệu đóng băng; báo cáo, slide, poster, ZIP đủ bộ |

### 7.2 Phân chia công việc

Ba người, mỗi người một nhánh code, sở hữu file của nhánh mình: **A = Khang, B = Huy, C = Hường.**
Bảng task đầy đủ nằm ở `tasks/todo.md`.

**A (Khang) — Lõi game, luật, tác tử cổ điển, giao diện.** 11 task (T3–T7, T23, T24, T27, T28, T33, T36); T3–T5 đã xong, còn 8 task, ước ~9 ngày công.

- **Mục tiêu:** bàn cờ và luật chơi đúng, bốn tác tử cổ điển cắm được vào bộ đánh giá, game chơi được bằng chuột.
- **Task và file sở hữu:** T3 `gomoku/board.py`; T4 `gomoku/rules/base.py` + `normal.py`; T5 `gomoku/features.py`; T6 `players/base.py|random.py|human.py` + `gomoku/runner.py` + `scripts/play.py`; T7 `players/mcts_pure.py`; T27 `gomoku/patterns.py`; T28 `players/heuristic.py`; T33 `players/alphabeta.py`; T36 `gomoku/game.py`; T23 `ui/pygame_app.py`; T24 `gomoku/rules/renju.py`.
- **Sản phẩm bàn giao:** `pytest` pass toàn bộ test luật, gồm 3-3, 4-4, quá 5 quân và ngoại lệ đúng 5 quân; `gomoku/game.py` có đủ bảy phương thức theo đề; `heuristic` thắng `random` ≥ 90%; `mcts_pure` thắng `random` > 60%; Alpha-Beta có cờ bật/tắt cắt tỉa cho cùng nước đi; bấm chuột đánh được và ván kết thúc hiện kết quả.
- **Bàn giao cho:** B nhận giao diện luật, đặc trưng và `gomoku/patterns.py`; C nhận các tác tử để dựng round-robin.
- **Không đụng:** `az/`, `scripts/train.py`, `scripts/tournament.py`, `configs/`.

**B (Huy) — Tầng AlphaZero và các cải tiến.** 11 task (T8–T12, T19, T20, T25, T29, T30, T35); T8–T12, T19, T20 đã xong, còn 4 task, ước ~7 ngày công cộng ~6 giờ máy.

- **Mục tiêu:** agent tự học từ dữ liệu tự chơi, không heuristic hardcode; biến thể chiến thuật dùng chung trọng số.
- **Task và file sở hữu:** T8 `az/model.py`; T9 `az/mcts.py`; T10 `az/replay_buffer.py`; T11 `az/selfplay.py`; T12 `az/trainer.py`; T19 `az/game_log.py`; T20 chạy train; T25 tích hợp Renju + fine-tune `az_renju`; T29 `az/player_tactical.py`; T30 `az/threat_space.py`; T35 chạy 2.400 ván.
- **Sản phẩm bàn giao:** `az_v0` có checkpoint và log đầy đủ (đã xong); `az_v1` với cờ `tactical.enabled=false` cho kết quả trùng `az_v0` cùng seed; `az_renju` fine-tune từ `last.pt`; log giải đấu đầy đủ.
- **Bàn giao cho:** C nhận checkpoint, log và số liệu thô để viết Chương 4.
- **Không đụng:** `gomoku/rules/`, `ui/`, `scripts/tournament.py`.

**C (Hường) — Hạ tầng, đánh giá, tài liệu.** 8 task (T1, T2, T13, T22, T26, T31, T32, T34), ước ~8 ngày công.

- **Mục tiêu:** mọi con số cuối cùng đo được, tra được, và tái lập được trên máy khác.
- **Task và file sở hữu:** T1 khung dự án; T2 `config.py` + `configs/`; T13 `az/evaluator.py` + `scripts/arena.py`; T31 `scripts/tournament.py` (round-robin, Wilson CI, p95); T32 `scripts/ablation.py`; T22 `docs/provenance.md` + `docs/README.md`; T26 `docs/report.md`; T34 poster, slide, ZIP.
- **Sản phẩm bàn giao:** round-robin chạy đủ ma trận với chia đều màu; hai ablation có bảng và hình; báo cáo có đủ số, mọi số tra được từ log; ZIP đúng quy cách tên.
- **Bàn giao cho:** giảng viên nhận báo cáo, slide, poster, ZIP.
- **Không đụng:** `az/model.py`, `az/mcts.py`, `az/trainer.py`, `gomoku/rules/`, `ui/`.

**Đã cắt khỏi phạm vi:** T14–T18 (Colab, `sync_drive`, selfplay shard, merge shard) và `scripts/simulate.py` (T21, gộp vào T31). Lý do: 1000 ván tự chơi chạy xong trong ~1 giờ trên một GPU tại chỗ, fine-tune Renju ~1,5 giờ, nên hạ tầng chạy phân tán không cần cho bất kỳ thí nghiệm nào trong kế hoạch.

**Mốc chặn giữa các nhánh**

| Mốc | Xong khi | Mở khoá cho |
|---|---|---|
| M1 | A xong T27: `gomoku/patterns.py` qua test | T24 (luật Renju) và T29, T30 |
| M2 | A xong T36 + T6: API game và bộ chạy ván | C dựng round-robin trên interface thống nhất |
| M3 | A xong T24: `renju.py` qua test | B tích hợp và fine-tune `az_renju` |
| M4 | C xong T31: round-robin chạy được | B chạy T35 |
| M5 | B xong T29 + T30 | C chạy ablation và đóng băng số liệu |

Quy ước phối hợp:

- Mỗi người sở hữu các file của nhánh mình. Task đụng nhiều file do người sở hữu file dẫn dắt, người khác gửi yêu cầu thay đổi.
- A và B đóng góp số liệu và log ván cho C viết báo cáo. C phụ trách Chương 4 (số liệu), B phụ trách Chương 2–3 (lý thuyết, phương pháp), A phụ trách Chương 3 phần engine và luật.
- Không ai sửa test hoặc cấu hình sau mốc đóng băng số liệu (Checkpoint N) để đổi kết quả.

## 8. Tài nguyên và rủi ro

**Tài nguyên:** môi trường huấn luyện chạy trên máy cá nhân có GPU, không phụ thuộc dịch vụ trả phí. Chi phí phần mềm bằng không: mọi thư viện dùng đều mã nguồn mở.

**Rủi ro chính**

| Rủi ro | Mức độ | Xử lý |
|---|---|---|
| Sai dấu giá trị khi truyền ngược trong tìm kiếm | Cao | Kiểm thử riêng cho bước truyền ngược, kiểm tra dấu ở mọi vòng lặp |
| Tìm kiếm chậm, không kịp 1000 ván | Cao | Đánh giá theo lô, đo số lượt gọi mạng mỗi ván, giảm số lượt mô phỏng nếu cần |
| Luật Renju tính nước cấm tốn thời gian | Trung bình | Tính một lần mỗi nước rồi lưu kết quả theo vị trí, vô hiệu hóa khi có nước mới |
| 11x11 khó, agent học chậm | Trung bình | Chạy thử 2 vòng trước khi chạy dài, theo dõi loss và win rate từng vòng |
| Lợi thế lượt đầu làm sai lệch đánh giá | Trung bình | Đánh giá luôn xen kẽ lượt đầu, báo cáo tách riêng hai chỉ số |
| Buffer quá lớn khiến dữ liệu cũ chi phối | Thấp | Giới hạn 50000 mẫu trước khi tăng cường |
| Máy cá nhân không chạy đủ lâu | Thấp | Lưu checkpoint sau mỗi vòng, nối tiếp được từ checkpoint |
| `az_v0` thua Alpha-Beta độ sâu 4 và thua heuristic | Cao | Đã lường trước: 1000 ván là ngân sách rất nhỏ so với AlphaZero gốc. Báo cáo nêu rõ cấp không tương ứng sức mạnh; không đổi luật hay giấu kết quả |
| Tactical override che lấp điểm yếu của mạng | Trung bình | Log tỉ lệ nước bị ghi đè; nếu > 30% số nước thì ghi rõ sức mạnh đến từ luật tay chứ không từ mạng |
| Bộ nhận diện mẫu sai làm hỏng cả luật Renju | Cao | T27 phải qua test trước T24 và T29; test cả hai màu và cả bốn hướng |
| Kết luận từ quá ít ván | Trung bình | Không kết luận dưới 100 ván/cặp; luôn báo khoảng tin cậy Wilson |

## 9. Tài liệu tham khảo

1. Silver, D. và cộng sự (2017). Mastering the game of Go without human knowledge. *Nature*, 550, 354-359.
2. Silver, D. và cộng sự (2018). Mastering the game of Chess, Shogi and Go through self-play. *Nature*, 537, 433-446.
3. Schrittwieser, J. và cộng sự (2020). Mastering Atari, Chess, Shogi and Go by planning with a learned model. *Nature*, 588, 683-689.
4. Ye, H. và cộng sự (2021). Mastering Atari games with a minimalist reinforcement learning framework. *Nature*, 596, 301-307.
5. Hoang, D. và cộng sự (2018). Is AlphaZero a general solution to game AI? *arXiv:1712.01815*.
6. Luật Renju: bộ luật của IFAA, và mục từ "Renju" trong Wikipedia.
7. Quan hệ đối xứng trong bài toán bàn cờ: Bussgang, J. và cộng sự (2020). The surprising creativity of digital evolution.

## 10. Phụ lục A: Truy xuất nguồn mã nguồn

Dự án lập bảng truy xuất cho từng module: đường dẫn, nguồn gốc (tự viết theo đặc tả hay lấy từ tài liệu hay kho mã nguồn), mục đích sử dụng, giấy phép. Bảng này hoàn thiện trong giai đoạn 8 và được kiểm thử tự động để không sót module nào.

| Thành phần | Nguồn | Dùng để làm gì | Giấy phép |
|---|---|---|---|
| `gomoku/board.py` | Tự viết theo đặc tả | Trạng thái bàn cờ 11x11, đặt/undo nước | Nội bộ |
| `gomoku/rules/` | Tự viết theo đặc tả | Giao diện luật, luật Normal, luật Renju theo bộ luật IFAA | Nội bộ |
| `gomoku/features.py` | Tự viết theo đặc tả | Mã hóa trạng thái thành 4 kênh, 8 phép đối xứng | Nội bộ |
| `az/model.py` | Tự viết theo kiến trúc ResNet chuẩn | Mạng chính sách và giá trị | Nội bộ |
| `az/mcts.py` | Tự viết theo thuật toán PUCT của AlphaZero | Tìm kiếm cây, đánh giá theo lô, nhiễu Dirichlet | Nội bộ |
| `az/selfplay.py`, `az/replay_buffer.py`, `az/trainer.py` | Tự viết theo đặc tả | Sinh ván, lưu dữ liệu, huấn luyện | Nội bộ |
| `az/evaluator.py` | Tự viết theo đặc tả | Đánh giá mô hình, cổng nâng cấp | Nội bộ |
| `players/` | Tự viết theo đặc tả | Người chơi ngẫu nhiên, người chơi người, MCTS thuần, agent AlphaZero | Nội bộ |
| `ui/` | Tự viết theo đặc tả | Giao diện đồ họa pygame | Nội bộ |
| `scripts/` | Tự viết theo đặc tả | Huấn luyện, chơi, đánh giá, mô phỏng | Nội bộ |
| Thư viện: torch, numpy, pygame, pyyaml, pytest | Gói mã nguồn mở | Học sâu, tính toán, giao diện, cấu hình, kiểm thử | Theo giấy phép từng gói |

## 11. Phụ lục B: Sản phẩm bàn giao

| Sản phẩm | Mô tả |
|---|---|
| Mã nguồn | Kho Git, có kiểm thử tự động, tài liệu hướng dẫn cài đặt và chạy |
| Mô hình đã huấn luyện | Checkpoint Normal (`az_v0`) và Renju (`az_renju`), kèm log huấn luyện |
| Bộ công cụ đánh giá | `scripts/tournament.py` round-robin có ma trận score và khoảng tin cậy; `scripts/ablation.py` cho hai ablation; log ván thô |
| Giao diện đồ họa | Chơi được với tác tử, chọn được cấp độ, chạy máy–máy để quay demo |
| Báo cáo | Mô tả phương pháp, kết quả thực nghiệm, hai ablation, phân tích nguyên nhân thua |
| Slide, poster, ZIP | Slide thuyết trình, poster tóm tắt, ZIP đúng quy cách `NhomXX_Gomoku_TTNTNC_<HoTen>_<MaSV>.zip`, kèm bảng phân công và tỉ lệ đóng góp |
