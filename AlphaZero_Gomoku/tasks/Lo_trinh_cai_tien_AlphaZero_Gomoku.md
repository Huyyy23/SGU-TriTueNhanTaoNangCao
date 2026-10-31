# Lộ trình cải tiến AlphaZero Gomoku 11x11

Tài liệu này chỉ nói về **phần cải tiến** (`az_v0` → `az_v1`). Phần bắt buộc của đồ án
(tác tử cấp 1–4, engine, giải đấu, báo cáo) nằm ở `tasks/proposal.md` và `tasks/todo.md`.

## 0. Trạng thái xuất phát (số đo thật, không phải giả định)

- `az_v0` đã chạy xong: 20 vòng × 50 ván = 1000 ván tự chơi trên luật Normal.
- Nguồn: `logs/train_20261028-063106.jsonl`, `checkpoints/last.pt`, `checkpoints/buffer_last.npz`.
- Máy: RTX 4060 Laptop. Tự chơi 200 lượt mô phỏng/nước: **~170 s/vòng**, train **~2,5 s/vòng**,
  tổng **~1 giờ** cho 1000 ván. Trung bình **23 nước/ván**, **~4.500 lượt suy luận/ván**.
- `loss_eval_on_fixed_batch` đi ngang ở 5,13–5,14 sau vòng 17. `policy_loss` ~3,77 so với
  ln(121) = 4,79, tức phân phối chính sách chỉ còn hơi nhỉnh hơn đều.
- **Chưa có** bất kỳ số đo sức mạnh thực chiến nào của `az_v0` (chưa có arena, chưa có
  `players/random.py`, `players/heuristic.py`, `players/alphabeta.py`, `players/mcts_pure.py`).
- **Kết luận phải chấp nhận trước khi làm:** với 1000 ván, `az_v0` rất có thể thua tác tử
  heuristic và Alpha-Beta. Đó không phải lỗi của đồ án; đề bài ghi rõ cấp không tương ứng
  sức mạnh. Nhưng kế hoạch phải tính trước kịch bản này, không phát hiện vào tuần 3.

## 1. Câu hỏi nghiên cứu của phần cải tiến

> **Trong cùng ngân sách 200 lượt mô phỏng mỗi nước, việc bổ sung tri thức chiến thuật
> Gomoku (bộ nhận diện mẫu, tactical override, threat-space search) có làm tăng score của
> AlphaZero so với chính nó khi không có các thành phần đó hay không?**

Ba hệ quả bắt buộc của câu hỏi này:

1. **Cùng trọng số.** `az_v1` dùng đúng `checkpoints/last.pt` của `az_v0`. Không train lại.
   Nếu train lại thì hai biến thay đổi cùng lúc và không trả lời được câu hỏi.
2. **Cùng số lượt mô phỏng.** Chênh lệch thời gian được báo cáo như một kết quả, không được
   bù bằng cách tăng sims cho bên yếu hơn.
3. **Một biến mỗi lần.** Bộ nhận diện mẫu là nền chung; bật tactical override và threat-space
   thành hai mốc riêng, mỗi mốc đo lại.

## 2. Trong và ngoài phạm vi

**Trong phạm vi**

| Giai đoạn | Nội dung | Task |
|---|---|---|
| 2 | Bộ nhận diện mẫu + tactical override (`az_v1`) | T27, T29 |
| 3 | Threat-space search | T30 |
| 4 | Đo sức mạnh và chi phí của từng mốc | T31, T35 |

**Ngoài phạm vi — cắt có lý do, không phải quên**

| Kỹ thuật | Lý do cắt |
|---|---|
| PPO / A2C / DQN | Không phải thành phần của AlphaZero; cần thiết kế lại pipeline huấn luyện, ~3–4 ngày, không phục vụ câu hỏi nghiên cứu |
| Prioritized Experience Replay | Cần định nghĩa priority cho dữ liệu AlphaZero; lợi ích chỉ thấy sau hàng nghìn ván, ngân sách hiện tại không đủ để đo |
| Curriculum Learning | Cần đối thủ có độ khó xếp hạng; chưa có tác tử nào mạnh hơn `az_v0` để làm mốc |
| Train lại `az_v1` từ self-play có hướng dẫn heuristic | Trộn thay đổi tầng tìm kiếm với phân phối dữ liệu; chỉ xét nếu tuần 3 còn dư, và khi đó ghi là hướng phát triển |
| Nhân bản chuỗi cải tiến sang Renju | Renju là bằng chứng tính tổng quát, không phải nơi trả lời câu hỏi nghiên cứu |

## 3. Giai đoạn 2 — Bộ nhận diện mẫu và tactical override

**Mục tiêu:** `az_v1` nhận ra nước thắng ngay và nước bắt buộc phải chặn, thứ mà
`az_v0` với 200 lượt mô phỏng bỏ sót.

### 3.1 Bộ nhận diện mẫu (T27, người A)

Một module duy nhất `gomoku/patterns.py`, thuần numpy, không import torch. Dùng chung cho
ba nơi: luật Renju (phát hiện 3-3), tác tử heuristic cấp 2, và tầng chiến thuật của `az_v1`.

Mẫu cần nhận diện cho một màu tại một ô, theo cả 4 hướng:

- năm liền (thắng), overline (6+)
- tứ mở (hai đầu trống), tứ bị chặn một đầu
- tam mở, tam bị chặn, tam gãy (broken three)
- **không** nhận diện: các mẫu từ 3 hướng trở lên cùng lúc (đó là việc của luật Renju, T24)

Tiêu chí nghiệm thu nằm ở `tasks/todo.md` T27. Điều kiện quan trọng nhất: phải có test cho
cả hai màu và cả 4 hướng, vì đây là module mà 4 nơi cùng phụ thuộc.

### 3.2 Tactical override (T29, người B)

Thứ tự kiểm tra, dừng ở điều kiện đầu tiên đúng:

1. Có nước tạo năm liền ngay → đánh nước đó.
2. Đối thủ có nước tạo năm liền ngay ở nước kế tiếp → chặn.
3. Không có gì → trả quyền chọn nước cho tầng tìm kiếm.

Ràng buộc:

- Chỉ áp cho nước **hợp lệ** theo `rules.legal_moves`; nước cấm của Renju không bao giờ
  được chọn, kể cả khi nó tạo năm liền (Renju: nước tạo đúng 5 luôn hợp lệ, nhưng nước tạo
  overline thì không).
- Bật/tắt bằng cờ cấu hình `tactical.enabled`, mặc định **tắt**. `az_v0` chạy với cờ tắt,
  `az_v1` chạy với cờ bật. Không có hai bản mã khác nhau.
- Đo được: tỉ lệ nước bị ghi đè, số lần ghi đè dẫn tới thắng, thời gian thêm mỗi nước.

**Thứ tự tích hợp — làm đúng thứ tự này, đo sau mỗi bước:**

1. Tactical override (rẻ nhất, hiệu quả rõ nhất, ~0,5 ngày).
2. Move ordering: sắp nước theo điểm mẫu trước khi mở rộng cây, để 200 lượt mô phỏng dồn vào
   nhánh có đe dọa. Đo bằng score, không đo bằng cảm giác.
3. Candidate move generation: chỉ xét các ô trong bán kính 2 quanh quân đã có. **Chỉ làm nếu**
   bước 1–2 cho thấy thời gian/nước là nút cổ chai; bước này làm giảm chất lượng nước đi nên
   phải chứng minh bằng số.

## 4. Giai đoạn 3 — Threat-space search (T30, người B)

**Điều kiện kích hoạt:** chỉ chạy khi bộ nhận diện mẫu báo có tứ mở, hoặc có chuỗi đe dọa
bắt buộc (đối thủ buộc phải chặn). Ngoài các tình huống đó, `az_v1` dùng PUCT như cũ.

**Giới hạn:** số node ≤ 5000 hoặc 150 ms mỗi nước, cái nào tới trước. Ghi lại số lần kích hoạt
và số node đã duyệt.

**Cấm làm:**

- Chạy Alpha-Beta trên toàn bộ cây MCTS. Đây là sai lầm phổ biến nhất và làm chậm gấp nhiều
  lần mà không tăng sức mạnh.
- Dùng quy ước dấu khác với PUCT. Giá trị trả về phải theo góc nhìn người đang đi.
- Kích hoạt mà không có test cho trường hợp "đối thủ có hai đe dọa cùng lúc" (khi đó phải
  thua, không được báo thắng).

## 5. Giai đoạn 4 — Đo lường

Không kết luận gì từ dưới 100 ván/cặp. Mọi con số lấy từ `logs/`, không nhập tay.

| Việc | Script | Task |
|---|---|---|
| Round-robin 6 tác tử, ma trận score, Wilson CI | `scripts/tournament.py` | T31 |
| Ablation: cắt tỉa Alpha-Beta bật/tắt; quét sims của `mcts_pure` | `scripts/ablation.py` | T32 |
| Chạy thật 2.400 ván và đóng băng số liệu | — | T35 |

Ba mốc so sánh, theo đúng thứ tự này:

1. `az_v0` (không thành phần chiến thuật)
2. `az_v1` mốc 2.1: chỉ tactical override
3. `az_v1` mốc 2.2: tactical override + threat-space

Mỗi mốc phải báo cáo: score, score khi cầm Đen, score khi cầm Trắng, thời gian trung bình/nước,
p95 độ trễ, số lượt suy luận/ván, và chênh lệch so với mốc trước kèm khoảng tin cậy.

## 6. Ánh xạ sang yêu cầu của đề bài

| Cấp | Tác tử | Task | Trạng thái |
|---|---|---|---|
| 1 | Random | T6 | Chưa có |
| 2 | Rule-based / Heuristic | T28 | Chưa có |
| 3 | Minimax / Alpha-Beta | T33 | Chưa có |
| 4 | MCTS (UCT, rollout) | T7 | Chưa có |
| 6 | Học tăng cường bằng self-play | T8–T12, T20 | **Đã có** (`az_v0`) |
| 7 | Mạng policy/value + PUCT | T9 | **Đã có** |
| 8 | Vòng lặp AlphaZero-inspired | T11, T12, T13 | **Đã có**, thiếu prioritized replay và curriculum (đã cắt, ghi rõ trong báo cáo) |

Cấp 5 (học có giám sát từ chuyên gia) **không làm**: cấp 6 đã thoả yêu cầu "tối thiểu một tác tử
học được ở cấp 5 hoặc 6", và pipeline nhãn chuyên gia tốn 2–3 ngày mà không phục vụ câu hỏi
nghiên cứu.

## 7. Thứ tự ưu tiên và điều kiện cắt

| Ưu tiên | Hạng mục | Điều kiện cắt nếu trễ |
|---|---|---|
| 1 | Tác tử cấp 1–4 + arena + giải đấu | **Không cắt** — đây là điểm bắt buộc |
| 2 | `az_v0` đo được sức mạnh | **Không cắt** — không có số thì không có báo cáo |
| 3 | Tactical override (T29) | Cắt được, nhưng mất câu hỏi nghiên cứu |
| 4 | Threat-space (T30) | Cắt trước tiên; khi đó câu hỏi nghiên cứu trả lời bằng mốc 2.1 |
| 5 | Ablation thứ hai (quét sims `mcts_pure`) | Cắt được nếu đã có ablation cắt tỉa |
| 6 | `az_renju` (T25) | Cắt fine-tune, giữ luật Renju + demo với tác tử cổ điển, ghi rõ trong báo cáo |

## 8. Tiêu chí giữ một cải tiến

Chỉ giữ khi thoả cả bốn:

1. Score cao hơn mốc trước trong điều kiện công bằng (cùng trọng số, cùng sims, chia đều màu).
2. Mức tăng ổn định qua đủ 100 ván/cặp, khoảng tin cậy không chứa 0,5.
3. Thời gian tăng thêm chấp nhận được so với ngân sách 1 giây/nước.
4. Không sinh nước không hợp lệ và không làm hỏng test luật.

Nếu tăng thời gian mà score không tăng: **loại bỏ và ghi vào báo cáo như một kết quả phủ định**.
Kết quả phủ định vẫn được điểm ở mục thảo luận, miễn là có số.

## 9. Rủi ro đã biết

| Rủi ro | Xử lý |
|---|---|
| `az_v0` thua Alpha-Beta độ sâu 4 và thua heuristic | Viết trước vào báo cáo: cấp ≠ sức mạnh; 1000 ván là ngân sách nhỏ so với AlphaZero gốc. Không giấu, không đổi luật chơi để cứu |
| Lợi thế đi trước của Gomoku Normal | Luôn chia đều màu; báo cáo score khi cầm Đen và khi cầm Trắng riêng |
| Tactical override che lấp điểm yếu của mạng | Đo tỉ lệ nước bị ghi đè. Nếu >30% số nước, ghi rõ trong báo cáo rằng phần lớn sức mạnh đến từ luật tay chứ không từ mạng |
| Threat-space phình thời gian | Giới hạn cứng 150 ms/nước; nếu vượt, tắt và ghi nhận |
| Bộ nhận diện mẫu sai làm hỏng cả luật Renju | T27 phải xong và test pass **trước** T24 và T29 |
