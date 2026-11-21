# Kiểm định chất lượng AlphaZero Normal sau 50k self-play

## Mục tiêu và định danh

Không huấn luyện thêm, self-play thêm hoặc promote checkpoint trong kiểm định này. Mốc 50k là checkpoint sau 25.000 ván mới, khởi tạo từ checkpoint parent 25k; mốc 40k là sau 15.000 ván mới.

- **40k:** `checkpoints/az_25k_continuation_v1/generations/71e7aedb1f394ee782dc84832b130b6a/checkpoint.pt`, SHA-256 `89fddabc55a1b79360ed91e0f1263405139ae81e9691e0a15e4e12cdec33f909`.
- **50k:** `checkpoints/az_25k_continuation_v1/generations/f43be9e336474950a382a64f15f2ca80/checkpoint.pt`, SHA-256 `28ea9001fd83c1c7b4d62117a4d7e4a1726c95eb02b8a37f39298614ea35c5cd`.

Cả hai là checkpoint candidate thực tế tại gate 40k/50k; `resolve_generation` xác thực manifest/checkpoint/replay hashes và SHA tương ứng event SQLite. Checkpoint `latest.json` sau gate có SHA file khác vì metadata checkpoint được lưu lại; không dùng sự khác SHA file đó làm bằng chứng weights khác.

## Protocol đối kháng

Hai so sánh độc lập: 50k–40k và 50k–Heuristic, mỗi so sánh đúng 400 ván với 200 opening pairs, đổi màu trong từng cặp. Normal, Python MCTS 200 simulations, không root noise, temperature 0; Heuristic backend Python. Opening bốn nước ngẫu nhiên hợp lệ; vì mỗi màu mới có hai quân, opening không terminal và chưa có nước thắng ngay. Seed 508001, `paired-crc32`, hai worker/so sánh, một torch thread/worker. RNG opening phụ thuộc label của cặp nên hai đối thủ không nhất thiết dùng cùng tập opening, nhưng hai ván đổi màu trong mỗi cặp dùng cùng opening. Checkpoint, config và raw games/telemetry được lưu riêng:

- `data/eval_50k_vs_40k_400.sqlite`, experiment `eval50k-vs40k-400-seed508001`, log cùng tên `.log`.
- `data/eval_50k_vs_heuristic_400.sqlite`, experiment `eval50k-vs-heuristic-400-seed508001`, log cùng tên `.log`.

CI dùng bootstrap theo **opening pair**; kết quả 400 ván chỉ áp dụng protocol và tập opening trên. Elo relative được ước lượng bằng `az.elo.paired_davidson_ci`; đây là pairwise Davidson có pseudocount, không phải leaderboard multi-agent hoặc bằng chứng sức chơi ngoài phân phối này.

### 50k vs 40k — hoàn tất

SQLite `data/eval_50k_vs_40k_400.sqlite`: 400 game ID, 200 opening pairs, 200 ván mỗi màu; `PRAGMA quick_check=ok`. Bootstrap 10.000 lần theo cặp opening, seed 508001: **278 thắng / 5 hòa / 117 thua**, win rate 69,5%, score 70,125%. ΔElo **+149,91**, CI 95% **[+114,88; +188,51]** so với 40k. CI không chứa 0: có bằng chứng 50k mạnh hơn 40k **trong protocol và phân phối opening này**, không phải cam kết tổng quát cho các luật hay opening khác.

| Màu của 50k | Số ván | Thắng | Hòa | Thua | Win rate |
|---|---:|---:|---:|---:|---:|
| Đen | 200 | 157 | 4 | 39 | 78,5% |
| Trắng | 200 | 121 | 1 | 78 | 60,5% |

`end_reason`: 395 five-in-row và 5 full-board, không có illegal/forfeit. So với 40k, Elo ở đây là **chênh lệch pairwise**, không phải rating tuyệt đối neo Heuristic.

### 50k vs Heuristic — hoàn tất

SQLite `data/eval_50k_vs_heuristic_400.sqlite`: 400 game ID, 200 opening pairs, 200 ván mỗi màu; `PRAGMA quick_check=ok`. `az.elo.paired_davidson_ci`, bootstrap 10.000 lần, seed 508001: **213 thắng / 3 hòa / 184 thua**, win rate 53,25%, score tính nửa điểm hòa 53,625%. ΔElo **+25,36**, CI 95% **[−7,86; +58,00]**; Heuristic quy ước 1200 thì rating tương đối ước lượng 1225,36 [1192,14; 1258,00]. CI chứa 0: **chưa đủ bằng chứng 50k mạnh hơn Heuristic** trong protocol này.

| Màu của 50k | Số ván | Thắng | Hòa | Thua | Win rate |
|---|---:|---:|---:|---:|---:|
| Đen | 200 | 126 | 3 | 71 | 63,0% |
| Trắng | 200 | 87 | 0 | 113 | 43,5% |

`end_reason`: 397 five-in-row và 3 full-board; không có game kết thúc do illegal/forfeit. Khoảng tin cậy resample từng cặp hai màu, không coi 400 game là độc lập.

## Self-play: mất cân bằng màu và phòng thủ Trắng

Nguồn `data/az_25k_continuation_v1.sqlite`, run `az-25k-continuation-v1`: 25.000 ván continuation Normal, 25.000 game ID và seed ván khác nhau, 247.517 bản ghi nước MCTS; mỗi nước ghi 200 simulations. Hai nước mở đầu ngẫu nhiên được ghi trong `provenance.opening_actions` và không phải lựa chọn MCTS.

| Nhóm iteration | Đen thắng | Trắng thắng | Hòa |
|---|---:|---:|---:|
| 0–99 | 3.919/5.000 (78,38%) | 1.081 | 0 |
| 100–199 | 4.181/5.000 (83,62%) | 819 | 0 |
| 200–299 | 4.324/5.000 (86,48%) | 676 | 0 |
| 300–399 | 4.317/5.000 (86,34%) | 683 | 0 |
| 400–499 | 4.089/5.000 (81,78%) | 908 | 3 |
| **Tổng** | **20.830/25.000 (83,32%)** | **4.167** | **3** |

Độ dài tính cả opening: 6.043 ván dài 9–10 ply; 12.823 dài 11–12; 4.315 dài 13–15; 1.154 dài 16–20; 665 dài 21–121. Có 11.859 cặp opening có thứ tự khác nhau; cặp phổ biến nhất xuất hiện 9 lần. Các cặp chưa gộp đối xứng, không chứng minh coverage đều.

**Mẫu phòng thủ:** lấy ngẫu nhiên 200 ván không hoàn lại từ từng nhóm 5.000, seed `20261120`, sau khi sắp game ID; tổng 1.000 ván. Replay hai opening trước rồi mới replay move records. Oracle độc lập quét mọi đoạn năm ô ngang/dọc/chéo, không gọi rules, model hay MCTS; kiểm tra vị trí, lượt, terminal và winner trên mẫu. Tại lượt Trắng có đúng một ô thắng ngay của Đen: 120 lần Trắng chặn, một lần Trắng chọn thắng ngay, **161 lần không chặn và không chọn thắng ngay**, xảy ra trong 155 ván. Đây là 161/281 cơ hội chỉ có một ô bị đe dọa sau khi loại nước thắng ngay được chọn; không phải tỷ lệ lỗi trên mọi nước. Trong 701 tình huống Đen có ít nhất hai ô thắng, một nước không thể chặn đồng thời cả hai: không gộp vào missed single-block.

Ở 679 lượt Trắng trước đó, Đen chưa có nước thắng ngay nhưng nước Đen thực tế tiếp theo tạo ít nhất hai ô thắng. Ô setup còn trống và Trắng không có nước thắng ngay; chiếm ô đó là phản thực tế ngăn **nước setup cụ thể**, không chứng minh ngăn được mọi mối đe dọa hoặc cứu được ván. Ví dụ missed single-block: game ID `7bfa36c9303f9e21d0bdabfb4f1a2c76fbb0bdd717f8ee9ad857828ffb10c423`, opening `[33,76]`, ply 9 Trắng chọn 57 trong khi ô thắng duy nhất của Đen là 85; action = `11*row+col`.

Kết quả màu phụ thuộc mạnh vị trí opening: theo khoảng cách Chebyshev từ nước Đen đầu đến tâm, tỷ lệ Đen thắng từ 75,08% ở khoảng cách 5 đến 99,34% ở khoảng cách 1. Đây là tương quan theo opening, không phải thí nghiệm nhân quả. Cùng model chơi cả hai màu, nhưng opening không ghép cặp hoán màu và trạng thái/lượt khác nhau; không thể tách riêng lợi thế tiên thủ của Normal khỏi thiên lệch model, dữ liệu hay search. Mẫu tactical không đại diện độc lập cho toàn bộ 25.000 ván.

## Bốn lỗi block tại gate 50k

Nguồn telemetry là event `promotion_evaluation_500` trong SQLite self-play, checkpoint 50k nêu trên; Python MCTS 200 simulations, seed 5314159, noise tắt, temperature 0. Oracle độc lập quét các tia liên tiếp theo bốn hướng, không dùng rules/model/MCTS. Cả bốn state đều hợp lệ và chưa terminal; bên đi không có thắng ngay, đối phương có đúng một ô thắng: **action 60 = (5,5)**. Model chọn action 63 = (5,8); đối phương có thể đi 60 hoàn tất hàng `[58,59,60,61,62]`. Chặn 60 chỉ được chứng minh tránh thua ngay, không bảo đảm cứu được ván.

Encoding khớp state: bốn plane là quân bên đi, quân đối thủ, ô cấm và màu; trong Normal plane cấm rỗng, plane màu gồm 121 số 1 cho Đen hoặc 0 cho Trắng. Không phát hiện lỗi encoding trên bốn state.

| State | Tổng plane quân mình/địch/cấm/màu | Prior P(60)/P(63) | Visits N(60)/N(63) | Q(60)/Q(63), góc nhìn root | NN value |
|---|---|---|---|---|---:|
| Đen | 4/4/0/121 | 0,02355 / 0,05264 | 6 / 13 | −0,99553 / −0,99998 | −0,99079 |
| Đen variant 1 | 5/5/0/121 | 0,03323 / 0,07452 | 9 / 18 | −0,98979 / −0,99995 | −0,98199 |
| Đen variant 2 | 5/5/0/121 | 0,03735 / 0,06194 | 9 / 15 | −0,99688 / −0,99998 | −0,99045 |
| Trắng | 4/5/0/0 | 0,03552 / 0,05747 | 12 / 14 | −0,97683 / −0,99926 | −0,98533 |

Prior của nước sai cao hơn nước chặn trong cả bốn case; Q của nước chặn tốt hơn nhưng cả hai gần −1, còn visits nghiêng về nước sai. Tổng root visits 199 vì simulation đầu expand root; budget vẫn là 200. Đây là cơ chế quan sát được, **chưa chứng minh** prior hoặc value riêng lẻ là nguyên nhân nhân quả, cũng chưa chứng minh value sau block sai vì chưa giải toàn bộ ván. Bốn biến thể gần nhau không phải bốn family độc lập. Gate 40k đạt 6/6 ở cùng family, gate 50k 2/6; seed gate khác nhau nên không quy toàn bộ chênh lệch cho checkpoint.

## Tactical mở rộng

Artifact tái lập `tactical_50k_analysis_20261120/analyze.py` và `raw.json`: 12 state thuộc sáu family win/block, cân bằng sáu state mỗi màu; kiểm tra các prefix nonterminal và đáp án bằng hai oracle độc lập. Không có state trùng D4 với 72 fixture benchmark cũ hoặc hai snapshot replay 40k/50k, mỗi snapshot 50.000 mẫu. Không kiểm tra replay đã bị eviction hoặc toàn bộ lịch sử, không loại được near-duplicate hay pattern leakage.

Với cùng seed 5314159, Python MCTS 200 simulations, noise tắt, temperature 0: **40k 12/12 và 50k 12/12**; mỗi màu 6/6, win 6/6, block 6/6 ở cả hai. Tập này được tạo sau khi biết kết quả gate và dùng hình thái từ thí nghiệm trước, nên chỉ là exploratory; 12 state từ sáu family và một seed không đủ chứng minh tổng quát hóa hay ngang sức. Regression block của family gate cũ không lặp lại trên ba family block mới.

## Trạng thái resume

Hai tiến trình đã được dừng theo yêu cầu và sau đó resume cùng run ID, checkpoint, seed, config và backend. Trước khi resume, `PRAGMA quick_check` trả `ok` cho cả hai SQLite:

| So sánh | Ván đã ghi / kế hoạch | Cặp hoàn chỉnh | Cặp dở |
|---|---:|---:|---:|
| 50k–40k | 130/400 | 65 | 0 |
| 50k–Heuristic | 233/400 | 116 | 1 |

Khi resume, driver xác thực config và game ID đã ghi, rồi bỏ qua các ván đó; phần còn lại đã hoàn tất. Kiểm tra cuối cho mỗi SQLite: 400 ID duy nhất, 200 cặp, mỗi cặp có hai màu và cùng bốn nước opening/seed; không có cặp sai.

SHA-256 SQLite sau khi kết thúc: 50k–40k `4ea904cbe1d03cc68a787e050c0897bf8610a4dd745baa406c7f7c95ce1bb6`; 50k–Heuristic `76ca135f99aca7d3fae4e3f3ee00cfadf9eb9bcd4c8817176f64416ea13be2df`. Mỗi game lưu lịch sử và telemetry; agent records lưu SHA checkpoint. Các phép kiểm là trên dữ liệu hai tournament, không chứng minh parity Python/C++ toàn diện.

## Kết luận và hướng tiếp theo

1. **Đã xác nhận:** 50k vượt 40k ở đối kháng ghép cặp trong protocol đã đăng ký, ΔElo +149,91 với CI 95% dương. So với Heuristic, kết quả 53,625% score có CI ΔElo chứa 0: chưa chứng minh vượt đối thủ này.
2. **Đã xác nhận:** 50k bỏ block duy nhất trong bốn biến thể gate; prior và visits ưu tiên nước sai dù Q của nước chặn cao hơn. Trong mẫu 1.000 ván self-play, 161 lượt Trắng không chọn chặn ô thắng đơn và cũng không chọn thắng ngay. Không phát hiện lỗi luật hay encoding trong bốn fixture; chưa chứng minh nguyên nhân gốc của value/prior hoặc mức độ đóng góp vào lệch màu.
3. **Đề xuất:** xây tập đánh giá phòng thủ mới từ game theo split độc lập theo game/seed và family, gồm các state trước nước tạo fork; khóa tập trước khi chọn model, kiểm tra D4 và provenance, báo riêng hai màu. Không dùng 12 state exploratory làm bằng chứng tổng quát.
4. **Đề xuất:** chạy ablation có đối chứng offline trên các state sai để cô lập prior và value, giữ checkpoint, seed, MCTS 200 và oracle; sau đó mới thử can thiệp training có baseline và ngân sách khớp. Không tích hợp tactical override chỉ vì tăng accuracy trên fixture.
5. **Đề xuất:** đánh giá thêm Heuristic trên opening/seed độc lập nếu cần kết luận hơn đối thủ, đồng thời so sánh 40k và 50k theo cùng protocol. Tăng số ván training hay ép tỷ lệ màu về 50/50 không tự giải quyết lỗi phòng thủ.

Không training, self-play hoặc promotion trong kiểm định này. Tập exploratory không loại được near-duplicate hay historical exposure; một seed tournament và opening bốn nước giới hạn phạm vi kết luận.
