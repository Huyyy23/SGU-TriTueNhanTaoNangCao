# Kiểm định value calibration AlphaZero Normal 50k

## Phạm vi

Checkpoint Normal 50k `checkpoints/az_25k_continuation_v1/generations/f43be9e336474950a382a64f15f2ca80/checkpoint.pt`, SHA-256 `28ea9001fd83c1c7b4d62117a4d7e4a1726c95eb02b8a37f39298614ea35c5cd`. Không self-play, training, thay weights, promote hay đổi cấu hình production. Nguồn ván đánh giá là hai tournament 400 ván sau checkpoint: `data/eval_50k_vs_40k_400.sqlite` và `data/eval_50k_vs_heuristic_400.sqlite`.

Tournament logs không ghi trực tiếp NN root value: `scripts/tournament.py:Probe.select_move` ghi thời gian, simulation và NN calls, không ghi value; cần tái dựng board từ history và inference lại checkpoint. Không dùng MCTS root mean như thể là NN value. Nhãn là outcome ván thực tế theo người đi, thắng/hòa/thua = 1/0,5/0; `p=(v+1)/2` biểu diễn **expected score**, không phải xác suất thắng thuần khi có hòa. Outcome tùy chính sách hai bên, opening và search; không phải ground truth minimax.

Phân tích tách riêng cohort thông thường sau opening và tactical single-block/pre-fork; báo màu và đối thủ. Dùng game/opening pair làm cụm cho độ bất định, không coi nhiều state cùng game độc lập. Các state tactical được chọn sau khi quan sát trajectory có thể thiên lệch, đặc biệt pre-fork; holdout và kiểm tra exact D4 không loại được historical/near-duplicate exposure. Mọi kết luận theo cohort và protocol, không suy ra model sẽ mạnh hơn sau fine-tune.

## Dữ liệu và xác minh

`value_calibration_50k_data/predictions_v1.json`, SHA-256 `a3b0d3c272b97d0c42dc8d61a9195a3b3ff62e8138dffb19a6f3b3d46af3bfd2`; mã tái lập `value_calibration_50k_data/build.py`. Tổng **865 state**: 800 state thông thường và 65 tactical. Cohort thông thường lấy một state tại quyết định đầu tiên của 50k mỗi ván, sau 4 nước opening nếu Đen hoặc 5 nếu Trắng; không đại diện toàn bộ middlegame. Hai tournament có 800 ván và 8 ván hòa.

| Cohort | Development Đen / Trắng | Holdout Đen / Trắng |
|---|---:|---:|
| Thông thường sau opening | 202 / 202 | 198 / 198 |
| Single-block | 7 / 10 | 2 / 10 |
| Pre-fork | 6 / 10 | 10 / 10 |

Split chung giữa cohort không giao game ID hoặc opening seed; mỗi màu/đối thủ trong cohort thông thường holdout có 99 ván. Các cohort vẫn có thể chứa state khác nhau từ cùng game ở **cùng split**, không coi chúng là mẫu độc lập. Lấy mẫu không dựa outcome hay prediction. 65 tactical state dùng split family cũ; 800 state thường lấy đúng một/game.

Scan exact D4 trên mỗi snapshot replay 40k/50k 50.000 features: 0 match ở 40k; **một state ordinary development** trùng D4 với một replay sample 50k. Đây không chứng minh trùng game; near-duplicate và replay bị eviction/historical 25k chưa kiểm tra. Không gọi dataset hoàn toàn không leakage.

Kiểm tra độc lập ba state, mỗi loại một: player và outcome khớp SQLite, raw value tái inference khớp trong sai số tối đa **0,000271** giữa CUDA batch, CUDA single và CPU single trên mẫu đó. Sai khác số và gán bin sát ranh có thể xảy ra; không khẳng định bitwise reproducibility giữa device/batch. Dữ liệu agent: inference 865 state bằng CUDA, batch 128, 7 calls, 0,241 s inference; tổng trích xuất trước ghi 1,337 s. Đây không bao gồm thời gian tạo tournament hay soát mọi replay lịch sử.

## Calibration: state thông thường sau opening

Holdout có 99 game mỗi tổ hợp màu/đối thủ. `p=(v+1)/2`, outcome 1/0,5/0; sai lệch = mean(p − outcome), MSE = mean((p − outcome)²). CI 95% bootstrap 2.000 lần theo opening seed, seed bootstrap 508050. Kết quả **thăm dò**, chưa hiệu chỉnh nhiều so sánh; dữ liệu tournament không phải minimax.

| Màu / đối thủ | Mean p / outcome | Sai lệch, CI 95% | MSE | Sai lệch development |
|---|---:|---:|---:|---:|
| Đen vs 40k | 0,704 / 0,788 | −0,084 [−0,189; +0,019] | 0,276 | −0,117 |
| Đen vs Heuristic | 0,696 / 0,646 | +0,049 [−0,056; +0,156] | 0,300 | +0,118 |
| Trắng vs 40k | 0,313 / 0,606 | **−0,293 [−0,411; −0,178]** | 0,431 | −0,286 |
| Trắng vs Heuristic | 0,390 / 0,434 | −0,044 [−0,168; +0,080] | 0,401 | −0,131 |

Trong protocol này value của Trắng **bi quan tương đối so với outcome thực khi gặp 40k**, với cùng chiều trên development và holdout. Không thấy bằng chứng tương tự rõ ràng ở nhóm Trắng–Heuristic; không suy ra cần tăng value toàn cục. Đối chiếu độc lập 99 opening pairs ở holdout vs 40k: chênh sai lệch Trắng trừ Đen −0,209, bootstrap theo cặp 10.000 lần seed 508050, CI 95% [−0,383; −0,025]. Đây là phân tích thăm dò sau phân nhóm, không khẳng định sai số causal hay đã hiệu chỉnh multiplicity.

Ví dụ điều kiện: Trắng vs 40k có 48/99 state p≤0,1, p trung bình 0,034 nhưng score thực trung bình 0,604; ngược lại Trắng vs Heuristic có 24/99 state p≥0,9, p trung bình 0,960 nhưng score thực 0,333. Các mẫu và đối thủ khác nhau; không diễn giải hai nhóm này là outcome cùng một state.

## Calibration tactical holdout

Các nhóm nhỏ, một số state cùng game/seed, chỉ báo mô tả; không gộp với cohort thông thường như mẫu xác nhận độc lập. Pre-fork được chọn có điều kiện theo fork quan sát về sau; một số state có outcome hòa.

| Nhóm | State / game / seed | Mean p / outcome | Sai lệch | MSE |
|---|---:|---:|---:|---:|
| Single-block Đen | 2 / 2 / 2 | 0,602 / 0,250 | +0,352 | 0,323 |
| Single-block Trắng | 10 / 8 / 8 | 0,526 / 0,200 | +0,326 | 0,491 |
| Pre-fork Đen | 10 / 10 / 10 | 0,486 / 0,350 | +0,136 | 0,362 |
| Pre-fork Trắng | 10 / 10 / 10 | 0,590 / 0,400 | +0,190 | 0,342 |

Các nhóm tactical trong mẫu có chiều lạc quan, trái chiều với Trắng–40k đầu opening; cỡ mẫu nhỏ và selection khác nhau ngăn kết luận có một bias chung. Tập thông thường và tactical cùng sử dụng **61 game và 58 seed** ở cùng split; không phải hai mẫu xác nhận độc lập. Raw bins, development và phân tầng đối thủ tại `value_calibration_50k_stats/summary.json`, SHA-256 `1379239a8aed34dbe85eb62bd1d0e221ae7f509eff2a074ea2d515e35b873829`.

## Truy nguyên hai state single-block

Trace tái lập tại `value_calibration_50k_mcts/trace.py`, raw `trace_v1.json`; observer không thay MCTS. Python MCTS 200, noise tắt; replay khớp 121 visits và priors/Q với kết quả ablation. `:387` có hai state cùng game ID, state lỗi ở **ply 75**, block 43; `:217` ở **ply 49**, block 26.

| State | Root NN value | Prior block / chọn sai | Root N block / chọn | Root Q block / chọn | Terminal đối thủ thắng trong nhánh sai |
|---|---:|---:|---:|---:|---:|
| Heuristic `:217`, chọn 75 | +0,695 | 0,163825 / 0,015152 | 23 / 58 | −0,330 / −0,190 | 26/58 visits |
| 40k `:387`, chọn 101 | +0,423 | 0,000046 / 0,035221 | 0 / 45 | chưa ước lượng / +0,335 | 4/45 visits |

`:217`: 32 nonterminal NN leaf trong nhánh sai có mean +0,4685 theo góc nhìn root; cùng 26 terminal backup −1 tạo Q trung bình −0,1898. Terminal thắng của đối thủ xuất hiện lần đầu ở simulation 128. `:387`: 41 NN leaf có mean +0,4654, cùng 4 terminal −1 tạo Q +0,3352; lần đầu thấy terminal ở simulation 185. Block `:387` có prior cực thấp và không được root thăm ngay cả trong raw ablation 400 offline. Các Q là trung bình backup, không phải NN value trực tiếp; terminal detection và sign trên hai trace khớp. Chúng giải thích phân bổ search trong hai state, **không** chứng minh block thắng ván hoặc value sai so minimax.

Snapshot replay 50k có 50.000 sample, median 7 quân: chỉ **334 (0,668%)** có ≥49 quân và **164 (0,328%)** có ≥75 quân. Hai trace ở ply 49/75 vì vậy nằm trong vùng độ sâu hiếm của snapshot; đây là tương quan và snapshot ring, không chứng minh thiếu coverage của toàn bộ lịch sử training hay là nguyên nhân duy nhất.

## Bằng chứng, giả thuyết và quyết định

- **Đã quan sát:** calibration theo outcome có sai lệch âm lớn ở Trắng–40k sau opening, tái hiện cùng chiều trên development/holdout. Điều này là miscalibration **theo policy/đối thủ được đo**, không tự nó chứng minh self-play value target sai hay minimax value sai. Nhóm Trắng–Heuristic không cho bằng chứng cùng chiều rõ; tactical nhỏ lại có chiều dương.
- **Đã quan sát:** hai lỗi phòng thủ có cả vấn đề allocation/prior và sự cạnh tranh của giá trị leaf trong trung bình Q. Không thấy lỗi terminal/sign ở hai trace; không suy rộng thành audit mọi MCTS state. Zero-value sửa bốn gate cũ nhưng trên holdout chặt sửa ba và gây ba regression; cả hai single-block trên vẫn sai.
- **Giả thuyết chưa kiểm chứng:** quá ít state sâu trong replay snapshot có thể làm value/policy ở ply 49–75 kém ổn định. Tỷ lệ snapshot không chứng minh thiếu sample trong toàn bộ lịch sử hay quan hệ nhân quả. Lệch giữa các đối thủ có thể phản ánh khác biệt policy/outcome chứ không chỉ lỗi calibration của mạng.

### Một phương án ưu tiên: pilot value reweight theo độ sâu

**Đề xuất thí nghiệm có đối chứng, chưa thực thi:** dùng checkpoint 50k và replay generation có provenance, giữ architecture, policy targets, `z`, uniform replay sampling, optimizer reset, batch, LR, seed và số update giống nhau giữa baseline/candidate. Candidate chỉ thay **trọng số loss value theo state**: `w=2` nếu số quân ≥25, ngược lại `w=1`, chuẩn hóa mean `w=1` trong mỗi batch; policy loss không đổi. Baseline dùng `w=1` cho mọi state. Đây là phép thử một biến nhằm tăng tín hiệu value cho vùng chỉ chiếm **1.437/50.000 sample (2,874%)** trong snapshot; ngưỡng 25 và hệ số 2 là thiết kế pilot bảo thủ, **không phải hyperparameter đã được chứng minh tối ưu**. Không cân bằng kết quả thắng/thua, không thay luật hay đưa tactical override vào search.

Đề xuất ngân sách ban đầu: **3 seed × 100 optimizer updates × 2 nhánh = 600 updates**, batch 32, cùng hardware và cùng chuỗi minibatch tương ứng; số liệu thời gian training mới chưa đo. Mỗi nhánh bắt đầu từ checkpoint bất biến, replay đóng băng, lưu hash và generation riêng; không promote tự động. Value loss tác động backbone dùng chung nên policy vẫn có thể đổi dù policy loss giữ nguyên.

Trước khi chạy, khóa một tập đánh giá mới theo game/opening seed/family, không dùng lại 32 state holdout này làm chứng cứ chọn nhánh. Đăng ký trước biên regression và cỡ mẫu dựa trên độ biến thiên theo cặp, không tự chọn ngưỡng sau khi xem kết quả. So sánh calibration theo màu/đối thủ/độ sâu và outcome, policy drift, hai oracle tactical, cùng đối kháng paired với 40k và Heuristic ở 200 simulations. Dừng/rollback nếu integrity hoặc budget sai, NaN, hay hiệu ứng bất lợi vượt biên đã đăng ký; không coi kết quả 32 state hoặc một seed là bằng chứng sức chơi.

Rủi ro chính: nhấn mạnh state sâu có thể làm giảm chất lượng opening và không giải quyết prior cực thấp; outcome của Heuristic/40k khác self-play, nên reweight self-play không chắc sửa bias theo đối thủ. Tactical policy training trước đây cải thiện dev nhưng holdout 5/8 so control 6/8, không chứng minh generalization; không lặp lại như một kết luận đã thành công. Chi phí đo được lần này chỉ là **0,241 s** cho 865 raw NN predictions batched CUDA và trace hai state tổng process **2,48 s**; không tính chi phí tạo tournament hoặc training. Không ngoại suy wall time của pilot 600 updates.

**Kết luận:** ưu tiên pilot reweight value theo độ sâu để kiểm tra giả thuyết coverage, không khẳng định đó là cấu hình tốt nhất hay cho phép fine-tune production ngay. Chưa có bằng chứng chọn zero value, tăng simulations hoặc sửa production.
