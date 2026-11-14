# Báo cáo Gomoku 11×11 với AlphaZero

## 1. Phạm vi và nguồn bằng chứng

Báo cáo phân biệt ba loại thông tin: hành vi đọc trực tiếp từ mã nguồn, cấu hình lưu trong artifact và kết quả ghi trong log. Cấu hình đặt mục tiêu chạy không đồng nghĩa mục tiêu đã hoàn thành; loss giảm không đồng nghĩa tỷ lệ thắng tăng. Các phần thuật toán và huấn luyện dưới đây dựa trên mã nguồn hiện có cùng log đã lưu, không phải một đợt huấn luyện mới.

Nguồn chính:

- [Mạng policy–value](../az/model.py), [MCTS](../az/mcts.py), [self-play](../az/selfplay.py).
- [Mã hóa đặc trưng](../gomoku/features.py), [replay buffer](../az/replay_buffer.py), [trainer](../az/trainer.py).
- [Tactical override](../az/player_tactical.py), [threat-space search](../az/threat_space.py), [chọn wrapper](../az/player_factory.py).
- [Normal training driver](../scripts/train.py), [Renju fine-tuning driver](../az/renju_finetune.py).
- [Log Normal 2026-10-28](../logs/train_20261028-063106.jsonl), [log từng ván cùng đợt](../logs/games_20261028-063106.jsonl), [log Renju 2026-11-12](../logs/renju_finetune_20261112-181734.jsonl).

Graphify đã được dùng để tìm `MCTS`, `Trainer`, `ReplayBuffer` và các wrapper trước khi đọc mã. Truy vấn đường có hướng giữa `MCTS` và `Trainer` không tìm thấy đường; điều đó không chứng minh hai thành phần độc lập. Luồng dữ liệu được xác nhận bằng các lời gọi trong mã nguồn, không suy diễn từ việc thiếu cạnh trong đồ thị.

## 2. Biểu diễn bàn cờ và mạng nơ-ron

### 2.1. Luật và đặc trưng đầu vào

Bàn cờ có 121 ô; hành động phẳng `a = 11r + c` tương ứng hàng `r`, cột `c`. [NormalRules](../gomoku/rules/normal.py) cho cả hai màu thắng khi có ít nhất năm quân liên tiếp. [RenjuRules](../gomoku/rules/renju.py) quy định Đen thắng với đúng năm, Trắng với ít nhất năm; Đen bị cấm overline, double-four và double-three, nhưng nước tạo đúng năm được xét hợp lệ trước các điều kiện cấm khác. Đây là hành vi của bộ luật trong dự án, không phải tuyên bố đã chứng minh tương đương toàn bộ quy định thi đấu Renju quốc tế.

`board_to_planes` tạo tensor `(4, 11, 11)` từ góc nhìn người sắp đi:

| Kênh | Nội dung |
| --- | --- |
| 0 | Quân của người sắp đi |
| 1 | Quân đối phương |
| 2 | Các ô cấm do bộ luật trả về |
| 3 | Toàn 1 nếu người sắp đi là Đen, toàn 0 nếu là Trắng |

Normal không có ô cấm; Renju chỉ cung cấp ô cấm cho lượt Đen. Không có kênh lịch sử nước đi hoặc kênh đánh dấu nước vừa đi. Đầu vào là bốn kênh cho cả hai luật, không phải bốn kênh Normal rồi thêm kênh thứ năm cho Renju.

### 2.2. Kiến trúc policy–value

`AZNet` mặc định dùng convolution 3×3 với 64 filters, BatchNorm và ReLU, tiếp theo sáu residual blocks. Mỗi block có hai convolution 3×3, BatchNorm, kết nối cộng tắt và ReLU. Hai đầu ra chia sẻ phần thân:

- Policy head: convolution 1×1 với hai filters, BatchNorm, ReLU, flatten và linear tạo 121 logits.
- Value head: convolution 1×1 với một filter, BatchNorm, ReLU, flatten, linear 64 nút, ReLU, linear một nút và `tanh`; giá trị thuộc `[-1, 1]`, theo góc nhìn người sắp đi.

`predict` dùng chế độ evaluation và `no_grad`, trả softmax của policy logits cùng value. Mask hợp lệ được áp dụng sau softmax trong MCTS rồi chuẩn hóa lại; không phải mask logits trước softmax. Loss huấn luyện cũng dùng đủ 121 logits, với policy target có khối lượng bằng 0 trên nước không được thăm.

## 3. Tìm kiếm MCTS và quy ước dấu

Mỗi node lưu prior, số lượt thăm, tổng value và các node con. `q_value` là trung bình value từ góc nhìn người sắp đi tại chính node đó. Khi chọn con `a` của trạng thái `s`, mã tối đa hóa PUCT:

$$
- Q(s_a) + c_{\mathrm{puct}} P(s,a)
\frac{\sqrt{\max(1,N(s))}}{1+N(s_a)}.
$$

Dấu trừ trước `Q(s_a)` là bắt buộc: người sắp đi tại con là đối thủ của người sắp đi tại cha. Các điểm bằng nhau được phá hòa bằng RNG. Node chưa được thăm có `Q = 0`.

Một simulation đi xuống cây bằng thao tác `play`; nếu cần đánh giá lá, `request()` trả đặc trưng và legal mask rồi hoàn tác đường đi. `receive(policy, value)` mask prior, chuẩn hóa, mở rộng đúng các nước hợp lệ và backup. Nếu tổng prior hợp lệ quá nhỏ, mã dùng phân phối đều trên nước hợp lệ. Khi backup từ một lá không kết thúc, value được cộng tại lá, sau đó đổi dấu ở từng cạnh về gốc.

Nếu nước vừa đi tạo thắng, người sắp đi tại lá đã thua nên terminal value là `-1`; hòa có value `0`. Lá kết thúc được backup trực tiếp, không gọi mạng. Vì vậy `nn_calls` đếm số đánh giá lá, không phải số simulation hay số lần gọi batch của mạng. Simulation đầu có thể chỉ mở rộng gốc, nên tổng visit ở các con không nhất thiết bằng ngân sách simulation.

Trong self-play, prior tại gốc được trộn nhiễu:

$$
P'(s,a)=(1-\varepsilon)P(s,a)+\varepsilon\eta_a,
\qquad \eta\sim\operatorname{Dirichlet}(\alpha).
$$

Cấu hình Normal lưu trong log dùng `c_puct = 1.5`, `alpha = 0.15`, `epsilon = 0.25` và 200 simulations mỗi nước. Nhiễu chỉ áp dụng khi mở rộng gốc, không áp dụng cho mọi node. Cây được `reset` sau mỗi nước thực; self-play hiện không tái sử dụng cây con của nước đã chọn.

## 4. Self-play, policy target và value target

`run_games` xen kẽ nhiều ván: mỗi ván có board, cây và danh sách mẫu riêng. Các yêu cầu đánh giá lá đang chờ được xếp vào một batch `model.predict`. `parallel_games` giới hạn số ván hoạt động, không bảo đảm mọi batch đạt kích thước đó. Bản nhiều process chia các ván thành nhóm, truyền trọng số CPU cho worker và gom mẫu về process cha; chỉ process cha thêm mẫu vào buffer. Đây là khả năng của mã hiện tại, không chứng cứ các đợt huấn luyện lịch sử đã dùng nhiều process.

Trước mỗi nước thực, self-play lưu `(planes, pi, player)`. Policy target được tạo từ số visit ở con của gốc:

$$
\pi(a\mid s)=\frac{N(s,a)^{1/\tau_{\mathrm{target}}}}
{\sum_b N(s,b)^{1/\tau_{\mathrm{target}}}}.
$$

`pi_target_temp = 1.0` trong cấu hình Normal ghi ở log, nên target là visit chuẩn hóa. Nhiệt độ dùng để chọn nước thực là biến khác: 15 nước đầu dùng nhiệt độ 1, sau đó dùng 0, chọn visit lớn nhất và phá hòa ngẫu nhiên. Không được nhầm việc chọn nước gần tất định cuối ván với việc biến policy target thành one-hot.

Sau khi ván kết thúc mới gán nhãn `z`: `+1` nếu người sắp đi tại mẫu là người thắng, `-1` nếu là người thua, `0` nếu hòa. Đây là kết quả cuối ván, không phải value dự đoán tại gốc. Các mẫu liên tiếp trong ván phân thắng bại do đó đổi dấu theo lượt. Mỗi vị trí trước nước đi tạo một mẫu gốc; không lưu thêm trạng thái kết thúc làm mẫu riêng.

## 5. Replay buffer, đối xứng và tách bộ luật

`ReplayBuffer` là bộ đệm vòng có sức chứa cố định; đầy thì ghi đè mẫu cũ. Đặc trưng lưu dạng `uint8`, policy dạng `float16`, value dạng `float32`. Buffer lưu mẫu trước augmentation, không nhân trước dung lượng lên tám lần.

Khi lấy một batch, mã chọn chỉ số có hoàn lại và chọn ngẫu nhiên một trong tám đối xứng của hình vuông: đồng nhất, ba phép quay, lật theo hai trục, chuyển vị và chuyển vị kèm lật hai trục. Cùng một biến đổi được áp dụng cho cả feature planes lẫn policy 121 ô; `z` giữ nguyên. Số lượt rút `batch_size × steps` vì vậy không phải số mẫu khác nhau. Trường `distinct_samples` chỉ đếm các slot buffer đã thực sự được rút trong iteration đó.

Mỗi buffer chỉ nhận một `rule_set`; `add` từ chối trộn Normal và Renju, `sample` kiểm tra nhãn luật. Khi fine-tune Normal sang Renju, `Trainer.from_pretrained` yêu cầu buffer Renju rỗng và kiến trúc khớp. Trọng số được chuyển giao, nhưng optimizer, `global_step` và `n_iter` được khởi tạo lại. Driver Renju ghi `renju_last.pt` và `renju_buffer_last.npz`, tách khỏi `last.pt` và `buffer_last.npz` của Normal.

## 6. Loss, tối ưu và đo trên batch cố định

Với batch kích thước `B`, loss dữ liệu thực tế trong `_total_loss` là:

$$
L_{\mathrm{policy}}=-\frac{1}{B}\sum_i\sum_a\pi_i(a)
\log\operatorname{softmax}(\ell_i)_a,
\qquad
L_{\mathrm{value}}=\frac{1}{B}\sum_i(v_i-z_i)^2,
$$

$$
L=w_pL_{\mathrm{policy}}+w_vL_{\mathrm{value}}.
$$

Cả hai trọng số mặc định bằng 1. Trainer dùng `torch.optim.Adam` với learning rate `0.001` và `weight_decay = 0.0001`; regularization được đưa qua optimizer, không cộng thêm một số hạng L2 trong `_total_loss`. Các trường `loss`, `policy_loss`, `value_loss` trong log là loss dữ liệu, không chứa giá trị phạt L2 được tính riêng. Không nên diễn giải `loss` đã ghi là tổng số học của CE, MSE và một khoản L2 hiển thị.

Cosine schedule dựa vào `global_step`, với tổng bước bằng `n_iter × train_steps_per_iter`. Resume phục hồi trọng số, optimizer, iteration và global step; đường learning rate còn phụ thuộc cấu hình truyền vào khi resume. Checkpoint lưu metadata kiến trúc và cấu hình, còn replay buffer cùng batch đo cố định được lưu cạnh checkpoint nếu bật `checkpoint_buffer`.

`freeze_eval_batch` chụp một batch từ buffer một lần; `loss_on_fixed_batch` đo loss với BatchNorm ở chế độ evaluation, không augmentation. Trainer loại các chỉ số nguồn đã giữ lại khỏi lượt lấy mẫu huấn luyện nếu vẫn còn slot khác để rút. Cần giữ ba giới hạn:

1. Batch này lấy từ self-play của chính run, không phải tập kiểm thử độc lập theo ván hoặc theo đối thủ; các trạng thái liên quan trong cùng ván vẫn có thể xuất hiện ở phần huấn luyện.
2. Lấy batch cố định có hoàn lại nên 512 phần tử không bảo đảm là 512 trạng thái khác nhau. Khi tất cả slot đều bị loại, sampler không thực hiện loại trừ để vẫn lấy được batch.
3. Khi buffer vòng ghi đè, việc loại trừ vẫn theo chỉ số slot, không theo danh tính nội dung mẫu. Cơ chế này không bảo đảm holdout theo nội dung xuyên suốt mọi vòng ghi đè.

Vì vậy chỉ dùng loss batch cố định để so sánh đúng metric đã giữ lại trong một run; không suy ra trực tiếp sức chơi, khả năng tổng quát hóa hoặc mức tiến bộ giữa hai bộ luật.

## 7. Tactical override và threat-space search

`TacticalAZPlayer` là wrapper ở bước chọn nước, không thay kiến trúc hay huấn luyện mạng. Khi bật, nó duyệt nước hợp lệ: ưu tiên thắng ngay, tiếp theo chặn một nước đối phương có thể thắng ngay; nếu không có thì gọi base player. Tập nước hợp lệ được tính riêng theo người đang đi và đối phương, nên không coi nước cấm Renju của đối phương là một đe dọa thắng hợp lệ. Nước chặn vẫn phải hợp lệ cho bên đang đi. Nếu đối phương có nhiều nước thắng độc lập, chặn một nước không bảo đảm cứu được ván.

`ThreatSpaceAZPlayer` thêm tìm kiếm chuỗi đe dọa bốn quân, dựa trên `open_four` và `blocked_four`, ở đường fallback. Nếu tactical cũng bật, thắng/chặn một nước được xét trước threat-space. Mặc định threat-space tắt; searcher mặc định giới hạn 5.000 node và 150 ms. Nó ưu tiên open-four, thử các ô hoàn thành để mô phỏng chặn, đồng thời kiểm tra một số nước thắng tức thì của đối thủ.

Kết quả `0` nghĩa chưa biết, không phải chứng minh hòa. Pattern chỉ đề xuất nước; searcher kiểm tra lại các ô hoàn thành có thật sự hợp lệ và thắng theo bộ luật. Một chuỗi chỉ được chấp nhận nếu mọi phản ứng chặn bắt buộc được xét đều còn tiếp diễn thắng; phản ứng thắng ngay của đối phương bác bỏ nhánh đó. Hai ô thắng hợp lệ khác nhau tạo đe dọa không thể chặn bằng một nước.

Đây không phải solver Gomoku đầy đủ: không mở rộng mọi nước yên lặng hoặc mọi chuỗi đe dọa ba quân. Hết ngân sách trả chưa biết, không trả chứng minh thắng. Deadline dùng chung cho tactical, threat-space và base MCTS; phần trước đã dùng bao nhiêu thì base chỉ nhận thời gian còn lại. Deadline mềm: thao tác sinh nước, dò pattern hoặc một lần suy luận đang chạy không bị ngắt cưỡng bức. Ở chế độ cố định simulations, tactical/threat vẫn thêm công việc ngoài ngân sách MCTS; phải báo cả độ trễ lẫn tỉ lệ override.

Mỗi nước reset counters, ghi số simulation, số neural leaves và lý do override. Số node MCTS thực không có counter tương ứng nên ghi `null`, không gán số simulation thành số node. Các wrapper không được gọi trong `run_games`, nên log self-play bên dưới không chứng minh tác động của tactical hoặc threat-space lên quá trình huấn luyện.

## 8. Huấn luyện đã ghi nhận và nguồn gốc artifact

### 8.1. Normal

[Log ngày 2026-10-28](../logs/train_20261028-063106.jsonl) có 20 iteration, mỗi iteration 50 ván, tổng cộng 1.000 ván được báo cáo. Dòng từng ván đã kiểm tra trong [game log](../logs/games_20261028-063106.jsonl) ghi Normal, 200 simulations, 16 ván xen kẽ, sáu blocks, 64 filters, batch 128 và 200 bước train/iteration. Training log ghi thiết bị `NVIDIA GeForce RTX 4060 Laptop GPU`.

| Metric | Iteration 1 | Iteration 20 |
| --- | ---: | ---: |
| Mẫu trong buffer | 2.005 | 30.452 |
| Global step | 200 | 4.000 |
| Loss train trung bình | 5.19370 | 4.09124 |
| Policy loss | 4.79342 | 3.76979 |
| Value loss | 0.40028 | 0.32145 |
| Loss batch cố định | 5.05494 | 5.14099 |
| Số nước trung bình/ván | 40.10 | 23.10 |
| Đánh giá lá mạng trung bình/ván | 7.877,7 | 4.408,2 |

Loss train cuối run thấp hơn đầu run, nhưng loss batch cố định cuối run **cao hơn** đầu run và dao động giữa các iteration. Bằng chứng này không hỗ trợ tuyên bố loss đánh giá giảm hay mô hình hội tụ tốt hơn. Ván ngắn hơn cũng không tự chứng minh sức chơi tăng.

[Log ngày 2026-11-01](../logs/train_20261101-163215.jsonl) chỉ có một iteration với hai ván, 95 mẫu và 200 global steps. Không cộng nó thành một đợt huấn luyện lớn hoặc dùng thay bằng chứng của run 1.000 ván. [Cấu hình `normal_5000.yaml`](../configs/normal_5000.yaml) đặt 100 iterations và 1.000 bước train/iteration; riêng tệp cấu hình và thư mục `checkpoints/az_v0b/` không chứng minh đã chạy đủ 5.000 ván.

### 8.2. Fine-tune Renju

[Log ngày 2026-11-12](../logs/renju_finetune_20261112-181734.jsonl) ghi 10 iterations, mỗi iteration 50 ván, tổng 500 ván; buffer cuối có 10.715 mẫu. Trường `source_checkpoint` chỉ tới `checkpoints/last.pt` trên máy chạy. Driver được đọc dùng buffer mới và `Trainer.from_pretrained`, không trộn replay Normal sang Renju.

| Metric | Iteration 1 | Iteration 10 |
| --- | ---: | ---: |
| Mẫu trong buffer | 1.081 | 10.715 |
| Loss train trung bình | 3.16399 | 3.57626 |
| Policy loss | 3.01036 | 3.12071 |
| Value loss | 0.15362 | 0.45555 |
| Loss batch cố định | 3.33481 | 3.45334 |
| Đánh giá lá mạng trung bình/ván | 4.110,60 | 3.650,68 |

Cả loss train lẫn loss batch cố định cuối run đều cao hơn đầu run. Log xác nhận có thực hiện và ghi nhận fine-tuning, không xác nhận Renju mạnh hơn Normal hay mạnh hơn trước fine-tuning. Không so trực tiếp trị số loss của hai bảng như một thước đo chất lượng giữa hai tập dữ liệu khác nhau.

### 8.3. Nhận dạng checkpoint và giới hạn truy xuất nguồn gốc

Metadata và SHA-256 do tiến trình tích hợp đọc trực tiếp, không suy ra từ tên tệp:

| Artifact | Metadata quan sát | SHA-256 |
| --- | --- | --- |
| `checkpoints/last.pt` | format 1; Normal; iteration 20; global step 4.000; 6 blocks, 64 filters | `6e29a4e3b8c235449898450de0f44b520829eda097ee74f97ef599238e52ffbc` |
| `checkpoints/renju_last.pt` | format 1; Renju; iteration 10; global step 2.000; 6 blocks, 64 filters | `42bdd1c941658ef91591224b0156d90ad6eeae0473e30f79eeda849392755870` |

Cấu hình trong cả hai checkpoint ghi 50 ván/iteration, 200 simulations, batch 128 và 200 bước train/iteration. Các metadata này phù hợp quy mô của hai log trên nhưng không chứng minh quan hệ nguồn gốc bằng mật mã giữa log và trọng số. Log Renju lưu đường dẫn nguồn, không lưu hash checkpoint nguồn lúc fine-tune; không thể chứng minh tệp Normal hiện tại chính là byte nguồn đã dùng ngày đó.

Môi trường tiến trình tích hợp quan sát: Python 3.10.6, PyTorch `2.14.1+cu126`, CUDA với RTX 4060 Laptop GPU. Đây là môi trường kiểm tra hiện tại, không phải bản kê môi trường lịch sử của từng run. Log không ghi đủ phiên bản thư viện, seed khởi tạo mạng và hash mã nguồn để bảo đảm tái lập bit-for-bit. `last.pt` là tên có thể bị ghi đè; muốn đối chiếu đúng artifact phải dùng hash, không chỉ tên.

Các số liệu trên là số đã có trong JSONL hoặc metadata được đọc, không phải số giả lập. Việc đọc nguồn và artifact không thay thế kiểm thử thực thi. Đánh giá sức chơi phải dựa trên các ván đối kháng thật, ghi rõ bộ luật, checkpoint, ngân sách chọn nước và kết quả đổi màu, tách khỏi kết luận từ loss huấn luyện.

## 9. Quy trình kiểm chứng thực thi

Các kết quả dưới đây do tiến trình tích hợp thực thi và báo lại; phần soạn báo cáo không chạy lại các phép kiểm tra. Chúng kiểm chứng tính đúng và khả năng chạy, không thay thế đánh giá sức chơi bằng tournament.

### 9.1. Các kiểm tra đã hoàn thành

| Nhóm kiểm tra | Kết quả quan sát | Thời gian |
| --- | --- | ---: |
| Điều kiện tiên quyết | 85 passed | 12,50 s |
| Evaluation, arena và tournament | 28 passed | 2,86 s |
| Hồi quy tích hợp | 1 passed | 1,50 s |
| Hồi quy AZ | 23 passed | 1,64 s |
| Hai gate `RUN_SLOW` | Cả hai đạt | 143,63 s |
| Alpha-Beta, MCTS Pure, CLI và UI | 14 passed | 4,70 s |

Hai gate `RUN_SLOW` chạy trên snapshot module fixed-simulation cũ, không phải bằng chứng trực tiếp cho toàn bộ mã cuối sau sửa ngân sách. Chế độ fixed-simulation cuối có kết quả hồi quy 23 passed; không gộp hai mốc kiểm tra thành tuyên bố mọi chế độ của mã cuối đều đã qua gate chậm.

### 9.2. Smoke run thật và phạm vi kết luận

Tiến trình tích hợp đã kiểm chứng smoke run thật của các biến thể AZ với cùng checkpoint, đối chiếu bằng hash, và ngân sách 200 simulations. Sáu ván có đổi màu cân bằng. Số lần override ghi nhận theo thứ tự sáu ván là `3, 1, 4, 2, 5, 3`. Đây là dấu vết wrapper thực sự hoạt động trong ván chạy, không phải số liệu giả lập hoặc chỉ kiểm tra khởi tạo đối tượng.

Sáu ván chỉ đủ làm smoke check đường thực thi và ghi log; không dùng để kết luận biến thể nào mạnh hơn, ước lượng tỷ lệ thắng đáng tin cậy hoặc xác nhận lợi ích của tactical/threat-space.

### 9.3. Quy tắc công bố kết quả T35

Mục này ghi quy tắc công bố từ giai đoạn kiểm chứng ban đầu. T35 hiện đã hoàn tất và được đối chiếu ở mục 10; các kết quả smoke hoặc tactical-only ngoài selection không cộng vào 2.400 ván chính thức.

Khoảng Wilson cho `score = (W + 0,5D)/n` là xấp xỉ với hòa tính nửa điểm, không phải khoảng Bernoulli chính xác hay khoảng cho win rate `W/n`. Nó chưa mô hình hóa đầy đủ hòa, phụ thuộc giữa các ván hoặc việc so nhiều cặp. Khoảng chứa 0,5 nghĩa chưa đủ bằng chứng score khác 0,5, không chứng minh hai tác tử tương đương. Seed cố định và kết quả theo cặp không bảo đảm suy rộng sang mọi đối thủ.

Kết quả đối kháng chỉ được đưa vào báo cáo khi run hoàn tất, có log từng ván cùng summary đối chiếu được. Mỗi kết quả cần giữ rõ bộ luật, cặp đối thủ và biến thể, hash checkpoint áp dụng, seed, cách phân màu, ngân sách chọn nước, số ván hoàn tất và thắng–hòa–thua. Cần phân biệt ngân sách fixed-simulation với fixed-time; không gọi hai phép đo là cùng điều kiện chỉ vì cùng tên agent. Mọi kết luận sức chơi phải giới hạn trong cặp đối thủ, bộ luật và ngân sách thực sự được đo.

### 9.4. Kiểm tra CLI trên mã cuối

Tiến trình tích hợp đã chạy lệnh thực sau trên mã cuối:

```bash
python scripts/play.py --mode ai-vs-ai --black-agent az_v1_tactical --white-agent az_v1_threat --black-checkpoint checkpoints/last.pt --white-checkpoint checkpoints/last.pt --time-budget 0.02 --seed 20261113
```

Quan sát: `White (O) wins` sau 22 nước, exit code 0; toàn bộ lệnh mất 2,93 s. Kết quả kiểm chứng đường CLI chọn hai biến thể, nạp checkpoint và hoàn tất ván thật. Tham số `--time-budget 0.02` không phải bằng chứng mỗi nước hoàn tất trong hạn cứng 0,02 s; thời gian toàn lệnh cũng không phải thời gian riêng của một lần chọn nước. Một ván này không chứng minh threat-space mạnh hơn tactical.

CLI tác tử cổ điển cũng đã chạy thật với `--black-agent alphabeta --white-agent mcts-pure --time-budget 0.03 --seed 20261113`: Trắng thắng sau 40 nước, exit 0, toàn lệnh 1,50 s. Alpha-Beta dùng iterative deepening và giữ kết quả của độ sâu hoàn tất cuối; MCTS Pure dừng rollout theo deadline. Hai tác tử giữ nguyên bàn đầu vào khi hết giờ. Deadline mềm vì không ngắt cưỡng bức một lần sinh nước hoặc đánh giá pattern đang chạy.

Các cặp `sims:200` được chạy song song theo tiến trình, mỗi tiến trình một thread CPU; độ trễ trong chế độ này phản ánh môi trường có tranh tài nguyên và không dùng làm benchmark thời gian cô lập. Chế độ `time:1` chạy riêng sau các run ngân sách cố định, không trộn vào cùng bảng score.

## 10. Kết quả T35 hoàn tất ngày 2026-11-14

### 10.1. Tập kết quả và thay đổi baseline

Đã hoàn tất **2.400 ván**, gồm Normal cố định 1.500, Renju cố định 600 và Normal 1 giây/nước 300. Mỗi cặp 100 ván, mỗi tác tử cầm Đen 50 ván; seed gốc `20261113`. Không có ván hòa trong tập được chọn.

Người dùng đã đổi baseline cố định từ Alpha-Beta depth 2 sang **depth 1**. Tám cặp có Alpha-Beta được chạy mới đủ 800 ván; 13 cặp không có Alpha-Beta giữ 1.300 ván vì tham số depth không tác động tới chúng. Log depth 2, smoke và run gián đoạn giữ riêng, không cộng vào 2.400 ván. `sims:200` áp dụng cho AZ/MCTS Pure; Alpha-Beta cố định dùng depth 1, không phải 200 node. Nhóm `time:1` dùng iterative deepening cho Alpha-Beta, **không khóa depth 1**.

Nguồn đóng băng: [selection](../evidence/colab_return_20261114T085015Z/evidence/depth1_selection.json), [checksum hoàn tất](../evidence/colab_return_20261114T085015Z/colab_completed.json), [phân tích máy đọc được](../evidence/colab_return_20261114T085015Z/t35_analysis.json). Công cụ [analyze_t35.py](../scripts/analyze_t35.py) kiểm tra checksum, 24 cặp × 100 ID liên tiếp, seed, màu, cấu hình ngân sách và số nước rồi tính trực tiếp từ JSONL; không dựa vào summary có thể cũ. Checkpoint theo SHA-256 ở mục 8.3.

Cặp Normal Random–AlphaBeta từng có 101 dòng do ID 88 bị ghi hai lần. Trước khi khôi phục đã xác nhận hai bản ghi chỉ khác timestamp, duration và latency từng nước; giữ bản xuất hiện đầu tiên, không chọn bản nhanh hơn. Backup và audit `dedup_*.json` nằm cạnh nguồn đã chuyển về. Chưa xác định chắc nguyên nhân ghi trùng. Tập cuối không có ID trùng hoặc thiếu.

### 10.2. Thắng–hòa–thua từng cặp

W/D/L và khoảng Wilson 95% đều từ góc nhìn **A**. Hai cột màu là số ván A thắng trong 50 ván A cầm Đen hoặc Trắng. Không gộp hai bộ luật hoặc hai ngân sách.

#### Normal — ngân sách cố định

| A | B | W/D/L của A | A thắng Đen/50 | A thắng Trắng/50 | Score A | Wilson 95% |
|---|---|---:|---:|---:|---:|---|
| random | heuristic | 0/0/100 | 0 | 0 | 0.00% | [0.000; 0.037] |
| random | alphabeta | 4/0/96 | 2 | 2 | 4.00% | [0.016; 0.098] |
| random | mcts-pure | 0/0/100 | 0 | 0 | 0.00% | [0.000; 0.037] |
| random | az_v0 | 2/0/98 | 2 | 0 | 2.00% | [0.006; 0.070] |
| random | az_v1_threat | 0/0/100 | 0 | 0 | 0.00% | [0.000; 0.037] |
| heuristic | alphabeta | 100/0/0 | 50 | 50 | 100.00% | [0.963; 1.000] |
| heuristic | mcts-pure | 100/0/0 | 50 | 50 | 100.00% | [0.963; 1.000] |
| heuristic | az_v0 | 100/0/0 | 50 | 50 | 100.00% | [0.963; 1.000] |
| heuristic | az_v1_threat | 100/0/0 | 50 | 50 | 100.00% | [0.963; 1.000] |
| alphabeta | mcts-pure | 2/0/98 | 1 | 1 | 2.00% | [0.006; 0.070] |
| alphabeta | az_v0 | 50/0/50 | 50 | 0 | 50.00% | [0.404; 0.596] |
| alphabeta | az_v1_threat | 0/0/100 | 0 | 0 | 0.00% | [0.000; 0.037] |
| mcts-pure | az_v0 | 99/0/1 | 50 | 49 | 99.00% | [0.946; 0.998] |
| mcts-pure | az_v1_threat | 31/0/69 | 12 | 19 | 31.00% | [0.228; 0.406] |
| az_v0 | az_v1_threat | 0/0/100 | 0 | 0 | 0.00% | [0.000; 0.037] |

#### Renju — ngân sách cố định

| A | B | W/D/L của A | A thắng Đen/50 | A thắng Trắng/50 | Score A | Wilson 95% |
|---|---|---:|---:|---:|---:|---|
| random | heuristic | 0/0/100 | 0 | 0 | 0.00% | [0.000; 0.037] |
| random | alphabeta | 2/0/98 | 0 | 2 | 2.00% | [0.006; 0.070] |
| random | az_renju | 0/0/100 | 0 | 0 | 0.00% | [0.000; 0.037] |
| heuristic | alphabeta | 100/0/0 | 50 | 50 | 100.00% | [0.963; 1.000] |
| heuristic | az_renju | 100/0/0 | 50 | 50 | 100.00% | [0.963; 1.000] |
| alphabeta | az_renju | 50/0/50 | 50 | 0 | 50.00% | [0.404; 0.596] |

#### Normal — 1 giây/nước

| A | B | W/D/L của A | A thắng Đen/50 | A thắng Trắng/50 | Score A | Wilson 95% |
|---|---|---:|---:|---:|---:|---|
| az_v0 | alphabeta | 50/0/50 | 20 | 30 | 50.00% | [0.404; 0.596] |
| az_v0 | az_v1_threat | 0/0/100 | 0 | 0 | 0.00% | [0.000; 0.037] |
| az_v1_threat | alphabeta | 87/0/13 | 47 | 40 | 87.00% | [0.790; 0.922] |

### 10.3. Xếp hạng trong từng đội hình

Score tổng = `(W + 0,5D) / số ván`. Mỗi agent Normal cố định chơi 500 ván, Renju 300 ván, Normal thời gian 200 ván. Không so trực tiếp score tổng giữa các đội hình khác nhau.

| Nhóm | Agent | W/D/L | Score |
|---|---|---:|---:|
| normal_fixed | heuristic | 500/0/0 | 100.00% |
| normal_fixed | az_v1_threat | 369/0/131 | 73.80% |
| normal_fixed | mcts-pure | 328/0/172 | 65.60% |
| normal_fixed | az_v0 | 149/0/351 | 29.80% |
| normal_fixed | alphabeta | 148/0/352 | 29.60% |
| normal_fixed | random | 6/0/494 | 1.20% |
| renju_fixed | heuristic | 300/0/0 | 100.00% |
| renju_fixed | az_renju | 150/0/150 | 50.00% |
| renju_fixed | alphabeta | 148/0/152 | 49.33% |
| renju_fixed | random | 2/0/298 | 0.67% |
| normal_time | az_v1_threat | 187/0/13 | 93.50% |
| normal_time | alphabeta | 63/0/137 | 31.50% |
| normal_time | az_v0 | 50/0/150 | 25.00% |

### 10.4. Phân tích sức chơi và giới hạn

- **Heuristic đứng đầu hai bảng cố định:** Normal 500/500 và Renju 300/300. AZ hiện tại chưa vượt baseline heuristic; thắng Random không đủ chứng minh sức chơi tốt.
- **Threat cải thiện lớp tìm kiếm ở cùng trọng số Normal:** thắng AZ gốc 100–0 ở cả ngân sách cố định và thời gian. Với MCTS Pure, threat thắng 69–31 (Wilson 95% cho threat khoảng [0,594; 0,772]), trong khi AZ gốc thua 1–99. Tuy vậy threat vẫn thua heuristic 0–100; không kết luận mạng đã học được chiến thuật đầy đủ.
- **Không diễn giải 50–50 là ngang sức:** Alpha-Beta depth 1 gặp AZ gốc Normal và AZ Renju đều có Đen thắng 100/100. Hai agent chia 50–50 chỉ vì đổi màu, không phải cân bằng trên mọi vị trí. Với nhóm thời gian, AZ gốc–AlphaBeta cũng 50–50 nhưng Đen chỉ thắng 40/100.
- **Nhóm 1 giây/nước:** threat thắng AlphaBeta 87–13, gồm 47/50 khi cầm Đen và 40/50 khi cầm Trắng; Wilson 95% [0,790; 0,922]. So với 100–0 ở depth 1 cố định, đây là baseline đối thủ/ngân sách khác, không phải hồi quy có kiểm soát của cùng phép đo.
- **Renju:** AZ Renju thắng Random 100–0, thua heuristic 0–100, chia 50–50 với AlphaBeta depth 1 do lợi thế màu như trên. Không có đối chứng trước fine-tune cùng luật và ngân sách nên chưa đo được lợi ích nhân quả của fine-tuning.
- **Độ độc lập hạn chế:** nhiều cặp có ván lặp tất định theo màu, khởi đầu bàn trống và seed cố định. 100 ván không đồng nghĩa 100 tình huống chiến thuật độc lập. Wilson chỉ là thống kê mô tả danh nghĩa, chưa hiệu chỉnh phụ thuộc hoặc so nhiều cặp; 100–0 không bảo đảm thắng tuyệt đối ngoài tập này.

### 10.5. Benchmark và môi trường

Nhóm cố định được chạy/tiếp tục trên Windows RTX 4060 Laptop, Colab Tesla T4 và có phiên Colab CPU trong hồ sơ môi trường. Không có nhãn thiết bị đáng tin cậy cho từng ván xuyên suốt các lần resume; không gộp latency cố định thành benchmark cùng máy, không dùng chênh lệch duration depth 1/depth 2 để tuyên bố tốc độ thuật toán đã tăng một tỷ lệ cụ thể.

**Toàn bộ 300 ván thời gian chạy local** trên Windows, Python 3.10.6, PyTorch 2.14.1+cu126, RTX 4060 Laptop, 16 CPU logic; 3 tiến trình chia sẻ CPU/GPU, mỗi tiến trình một thread PyTorch. Tên thư mục còn `colab` vì giữ tương thích đường dẫn, không phải vị trí thực thi. Nguồn môi trường: `local_return_20261114T085301Z.json` và `colab_environment_20261114T085303143443Z.json` trong thư mục bằng chứng.

Nhóm thời gian bắt đầu khoảng 16:43:39, hoàn tất 17:23:11 giờ local: khoảng **39 phút 32 giây**, throughput khoảng **455 ván/giờ**. Tổng duration của 300 ván là **6.574,36 giây (109,57 phút)**, là tổng công việc các tiến trình chồng lấp, không phải wall-clock. Tỷ số khoảng 2,77 giữa tổng duration và wall-clock không phải speedup được benchmark: không chạy đối chứng tuần tự trên cùng điều kiện. Lần tiếp tục local từ 1.914 đến 2.400 ván mất 5.411,73 giây, khoảng 90,20 phút.

Xem [benchmark từng cặp, mean/p95 latency và lượng tìm kiếm](benchmarks.md#t35--benchmark-giai-dau-hoan-tat-2026-11-14). Deadline 1 giây là mềm; nước tactical kết thúc sớm, thao tác đang thực thi có thể vượt deadline. MCTS nodes không có counter thì giữ `null`, không coi là 0; nodes của threat và AlphaBeta cũng không tương đương simulations.
