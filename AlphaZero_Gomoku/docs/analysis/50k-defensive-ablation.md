# Nghiên cứu phòng thủ AlphaZero Normal 50k

## Phạm vi và protocol

Checkpoint cố định `checkpoints/az_25k_continuation_v1/generations/f43be9e336474950a382a64f15f2ca80/checkpoint.pt`, SHA-256 `28ea9001fd83c1c7b4d62117a4d7e4a1726c95eb02b8a37f39298614ea35c5cd`. Không self-play, training, đổi weights, promote hoặc tích hợp tactical override.

Tập đánh giá lấy từ SQLite tournament được tạo **sau checkpoint 50k**, không tự gọi dữ liệu self-play từng dùng training là holdout độc lập. Chia theo family, game và opening seed; kiểm tra D4 và provenance. Kiểm tra exact D4 so với replay snapshot không loại trừ gần trùng hoặc lịch sử training đã bị eviction. State trong tournament do 50k/opponent sinh ra nên có selection bias: không khẳng định đại diện mọi opening hoặc trạng thái có thể gặp.

Bốn nhánh offline, cùng checkpoint, Python MCTS, cùng state/seed, Normal, no root noise, temperature 0, `c_puct=1.5`: baseline 200; budget 400; prior đồng đều **chỉ tại root** với 200; value mạng thay bằng 0 tại mọi leaf nonterminal với 200. Terminal value/rules không đổi. Seed từng state lấy từ SHA-256 game ID và history, giữ giống nhau giữa nhánh. Budget 400 là **ngoại lệ thí nghiệm offline do người dùng yêu cầu**, không thay đổi contract `SPEC.md` hay cấu hình production 200. Uniform prior và zero value là can thiệp chẩn đoán, không phải phương án training hay tactical override.

Báo từng màu và family, số case sửa được (baseline sai, intervention đúng) và regression mới (baseline đúng, intervention sai), chi phí wall-clock và NN calls. Nhãn chỉ chứng nhận tactical horizon của oracle, không chứng minh minimax outcome đầy ván; các state cùng game/family không độc lập. Không suy ra causal training mechanism hoặc strength từ một benchmark.

Lần đầu chạy 32 state trên CPU hết deadline 1.800 s sau 27 state; runner cũ chỉ ghi file khi hoàn tất nên không còn raw results để dùng. Chạy lại trên CUDA với cùng checkpoint, seed và nhánh; ghi thêm JSONL sau từng state. Không so sánh trực tiếp timing hay các case sát ngưỡng giữa GPU và CPU như thể là cùng execution environment.

## Dataset và giới hạn độc lập

Artifact `defensive_holdout_50k_data/dataset_v1.json`; mã tái lập `defensive_holdout_50k_data/build.py`. Nguồn là 800 ván trong hai SQLite tournament 50k–40k và 50k–Heuristic, đều bắt đầu sau khi checkpoint 50k được ghi. Mỗi case lưu history, game ID, opening seed, SHA checkpoint/SQLite và fingerprint state. Lựa chọn theo hash, family và màu, **không chọn theo model đúng hay sai**.

| Split | Single-block Đen / Trắng | Pre-fork Đen / Trắng | Tổng |
|---|---:|---:|---:|
| Development | 7 / 10 | 6 / 10 | 33 |
| Holdout | 2 / 10 | 10 / 10 | 32 |

Tập 65 state không trùng game, opening seed, family signature hoặc state D4 chính xác giữa hai split; không trùng D4 chính xác với hai snapshot replay 40k/50k, mỗi snapshot 50.000 features. Không kiểm tra phần replay đã bị eviction, history 25k ban đầu hoặc near-duplicate. Single-block Đen chỉ có **2** case holdout; không thể suy ra độ chính xác tổng quát theo màu từ nhóm này.

Oracle quét độc lập 252 đoạn năm ô. Single-block yêu cầu bên đi không có thắng ngay và đối thủ chỉ có một ô thắng. Pre-fork lấy state trước một nước tạo fork thực tế của đối thủ; tất cả nước phòng thủ được liệt kê, chấp nhận nước thắng ngay hoặc nước để đối thủ không có immediate win/fork ở lượt kế. Bộ lọc fork hình học bảo thủ: đây là metric **ngăn đe dọa trong horizon ngắn**, không chứng minh mọi nước khác thực sự thua hay state cứu được cả ván. Trong tập candidate, 71 state không có defense được chứng minh theo định nghĩa này bị loại. Các case pre-fork bị điều kiện hóa vào trajectory tương lai, không đại diện toàn bộ phân phối state.

Đối chiếu oracle thứ hai `scripts.tactical_benchmark.oracle`: **12/12 single-block** có tập đáp án giống hệt; **13/20 pre-fork** giống hệt và **7/20** có tập đáp án dataset là tập con chặt. Oracle thứ hai cho phép thêm nước khi bên phòng thủ có thắng ngay ở lượt sau; dataset dùng tiêu chí hình học chặt hơn. Vì vậy case không đạt tiêu chí chặt **không tự động là nước thua**. Kết quả dưới đây phải phân biệt metric dataset chặt và metric oracle thứ hai; đáp án đối chiếu lưu tại `defensive_holdout_50k_eval/second_oracle_v1.json`.

## Ablation trên sáu case gate cũ — exploratory

Artifact `defensive_gate_ablation_50k/analyze.py` và `raw.json`. Sáu biến thể cùng family, seed 5314159, 3 Đen/3 Trắng; oracle quét cửa sổ xác nhận duy nhất ô chặn. Đây không phải holdout độc lập; 400 simulations là ngoại lệ offline.

| Nhánh | Tổng | Đen | Trắng | Sửa / regression so 200 | NN calls | Tổng search |
|---|---:|---:|---:|---:|---:|---:|
| Baseline 200 | 2/6 | 0/3 | 2/3 | — | 747 | 83,90 s |
| 400 | 2/6 | 0/3 | 2/3 | 0 / 0 | 815 | 87,17 s |
| Uniform root, 200 | 0/6 | 0/3 | 0/3 | 0 / 2 | 679 | 72,31 s |
| Zero nonterminal value, 200 | 6/6 | 3/3 | 3/3 | 4 / 0 | 728 | 77,69 s |

Trong đúng sáu state này, đặt value nonterminal bằng 0 thay đổi lựa chọn và sửa bốn lỗi; đặt prior root đồng đều tạo hai regression. Điều này cho thấy value **có ảnh hưởng nhân quả tới quyết định trong phép can thiệp**, không chứng minh value mạng sai theo kết quả ván hoàn chỉnh, là nguyên nhân duy nhất, hay zero value tốt cho sức chơi. Terminal không gọi network nên NN calls không bằng simulations; thời gian chịu tải máy, không phải benchmark tốc độ đáng tin cậy.

## Ablation trên 32 state holdout

`defensive_holdout_50k_eval/run.py`, raw `results_v1.json` SHA-256 `e5efe77ce4899e422dc0114878aa06caed4f0c4d8142e5a9431789127aca7c2f`; dataset SHA-256 `e45d33b50b849eed87af517e127afa7f116df20a3af2c90177e5f249b3c4c9aa`. Python MCTS, inference CUDA, một CPU thread; tất cả state thực hiện đúng budget. Tổng wall chạy lại 53,75 s; riêng nhánh search lần lượt 11,01 / 21,61 / 10,56 / 10,52 s. Lần CPU thất bại trước đó tốn thêm 1.800 s không có raw hoàn chỉnh.

### Chấm theo oracle dataset chặt

| Nhánh | Tổng | Đen | Trắng | Sửa Đen / Trắng | Regression Đen / Trắng | NN calls |
|---|---:|---:|---:|---:|---:|---:|
| Baseline 200 | 22/32 | 7/12 | 15/20 | — | — | 6.149 |
| 400 | 23/32 | 8/12 | 15/20 | 1 / 1 | 0 / 1 | 12.146 |
| Uniform root, 200 | 21/32 | 7/12 | 14/20 | 1 / 0 | 1 / 1 | 6.101 |
| Zero value, 200 | 22/32 | 6/12 | 16/20 | 1 / 2 | 2 / 1 | 5.863 |

Theo category, baseline single-block Đen 2/2, Trắng 8/10; pre-fork Đen 5/10, Trắng 7/10. Với 400, tương ứng 2/2, 8/10, 6/10 và 7/10; uniform root 1/2, 8/10, 6/10 và 6/10; zero value 1/2, 8/10, 5/10 và 8/10. Nhóm single-block Đen chỉ hai case.

### Chấm theo oracle đối chiếu

| Nhánh | Tổng | Đen | Trắng | Sửa Đen / Trắng | Regression Đen / Trắng |
|---|---:|---:|---:|---:|---:|
| Baseline 200 | 23/32 | 8/12 | 15/20 | — | — |
| 400 | 25/32 | 9/12 | 16/20 | 1 / 1 | 0 / 0 |
| Uniform root, 200 | 21/32 | 7/12 | 14/20 | 0 / 0 | 1 / 1 |
| Zero value, 200 | 26/32 | 8/12 | 18/20 | 1 / 3 | 1 / 0 |

Đây là hai tiêu chí tactical horizon khác nhau, không phải hai ước lượng độc lập. Chấm oracle đối chiếu cho phép các nước bỏ sót bởi tiêu chí hình học chặt; kết quả vẫn không chứng minh nước đi thắng hoặc hòa. 400 có gần gấp đôi NN calls và thời gian của baseline trong workload này, nhưng chỉ sửa hai case theo oracle thứ hai; không có cơ sở đưa 400 vào production hoặc kết luận cải thiện Elo. Zero value cải thiện metric oracle thứ hai nhưng có regression và không cải thiện tổng metric chặt; không tích hợp zero value.

## Truy nguyên hai lỗi single-block Trắng

Trace tái lập `defensive_holdout_50k_eval/trace_missed.py`, raw `missed_trace_v1.json`; cùng checkpoint, seed từng state và 200 simulations. Rules xác nhận nước đối thủ trong bảng thắng ngay sau lựa chọn sai.

| Game | Block đúng | Chọn sai | Prior block / chọn | Visits block / chọn | Q block / chọn | Đối thủ thắng: visits trong cây con |
|---|---:|---:|---:|---:|---:|---:|
| Heuristic `:217` | 26 | 75 | 0,163825 / 0,015152 | 23 / 58 | −0,32950 / −0,18977 | 26/58 |
| 40k `:387` | 43 | 101 | 0,000046 / 0,035221 | 0 / 45 | 0 / +0,33518 | 4/45 |

Case `:387` có nguyên nhân phân bổ quan sát được: prior block cực thấp và không được thăm ở root ngay cả tại 400; nhánh sai đã thấy nước đối thủ thắng nhưng chỉ 4/45 visits. Case `:217` có prior block cao hơn nước sai nhưng Q trung bình tại root và visit cuối ưu tiên nước sai; nhánh sai đã thấy terminal thắng của đối thủ 26/58 visits. Terminal detection hoạt động trên cả hai trace; MCTS chuẩn dùng trung bình value/visits, không chứng minh sai implementation chỉ từ việc chọn sai. Root NN value lần lượt +0,69475 và +0,42255 nhưng chưa có ground truth minimax để gọi value sai calibration. Những can thiệp thử không sửa hai single-block này; uniform root cho case `:387` thăm block 18 lần nhưng vẫn chọn sai.

## Kết luận và phương án ưu tiên

**Phương án đáng thử trước: thí nghiệm cải thiện/chẩn đoán value calibration theo game**, không thay value bằng 0 trong production. Lý do: zero-value sửa bốn case gate và, theo oracle đối chiếu, sửa bốn/thoái hóa một trên holdout; nhưng metric chặt không tăng và hai lỗi single-block Trắng vẫn còn. Đây là bằng chứng value ảnh hưởng quyết định, **chưa** chứng minh value sai hoặc tăng trọng số value loss sẽ cải thiện model.

Thiết kế tiếp theo: trước hết đo value prediction so với outcome trên các game/seed độc lập, tách màu và nhóm đe dọa; outcome phụ thuộc đối thủ nên không đồng nhất với minimax value. Sau đó, chỉ nếu phát hiện sai calibration có hệ thống, so sánh baseline với **một** thay đổi value-training dựa trên kết quả đó: cùng checkpoint 50k, dữ liệu có provenance, seed, số game/update, batch, 200 simulations và tài nguyên. Theo dõi đồng thời policy, value, single-block/pre-fork, regression theo màu và đối kháng paired với 40k/Heuristic. Không chọn trọng số hay lịch LR cụ thể từ số liệu hiện tại. Đóng băng một holdout mới theo game/seed/family trước khi chọn nhánh; tập 32 state ở đây đã được dùng để phân tích và không còn là kiểm định cuối độc lập cho lựa chọn này.

**Chi phí:** thí nghiệm offline bốn nhánh trên 32 state dùng 53,75 s CUDA sau lần CPU timeout 1.800 s; 400 simulations dùng 12.146 NN calls so với baseline 6.149 và gần gấp đôi thời gian search trong phép đo này. Một lần chấm 32 state ở 200 simulations tốn 6.149 calls, 11,01 s search ở workload đo được; con số không gồm mọi chi phí dựng dữ liệu, CPU timeout hay tournament. Chi phí pilot training chưa được đo, không có cơ sở dự đoán wall time hay lợi ích.

**Rủi ro:** điều chỉnh value loss có thể làm policy hoặc sức chơi giảm; ưu tiên những state chiến thuật có thể làm lệch phân phối replay và lặp lại thất bại tactical training trước đó. Hai lỗi block thực tế còn gợi ý cả thiếu prior lẫn giá trị trung bình/search, nên chỉ cải thiện value có thể không đủ. Nếu pilot không cải thiện holdout mới và paired games trong uncertainty đã đăng ký, giữ checkpoint 50k và không tích hợp thay đổi.

Không có bằng chứng để chọn 400 simulations, uniform root hoặc zero value làm cấu hình production. Không training, self-play, promote hoặc thay weights trong nghiên cứu này.
