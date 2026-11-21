# AlphaZero 25k — final readiness

## Quyết định

**Chưa nên bắt đầu continuation quy mô vừa.** Recovery đã vượt các kiểm thử process-crash, rollback và lineage được nêu dưới đây, nhưng chưa có bằng chứng rằng recipe training cải thiện phòng thủ hay sức chơi; evaluation chỉ có một training seed và bốn opening. Durability khi mất điện trên Windows và concurrent writers cũng chưa được chứng minh.

Giữ ε=0,25 và uniform replay sampling làm **baseline**, không tích hợp tactical override hoặc sampling thử nghiệm. Parent 25k chỉ dùng làm weights nguồn với SHA đã xác nhận; replay continuation phải mới, có game IDs và provenance, không coi legacy buffer là generation hợp lệ. Cấu hình ở cuối báo cáo là **đề xuất có điều kiện**, chưa phải lệnh cho phép chạy.

## 1. Recovery Windows và lỗi đã sửa

### Bằng chứng đã đạt

- Kiểm thử Windows dùng subprocess termination tại các seam game SQLite, generation pointer và restart; test sau game commit xác nhận không regenerate self-play, giữ nguyên π/z và payload, resume hai lần không nhân đôi metrics. Các test khác bao phủ outbox hash mismatch, source/game-ID collision, ring-buffer overwrite, lineage và rollback. Test của tác vụ này: `pytest -q -p no:cacheprovider tests/test_train_resume_logging.py tests/test_trainer.py tests/test_selfplay.py tests/test_logging_v2.py tests/test_native_mcts.py tests/test_tactical_benchmark.py` → **79 passed**.
- Trên bản sao mới `az_finalprobe_71723je1/rollback_copy`, rollback generation `63d4eb270cee4277bb166a25f44f2a71`, run ID `final-rollback-111`, hai worker: phục hồi iteration 1/step 8, tạo bốn ván, commit iteration 2/step 16. Manifest mới ghi model parent là generation được chọn, `commit_previous` là tip cũ `0c7d48ace7a141838ff53ec2b424ff79`; tip cũ vẫn resolve được. Cả bốn game ID mới có trong replay, mọi replay ID có record SQLite, worker tensor digest khớp checkpoint nguồn; replay có 78 positions. Đây là kiểm thử trên bản sao, không sửa run nguồn.
- Báo cáo [25k-recovery-and-tactics.md](25k-recovery-and-tactics.md) đã ghi kiểm thử `os._exit(73)` sau SQLite commit trên bốn ván self-play thực: hai lần resume giữ nguyên 36 targets và một metrics record. Bộ test mới chạy lại bao gồm kiểm thử process-exit đó.

### Bug đã tái hiện và sửa

1. **Resume nhầm run ID.** Với checkpoint đã có recovery metadata, trước sửa CLI cho phép STANDARD resume bằng run ID khác. Agent chạy regression đỏ trước sửa. Kiểm tra thêm tái hiện hai biến thể trên generation ban đầu chưa có recovery metadata: run ID không tồn tại được tạo mới, và run ID khác đã tồn tại với cùng config cũng được chấp nhận. `scripts/train.py` nay yêu cầu run ID gốc tồn tại trong SQLite và khớp owner lưu trong checkpoint hoặc recovery metadata. `az/trainer.py` lưu `run_id` từ generation đầu. Generation cũ thiếu cả hai identity phải dùng explicit `--generation` với run ID mới, không đoán owner. Regression mới thất bại trước sửa và pass trong bộ 79 test.
2. **Retry rollback sau crash trước generation commit bị chặn.** Test tái hiện: rollback run mới đã ghi game/outbox rồi process dừng; chạy lại đúng `--generation` và run ID bị từ chối vì run ID đã tồn tại. Nay metadata ghi generation và SHA nguồn rollback; retry trước commit được phép chỉ với cùng nguồn/config, dùng lại outbox. Sau commit, dùng `--resume` không kèm `--generation`; chọn nguồn khác bằng cùng run ID bị từ chối. Regression thất bại trước sửa và pass sau sửa; không xóa record, outbox hoặc nhánh cũ.

`SPEC.md` ghi contract mới. Không có test mất điện thật; `_sync_directory` trên Windows không chứng minh được độ bền directory entry. Một run/output directory chỉ hỗ trợ **một writer**; không khẳng định an toàn khi concurrent writers. Nếu crash trước outbox commit, game chưa commit không được bảo đảm phục hồi; retraining sau crash không được khẳng định bitwise-identical. Graphify đã cập nhật; CLI cảnh báo phiên bản package và skill khác nhau.

## 2. Dữ liệu self-play lịch sử

SQLite được mở read-only. Cohort có lineage trong `data/selfplay.sqlite` là run `az_v0-15000-continuation-20261119`, 15.000 record, model_iter 200–499. Hai cửa sổ dưới đây là **ván self-play trước mốc danh nghĩa**, không phải đánh giá độc lập của checkpoint tại mốc đó:

| Chỉ số | Iter 200–299, trước mốc 15k | Iter 400–499, trước mốc 25k |
|---|---:|---:|
| Ván | 5.000 | 5.000 |
| Đen thắng | 4.973 (99,46%) | 4.976 (99,52%) |
| Độ dài 9 / 11 ply | 3.157 / 1.416 | 3.251 / 1.276 |
| Nước đầu ở tâm | 4.996 | 4.982 |
| Prefix khác nhau sau 2 / 4 / 6 ply | 124 / 3.895 / 4.911 | 138 / 3.828 / 4.901 |
| Trajectory đầy đủ khác nhau | 5.000 | 5.000 |

Toàn cohort 15.000 ván: Đen thắng 14.931; 13.657 ván kết thúc sau 9 hoặc 11 ply. Chỉ một cặp trajectory hoàn toàn trùng trong toàn cohort: **không được đồng nhất opening tập trung với mọi ván trùng nhau**. Gomoku Normal có lợi thế đi trước, nhưng tỷ lệ ~99,5%, thời lượng ngắn và opening gần như cố định là bằng chứng lệch phân phối self-play rất mạnh; không định lượng được phần do lợi thế tự nhiên, không tự chứng minh lỗi luật hay nguyên nhân nhân quả của mức thắng Heuristic.

Oracle cửa sổ năm ô trên mẫu có hệ thống *một game local ID nhỏ nhất mỗi iteration*, 100 ván/cửa sổ: cửa sổ trước 15k có 97 lượt Trắng đối diện ít nhất hai nước thắng ngay của Đen; cửa sổ trước 25k có 100 lượt. Đen có thắng ngay 100 lượt và chọn thắng đủ 100 trong cả hai mẫu. Mẫu không ngẫu nhiên, không phải tỷ lệ toàn cohort; hai mối đe dọa thường hình thành **trước** lượt cuối nên không thể coi mọi thất bại là một nước không chặn đơn. Giả thuyết cần kiểm tra: policy/value tự củng cố các prefix và không ngăn hình thành fork; chưa chứng minh nguyên nhân bằng can thiệp.

`checkpoints/buffer_latest.npz` lịch sử SHA `908a22b249591fdbf64fb1f7430de879d0b94e2925be4e619a5db484e0170fb9`: 50.000 slots, 28.583 feature byte khác nhau, 42.793 bộ `(feature, π, z)` khác nhau; bàn trống 4.906 lần, trung vị ba action có policy mass dương. Exact bytes chưa gộp D4; z +1/−1 là 27.440/22.560 **theo người sắp đi**, không phải tỷ lệ thắng Đen/Trắng. Buffer thiếu game ID/cursor và chưa chứng minh ghép cùng checkpoint hay optimizer đã dùng từng sample; không sử dụng nó làm replay hoặc holdout đã xác minh. Candidate tournament lịch sử SHA `238404…` khác byte SHA `latest.pt` `f7f521…`; chưa chứng minh tensor giống hoặc khác.

## 3. Thí nghiệm training một biến

### Protocol và provenance

Dùng **12 game iteration 0** từ ba run ε=0,25 `az_contpilot_wzdps5fu/baseline_{111,222,333}` đã có generation và SQLite provenance. Đối chiếu SHA generation nguồn, tensor worker và parent `checkpoints/latest.pt` SHA-256 `f7f521c1cc37a5054ed854201b087e484e9fcef3d4d6b1877ab81dff2e51d7e4`. Không dùng legacy NPZ. Tám game seed 111/222 cho training: 82 positions; bốn game seed 333 cho holdout: 36 positions. Loại 12 state holdout trùng training dưới D4, còn **24 positions**; tách theo game và loại duplicate state, nhưng không biết exposure đối với lịch sử 25k.

Hai model độc lập khởi tạo cùng parent weights, Adam reset, Normal, cùng 82 state/π/z, cùng RNG seed 9173 và chuỗi augmentation, batch 32, **32 optimizer updates**, LR `1e-4`, cosine horizon 32; không đổi ResNet, luật hay 200 simulations. Nhánh **uniform** lấy sample đều; nhánh **late50** lấy 50% từ state có ít nhất 6 quân và 50% từ phần còn lại. Không sửa target, không thêm tactical override. Tỷ lệ late thực tế trong 1.024 draws: 39,36% so với 49,61%; mỗi nhánh cùng số forward/backward, nhưng thứ tự/slot sample khác và thời gian wall không bằng tuyệt đối.

Đây là thí nghiệm **offline trên replay self-play có provenance**, không phải hai run self-play mới hay evidence rằng late sampling sẽ tốt hơn khi tự sinh dữ liệu. Train loss khác batch không dùng làm quality gate. Training wall: uniform 0,454 s, late50 0,176 s; lần đầu có thể chịu GPU warmup, không suy ra speedup. Mã, split, checkpoints, hash và raw output ở `az_finalprobe_71723je1/experiment.py`, `results.json`, `evaluation.json`, `novel.json`. Checkpoint cuối uniform SHA `ba7a1ab5925d8d0089ac6c923d2565cf5d8044781d0c99d6016423bd07679ce7`; late50 SHA `579a860be8f5e2cc5dcb564a3dbdead96b26db8f425586894161168cdede1a4f`.

### Policy/value trên game holdout

| Metric trên 24 state từ bốn game giữ lại | Parent | Uniform | Late50 |
|---|---:|---:|---:|
| Policy cross-entropy theo target MCTS | 2,18323 | 2,18574 | 2,18431 |
| Value MSE theo outcome self-play | 0,000365 | 0,0000905 | 0,0000820 |
| Value MSE Đen, n=14 | 0,000525 | 0,000118 | 0,0000981 |
| Value MSE Trắng, n=10 | 0,000141 | 0,0000519 | 0,0000594 |

Các game đều đến từ phân phối self-play cực lệch Đen; value có thể khớp outcome theo màu mà không học đánh giá tổng quát. Số liệu không chứng minh calibration trước Heuristic. Policy CE nhỏ khác nhau không chứng minh late50 tốt hơn; các position trong cùng game không độc lập và chỉ có bốn game holdout.

### Tactical holdout

Mỗi search dùng C++ MCTS, **200 simulations**, seed 420, no noise. Tập 24 state holdout hiện có không trùng D4 với replay train; các biến thể cùng family tương quan và tập này đã được dùng trong nghiên cứu trước. Tập bổ sung mới gồm ba cấu hình ngang/dọc/chéo × win/block × hai màu, oracle cửa sổ 5 ô, history hợp lệ/nonterminal; 12 state phân biệt, không trùng D4 với replay train, 72 fixture cũ hoặc 12 fixture của thí nghiệm trước. Không chứng minh được tách khỏi historical 25k.

| Metric | Parent | Uniform | Late50 |
|---|---:|---:|---:|
| Holdout cũ tổng | 16/24 | 15/24 | 15/24 |
| Holdout cũ win / block | 0/6, 4/6 | 0/6, 3/6 | 0/6, 3/6 |
| Holdout mới tổng | 11/12 | 11/12 | 11/12 |
| Holdout mới Đen / Trắng | 6/6, 5/6 | 6/6, 5/6 | 6/6, 5/6 |
| Prior đáp án oracle trung bình, holdout cũ | 0,4451 | 0,4496 | 0,4451 |
| Root visits đáp án trung bình, holdout cũ | 0,4087 | 0,4343 | 0,4294 |

Không có cải thiện accuracy chiến thuật giữa hai nhánh trên các tập này, và cả hai vẫn 0/6 win trên họ broken-four cũ. Số case ít, các case cùng family tương quan; không suy ra can thiệp chắc chắn vô ích trên phân phối rộng hơn.

### Đối kháng paired và uncertainty

Bốn opening seed 2501–2504, mỗi opening bốn nước hợp lệ, không terminal hay có thắng ngay, vị trí opening không trùng D4 train; đổi màu theo cặp. Mỗi model gặp cùng Heuristic trên tám ván, C++ MCTS 200 simulations/nước, không override. Tổng 24 ván; history và kết quả từng ván trong `evaluation.json`.

| Model | AZ Đen | AZ Trắng | Tổng | Brier Đen / Trắng, n=4 mỗi màu |
|---|---:|---:|---:|---:|
| Parent | 0/4 | 0/4 | 0/8 | 0,829 / 0,00014 |
| Uniform | 0/4 | 2/4 | 2/8 | 0,942 / 0,499 |
| Late50 | 0/4 | 1/4 | 1/8 | 0,812 / 0,249 |

Brier dùng `(value+1)/2` ở lượt đầu AZ so với outcome thực tế gặp Heuristic; đây không phải minimax calibration. Chênh score late50−uniform **−0,125**, bootstrap 10.000 lần theo bốn cụm opening cho khoảng percentile **[−0,375; 0]**. Uniform−parent +0,25, khoảng [0; 0,5]; late50−parent +0,125, khoảng [0; 0,375]. Chỉ một training seed, bốn opening và mọi khoảng chạm 0: kết quả **inconclusive**, không kết luận tăng Elo, không chọn late50 hay tuyên bố hai recipe tương đương. Evaluation tactical + paired 86,12 s; tập mới thêm 5,98 s. Môi trường: Windows 10 22631, Python 3.10.6, PyTorch 2.14.1+cu126, RTX 4060 Laptop GPU.

## 4. Đề xuất continuation quy mô vừa — chưa được phép chạy

Nếu các gate bên dưới được đáp ứng, thử một run Normal mới từ parent SHA ở trên, replay rỗng, STANDARD logging/outbox, một writer, immutable generation từng iteration và **ε=0,25**, `n_simulations=200`, `c_puct=1.5`, `dirichlet_alpha=0.15`, uniform replay sampling, không tactical override. Một cấu hình bounded để bàn trước: `n_iter=25`, `games_per_iter=4`, `train_steps_per_iter=8`, `batch_size=32`, `parallel_games=4`, một worker, Adam reset, LR `1e-4`, cosine horizon 200 updates; tối đa **100 game mới và 200 updates**. Lựa chọn LR/reset là điều kiện thử nghiệm chưa được chứng minh tối ưu: pilot này không so LR hay optimizer, và replay rất nhỏ dễ bị lặp. Không nhập legacy buffer; lưu config hoàn chỉnh, parent/source hash, worker hash, game IDs, seed, environment và manifests.

- **Trước run:** xác nhận checkpoint/replay/SQLite identity; kiểm tra quyền ghi và đủ dung lượng cho generation/outbox, test crash và khôi phục trên đúng filesystem; tách game/seed holdout, đóng băng parent và Heuristic, protocol/CI và ngưỡng dừng trước khi thấy kết quả.
- **Mỗi 20 game** (5 iteration): validate hash generation/outbox và ID replay–SQLite, thống kê opening, độ dài, kết quả màu, policy/value và tactical holdout. Đối kháng candidate trực tiếp với parent và Heuristic bằng cùng opening, đổi màu; ví dụ 10 cặp với mỗi đối thủ = tối đa 40 game đánh giá/gate, 200 game nếu đi hết năm gate; CI theo opening cluster. Đây chỉ là sàng lọc, không thay thế protocol promotion v1.2 100 game trong SPEC; chi phí thực tế và power cần xác minh trước, 10 cặp có thể cho kết quả inconclusive.
- **Dừng ngay, không promote:** collision, thiếu outbox/lineage, sai target hợp lệ, hash mismatch, checkpoint không resume được, hoặc hồi quy chiến thuật/strength nghiêm trọng được xác nhận. Nếu CI không đủ kết luận, giữ checkpoint chuẩn và dừng thay vì tự động chạy tiếp. Không dùng training loss để promote. Chỉ promote theo gate hiện tại của SPEC: lower 95% paired ΔElo CI > 0 so với best/đối thủ quy định.
- **Rollback:** giữ mọi generation và SQLite record, chọn ancestor đã commit bằng `--resume --generation <id> --run-id <id_mới>`. Nếu crash trước commit branch, retry cùng run ID và cùng source; sau commit dùng `--resume` với run ID branch, không kèm `--generation`. Không overwrite hoặc xóa checkpoint/replay lịch sử.

**Còn thiếu trước khi cho phép run:** lặp training trên nhiều seed và game có provenance; thêm forced-defense holdout độc lập hơn; đủ paired openings để ước lượng uncertainty và phát hiện suy giảm; kiểm chứng vận hành recovery/dung lượng trên filesystem đích và chấp nhận rủi ro power loss chưa đo. Không ép thắng Đen/Trắng về 50/50 và không tăng simulations để che vấn đề dữ liệu.
