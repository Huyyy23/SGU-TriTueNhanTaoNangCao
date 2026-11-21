# Kiểm toán nguồn gốc checkpoint và 20.000 ván self-play

Ngày kiểm toán: 21/11/2026. Phạm vi: run `selfplay_20k_cpp_2w`, các checkpoint Normal liên quan, SQLite và recovery outbox. Chỉ đọc, thống kê, tính hash và so sánh tensor; không chạy self-play, training, promote hay sửa checkpoint hoặc SQLite. Source hiện tại có thay đổi chưa commit, nên không mặc định nó trùng với code đã chạy.

## Kết luận

1. **20.000 game records — Verified, tin cậy cao:** `data/selfplay.sqlite` chứa đúng 20.000 ván của run `selfplay_20k_cpp_2w`. Có 40 nhóm iteration, mỗi nhóm 500 ván. Các targets tương ứng vẫn tồn tại trong recovery outbox.
2. **20.000 ván tiếp tục từ AZ 60k — không được xác nhận; bằng chứng trực tiếp bác bỏ việc bắt đầu từ checkpoint 60k đã xác định, tin cậy cao.** Checkpoint khởi đầu run 20k khác **96/96 tensor** so với checkpoint AZ 60k, optimizer ban đầu rỗng và global step bằng 0. Run metadata không ghi bootstrap hay parent. Chuỗi run 20k thực sự có training riêng: checkpoint cuối và metrics ghi 40.000 optimizer steps. Không suy ra nguồn khởi tạo thực tế là ngẫu nhiên chỉ từ những dữ kiện này.
3. **AZ 80k — chưa có bằng chứng.** Có 20k records ở nhánh riêng; không cộng chúng vào lịch sử của model 60k để gọi checkpoint cuối là AZ 80k. Mốc 60k trước đó là **danh nghĩa**: parent lịch sử 25k cộng 35k ván continuation; bản thân 25k lịch sử chưa được đếm độc lập đầy đủ. Báo cáo [screening 60k](analysis/60k-vs-heuristic-100-games-20261120.md) định danh checkpoint này. Tính tổng công việc self-play trên các nhánh khác với số ván đã huấn luyện liên tục một model.
4. **Nguyên nhân gốc vận hành — UNKNOWN:** không tìm được lệnh khởi chạy hoặc console log xác thực của run 20k, và chưa xác định được source code chính xác lúc chạy. Với code hiện tại, chạy không có `--bootstrap` hoặc `--resume` sẽ tạo model mới và ghi parent null; điều này phù hợp hiện vật, nhưng không chứng minh chính lệnh hay ý định lịch sử. Không có căn cứ quy lỗi cho người chạy hay khẳng định lỗi phần mềm.

## Định danh checkpoint và lineage

SHA-256 sau đây được tính từ bytes checkpoint, đối chiếu với manifest; tensor so bằng `torch.equal` trên CPU theo tên, shape và dtype, không dùng hash file làm thay thế cho so sánh weights. Các checkpoint đều là artifact local; `torch.load(..., weights_only=False)` dùng pickle, chỉ nạp file tin cậy.

| Artifact | Generation | SHA-256 checkpoint | Iteration / global step | Optimizer |
|---|---|---|---|---|
| AZ 60k, `checkpoints/az_25k_continuation_v1` | `ba897320ab8d406c881de90496d7f3a0` | `73c7d292a1588255cbcfe9a2d1fbc6dd471cb3e2e6cedb9850d0b47f05e899e9` | 700 / 70.000 | 51 state, mỗi state step 70.000 |
| Run 20k đầu, `checkpoints_selfplay_20k_2w` | `9be38fbc09cb471187b72d14fa41cf97` | `3e92a1c31b07645de45f7e4a5cb6abe770ac6b2538e4d7e53c7752b82961c3e5` | 0 / 0 | rỗng |
| Run 20k cuối, cùng thư mục | `37e2ec37d753481cb5c580da70dc6145` | `492ca39d3ab916c79d4037e228cb96606766c1cbaf377b8f8c6730b84c2d82f7` | 40 / 40.000 | 51 state, mỗi state step 40.000 |
| Parent lịch sử, `checkpoints/latest.pt` | legacy | `f7f521c1cc37a5054ed854201b087e484e9fcef3d4d6b1877ab81dff2e51d7e4` | 500 / 484.000 | 51 state, mỗi state step 484.000 |
| Đầu continuation 60k, cùng thư mục AZ 60k | `eccf2e59f43c4da78d0fa167c29e0588` | `2d9f7331e346dceacac758f4f9913c81e31212e414f0187f6e8cda962708d46c` | 0 / 0 | rỗng |

- `checkpoints_selfplay_20k_2w/latest.json` trỏ generation cuối 20k. Duyệt `commit_previous`, dự phòng `previous`, tìm được 45 manifest về generation đầu `9be3…`, có `previous: null`. Đây là đầu **chuỗi manifest quan sát được**, không tự chứng minh mọi nguồn weights ngoài chuỗi.
- Outbox iteration 0 ghi source checkpoint SHA `3e92a1c3…`; khớp hash thực của generation `9be3…`. Như vậy đây là checkpoint nguồn của nhóm self-play đầu tiên theo các hiện vật đã ghi.
- Checkpoint đầu 20k khác **96/96** tensor so với lần lượt checkpoint cuối 60k, checkpoint đầu continuation, `checkpoints/latest.pt` và checkpoint cuối 20k. Checkpoint cuối 20k cũng khác 96/96 so với cuối 60k. Không kết luận từ điều này rằng không có tổ tiên xa khác chưa quan sát.
- Đối chứng phương pháp: checkpoint đầu continuation 60k bằng `checkpoints/latest.pt` ở **96/96** tensor, và metadata bootstrap ghi đúng SHA parent; optimizer được reset. Do đó khác file hash không mặc nhiên là khác weights.
- **UNKNOWN:** nguồn thực sự của weights khởi đầu 20k ngoài các checkpoint đã đối chiếu; lệnh khởi chạy, seed khởi tạo model và bản code chạy thực tế. Không thể loại trừ một nguồn khác hay thao tác ngoài pipeline không được ghi.

## SQLite, training và recovery

Mở `data/selfplay.sqlite` bằng SQLite URI `file:data/selfplay.sqlite?mode=ro`. `PRAGMA integrity_check` trả `ok`; `PRAGMA foreign_key_check` không trả dòng. Đối chiếu theo `run_id = 'selfplay_20k_cpp_2w'`:

- Có **20.000 game**, local ID 0–19.999; **310.179 moves** trong bảng và payload, `moves_count` khớp độ dài moves của từng game. Có 310.179 MCTS roots cho run này.
- `model_iter` và provenance iteration là 0–39, mỗi iteration 500 game; run ID và source SHA không thiếu hoặc lệch trong nhóm đã kiểm tra. Cả **40/40 source SHA** khớp giá trị khai báo trong các manifest checkpoint của cây 20k, và 40 recovery manifest khớp identity `(run_id, iteration, source_checkpoint_sha256)`. Đối chiếu này chưa băm lại toàn bộ 40 checkpoint lịch sử.
- `training_steps` có **40 dòng metrics**, iteration 1–40; `global_step` tăng 1.000 mỗi dòng đến 40.000. Checkpoint cuối xác nhận counter 40.000 và 51 optimizer state có step 40.000. Code hiện tại `az/trainer.py:137–151` tăng counter sau mỗi `optimizer.step()`: đây là bằng chứng mạnh về 40.000 bước được ghi nhận, không phải 40.000 log độc lập từng lệnh. Code lúc chạy chính xác vẫn UNKNOWN.
- Run bắt đầu `2026-11-20T18:02:04.854280+00:00`; game đầu/cuối theo payload `18:02:17`–`19:53:29` UTC; metrics cuối lúc `19:53:42.103344+00:00`. Event promotion cuối lúc `20:02:57.957634+00:00`. Event candidate và manifest latest là hai generation khác nhau; không mặc định cùng bytes hay cùng weights.
- 40 outbox chứa **20.000 records và 310.179 samples**. Đã tính hash `records.json`, `samples.npz`, `replay.npz` và đối chiếu manifest cả 40; số sample/game ID khớp records và SQLite, `pi`/`z` hữu hạn. Chưa kiểm định đầy đủ tính hợp lệ từng policy, tính đúng chiến lược hay chất lượng target.
- `replay.npz` generation cuối chỉ giữ **50.000 sample thuộc 4.060 game ID**, cursor 10.179: đây là ring buffer, không phải toàn bộ 310.179 sample. Không nhầm 50.000 sample với 50.000 ván.

Ví dụ truy vấn tái hiện, chạy bằng Python `sqlite3.connect('file:data/selfplay.sqlite?mode=ro', uri=True)`; có thể đặt `PRAGMA temp_store=MEMORY` trước phép nhóm:

```sql
SELECT json_extract(payload_json, '$.model_iter') AS iter,
       json_extract(payload_json, '$.config.provenance.source_checkpoint_sha256') AS source_sha,
       COUNT(*) AS games
FROM games WHERE run_id = 'selfplay_20k_cpp_2w'
GROUP BY 1, 2 ORDER BY 1;

SELECT iteration, global_step, recorded_at,
       json_extract(metrics_json, '$.games') AS games
FROM training_steps WHERE run_id = 'selfplay_20k_cpp_2w'
ORDER BY iteration;
```

So sánh weights có thể tái hiện bằng `.venv/Scripts/python.exe -B`: nạp hai file bảng trên với `torch.load(path, map_location='cpu', weights_only=False)`, đối chiếu key/shape/dtype rồi đếm `torch.equal(a['model_state_dict'][k], b['model_state_dict'][k])`. Hash file dùng `hashlib.sha256` trên bytes file. Chỉ đọc checkpoint tin cậy.

## Vì sao metadata null?

Trong source **hiện tại**, `scripts/train.py:384–406` nạp generation khi `--resume`, gọi `Trainer.bootstrap_normal` khi `--bootstrap`, và nếu không có cả hai thì dựng `AZNet` mới với replay rỗng. `scripts/train.py:414–434` lấy parent SHA từ bootstrap hoặc source khi tạo metadata resume mới; nhánh tạo mới ghi `parent_checkpoint_sha256: null`. Resume vào run đã có giữ metadata run cũ. `az/trainer.py:304–355` xác thực bootstrap và ghi parent hash, cấu hình nguồn, lựa chọn optimizer. Vì vậy hai trường null của run 20k **phù hợp** nhánh khởi tạo mới, không phải bằng chứng độc lập về CLI hay source lịch sử.

`manifest.previous` mô tả liên kết generation trong cây run, không tự biểu thị parent bootstrap ngoài cây. Source `scripts/train.py` và `az/trainer.py` đang có thay đổi chưa commit; chưa thể dùng source hiện tại để khẳng định tuyệt đối luồng đã chạy. Không tìm thấy lệnh hay console log lịch sử xác thực cho run này.

## Phương án tận dụng 20k dữ liệu — đề xuất, chưa thực hiện

**Ưu tiên:** dùng lại đầy đủ 310.179 target để thiết kế thí nghiệm offline mới từ bản sao AZ 60k, thay vì sinh lại 20.000 ván. Điều này cần phát triển và kiểm thử importer/loader; **không có lệnh sẵn đã được xác minh** để thực hiện chính xác. Không gọi kết quả là 20k self-play on-policy từ AZ 60k: targets được sinh bởi các checkpoint của run 20k riêng.

1. Đóng băng inventory, SHA-256 hai cây checkpoint/outbox; tạo bản sao SQLite nhất quán bằng cơ chế backup của SQLite khi được phép thực hiện. Chỉ dùng output, DB và run ID mới; không ghi đè, xóa hay chuyển artifact gốc. Kiểm tra đủ không gian và không có writer trên bản sao làm việc.
2. Xác minh lại hash manifest, checkpoint nguồn mỗi iteration, ruleset Normal, encoding planes/policy, game ID, kích thước, finite/nonnegative/normalized/legal policy, `z` và số samples/game; ghi nhận và cách ly bất kỳ sai lệch nào thay vì bỏ qua. Giữ lineage gốc của từng sample. Đã kiểm tra finite, count và hash như trên; các kiểm tra target sâu hơn còn **chưa thực hiện**.
3. Import/stream `samples.npz` từ 40 outbox theo iteration vào dataset mới có provenance; bảo đảm xử lý đủ 310.179 samples hoặc ghi rõ subset. Không nạp toàn bộ vào ring 50k rồi tuyên bố dùng đủ: ring ghi đè dữ liệu cũ. Tách evaluation theo game và opening/seed, kiểm tra exact symmetry overlap khi khả thi; tránh dùng cùng game cho train và holdout.
4. Khởi tạo candidate từ **bản sao checkpoint AZ 60k đã xác minh hash**; xác định trước reset hay restore optimizer/scheduler, LR, seed, sampling và số update; lưu parent SHA, dataset manifest, game IDs và cấu hình bất biến. Có thể so sánh với baseline 60k không update và phương án sampling khác chỉ nếu protocol được chốt trước. Không tự promote.
5. Đánh giá candidate và baseline bằng protocol matched, cùng luật, search budget, opening/seed và đổi màu theo cặp; đánh giá đối thủ độc lập, báo W/D/L theo màu và khoảng tin cậy theo cặp. Giữ nguyên 60k nếu không đạt gate đã định trước. Loss giảm hoặc nhiều samples không chứng minh strength tăng.

Cơ chế recovery sẵn có không phải importer này: `az/training_recovery.py:106–127` kiểm tra identity/hash outbox và nạp replay snapshot tương ứng; `scripts/train.py:489–523` phục hồi theo identity run/iteration/source checkpoint, không tự nối 40 file sample vào một run AZ 60k. Policy/value targets từ model khác là **off-policy**, có thể làm model quên hoặc yếu đi. Chưa chứng minh dữ liệu đa dạng, holdout độc lập hay cải thiện sức chơi. Không thể tái tạo một trajectory on-policy 20k từ AZ 60k chỉ bằng dữ liệu từ nhánh này.

## Phòng ngừa cho các run sau — chỉ đề xuất

- Bắt buộc khai báo rõ chế độ `fresh`, `bootstrap` hoặc `resume`; chỉ `fresh` được phép không có parent. Không dùng null mơ hồ làm chế độ.
- Với bootstrap/resume, yêu cầu **expected parent SHA-256** và generation/run ID khi có; xác minh hash file cùng manifest/replay và ruleset/architecture trước khi tạo output, ghi DB hoặc chơi ván đầu; thiếu/sai thì fail closed. Resume phải xác minh ownership và parent/replay tương thích, không chỉ đường dẫn.
- Ghi bất biến mode, parent SHA, đường dẫn, generation/run ID nguồn, model tensor digest, optimizer reset/restore, config, seed, code identity/dirty state và lệnh vào checkpoint cùng run DB; kiểm tra nhất quán ở mỗi resume. Ghi `fresh` tường minh. SHA file chỉ định danh bytes, không chứng minh sức chơi hoặc đầy đủ tổ tiên.
- Thêm kiểm thử fail-closed thiếu/sai SHA, mode mâu thuẫn, parent sai ruleset/shape, sai ownership, metadata checkpoint/DB lệch; test hành vi thực sự từ CLI và không tạo artifact khi validation thất bại. Các đề xuất này chưa được triển khai hay chạy kiểm thử trong cuộc audit.
