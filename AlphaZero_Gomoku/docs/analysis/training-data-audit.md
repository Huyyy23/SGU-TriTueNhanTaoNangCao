# Kiểm toán training và checkpoint 25k

## Phạm vi và kết luận

Kiểm toán tĩnh source hiện tại và đọc các artifact hiện có, không chạy training, tournament, test hay ghi vào dữ liệu hiện hữu. Cấu hình nghiên cứu bắt buộc 200 simulations mỗi lần gọi MCTS (`SPEC.md:193`, `configs/normal.yaml:4-5`); không đề xuất thay đổi ngân sách. Thống kê/chất lượng dữ liệu self-play được kiểm toán riêng. Source hiện tại có thể khác code đã tạo artifact lịch sử; các kết luận về quá khứ bên dưới dựa trên metadata/log, không suy diễn source hiện tại chính là code lịch sử.

**Quan sát, độ tin cậy cao:** `checkpoints/latest.pt` khai báo iteration 500, global step 484.000, Adam state step 484.000 và Normal; cấu hình checkpoint ghi 50 games/iteration, 1.000 steps/iteration và 200 simulations. 500 × 50 = 25.000 là số ván *danh nghĩa* theo iteration/config, không phải phép đếm ván độc lập. Không thể suy ra 500.000 optimizer updates từ tên mốc 25k: checkpoint và optimizer đều ghi 484.000. Log continuation có 298 dòng cho iteration 201–500, thiếu 300 và 480; đó là thiếu metric, không tự chứng minh thiếu update. Checkpoint có 483.810 tham số học được, 96 tensor trong state dict kể cả buffers, tất cả bộ đếm BatchNorm quan sát được là 484.000.

**Quan sát, độ tin cậy cao:** ở iteration 500 log ghi loss train 2,32103 (policy 2,30162; value 0,01941), fixed-batch loss 7,65346; iteration 201 tương ứng 2,91326 và 6,50930. Fixed loss tăng trong khoảng này dù train loss giảm; không thể dùng các số này chứng minh sức chơi hay generalization. Metric là trung bình theo bước, không phải loss trên cùng một tập train cố định.

## Findings

| Mức độ / tin cậy | Bằng chứng và tác động |
|---|---|
| **P1 / cao — nguy cơ integrity tiềm ẩn: checkpoint và replay không giao dịch cùng nhau** | `Trainer.save` ghi checkpoint trước, sau đó mới ghi NPZ (`az/trainer.py:145-167`); checkpoint được replace atomically (`az/model.py:130-136`), còn replay dùng `np.savez` trực tiếp (`az/replay_buffer.py:104-116`). Interrupt có thể để latest và replay thuộc hai thời điểm, hoặc NPZ hỏng. Resume chỉ kiểm tra hai file tồn tại trước load (`scripts/train.py:331-337`), không chứng minh chúng cùng generation. Đây là rủi ro recovery tiềm ẩn, chưa xác nhận sự cố hay hai artifact hiện tại sai cặp. |
| **P1 / cao — metric cố định không phải holdout độc lập** | `freeze_eval_batch` lấy có hoàn lại từ buffer, lưu bản sao và loại các *chỉ số slot* khỏi train (`az/trainer.py:110-122`, `az/trainer.py:93-104`). Các vị trí khác có thể cùng ván; sau khi ring overwrite, chỉ số slot không còn định danh mẫu ban đầu. Không được diễn giải fixed-batch loss như đánh giá tổng quát hoặc sức chơi (`SPEC.md:177`). |
| **P1 / cao — provenance và identity chưa đủ để tái lập hoàn toàn** | Run continuation lưu seed 0, backend/config và parent SHA `fb3c39a9b3db6aa70764f8427e2ba6accc7ef90326d2e0d10e8a291f5e765c6b`, nhưng `git_commit` và `engine_version` là NULL; checkpoint lưu RNG NumPy của Trainer, không lưu torch RNG hay seed khởi tạo model (`az/trainer.py:147-154`, `scripts/train.py:339-341`). Không có bằng chứng ở đây để tái tạo chính xác code, model initialization, môi trường và toàn bộ lịch sử parent. Chỉ load pickle checkpoint đáng tin cậy: loader dùng `weights_only=False` (`az/model.py:149`). |
| **P1 / cao — candidate SHA khác latest đang có** | Log promotion iteration 500 ghi candidate SHA `2384043ae275dfe5eef24cb9ec1f87d771ae0238ccfff01ffa14591baffeb731`; SHA byte hiện tại của `latest.pt` là `f7f521c1cc37a5054ed854201b087e484e9fcef3d4d6b1877ab81dff2e51d7e4`. Source hiện tại tính SHA trước evaluation rồi cập nhật `promotion_eval_games` và save lại (`scripts/train.py:195-196,225-228,250-253`), là một cơ chế có thể tạo byte SHA khác; **chưa chứng minh** chính cơ chế này xảy ra trong code lịch sử hoặc hai file có cùng weights. Đừng gán trực tiếp kết quả tournament cho byte identity hiện tại nếu không có candidate gốc/đối chiếu tensor. `best.pt` không tồn tại tại thời điểm kiểm toán. |
| **P1 / cao — log và checkpoint không phải một commit** | Games được ghi trước training; checkpoint lưu trước metric (`scripts/train.py:378-404,432-437`); interruption có thể gây sai lệch giữa các nguồn. Hai iteration 300 và 480 không có dòng metric continuation, nhưng các dòng 299/301 và 479/481 hoặc checkpoint cần được đối chiếu trước kết luận nguyên nhân. Không được tự điền số còn thiếu. |
| **P1 / cao — horizon LR thay đổi khi resume** | `_set_lr` lấy horizon từ config đang chạy, không phải horizon cố định theo checkpoint (`az/trainer.py:38-40,48-58,184-194`). Đổi `n_iter` hoặc steps/iteration khi continuation làm đổi đường cosine; >= horizon thì LR bằng 0. Với config checkpoint horizon 500.000 và global step 484.000, LR Adam lưu là `2.524806814795777e-06`, tương ứng bước cuối đã dùng trước tăng counter, không phải LR gốc 0,001. Không suy ra toàn bộ lịch sử LR từ config cuối. |
| **P2 / cao — config optimizer không được dispatch** | `optimizer: adam` là default (`configs/normal.yaml:23-25`), nhưng Trainer luôn khởi tạo `torch.optim.Adam` (`az/trainer.py:24-30`); đổi tên optimizer trong YAML không đổi implementation. Weight decay được truyền vào Adam, không cộng lần hai vào loss. |
| **P2 / cao — CLI cho phép đi lệch hợp đồng 200** | `--simulations` ghi đè config mà không kiểm tra 200 (`scripts/train.py:43-45,282-287`), trong khi SPEC yêu cầu cố định 200. Artifact đang xét ghi 200; không suy ra mọi run lịch sử tuân thủ. |

## Pipeline hiện tại và targets

Mỗi vị trí trước nước đi có policy target từ visit distribution và value target kết quả terminal theo người sắp đi (`SPEC.md:230-238`). Replay là ring 50.000 mẫu; lấy minibatch có hoàn lại, áp dụng cùng đối xứng cho feature và policy (`az/replay_buffer.py:13-16,61-102`; `az/trainer.py:93-104`). `distinct_samples` là số slot khác nhau trong iteration, không phải tổng số mẫu mới hoặc số ví dụ độc lập.

Với logits `l`, policy `π`, value `v`, outcome `z`, loss là mean cross entropy mềm `-Σπ log softmax(l)` cộng MSE `(v-z)²`, mỗi trọng số 1 theo config (`az/trainer.py:42-46`, `configs/normal.yaml:26-27`). Mỗi batch thực hiện zero-grad, backward, Adam step và tăng global step một (`az/trainer.py:60-79`). Driver self-play rồi freeze eval nếu chưa có, chạy train, đánh giá fixed batch, lưu và ghi metric (`scripts/train.py:363-404,410-438`). Model và optimizer state, iteration, global step, config, Trainer NumPy RNG, promotion counter lưu trong checkpoint; frozen eval và replay lưu NPZ (`az/trainer.py:145-167`). Resume khôi phục optimizer nếu có, nhưng không từ chối checkpoint thiếu optimizer state (`az/trainer.py:184-200`).

## Dữ liệu self-play và buffer thực tế

**P1, quan sát xác nhận; độ tin cậy cao, nguyên nhân và tác động nhân quả chưa xác nhận.** Truy vấn read-only `data/selfplay.sqlite` cho run `az_v0-15000-continuation-20261119`: 15.000 game records, 151.893 moves, trung bình 10,13 nước/ván, min–max 9–62; Đen thắng 14.931, Trắng 69, hòa 0. Có 9.563 ván dài 9 nước và 4.094 ván dài 11 nước; 14.960/15.000 ván bắt đầu tại tâm `(5,5)`, tổng cộng 35 action đầu khác nhau. Mỗi game có moves và MCTS roots, mỗi loại tổng 151.893. `moves.simulations` đều ghi 200 nhưng đó là config được ghi (`az/selfplay.py:171-177`), không tự chứng minh từng search đã chạy đủ 200. Sự tập trung khai cuộc và kết quả cực lệch là dấu hiệu cần kiểm tra diversity/learning, không tự chứng minh lỗi, policy collapse toàn cục hoặc nguyên nhân thua Heuristic. Tái hiện bằng truy vấn bên dưới; đề xuất đo trajectories và target entropy theo thời gian, tách màu/seed, kiểm tra khai cuộc.

Không cộng các run khác để khẳng định 25.000 ván của cùng checkpoint khi lineage chưa xác nhận. Tổng database có 49.270 game records bao gồm nguồn import/legacy, không phải số ván độc lập đã khử trùng. Những record được lưu không chứng minh từng mẫu đã được optimizer sử dụng.

`checkpoints/buffer_latest.npz` có 50.000/50.000 mẫu Normal; z −1/+1 là 22.560/27.440. Hash byte feature tìm được 28.583 state khác nhau và 42.793 bộ `(state, pi, z)` khác nhau; state xuất hiện nhiều nhất là bàn trống, 4.906 lần. Trung vị số quân 5, trung vị số action có π > 0 là 3. Exact bytes không gộp đối xứng; state trùng có thể khác target, không đồng nghĩa toàn bộ là bản sao. Renju buffer có 10.715 mẫu, 9.243 state khác nhau và 10.414 bộ khác nhau; không gộp hai bộ luật.

Trong hai buffer, π hữu hạn, không âm, tổng mỗi hàng từ 0,9996338 đến 1,0003662, không hàng nào lệch quá 0,001 và không có mass trên ô đã chiếm. Sai số nhỏ phù hợp lưu float16 (`az/replay_buffer.py:23-25,49-55`). **Chưa kiểm** forbidden Renju, nhãn z theo winner thực tế, target visit hay historical legality: không kết luận target đúng toàn bộ. Buffer không lưu game ID/timestamp theo mẫu (`az/replay_buffer.py:104-115`), cursor khi load không khôi phục thứ tự tuổi (`az/replay_buffer.py:118-130`); tỷ lệ dữ liệu cũ/mới và quan hệ với checkpoint là unknown. Đề xuất lưu lineage/mẫu theo game và kiểm chứng held-out theo game trước khi đổi recipe.

Tái hiện các số chính; mở database read-only, chỉ đọc NPZ:

```bash
.venv/Scripts/python.exe - <<'PY'
import hashlib, pathlib, sqlite3, numpy as np
c = sqlite3.connect(pathlib.Path("data/selfplay.sqlite").resolve().as_uri()+"?mode=ro", uri=True)
c.execute("pragma temp_store=memory")
r = "az_v0-15000-continuation-20261119"
for q in [
 "select winner,count(*),min(moves_count),avg(moves_count),max(moves_count),sum(moves_count) from games where run_id=? group by winner",
 "select moves_count,count(*) from games where run_id=? group by moves_count order by count(*) desc limit 10",
 "select m.action,count(*) from moves m join games g on g.game_id=m.game_id where g.run_id=? and m.move_index=0 group by m.action order by count(*) desc limit 10",
 "select count(*),min(m.simulations),max(m.simulations) from moves m join games g on g.game_id=m.game_id where g.run_id=?",
]:
 print(list(c.execute(q, (r,))))
for p in ["checkpoints/buffer_latest.npz", "checkpoints/renju_buffer_last.npz"]:
 with np.load(p) as d:
  f, raw, z = d["features"], d["pi"], d["z"]
  pi = raw.astype(np.float32); n = len(z)
  occupied = (f[:,0] + f[:,1] > 0).reshape(n,-1)
  states = {hashlib.sha256(x.tobytes()).digest() for x in f}
  full = {hashlib.sha256(f[i].tobytes()+raw[i].tobytes()+z[i].tobytes()).digest() for i in range(n)}
  print(p, n, len(states), len(full), np.unique(z,return_counts=True),
        np.isfinite(pi).all(), pi.min(), pi.sum(1).min(), pi.sum(1).max(),
        ((pi * occupied).sum(1)>0).sum())
PY
```

## Kết quả promotion lịch sử (không phải chạy lại)

Dòng metric iteration 500 ghi protocol `az-v0-promotion-v1.1`, mở đầu 4 plies, 400 ván / 200 cặp opening, 46 thắng, 354 thua, 0 hòa trước heuristic, score 0,115; paired Davidson ΔElo −352,87, bootstrap 95% CI [−405,06; −303,55], 2.000 bootstrap, seed 5.000.000, `promote=false`. Đây là kết quả log của candidate SHA nói trên, không phải evaluation mới và không chứng minh identity byte với latest hiện tại. Source hiện tại ghi protocol v1.2 (`scripts/train.py:234`), nên không đồng nhất protocol lịch sử và hiện tại. `best.pt` vắng mặt phù hợp với log `best_sha256=null` và không promote tại mốc đó, nhưng không chứng minh toàn bộ lịch sử best.

## Lệnh đọc và giới hạn

Từ project root, dùng interpreter có sẵn; SQLite mở read-only. Chỉ thực hiện với checkpoint local tin cậy vì PyTorch pickle có thể thực thi code khi load. Các lệnh dưới đây chỉ đọc, không chạy self-play/train/tournament:

```bash
graphify query "training loop optimizer learning rate loss targets checkpoints model provenance"
.venv/Scripts/python.exe -c 'import torch,hashlib,pathlib; p=pathlib.Path("checkpoints/latest.pt"); d=torch.load(p,map_location="cpu",weights_only=False); print(hashlib.sha256(p.read_bytes()).hexdigest()); print(d["meta"]); o=d["optimizer_state_dict"]; print(o["param_groups"]); print(sorted(set(int(v["step"].item()) for v in o["state"].values() if "step" in v))); print(len(d["model_state_dict"]),sum(v.numel() for k,v in d["model_state_dict"].items() if "running_" not in k and "num_batches_tracked" not in k))'
.venv/Scripts/python.exe -c 'import sqlite3,json; c=sqlite3.connect("file:data/selfplay.sqlite?mode=ro",uri=True); r="az_v0-15000-continuation-20261119"; print(c.execute("select run_id,started_at,git_commit,engine_version,seed,metadata_json from runs where run_id=?",(r,)).fetchone()); rows=c.execute("select iteration,global_step,metrics_json from training_steps where run_id=? order by iteration",(r,)).fetchall(); print("count",len(rows),"missing",[i for i in range(201,501) if i not in {x[0] for x in rows}]); [print(i,s,json.loads(m)) for i,s,m in rows if i in (201,499,500)]'
```

Unknown: checkpoint candidate gốc còn ở đâu, tensor candidate có trùng latest không; source commit/diff và torch initialization RNG lịch sử; replay và checkpoint hiện tại có cùng generation không; nguyên nhân thiếu metric 300/480; số ván thực sự dùng train, tỷ lệ cũ/mới và tính đúng đắn đầy đủ của target. Không chạy kiểm thử, training, tournament, hoặc benchmark trong kiểm toán này.
