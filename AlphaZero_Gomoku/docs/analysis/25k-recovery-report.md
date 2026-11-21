# Điều tra phục hồi AlphaZero Gomoku tại mốc 25k

## Quyết định vận hành

**Chưa nên tiếp tục self-play/training dài từ checkpoint hiện tại.** Đã sửa một lỗi resume của replay, nhưng NPZ hiện tại không có cursor lịch sử nên không thể khôi phục chính xác thứ tự tuổi, còn quan hệ generation giữa NPZ và checkpoint chưa được chứng minh. Các kết quả cũ không được ghi đè hoặc xóa. Cần kiểm chứng provenance, dữ liệu và đánh giá độc lập trước khi quyết định một continuation có kiểm soát; không mặc định tăng simulations hay số ván. Đây là quyết định quản trị rủi ro, không phải khẳng định lỗi cursor đã gây ra tỷ lệ thắng 11,5%.

## Lỗi đã xác nhận và sửa

**P1 — sai thứ tự FIFO của replay sau resume:** `az/replay_buffer.py`, `ReplayBuffer.save_npz` / `load_npz`. Ring đầy đã wrap lưu mảng theo vị trí vật lý, nhưng code cũ không lưu `_cursor` và luôn phục hồi cursor về 0. Khi cursor thực là 1, thao tác thêm tiếp có thể xóa mẫu mới hơn thay vì mẫu cũ nhất. Regression test `tests/test_replay_buffer.py::test_resume_full_buffer_preserves_oldest_sample_order` thất bại trước sửa (1 failed) và sau sửa nằm trong nhóm 27 tests passed. Đây là lỗi implementation tái hiện được, **không có chứng cứ nó là nguyên nhân trực tiếp** của 99,54% Đen thắng hay 11,5% thắng Heuristic.

Sửa tối thiểu: NPZ mới lưu cursor và loader khôi phục/kiểm tra giá trị. NPZ cũ chưa đầy suy ra cursor từ size; NPZ cũ đã đầy không có thông tin thứ tự tuổi, loader phát `UserWarning` và giữ cursor 0 như hành vi cũ. Không sửa hay ghi lại NPZ lịch sử. Cảnh báo không làm dữ liệu cũ trở thành đúng FIFO; không chứng minh checkpoint và replay là cùng generation. `SPEC.md` ghi rõ giới hạn này.

Tái hiện test đỏ trên source cũ và kiểm tra xanh trên source sửa:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/test_replay_buffer.py::test_resume_full_buffer_preserves_oldest_sample_order
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/test_replay_buffer.py tests/test_trainer.py tests/test_native_mcts.py
```

Kết quả lần chạy này: test đầu **1 failed trước sửa**; lệnh thứ hai **27 passed in 6.40s sau sửa**, bao gồm kiểm thử parity native hiện có. Các kiểm thử native không chứng minh parity trên toàn bộ trajectories, seeds hay strength. `graphify update .` đã chạy sau thay đổi mã.

## Đồng bộ mô hình, LR và ranh giới bằng chứng

Code hiện tại `scripts/train.py` hoàn tất self-play trước train; `az/selfplay.py` chụp `state_dict` model hiện tại cho từng iteration và worker load snapshot đó. Vì vậy source hiện tại không cho thấy worker cố tình giữ model cũ giữa các iteration. Trong database continuation, 15.000 ván có `model_iter` 200–499; nhãn không phải hash tensor của từng worker. Source lịch sử dirty và log không có hash per-worker: **không thể xác nhận identity trọng số thực tế của mọi ván lịch sử**.
Promotion 400 ván ghi candidate SHA `2384043ae275dfe5eef24cb9ec1f87d771ae0238ccfff01ffa14591baffeb731`; `latest.pt` được audit trước có SHA `f7f521c1cc37a5054ed854201b087e484e9fcef3d4d6b1877ab81dff2e51d7e4`. Khác hash byte không chứng minh tensor khác: source có thể lưu lại metadata sau evaluation. Chưa tìm được candidate đúng hash để so tensor. Checkpoint ghi iteration 500 / 484.000 optimizer steps; LR Adam lưu khoảng `2,52e-6`. Horizon cosine thay đổi khi cấu hình continuation thay đổi; số steps và loss không đủ kết luận quality.

Code MCTS hiện tại dùng value của người sắp đi, đổi dấu khi backup; self-play bật Dirichlet noise và temperature ở giai đoạn đầu, evaluation tắt noise. Chưa chứng minh bug sign trong run lịch sử; các phát hiện về giới hạn policy validation và checkpoint/replay không giao dịch trong [audit trước](mcts-correctness-audit.md) và [audit training](training-data-audit.md) vẫn là rủi ro, không tự chứng minh nhân quả. Giữ nguyên luật và 200 simulations cho mọi search AZ đã nghiên cứu.

## Trajectory, exploration và replay thực tế

Run `az_v0-15000-continuation-20261119` có 15.000 ván ghi ở model iterations 200–499; 14.931 Đen thắng, 69 Trắng thắng. Độ dài 9: 9.563 ván, 11: 4.094 ván; tổng 13.657/15.000 (91,05%), trung bình 10,1262. Kiểm tra độc lập bốn hướng trên toàn bộ trajectory cho thấy không có ô chiếm lặp, sai lượt, thắng trước kết thúc, sai winner hoặc sai số nước. Ván `10000`: `[60,72,71,112,82,118,49,22,38]`; Đen tạo năm quân dọc sau chín ply.

Không phải lặp lại nguyên ván theo nghĩa byte: có **14.999 trajectory đầy đủ khác nhau**; chỉ một cặp trùng. Ngược lại, prefix hai ply chỉ có 160 tổ hợp; sau bốn ply có 9.422, sau sáu ply có 14.555. Ở run trước `az_v0b-5000-20261119`, iteration 20 dài trung bình 20,56; iteration 35 là lần đầu ít nhất một nửa 50 ván kết thúc ≤11 ply. Continuation iteration 200 đã ngắn trung bình 10,08. Các run có lineage chưa chứng minh và có thể overlap; đây là trình tự quan sát, không phải chuỗi nhân quả đã chứng minh.

151.893 root blobs đều có tổng 199 child visits, nhất quán với root expansion tiêu thụ một trong 200 simulations; policy/target/prior được ghi không có giá trị âm/không hữu hạn, chuẩn hóa sai, hay mass ngoài legal mask. Có 78.646 nước được chọn không phải argmax visits; nhiệt độ 1 trong 15 nước đầu cho phép điều đó, không tự là lỗi. Không có prior trước noise và noise realization riêng nên không thể chứng minh historical noise chỉ từ log. Cấu hình ghi native C++, α=0,15, ε=0,25, 200 simulations; `simulations` là metadata, không phải audit độc lập toàn bộ thuật toán.

Replay Normal hiện tại có 50.000 mẫu, 28.583 state byte khác nhau và 42.793 bộ `(state, π, z)` khác nhau; 4.906 là state bàn trống. Xem [audit dữ liệu trước](training-data-audit.md) cho các lệnh kiểm tra, cùng giới hạn thiếu game ID/timestamp và không biết tuổi mẫu. Dữ liệu có tập trung trạng thái đầu và thiên lệch outcome rất mạnh, nhưng chưa chứng minh nguyên nhân là RNG, target hay MCTS backend; cũng không gọi các trajectory đầy đủ là bản sao hàng loạt.

Truy vấn tái lập các đếm chính:

```bash
.venv/Scripts/python.exe - <<'PY'
import collections, sqlite3
c = sqlite3.connect('file:data/selfplay.sqlite?mode=ro', uri=True)
r = 'az_v0-15000-continuation-20261119'
print('outcome/length', c.execute('select winner,count(*),avg(moves_count) from games where run_id=? group by winner',(r,)).fetchall())
print('lengths', c.execute('select moves_count,count(*) from games where run_id=? group by moves_count order by count(*) desc limit 10',(r,)).fetchall())
rows=c.execute('select g.game_id,m.action from games g join moves m on m.game_id=g.game_id where g.run_id=? order by g.game_id,m.move_index',(r,))
games=collections.defaultdict(list)
for gid,a in rows: games[gid].append(a)
for n in (2,4,6,8,None):
 counts=collections.Counter(tuple(v[:n]) for v in games.values())
 print('prefix',n,'unique',len(counts),'repeated_games',sum(v for v in counts.values() if v>1))
PY
```

## Đánh giá ngoài dữ liệu train: giới hạn holdout

**Chưa tìm được holdout self-play độc lập có thể chứng minh**: replay không có game ID/timestamp theo mẫu, historical lineage không đầy đủ; fixed batch lấy từ chính replay. Không gọi phân tích dưới đây là held-out training hay đánh giá chọn model độc lập.

Thay thế mang tính mô tả: 40 game đầu theo ID của tournament promotion 25k, 20 cặp opening; 149 vị trí AZ sắp đi tại các mốc ply 4/5, 8/9, 12/13, 16/17 nếu tồn tại. Dùng `latest.pt` SHA `f7f521…`, không chứng minh trùng candidate lịch sử `238404…`. Không search, chỉ network CPU; target là kết quả thực tế của ván theo người đi, Brier dùng `(value+1)/2`. Các vị trí trong cùng ván và hai ván cùng opening không độc lập; đây là tournament đã dùng trong promotion.

| Nhóm | Vị trí | Xác suất thắng dự báo TB | Tỷ lệ thắng thực tế | Value MSE | Brier |
|---|---:|---:|---:|---:|---:|
| Tất cả | 149 | 0,6212 | 0,1342 | 1,9154 | 0,4789 |
| AZ Đen | 76 | 0,8799 | 0,2105 | 2,7182 | 0,6795 |
| AZ Trắng | 73 | 0,3519 | 0,0548 | 1,0797 | 0,2699 |

Trung bình có trọng số bằng nhau theo từng game: MSE 1,8831, Brier 0,4708. Đây là dấu hiệu value quá lạc quan trên mẫu đối kháng này, không phải calibration theo optimal play, không chứng minh tổng quát hay nhân quả; raw policy top-1 trùng nước MCTS lịch sử 69,8% không phải so với visit target. Cần game holdout được đóng băng, có game ID, seed, provenance và chưa dùng chọn model.

Lệnh tái lập ước lượng theo vị trí (chỉ tải checkpoint local tin cậy, đọc SQLite):

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe - <<'PY'
import collections, hashlib, json, sqlite3, numpy as np, torch
from az.model import load_checkpoint
from gomoku.board import Board
from gomoku.features import board_to_planes
from gomoku.rules import get_rules
p = 'checkpoints/latest.pt'
print('sha256', hashlib.sha256(open(p,'rb').read()).hexdigest())
torch.set_num_threads(2)
model, _ = load_checkpoint(p, device='cpu')
c = sqlite3.connect('file:data/tournament.sqlite?mode=ro', uri=True)
exp = 'az_v0-15000-continuation-20261119-promotion-500-heuristic-0'
rows = c.execute('select local_game_id,winner,payload_json from games where experiment_id=? order by cast(local_game_id as integer) limit 40',(exp,)).fetchall()
rules = get_rules('normal'); features = []; labels = []
for gid, winner, payload in rows:
 d = json.loads(payload); board = Board()
 az = 1 if d['first_player'].startswith('az_v0') else 2
 for ply, action in enumerate(d['moves']):
  if ply in (4,5,8,9,12,13,16,17) and board.current_player == az:
   features.append(board_to_planes(board, rules, az))
   z = 1 if (winner == 'black') == (az == 1) else -1
   labels.append((az,z))
  board.play(*divmod(action,11))
values=[]
for i in range(0,len(features),64):
 _, v = model.predict(np.stack(features[i:i+64])); values.extend(v)
for color in (None,1,2):
 data=[(float(v),z) for (side,z),v in zip(labels,values) if color is None or side==color]
 print('color',color,'n',len(data),'pwin',np.mean([(v+1)/2 for v,z in data]),
       'actual',np.mean([(z+1)/2 for v,z in data]),
       'mse',np.mean([(v-z)**2 for v,z in data]),
       'brier',np.mean([((v-z)/2)**2 for v,z in data]))
PY
```


## Hai sai lầm trước khi hình thành đe dọa kép

Oracle độc lập liệt kê các cửa sổ năm ô theo bốn hướng, thử nước đối thủ và tìm ít nhất hai ô thắng riêng. Không dùng `gomoku.patterns` làm oracle. Cả hai ví dụ thuộc experiment promotion 25k, action phẳng `11*r+c`, ply 0-based; prefix và assertion được cố định trong `tests/test_tactical_oracle.py`.

| Game / trạng thái | Nước thực tế | Phản ứng Heuristic | Hậu quả kiểm chứng |
|---|---:|---:|---|
| `...-heuristic-0:0`, trước ply 24, Đen đi | 52 | 17 | Trắng có hai ô thắng `5,65`; Đen không có thắng ngay. Các ô `5,17,65` ngăn fork nông này. |
| `...-heuristic-0:1`, trước ply 25, Trắng đi | 96 | 69 | Đen có hai ô thắng `29,79`; Trắng không có thắng ngay. Các ô `29,69` ngăn cả hai fork nông đã liệt kê. |

Đây là sai lầm chiến thuật xác nhận trên trajectory, **không chứng minh các nước phòng thủ nêu trên cứu được ván**: oracle không giải hết cây sâu. Sau mỗi fork, test cho bên phòng thủ chặn từng ô rồi kiểm tra `NormalRules.is_win` xác nhận đối thủ vẫn thắng ở ô còn lại. Lệnh `PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/test_tactical_oracle.py` cho **2 passed in 0.17s**. Đây là test chứng minh tính chất bàn cờ, không phải khẳng định checkpoint lịch sử sẽ tự tìm đúng nước hay so sánh parity strength.

Trên bốn vị thế, dùng `latest.pt` SHA `f7f521…`, Normal, seed 0, CPU một thread, noise tắt và 200 simulations/search, **cả Python lẫn C++ chọn cùng nước**. Với hai vị thế trên, checkpoint hiện tại cũng chọn 52 và 96: ở ván 0 các nước 5/17/65 chỉ nhận lần lượt 1/1/1 visits, priors 0,006234/0,007361/0,008374; ở ván 1 các nước 29/69 đều 0 visits, priors 0,008804/0,015224. Mỗi search có 199 child visits. Đây là bằng chứng khả năng phòng thủ nông kém trên hai thế cụ thể và prior/visits thấp, không chứng minh vì sao model học sai, cũng không chứng minh candidate lịch sử có cùng tensor. Hai fixture khác cũng chọn cùng nước giữa backend; không khái quát parity toàn diện.

Tái lập riêng hai thế (mỗi backend hai search, 200 simulations mỗi search):

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe - <<'PY'
import numpy as np, torch
from az.mcts import MCTS, NativeMCTS
from az.model import load_checkpoint
from gomoku.board import Board
from gomoku.rules.normal import NormalRules
torch.set_num_threads(1)
model, _ = load_checkpoint('checkpoints/latest.pt', device='cpu')
cases = [
 ([20,72,31,58,61,60,9,48,42,53,84,38,28,49,71,50,51,41,81,91,83,82,30,29], [5,17,65]),
 ([20,72,31,58,42,53,9,18,60,70,49,27,71,82,46,94,106,50,38,61,83,48,59,95,39], [29,69]),
]
for moves, defenses in cases:
 for cls in (MCTS, NativeMCTS):
  board = Board()
  for action in moves: board.play(*divmod(action,11))
  search = cls(NormalRules(),n_simulations=200,root_noise=False,rng=np.random.default_rng(0))
  search.reset(board)
  while (request := search.request()) is not None:
   p,v=model.predict(request.planes[None]); search.receive(p[0],float(v[0]))
  visits,priors,_,_=search.root_statistics()
  print(cls.__name__, 'move',search.select_move(0),'visits',visits[defenses],
        'priors',priors[defenses],'total',visits.sum())
PY
```

## Tối đa ba ưu tiên cải tiến / thí nghiệm

1. **P1 — xác định và kiểm soát mẫu self-play suy giảm:** lưu provenance game/worker và thống kê first-player bias, trajectory length, root pre-/post-noise cùng entropy/visits theo iteration; phân tích tập game đã ghi và thiết lập canary tactic trước khi sửa recipe. Giả thuyết cần kiểm tra: policy/value học lệch khiến Trắng không chống được các threat sớm; chưa chứng minh nhân quả. Không tăng simulations.
2. **P1 — đánh giá độc lập theo game và checkpoint bất biến:** dựng holdout với game ID, seed và checkpoint hash tách khỏi train và promotion; đo calibration theo màu/ply và tactical oracle, sau đó paired opening/color evaluation nhỏ theo protocol đã đăng ký. Tập 40 game ở trên chỉ là quan sát sau chọn mẫu, không thay thế holdout.
3. **P1 — bảo vệ resume và provenance:** lưu cursor đã sửa cho NPZ mới; cần thiết kế/check generation giữa checkpoint và replay, hash tensor worker/parent và xử lý legacy buffer tuổi không rõ trước lần chạy thật. Việc thay đổi checkpoint schema hoặc quy trình khôi phục cần phê duyệt theo `SPEC.md`; không ghi đè hiện vật hiện tại.

Không khẳng định đã sửa mọi nguyên nhân; chưa có thí nghiệm đối chứng chứng minh cải tiến sức chơi. Không chạy training, self-play hay tournament quy mô lớn trong lần điều tra này.
