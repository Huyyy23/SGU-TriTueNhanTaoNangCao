# Kiểm toán tính đúng đắn MCTS Python/C++

## Phạm vi và kết luận

Kiểm toán mã hiện tại, không sửa thuật toán; đối chiếu `SPEC.md:179-228,254-275`, đặc biệt mỗi lần tìm kiếm AZ thực sự phải dùng **200 simulations** (`SPEC.md:193`). Không suy diễn hành vi lịch sử của checkpoint từ mã hiện tại. Mức tin cậy nói về kết luận cụ thể, không phải chứng nhận toàn hệ thống.

| Kết luận | Ưu tiên / tác động | Tin cậy |
|---|---|---|
| Luồng tournament/evaluation cho phép budget khác 200; time mode còn thay đổi budget từng simulation và không thực sự chặn ở cap được ghi chú. | P1 — vi phạm contract khi chọn tùy chọn tương ứng | Cao, đối chiếu mã |
| Cả hai backend không xác thực đầy đủ shape/miền của policy tại consumer; dữ liệu lỗi có thể bị biến thành uniform hoặc prior âm. | P2 — rủi ro phụ thuộc đầu vào lỗi chưa quan sát | Cao về control flow; chưa quan sát lỗi model thực tế |
| Với fixture Normal bàn trống và evaluator cố định, 6 cặp tìm kiếm 200 simulations cho thống kê Python/C++ trùng tuyệt đối. | Bằng chứng tích cực có giới hạn, không phải finding | Cao cho fixture đã chạy |
| Đảo dấu, terminal, mask, noise và nhiệt độ mặc định phù hợp công thức khi đầu vào hợp lệ. | Không phát hiện lỗi trong phạm vi đọc | Trung bình; không vét cạn trạng thái |

## Phát hiện

### 1. P1 — Budget evaluation có thể vi phạm contract

`config.py:8-17` đặt mặc định 200 nhưng `az/selfplay.py:111-122` chuyển trực tiếp giá trị cấu hình cho MCTS, không ép bằng 200. `scripts/tournament.py:145-166` nhận `sims` hoặc budget mode và chuyển số đó vào factory; `players/factory.py:47-48,86-87` chỉ yêu cầu số nguyên dương. Vì vậy mặc định 200 không phải ràng buộc 200. Trong chế độ thời gian, `players/az_player.py:39-62` đặt lại `mcts.n_simulations = mcts.simulations_done + 1` trong mỗi vòng; vòng lặp dừng theo deadline chứ không theo cap ban đầu, trái với mô tả cap tại `scripts/tournament.py:162-165`. Khi không có child visit, player chọn ngẫu nhiên một legal move (`players/az_player.py:68-71`), không phải kết quả 200 simulations. Đây là suy luận trực tiếp từ control flow, chưa chạy tournament hay deadline experiment. Tactical override trước MCTS được contract cho phép nếu ghi 0; không đồng nhất trường hợp đó với budget biến đổi.

**Tác động:** các kết quả tạo với budget tùy chọn không tuân thủ protocol 200, có thể không so sánh được; không có bằng chứng rằng việc này đã gây ra kết quả của run 25k. **Tin cậy:** cao về khả năng từ mã, chưa đo tần suất hoặc tác động thực tế.

Tái lập bằng đọc mã: đối chiếu các dòng trên với `SPEC.md:193`; không cần chạy tournament. Khuyến nghị kiểm soát budget tại tất cả entry point AZ, xử lý time mode theo contract hoặc tuyên bố rõ nó không phải protocol đánh giá 200. Không nhầm rollout MCTS Pure với AZ.

### 2. P2 — Policy không được kiểm tra đầy đủ

`SPEC.md:220-222` yêu cầu policy/value có shape và miền hợp lệ tại consumer. Python `az/mcts.py:163-174` và native adapter `az/mcts.py:395-407` nhân rồi chuẩn hóa policy mà không kiểm tra shape, finite hoặc không âm. Ví dụ NaN làm tổng NaN, nhánh `> 1e-8` sai và dẫn tới uniform; policy có phần tử âm nhưng tổng dương tạo prior âm. Sai shape có thể broadcast hoặc phát lỗi không rõ nghĩa. `az/model.py:86-104` dùng softmax nên đầu ra bình thường không âm, nhưng điều đó không chứng minh tính hữu hạn cho mọi checkpoint/evaluator; tác động thực tế chưa được đo. Python kiểm tra value bằng `assert` (`az/mcts.py:190-192`), có thể bị tắt với Python `-O`; native dùng kiểm tra tường minh (`az/mcts.py:389-394`). Khuyến nghị kiểm tra shape `(121,)`, hữu hạn và không âm trước masking tại ranh giới consumer, kiểm tra value tường minh đồng nhất hai backend.

**Tác động:** evaluator hỏng có thể âm thầm tạo phân phối tìm kiếm sai hoặc gây lỗi về sau; chưa quan sát policy lỗi hay chứng minh tác động đến run 25k. **Tin cậy:** cao về control flow; thấp về tần suất thực tế. **Tái hiện tĩnh:** đối chiếu `SPEC.md:220-222` với hai nhánh normalize tại `az/mcts.py:163-174,389-407`: NaN khiến tổng NaN và đi vào nhánh uniform; một phần tử âm với tổng dương được giữ lại sau chia. Không cần chạy tìm kiếm.

## Đối chiếu cơ chế

- **PUCT và dấu:** Q của node là theo người sắp đi (`az/mcts.py:32-37`); selection dùng `-child.q_value + c_puct * prior * sqrt(max(1,N))/(1+n)` (`az/mcts.py:279-299`; C++ `native/gomoku_mcts_native.cpp:170-190`). Với leaf value theo người tại leaf, backup NN đổi dấu mỗi cạnh (`az/mcts.py:190-200`; C++ `native/gomoku_mcts_native.cpp:99-114`). Terminal sau nước thắng có value -1 cho người tiếp theo, draw 0, không gọi NN; terminal backup luân phiên dấu (`az/mcts.py:108-147,301-309`; native adapter `az/mcts.py:363-378`, C++ `native/gomoku_mcts_native.cpp:41-76,193-201`). Đọc mã không chứng minh mọi cây sâu, trạng thái hoặc quy tắc.
- **Mask và ưu tiên:** legal mask là legal trừ forbidden tại mỗi leaf (`az/mcts.py:324-328`), policy được mask và chuẩn hóa hoặc uniform legal nếu tổng nhỏ (`az/mcts.py:163-186,395-407`). C++ tree nhận danh sách hợp lệ từ cùng adapter/rules Python; không có rules C++ độc lập (`SPEC.md:254-260`). Không có nước hợp lệ được backup như hòa (`az/mcts.py:140-147,374-378`).
- **Noise:** chỉ root, chỉ các nước legal, theo công thức trộn Dirichlet (`az/mcts.py:176-186,403-407`). Self-play bật noise (`az/selfplay.py:111-122`); `AZPlayer` tắt (`players/az_player.py:40-49`). Random tie dùng cùng RNG, không đảm bảo parity xác suất trên mọi môi trường.
- **Temperature:** `select_move(0)` chọn ngẫu nhiên trong các nước có visit cao nhất; với nhiệt độ dương lấy lũy thừa `N^(1/τ)` (`az/mcts.py:228-248,419-429`). Self-play tạo target riêng bằng `pi_target_temp`, sau đó chọn action τ=1 cho 15 nước đầu, 0 về sau (`az/selfplay.py:152-160`; `config.py:17,34`). Evaluation dùng τ=0 (`players/az_player.py:68-71`). Các giá trị τ cực nhỏ hoặc không hợp lệ chưa được kiểm chứng; lũy thừa trực tiếp có rủi ro overflow/chia 0 ngoài cấu hình mặc định.
- **Phạm vi train/eval:** cùng MCTS/backend adapter, nhưng self-play batch nhiều game (`az/selfplay.py:84-102`) và evaluation gọi từng leaf (`players/az_player.py:52-62`); noise, sampling và budget mode khác như trên. Đây không phải khẳng định parity toàn bộ trajectory hay sức chơi. Reset lại cây sau mỗi nước (`az/selfplay.py:178-183`). Root expansion tính một simulation nên tổng child visits thường là 199, không phải 200 (`SPEC.md:223`).

## Kiểm thử hiện có

Đã chạy `PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/test_mcts.py::test_backup_negates_child_value tests/test_mcts.py::test_forbidden_moves_masked_and_never_expanded tests/test_mcts.py::test_dirichlet_noise_only_at_root tests/test_mcts.py::test_win_in_1`: **4 passed in 1.21s**. Đây là fixture stub giới hạn, không chứng minh mọi trạng thái hay toàn bộ pipeline. Các test khác và full suite chưa chạy. `tests/test_native_mcts.py:40-136` có fixture differential, Renju và terminal, nhưng không chạy trong lượt này; benchmark 200 simulations bên dưới là riêng. Cần thêm differential đa seed và state hợp lệ, cây sâu/terminal, pi/z trajectory và quality gate trước khi kết luận C++ parity rộng.

## Microbenchmark có kiểm soát

Đã chạy trên Windows 10 build 22631, Python 3.10.6, NumPy 2.2.6, extension C++ hiện có; evaluator giả trả `linspace(1,2,121)` và value 0.2, Normal board trống, cùng seed 11/12/13, reset mới mỗi lần, 200 simulations/search. Không inference checkpoint, không training/tournament; đo wall-clock toàn search gồm Python rules/features, không có warm-up riêng. Thứ tự chạy Python trước C++, chỉ 3 mẫu, máy có tải chưa kiểm soát nên không khái quát speedup thực tế.

| Noise | Backend | Thời gian từng lần (s) | Median (s) |
|---|---|---|---|
| tắt | Python | 0.0364, 0.0633, 0.0646 | 0.0633 |
| tắt | C++ | 0.0093, 0.0087, 0.0089 | 0.0089 |
| bật | Python | 0.0674, 0.0349, 0.0707 | 0.0674 |
| bật | C++ | 0.0108, 0.0097, 0.0092 | 0.0097 |

Tất cả 12 search có `simulations_done=200`, 200 NN calls giả và tổng 199 child visits. Với từng seed/noise, hai backend có visit counts và bốn mảng root statistics giống tuyệt đối (max sai khác 0), root value sai khác 0. Kết quả chỉ cho workload/seed này; không đại diện end-to-end GPU, Renju, sức chơi hoặc chi phí theo batch.

Lệnh tái lập (thời gian sẽ khác):

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe - <<'PY'
import time, platform, sys, numpy as np
from az.mcts import MCTS, NativeMCTS
from gomoku.board import Board
from gomoku.rules.normal import NormalRules
print('python',sys.version.split()[0], 'numpy',np.__version__, 'platform',platform.platform())
policy=np.linspace(1,2,121,dtype=np.float32)
for noise in (False,True):
 out=[]
 for cls in (MCTS,NativeMCTS):
  times=[]; data=[]
  for seed in (11,12,13):
   b=Board(); s=cls(NormalRules(),n_simulations=200,root_noise=noise,rng=np.random.default_rng(seed)); t=time.perf_counter(); c=s.search(b,lambda r:(policy,0.2)); times.append(time.perf_counter()-t); data.append((c.copy(),s.root_statistics(),s.root_value(),s.nn_calls,s.simulations_done))
  out.append(data); print('noise',noise,cls.__name__,'seconds',*[round(x,4) for x in times],'median',round(float(np.median(times)),4),'childvisits',[int(x[0].sum()) for x in data],'nn',[x[3] for x in data],'sims',[x[4] for x in data])
 for j in range(3):
  a,b=out[0][j],out[1][j]; print('seed',11+j,'visits_equal',np.array_equal(a[0],b[0]),'stats_maxdiff',max(float(np.max(np.abs(x.astype(float)-y.astype(float)))) for x,y in zip(a[1],b[1])),'value_delta',abs(a[2]-b[2]))
PY
```

### Chứng cứ checkpoint độc lập do kiểm toán chính cung cấp

Trên CPU một thread với `latest.pt` SHA-256 `f7f521c1cc37a5054ed854201b087e484e9fcef3d4d6b1877ab81dff2e51d7e4` (metadata n_iter 500/global_step 484000), Normal, seed 0, Python MCTS 200 không noise: vị thế win từ `Board.play` lần lượt `[(5,1),(5,0),(5,2),(0,0),(5,3),(0,2),(5,4),(1,4)]` chọn (5,5), policy 0.841874, value 0.998483, 199/199 visits; vị thế block `[(5,0),(5,1),(0,0),(5,2),(0,2),(5,3),(1,4),(5,4)]` chọn (5,5), policy 0.242531, value -0.939954, 68/199 visits. Đây là kết quả do agent chính báo, không phải thí nghiệm tôi tự chạy; hai tình huống tổng hợp không chứng minh sức chơi, tính calibration, lịch sử training hoặc parity C++. Tái lập cần đúng file theo SHA, `AZNet` từ checkpoint, `NormalRules`, tuần tự `Board.play` trên, `np.random.default_rng(0)`, `n_simulations=200`, `root_noise=False`, và `torch.set_num_threads(1)`; record policy/value root trước masking và root visits. Lệnh đầy đủ và output gốc thuộc báo cáo agent chính.

Tái lập hai thế checkpoint (file phải có SHA đã nêu):

```bash
.venv/Scripts/python.exe - <<'PY'
import numpy as np, torch
from az.model import load_checkpoint
from az.mcts import MCTS
from gomoku.board import Board
from gomoku.rules.normal import NormalRules
from gomoku.features import board_to_planes
torch.set_num_threads(1)
model, _ = load_checkpoint('checkpoints/latest.pt', device='cpu')
rules = NormalRules()
cases = {
 'win': [(5,1),(5,0),(5,2),(0,0),(5,3),(0,2),(5,4),(1,4)],
 'block': [(5,0),(5,1),(0,0),(5,2),(0,2),(5,3),(1,4),(5,4)],
}
for name, moves in cases.items():
 board = Board()
 for r, c in moves:
  board.play(r, c)
 p, v = model.predict(board_to_planes(board, rules, board.current_player)[None])
 search = MCTS(rules=rules, n_simulations=200, root_noise=False, rng=np.random.default_rng(0))
 search.reset(board)
 while (request := search.request()) is not None:
  probs, values = model.predict(request.planes[None])
  search.receive(probs[0], float(values[0]))
 counts = search.visit_counts()
 print(name, 'prior', p[0,60], 'value', v[0], 'move', divmod(int(search.select_move(0)),11),
       'target_visits', counts[60], 'total_child_visits', counts.sum())
PY
```
