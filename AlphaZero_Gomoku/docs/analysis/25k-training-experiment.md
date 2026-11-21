# Thí nghiệm training và phục hồi AlphaZero 25k

## Kết luận vận hành

**Chưa đủ bằng chứng để tiếp tục self-play dài từ checkpoint 25k.** Checkpoint `checkpoints/latest.pt` SHA-256 `f7f521c1cc37a5054ed854201b087e484e9fcef3d4d6b1877ab81dff2e51d7e4` có thể dùng làm parent cho một nghiên cứu mới, không đồng nghĩa có thể resume trạng thái 25k chính xác. Replay legacy `buffer_latest.npz` thiếu cursor và chứng cứ gắn checkpoint, không được coi là replay đã xác minh. Không xóa hay ghi đè hiện vật gốc.

Thiếu trước continuation: (1) replay mới có game ID, seed, checkpoint parent và ruleset, tách biệt rõ training/evaluation; (2) một đường bootstrap Normal từ weights 25k với replay mới và lựa chọn optimizer/LR được ghi rõ — CLI hiện tại chỉ tạo model mới hoặc resume generation đã xác minh (`scripts/train.py:31-50,297-349`); (3) thắng không suy giảm trên tactical holdout theo nhóm và tournament paired đủ mẫu/uncertainty; (4) kiểm chứng power-loss durability nếu deployment yêu cầu. Không tự gắn hash mới cho replay cũ để tuyên bố tương thích lịch sử.

## Kiểm chứng generation trên Windows

Bổ sung `tests/test_trainer.py` thử dừng tiến trình con thật tại điểm ngay trước/sau commit pointer trong thư mục tạm. Phép thử Windows cho thấy pointer vẫn trỏ generation cũ khi dừng trước commit; sau commit chọn generation mới; orphan chưa commit bị từ chối; rollback về generation cũ giữ nguyên hash checkpoint/replay và các file legacy mẫu. Không phát hiện bug mới trong phạm vi này, không sửa logic recovery.

Regression trên Windows: `PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/test_trainer.py tests/test_replay_buffer.py tests/test_train_resume_logging.py tests/test_tactical_benchmark.py tests/test_tactical_oracle.py tests/test_native_mcts.py` → **49 passed in 11.98s**; bao gồm phép thử subprocess và native parity hiện có. `graphify update .` đã chạy sau thay đổi code.

Đây là bằng chứng cho termination có kiểm soát ở các seam cụ thể, không chứng minh mọi vị trí crash hay an toàn mất điện vật lý: Windows directory-entry fsync chưa được chứng minh. Immutable manifest + hash phát hiện file bị thay, không chứng nhận chất lượng dữ liệu hoặc lineage 25k. Các generation cũ và dữ liệu người dùng không bị xóa.

## Benchmark development/holdout mới

`scripts/tactical_benchmark.py` có 16 vị trí Normal: 8 development, 8 holdout; mỗi split có một vị trí cho mỗi cặp bốn loại win/block/fork/forced-defense và màu. Nhóm construction family được phân trước; kiểm thử `tests/test_tactical_benchmark.py` kiểm tra không có state tương đương theo tám đối xứng D4 giữa hai split, trạng thái đi hợp lệ và oracle cửa sổ năm ô độc lập. Forced-defense chỉ đảm bảo chặn thắng ngay/fork ở horizon nông, không phải lời giải game. Negative controls kiểm tra riêng trong tests, không tính vào accuracy tám thế.

Holdout **độc lập với các sample development của thí nghiệm mới**, nhưng provenance training 25k không đầy đủ nên không chứng minh chưa gặp các trạng thái này trước đó. 8 thế, chỉ một thế mỗi category/màu, không đủ kết luận tổng quát hay ước lượng uncertainty chắc chắn. Cùng hình dạng chiến thuật và số lượng nhỏ cũng có thể khiến chuyển giao giữa hai split dễ hơn bài toán thực tế.

Kiểm tra thực tế toàn bộ 50.000 feature trong NPZ legacy SHA `908a22…`: không có fixture nào trong 16 thế khớp byte với bất kỳ một trong tám đối xứng của nó; 28.583 feature khác nhau. Đây chỉ loại overlap exact trên **buffer hiện tại**, không loại overlap các thế từng bị ring ghi đè, augmentation/biến thể gần giống, hoặc chính checkpoint 25k đã học từ game tương đương.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe - <<'PY'
import numpy as np
from scripts.tactical_benchmark import FIXTURES, board_for
from gomoku.features import board_to_planes, symmetry_transform
from gomoku.rules.normal import NormalRules
with np.load('checkpoints/buffer_latest.npz') as d:
 keys={x.tobytes() for x in d['features']}
 print('stored',len(d['features']),'distinct',len(keys))
for f in FIXTURES:
 b=board_for(f); x=board_to_planes(b,NormalRules(),b.current_player)
 print(f.split,f.name,[k for k in range(8) if symmetry_transform(x,k).tobytes() in keys])
PY
```

## Cô lập prior, value và 200-simulation MCTS

Ngoài hai canary trong [báo cáo trước](25k-improvement-report.md), kiểm tra thế `fork_white` của benchmark cũ (đáp án oracle nông `{60}`), checkpoint SHA nêu trên, Normal, C++ MCTS, seed 42, noise tắt, CPU một thread, 200 simulations mỗi search:

| Intervention | Nước chọn | Visits ô 60 | Prior ô 60 | Root value | NN calls |
|---|---:|---:|---:|---:|---:|
| Baseline | 57 | 0 | 0,034148 | +0,786808 | 200 |
| Uniform chỉ tại root | 50 | 0 | 0,008772 | +0,919450 | 200 |
| Value NN tại lá không terminal = 0 | 57 | 3 | 0,034148 | +0,090000 | 182 |

Zero value chỉ thay evaluation phi terminal; terminal value vẫn do rules/search xử lý. Cả ba đều bỏ lỡ fork nông. Uniform root làm prior ô đúng giảm ở vị trí này; zero value làm ô đúng được thăm ba lần nhưng chưa thắng cạnh tranh visits. Q = 0 khi chưa thăm không phải dự đoán value. Kết luận chỉ áp dụng vị trí/seed này, không chứng minh causal training hay giá trị game-theoretic. Bằng chứng trước đó cho thấy canary khác có thể hưởng lợi uniform root; không tích hợp uniform prior hoặc zero value vào production.

Lệnh tái lập thế này (`dev_gap_three_white` là chính fixture `fork_white` cũ):

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 .venv/Scripts/python.exe - <<'PY'
import numpy as np, torch
from scripts.tactical_benchmark import FIXTURES, board_for, oracle
from az.model import load_checkpoint
from az.mcts import NativeMCTS
from gomoku.rules.normal import NormalRules
torch.set_num_threads(1); model,_=load_checkpoint('checkpoints/latest.pt',device='cpu')
f=next(f for f in FIXTURES if f.name=='dev_gap_three_white')
b=board_for(f); answers=sorted(oracle(b.grid,f.player,f.category))
for mode in ('baseline','uniform_root','zero_value'):
 s=NativeMCTS(NormalRules(),n_simulations=200,root_noise=False,rng=np.random.default_rng(42)); s.reset(b); calls=0
 while (r:=s.request()) is not None:
  p,v=model.predict(r.planes[None]); policy=np.ones_like(p[0]) if mode=='uniform_root' and calls==0 else p[0]
  s.receive(policy,0.0 if mode=='zero_value' else float(v[0])); calls+=1
 visits,priors,q,_=s.root_statistics()
 print(mode,s.select_move(0),visits[answers],priors[answers],q[answers],s.root_value(),calls,s.simulations_done)
PY
```


## Training nhỏ có đối chứng

Hai nhánh cùng khởi tạo từ checkpoint 25k SHA nêu trên; cùng tám vị trí development, seed 9173, tám bước Adam `lr=1e-4`, full batch, BatchNorm đóng băng và cùng target value là prediction của parent. Control dùng policy mềm của parent; nhánh tactical dùng policy đều trên tập nước oracle nông. Không đưa tám vị trí holdout vào updates. Hai nhánh có cùng số bước và dữ liệu đầu vào, nhưng khác policy target; mục tiêu chỉ là can thiệp thăm dò, không phải pipeline self-play tiêu chuẩn.

Trên tám vị trí holdout mới, mỗi vị trí dùng 200 simulations, seed đã cố định, cả Python và C++: control **6/8** (Đen 3/4, Trắng 3/4), tactical **5/8** (Đen 3/4, Trắng 2/4). Hai backend cho cùng accuracy trong các search này; không khẳng định mọi visit/tree trùng nếu chưa so đủ. Kết quả này không cho thấy tactical target cải thiện tổng quát; sample quá nhỏ và holdout không chứng minh độc lập với lịch sử 25k. Không tích hợp thay đổi thuật toán hoặc recipe training dựa trên thí nghiệm này.

Theo category, control và tactical lần lượt: win **0/2 vs 0/2**, block **2/2 vs 1/2**, fork **2/2 vs 2/2**, forced defense **2/2 vs 2/2**. Ở block Trắng, answer prior tăng 0,00365 → 0,09071 nhưng answer visit fraction giảm 0,16080 → 0,05528 và nước chọn đổi từ đúng sang sai; prior cao hơn không tự đảm bảo quyết định tốt hơn. Cả hai nhánh bỏ lỡ win-in-one ở hai màu: ô thắng 60, chọn 63; control prior 0,03070/0,01064 và 0 visits; tactical prior 0,32473/0,24574 nhưng chỉ 36,18%/26,13% visits và vẫn không được chọn. Root value gần +1 không chứng minh agent thật sẽ thắng.

Training trên CPU một thread: control 4,689 giây, tactical 4,214 giây, mỗi nhánh tám updates, batch 8. Tactical policy loss đo trước update giảm 1,8568 → 0,5350 **trên development**; đây không phải bằng chứng holdout hoặc đối kháng cải thiện. Cả hai nhánh dùng value target bằng prediction parent, do đó loss value trong training không phải calibration so outcome. Lượt chạy đầu có deadline 300 giây, hoàn thành 32 search holdout (2 model × 2 backend × 8 vị trí × 200 simulations) và một ván lẻ control; phần tournament bị timeout. Không gộp ván lẻ đó vào kết quả paired bên dưới.

## Tournament paired nhỏ và calibration

Hai opening hợp lệ nhưng cố ý gần kết thúc: `[44,77,45,78,46,79,47,80]` và `[22,88,23,89,24,90,25,91]`; mỗi opening đổi màu, seed theo cặp 100/101, mỗi nhánh gặp cùng Heuristic, C++ MCTS 200 simulations mỗi search. Hai opening đều có thế thắng ngay cho Đen, **không đại diện** phân phối bắt đầu bàn trống; đây chỉ là kiểm tra cực nhỏ, không phải estimate strength thực tế. Tổng tám ván, bốn mỗi nhánh.

| Nhánh | W/D/L | Score | ΔElo mô tả so Heuristic | Simulations | Thời gian game |
|---|---:|---:|---:|---:|---:|
| Control | 2/0/2 | 0,50 | 0 | 400 | 0,15 s |
| Tactical | 1/0/3 | 0,25 | −190,85 | 3.000 | 104,36 s |

Elo ở đây dùng `400 log10(score/(1-score))`, không điều chỉnh màu và không phải Davidson estimate; điểm 0 hoặc 1 cho giá trị biên vô cực, không tự bịa số hữu hạn. Bootstrap percentile nếu coi hai opening là hai clusters cho tactical score `[0, 0,5]`, tương ứng Elo `[−∞, 0]`; hai opening cùng dạng chiến thuật nên giả định cluster độc lập cũng đáng ngờ. CI rất thô, không đủ cơ sở infer non-inferiority, strength hay promotion. Control có score cố định 0,5 ở cả hai cặp nên bootstrap suy biến; không coi `[0,0]` là CI strength đáng tin. Thời gian/simulations khác vì trajectory khác: không so tốc độ. Training 8 bước/mỗi nhánh là chi phí thêm; tổng tournament 3.400 simulations, giữ 200 mỗi search AZ.

Đánh giá value theo outcome thật tại đầu hai opening ở các ván **AZ cầm Đen** (hai positions/nhánh): control Brier trung bình **0,36525**, value MSE **1,46101**; tactical Brier **0,57799**, MSE **2,31196**. Giá trị/outcome từng cặp: control `(-0,56952, thắng)`, `(0,32277, thắng)`; tactical `(0,09241, thắng)`, `(0,94941, thua)`. Dùng `p=(v+1)/2`, z=±1 theo AZ sắp đi. Mẫu n=2 cho mỗi nhánh ở màu Đen quá nhỏ để gọi là calibration. AZ Trắng không có lượt nào sau opening trong bốn ván này vì Heuristic cầm Đen thắng ngay, nên **không có calibration Trắng mới**, không bịa số; các kết quả theo hai màu trên 40 game promotion cũ chỉ mang tính mô tả và không độc lập như [báo cáo trước](25k-improvement-report.md).

Kết quả này không chứng minh tactical training làm kém hay tốt trong điều kiện thực; hai opening quá nhân tạo, mẫu nhỏ, thiếu training seeds lặp và holdout có cùng họ hình học hạn chế. Phương án an toàn nhất hiện tại là **không tích hợp intervention**, chưa chọn cấu hình continuation mới.

### Tái lập thí nghiệm

Lệnh sau huấn luyện hai model chỉ trong RAM rồi chạy 32 search holdout và tám ván paired; mỗi search 200 simulations. Lượt đo trên máy này mất nhiều phút; nếu chỉ cần kiểm tra một phần, giảm vòng case/game nhưng không thay ngân sách simulation. Dữ liệu lịch sử chỉ đọc và không tạo checkpoint hay SQLite:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 .venv/Scripts/python.exe - <<'PY'
import copy, hashlib, json, random, time
import numpy as np, torch, torch.nn.functional as F
from az.model import load_checkpoint
from gomoku.features import board_to_planes
from gomoku.rules.normal import NormalRules
from gomoku.board import Board
from players.az_player import AZPlayer
from players.factory import create_player
from scripts.tactical_benchmark import FIXTURES, board_for, oracle, run_case
torch.set_num_threads(1); torch.manual_seed(9173); np.random.seed(9173); random.seed(9173)
p='checkpoints/latest.pt'; sha=hashlib.sha256(open(p,'rb').read()).hexdigest()
assert sha=='f7f521c1cc37a5054ed854201b087e484e9fcef3d4d6b1877ab81dff2e51d7e4'
rules=NormalRules(); parent,_=load_checkpoint(p,device='cpu'); parent.eval()
dev=[f for f in FIXTURES if f.split=='dev']; hold=[f for f in FIXTURES if f.split=='holdout']
boards=[board_for(f) for f in dev]
x=torch.tensor(np.stack([board_to_planes(b,rules,b.current_player) for b in boards]),dtype=torch.float32)
with torch.no_grad():
 logits,z=parent(x); control_target=logits.softmax(-1).detach(); z=z.detach()
tactical_target=torch.zeros_like(control_target)
for i,(f,b) in enumerate(zip(dev,boards)):
 answers=sorted(oracle(b.grid,f.player,f.category)); assert answers and f.expected in answers
 tactical_target[i,answers]=1/len(answers)
models={}
for name,target in [('control',control_target),('tactical',tactical_target)]:
 m=copy.deepcopy(parent); m.eval(); opt=torch.optim.Adam(m.parameters(),lr=1e-4)
 start=time.perf_counter()
 for _ in range(8):
  opt.zero_grad(); policy,value=m(x)
  lp=-(target*F.log_softmax(policy,dim=-1)).sum(-1).mean(); lv=F.mse_loss(value,z)
  (lp+lv).backward(); opt.step()
 print('train',name,'seconds',time.perf_counter()-start,'last_loss_preupdate',float(lp.detach()),float(lv.detach()))
 models[name]=m
for name,m in models.items():
 for backend in ('python','cpp'):
  for f in hold:
   r=run_case(f,m,backend,42)
   print('holdout',name,backend,f.name,r['correct'],r['selected'],r['answers'],
         r['answer_prior_mass'],r['answer_visit_mass'],r['root_nn_value'],r['root_value'],r['simulations'])
openings=[(44,77,45,78,46,79,47,80),(22,88,23,89,24,90,25,91)]
for name,m in models.items():
 for pair,opening in enumerate(openings):
  for color in (1,2):
   board=Board()
   for action in opening: board.play(*divmod(action,11))
   assert not rules.is_terminal(board)
   agent=AZPlayer(p,rules,cfg={'n_simulations':200,'mcts_backend':'cpp'},seed=100+pair,device='cpu')
   agent.model=m; opponent=create_player('heuristic',rules,seed=100+pair,patterns_backend='python')
   searches=0; start=time.perf_counter()
   while not rules.is_terminal(board):
    is_az=board.current_player==color; player=agent if is_az else opponent
    row,col=player.select_move(board); assert row*11+col in rules.legal_moves(board)
    if is_az:
     assert player.simulations_run==200
     searches+=1
    board.play(row,col)
   winner=3-board.current_player if rules.is_win(board) else 0
   print('game',name,pair,color,winner,'plies',len(board.history),'simulations',200*searches,
         'seconds',time.perf_counter()-start)
PY
```


## Kế hoạch continuation cụ thể, chưa thực thi

1. Giữ nguyên 25k `.pt`/`.npz`; tạo run ID, thư mục generation và database **mới**. Chốt parent checkpoint SHA ở trên; chọn rõ optimizer mới hoặc optimizer lịch sử là hai protocol khác nhau. Không tiếp tục global-step/cosine một cách ngầm định. Hiện CLI chưa có đường bootstrap Normal 25k + replay rỗng; cần thiết kế, test và phê duyệt trước khi dùng, không ghép legacy giả làm generation.
2. Tạo replay Normal mới từ game/seed có provenance: checkpoint hash per worker/iteration, game ID, seeds, rules, openings, code/config hash, target π/z và generation. Dành riêng game/seed/opening family, cùng mọi biến thể D4 của chúng, cho evaluation; không đưa holdout vào replay hoặc chọn checkpoint. Tách riêng tập theo đối thủ; không ép cân bằng kết quả Đen/Trắng.
3. Dùng cấu hình `configs/normal.yaml` hiện có chỉ làm **control tham khảo**, không tuyên bố tốt nhất; giữ ResNet, rules và 200 simulations. Sau khi chốt optimizer/LR, replay/steps và compute, chạy pilot nhỏ cùng parent và seeds cho control và một can thiệp. Tactical oracle targets phải có scope/horizon và một tập không dùng để tuning; kiểm tra policy target và value theo outcome. Thí nghiệm tám bước trên chưa đủ lựa chọn cấu hình.
4. Trước khi mở rộng, yêu cầu recovery tests trên filesystem triển khai, provenance/hash không thiếu, parity Python/C++ trên tập đa trạng thái, không có regression luật/target, tactical holdout đa dạng hơn kèm CI theo game/nhóm, value calibration theo **cả hai màu** và tournament matched openings/đổi màu với CI theo opening clusters. Pilot gần terminal ở đây không thỏa điều kiện strength. Quyết định số game confirmatory phải dựa trên pilot/power và được phê duyệt trước.

Không có policy/training production mới được tích hợp; không chạy self-play dài hay tournament lớn, không commit, không sửa checkpoint/replay lịch sử.
