# Phục hồi và thí nghiệm AlphaZero 25k

## Quyết định

**Chưa nên tiếp tục training dài trực tiếp từ cặp 25k hiện tại.** Checkpoint `checkpoints/latest.pt` SHA-256 `f7f521c1cc37a5054ed854201b087e484e9fcef3d4d6b1877ab81dff2e51d7e4` có thể dùng để inference và làm điểm xuất phát nghiên cứu. `checkpoints/buffer_latest.npz` SHA `908a22b249591fdbf64fb1f7430de879d0b94e2925be4e619a5db484e0170fb9` là buffer legacy đầy không có cursor hoặc bằng chứng hash liên kết lịch sử: **không dùng nó như replay resume đã xác minh**, không xóa hay gán chứng nhận bằng cách hash hậu nghiệm. Chưa có replay nào được chứng minh tốt nhất và chưa có cấu hình training nào có đủ bằng chứng tốt nhất.

Để mở một continuation mới, cần quy trình được chấp thuận rõ ràng cho khởi tạo replay mới/có provenance, tách bạch việc giữ hay làm mới optimizer, lưu parent checkpoint hash và giữ bản gốc; đánh giá so với checkpoint 25k trên tập game/seed chưa dùng chọn cấu hình. Không kết luận weights 25k không hữu ích, cũng không khẳng định tự chơi thêm sẽ cải thiện.

## Immutable generations và recovery

`Trainer.save` ghi checkpoint, replay và manifest có SHA-256 vào `generations/<UUID>/`, sau đó commit bằng thay thế atomically pointer `latest.json`; `best.json` trỏ generation được promote. `resolve_generation` kiểm tra hash, đường dẫn và ancestry; `scripts/train.py --resume --generation <id>` chọn generation ancestor đã commit để rollback. Rollback với STANDARD logging yêu cầu run ID mới để không ghi đè log cũ. `az/renju_finetune.py` chuyển sang generation API. Không có migration tự động từ 25k legacy, không xóa generation hay sửa `.pt`/`.npz` lịch sử. Xem `SPEC.md:230-252` và tests về crash/rollback/tampering.

Kiểm thử đầu tiên phát hiện `os.fsync` trên handle read-only bị Windows từ chối (4 failed). Đã đổi sang handle đọc/ghi; sau cutover các caller, focused suite **110 passed, 2 skipped in 22.28s**. Test mô phỏng crash trước/sau commit, không phải kiểm chứng mất điện vật lý. Windows directory fsync/durability khi mất điện chưa được chứng minh; không tuyên bố power-loss safety tuyệt đối.

Lệnh:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/test_trainer.py tests/test_replay_buffer.py tests/test_train_resume_logging.py tests/test_native_mcts.py tests/test_tactical_benchmark.py tests/test_tactical_oracle.py tests/test_model.py tests/test_tournament.py tests/test_arena.py tests/test_az_player.py tests/test_agent_factory.py
```

## Benchmark chiến thuật mới

`scripts/tactical_benchmark.py` dùng oracle cửa sổ năm ô độc lập với detector production, 8 fixture Normal mới, cân bằng Đen/Trắng cho bốn loại: win, block, fork, forced defense nông. Mỗi search 200 simulations, seed 42, no noise, temperature 0; oracle shallow không chứng minh forced win theo toàn bộ ván. Fixture khác hai canary cũ nhưng **chưa thể loại trùng exposure của replay lịch sử**, do đó không gọi benchmark là holdout training độc lập. Mẫu một thế/màu/loại quá nhỏ để ước lượng tổng quát.

Chạy C++ trên checkpoint 25k: **7/8**, win 2/2, block 2/2, fork 1/2, forced defense 2/2. Riêng fork Trắng: đáp án 60, chọn 57; prior hợp lệ trên đáp án 0,03415, 0/199 child visits; root NN value +0,99717 và root search value +0,78681. Vì vậy value cao không đảm bảo tìm được fork; một fixture không cho phép suy ra nguyên nhân thống kê hay lỗi sign. Các kết quả mỗi case và visits/prior/value được CLI xuất JSON:

```bash
OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe scripts/tactical_benchmark.py --checkpoint checkpoints/latest.pt --backend cpp --seed 42
```

Python cũng đạt 7/8, chọn cùng tám nước và có cùng tổng visit mass trên đáp án đến sáu chữ số với C++ trong run seed 42; chỉ so các thống kê đó, không kết luận cây hoàn toàn giống. Lệnh tương tự với `--backend python`.

## Thí nghiệm policy/training nhỏ

Thí nghiệm exploratory trong RAM, không lưu checkpoint hoặc sửa replay: từ đúng checkpoint 25k, copy model, đóng băng BatchNorm ở eval, 8 bước Adam `lr=1e-4`, tối ưu policy cross-entropy với target đều trên {5,17,65} ở **canary 0 đã biết**; không train value trực tiếp. So cùng model gốc, seed 0, Normal, Python MCTS, no noise, 200 simulations/search trên canary 0 đã train và canary 1 đã biết nhưng không train. Candidate có thêm chi phí 8 optimizer steps; search budget giống nhau. Target chỉ ngăn fork nông, không chứng minh cứu ván. Hai canary từng được nghiên cứu, không phải holdout độc lập.

| Model / thế | Nước chọn | Visits trên các defense | Root value |
|---|---:|---|---:|
| Baseline, canary 0 | 52 | 1/1/1 | −0,269665 |
| Candidate, canary 0 | 65 | 1/17/87 | −0,832403 |
| Baseline, canary 1 | 96 | 0/0 | +0,393148 |
| Candidate, canary 1 | 41 | 0/23 | +0,365749 |

Candidate chọn một defense nông trên đúng vị trí đã train nhưng vẫn không chọn defense ở canary 1; kết quả có thể là ghi nhớ. Backbone dùng chung nên policy-only update cũng có thể đổi value. Không có bằng chứng tăng sức chơi hay generalization, không tích hợp recipe/trọng số này.

Tái lập pilot train trong RAM, không lưu model:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe - <<'PY'
import copy, hashlib, numpy as np, torch, torch.nn.functional as F
from az.model import load_checkpoint
from az.mcts import MCTS
from gomoku.board import Board
from gomoku.features import board_to_planes
from gomoku.rules.normal import NormalRules
torch.set_num_threads(1); torch.manual_seed(0)
path='checkpoints/latest.pt'; print('sha',hashlib.sha256(open(path,'rb').read()).hexdigest())
base,_=load_checkpoint(path,device='cpu'); candidate=copy.deepcopy(base); rules=NormalRules()
cases=[([20,72,31,58,61,60,9,48,42,53,84,38,28,49,71,50,51,41,81,91,83,82,30,29],[5,17,65]),
       ([20,72,31,58,42,53,9,18,60,70,49,27,71,82,46,94,106,50,38,61,83,48,59,95,39],[29,69])]
boards=[]
for moves,_ in cases:
 b=Board()
 for a in moves: b.play(*divmod(a,11))
 boards.append(b)
x=torch.tensor(board_to_planes(boards[0],rules,boards[0].current_player)[None],dtype=torch.float32)
target=torch.zeros(121); target[cases[0][1]]=1/3
candidate.eval(); opt=torch.optim.Adam(candidate.parameters(),lr=1e-4)
for _ in range(8):
 opt.zero_grad(); logits,_=candidate(x)
 loss=-(target*F.log_softmax(logits[0],dim=0)).sum(); loss.backward(); opt.step()
print('steps',8,'final_training_loss',float(loss.detach()))
for name,model in [('baseline',base),('candidate',candidate)]:
 for i,(board,(_,answers)) in enumerate(zip(boards,cases)):
  s=MCTS(rules,n_simulations=200,root_noise=False,rng=np.random.default_rng(0)); s.reset(board)
  while (req:=s.request()) is not None:
   p,v=model.predict(req.planes[None]); s.receive(p[0],float(v[0]))
  visits,priors,_,_=s.root_statistics()
  print(name,i,'move',s.select_move(0),'visits',visits[answers],
        'priors',priors[answers],'root_value',s.root_value(),'simulations',s.simulations_done)
PY
```

Một probe trên 128 replay samples cùng seed 0: cross-entropy theo target π của policy mạng 2,2972 so với uniform legal 4,7550; đây là độ khớp target có liên quan training, không phải tactical accuracy hay strength, không chứng minh policy tốt khi gặp Heuristic. Ablation trước đây `25k-next-steps.md` cho thấy uniform root prior cứu một canary nhưng không cứu canary còn lại; zero value đơn lẻ không cứu được hai canary đó. Không có cơ sở tích hợp uniform prior toàn cục.

## Đấu đối kháng và calibration

Pilot 4 ván bàn trống, seed 0, Normal, checkpoint trên, 1 CPU thread, native C++ MCTS và Python patterns; mỗi lần AZ search đúng 200 simulations. `az_v0` thua Heuristic khi cầm Đen và Trắng (26/27 ply); `az_v1_tactical` cũng thua cả hai (36/27 ply; override lần lượt 4/1, các nước override không chạy MCTS). Tổng 10.400 simulations, 15,36 giây trong process. Đây là **0/2 vs 0/2**, không đủ chứng minh tactical override tốt hơn/kém hơn về sức chơi. Cùng seed và bàn trống không biến hai ván thành mẫu thống kê độc lập đủ lớn.

Tái lập pilot đối kháng (ghi kết quả ra stdout, không ghi database):

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe - <<'PY'
import hashlib, torch
from players.factory import create_player
from gomoku.board import Board
from gomoku.rules import get_rules
p='checkpoints/latest.pt'; print('sha256',hashlib.sha256(open(p,'rb').read()).hexdigest())
torch.set_num_threads(1); rules=get_rules('normal')
for variant in ('az_v0','az_v1_tactical'):
 for color in (1,2):
  agent=create_player(variant,rules,checkpoint=p,seed=0,n_simulations=200,mcts_backend='cpp',patterns_backend='python')
  opponent=create_player('heuristic',rules,seed=0,patterns_backend='python')
  board=Board(); searches=overrides=0
  while not rules.is_terminal(board):
   az=board.current_player==color; player=agent if az else opponent
   r,c=player.select_move(board); assert r*11+c in rules.legal_moves(board)
   if az:
    assert player.simulations_run in (0,200)
    searches+=bool(player.simulations_run); overrides+=bool(getattr(player,'last_override_applied',False))
   board.play(r,c)
  winner=3-board.current_player if rules.is_win(board) else 0
  print(variant,'az_color',color,'winner',winner,'plies',len(board.history),
        'searches',searches,'overrides',overrides,'moves',board.history)
PY
```

Đánh giá value trước đó trên 40 game promotion, 149 vị trí: dự báo thắng trung bình Đen 0,8799 so outcome 0,2105, Brier 0,6795 (76 vị trí); Trắng 0,3519 so outcome 0,0548, Brier 0,2699 (73 vị trí). Game đã dùng promotion, vị trí phụ thuộc cùng game, historical candidate hash khác latest: đây là mô tả hiện tượng lạc quan trên tập chọn, **không phải calibration holdout**. Chi tiết và lệnh tái lập tại [báo cáo recovery](25k-recovery-report.md).

## Điều kiện trước training tiếp

1. Chứng minh process-crash recovery, cross-file hashes, rollback và test phù hợp môi trường triển khai; nếu cần bảo đảm mất điện thật phải kiểm chứng durability filesystem/Windows và có chiến lược sao lưu riêng.
2. Đóng băng checkpoint, seed, opponent, luật, game IDs, opening family và split theo cả game/seed/đối xứng; tập test phải tách hẳn training, promotion và tuning. Tạo tactical set đủ số thế mỗi loại/màu, oracle nêu horizon và các negative controls.
3. Chạy pilot có phê duyệt với replay có lineage và các recipe đối chứng cùng parent, games, updates, compute, 200 simulations. Báo tactical accuracy, value Brier/MSE theo màu và game, paired W/D/L và uncertainty theo opening cluster. Không ép tỉ lệ màu về 50/50; không chọn cấu hình chỉ vì loss giảm hay khớp hai canary.

Không có thay đổi policy/training production được tích hợp: thí nghiệm chưa chứng minh nâng sức mạnh, và các kiểm tra parity hiện tại không chứng minh tương đương trên mọi trajectory.
