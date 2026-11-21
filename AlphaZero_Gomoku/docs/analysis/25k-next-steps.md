# Bước phục hồi tiếp theo sau mốc 25k

## Quyết định

**Không resume training/self-play từ cặp artifact hiện tại như một cặp đã được xác minh.** `latest.pt` và `buffer_latest.npz` là hiện vật legacy không có bằng chứng hash liên kết; replay đầy còn thiếu cursor. Lỗi đồng bộ đã được sửa cho các cặp lưu mới nhưng không thể tái tạo thông tin đã mất. Vẫn có thể nạp checkpoint để inference và nghiên cứu; việc tạo continuation mới cần một quy trình khôi phục có phê duyệt, bảo toàn bản gốc, công bố dữ liệu/optimizer/replay được sử dụng và tính không chắc chắn của lineage. Không tự tuyên bố replay cũ thuộc checkpoint bằng cách ký hash hậu nghiệm.

## Lỗi đồng bộ checkpoint/replay

**Xác nhận — P1, chưa chứng minh là nguyên nhân thua Heuristic.** Source trước sửa `Trainer.save` lưu checkpoint và replay riêng, `Trainer.load` chấp nhận mọi buffer truyền vào. `tests/test_trainer.py::test_resume_rejects_replay_from_another_checkpoint` lưu hai cặp hợp lệ rồi ghép chéo: trước sửa **FAILED: DID NOT RAISE**. Hai test legacy cũng thất bại trước sửa: checkpoint không có identity và full replay không có cursor vẫn được nhận để resume.

Sửa tại `az/trainer.py`, `az/replay_buffer.py`: lưu replay trước, đưa SHA-256 byte NPZ vào metadata checkpoint; loader tính SHA-256 của file replay và `Trainer.load` từ chối hash thiếu/sai, hoặc buffer đầy legacy không biết cursor. Inference không cần replay vẫn dùng `load_checkpoint` bình thường. `SPEC.md` cập nhật contract. Hash khớp chứng minh byte hiện tại khớp hash lúc lưu mới, **không chứng minh** dữ liệu có nhãn đúng hoặc checkpoint lịch sử thực sự đi cùng buffer. `ReplayBuffer.load_npz` vẫn cho phép đọc legacy để nghiên cứu, kèm cảnh báo; `Trainer.load` không cho resume không có identity/cursor.

**Lỗi ghi replay được tái hiện:** `ReplayBuffer.save_npz` trước sửa ghi trực tiếp lên file cũ; test `test_failed_replay_save_preserves_previous_file` giả lập lỗi giữa lúc ghi và thấy file cũ bị thay bằng `incomplete`. Bản sửa ghi file tạm cùng thư mục rồi `os.replace` khi ghi xong; test kiểm tra giữ nguyên file cũ khi lỗi ghi. Đây là atomic replace từng file, không chứng minh dữ liệu đã durable qua mất điện.

**Giới hạn quan trọng:** hai file chưa phải transaction. Crash sau khi replace NPZ mà trước khi checkpoint mới được thay thế có thể làm cặp hiện tại không dùng được; hash giúp từ chối cặp lệch, không cung cấp rollback hay crash durability. Không gọi sửa này là transactional recovery. Một thiết kế immutable generation-specific replay + checkpoint là commit point, giữ generation trước và test crash/flush là bước tiếp theo trước vận hành dài; không được tự dọn các generation cũ.

Kiểm thử sau sửa: `PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/test_trainer.py tests/test_replay_buffer.py tests/test_native_mcts.py tests/test_tactical_oracle.py tests/test_train_resume_logging.py` → **37 passed in 6.31s**. Bao gồm regression mismatch/legacy, lỗi ghi replay và parity native hiện có; không chứng minh recovery qua mất điện hay parity toàn bộ trạng thái.

Hiện vật đã kiểm toán trước: checkpoint SHA `f7f521c1cc37a5054ed854201b087e484e9fcef3d4d6b1877ab81dff2e51d7e4`, iteration 500, global step 484000; NPZ SHA `908a22b249591fdbf64fb1f7430de879d0b94e2925be4e619a5db484e0170fb9`, 50.000/50.000 mẫu, không cursor hoặc generation. Hash là định danh snapshot, không phải chứng nhận hai file thuộc cùng lần save. Không sửa hay xóa các file đó.

## Ablation có kiểm soát trên hai tactical canary

Hai trạng thái và oracle là các ván thua đã biết trong [báo cáo trước](25k-recovery-report.md), không phải holdout. Đã chạy 16 search: 2 vị thế × 4 intervention × Python/C++, cùng `latest.pt` SHA `f7f521…`, Normal, seed 0, CPU một thread, noise tắt, 200 simulations/search. **Uniform root** thay policy lần nhận root đầu tiên bằng đều rồi mask legal; NN policy ở các lá sau giữ nguyên. **Zero value** thay toàn bộ value do evaluator cung cấp ở lá không-terminal bằng 0; terminal value vẫn do MCTS xử lý. Không sửa weights hay thuật toán production.

| Canary | Uniform root | Zero value | Nước chọn | Visits các nước phòng thủ |
|---|---|---|---:|---|
| 0: phòng thủ {5,17,65} | Không | Không | 52 | 1, 1, 1 |
| 0 | Có | Không | 47 | 1, 1, 1 |
| 0 | Không | Có | 47 | 1, 1, 2 |
| 0 | Có | Có | 118 | 2, 2, 2 |
| 1: phòng thủ {29,69} | Không | Không | 96 | 0, 0 |
| 1 | Có | Không | **29** | 175, 0 |
| 1 | Không | Có | 96 | 1, 3 |
| 1 | Có | Có | 118 | 2, 2 |

Mỗi dòng Python/C++ trùng nước chọn, root value và các thống kê phòng thủ được liệt kê; mỗi search có 199 child visits. Đây là parity trên những phép thử này, không phải toàn bộ cây hoặc phân bố.

**Diễn giải:** thay root prior làm canary 1 chọn nước 29 thay vì 96; thay value bằng 0 đơn lẻ không làm canary nào chọn defense. Với canary 0, value NN ở các trạng thái sau nước 5/17/65 theo góc nhìn đối thủ lần lượt khoảng +0,999955/+0,999726/+0,999768, tương ứng Q từ góc nhìn root khoảng −1 sau một visit. Nhưng defense nông không chắc cứu được toàn ván, nên không thể kết luận các value này sai tuyệt đối. Với canary 1, value sau 29/69 theo góc nhìn đối thủ khoảng −0,999617/−0,999858, song 0 visits ở baseline; uniform prior dẫn đến 175 visits cho 29. Chỉ ra tương tác prior/value/search tại hai vị trí, **không chứng minh nguyên nhân huấn luyện hoặc một can thiệp sẽ cải thiện Elo**. Root prior được chuẩn hóa theo legal mask; Q của action chưa thăm bằng 0 không phải value đã được đánh giá.

Tái lập (checkpoint local tin cậy; 16 search có thể mất vài phút):

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe - <<'PY'
import hashlib, numpy as np, torch
from az.mcts import MCTS, NativeMCTS
from az.model import load_checkpoint
from gomoku.board import Board
from gomoku.features import board_to_planes
from gomoku.rules.normal import NormalRules
torch.set_num_threads(1)
path='checkpoints/latest.pt'
print('checkpoint_sha256',hashlib.sha256(open(path,'rb').read()).hexdigest())
model,_=load_checkpoint(path,device='cpu')
cases=[([20,72,31,58,61,60,9,48,42,53,84,38,28,49,71,50,51,41,81,91,83,82,30,29],[5,17,65]),([20,72,31,58,42,53,9,18,60,70,49,27,71,82,46,94,106,50,38,61,83,48,59,95,39],[29,69])]
for ci,(moves,defs) in enumerate(cases):
 rules=NormalRules(); board=Board()
 for a in moves: board.play(*divmod(a,11))
 _,v=model.predict(board_to_planes(board,rules,board.current_player)[None]); print('case',ci,'root_nn_value',float(v[0]))
 for a in defs:
  board.play(*divmod(a,11))
  _,v=model.predict(board_to_planes(board,rules,board.current_player)[None])
  board.undo(); print('case',ci,'defense',a,'child_value',float(v[0]))
 for uniform,zero in ((False,False),(True,False),(False,True),(True,True)):
  for cls in (MCTS,NativeMCTS):
   search=cls(rules,n_simulations=200,root_noise=False,rng=np.random.default_rng(0)); search.reset(board); calls=0
   while (req:=search.request()) is not None:
    p,v=model.predict(req.planes[None]); policy=np.ones_like(p[0]) if uniform and calls==0 else p[0]
    search.receive(policy,0.0 if zero else float(v[0])); calls+=1
   visits,priors,q,legal=search.root_statistics()
   print(ci,uniform,zero,cls.__name__,search.select_move(0),search.root_value(),calls,
         int(visits.sum()),[(a,int(visits[a]),float(priors[a]),float(q[a])) for a in defs])
PY
```

## Thiết kế đánh giá độc lập

Không dùng các ván promotion hay fixed eval batch hiện tại làm confirmatory holdout: chúng đã tham gia chọn model, replay thiếu game ID và historical lineage không đầy đủ. Hai canary thua Heuristic đã biết chỉ thuộc development.

- Trước khi thấy kết quả, đóng băng checkpoint/model SHA, rules, code/config/opponent/backend, manifest game ID, seed, opening, tiêu chí loại trừ và seed bootstrap; chỉ tạo artifacts/SQLite mới, không ghi đè lịch sử.
- Phân nhóm cả ván, hai màu cùng opening và các biến thể đối xứng vào cùng partition; tách hẳn development và holdout theo family opening và seed. Kiểm tra va chạm với trajectory có sẵn; không nói đã loại hết leakage lịch sử khi provenance không đủ.
- Tách hai strata bàn trống và opening bốn ply: chúng đo hai protocol khác nhau. Cố định Heuristic làm đối thủ chính và một đối thủ độc lập thứ hai làm kiểm tra phụ; match checkpoint, opening, seed và hai màu. Mỗi search AZ dùng 200 simulations, no root noise, temperature 0; override ghi riêng nếu có.
- Tactical set mới: win-in-1, block-in-1, tạo/ngăn fork nông, forced-loss control và không threat, có cả hai màu và oracle độc lập; đặt nhãn theo game và nhóm đối xứng. Báo accuracy theo nhóm/màu, prior/visit mass trên tập đáp án hợp lệ và số vị trí. Fork nông không chứng minh cứu ván toàn cục.
- Theo game, báo W/D/L, score, illegal/abort và màu; value MSE/Brier theo game với trọng số game bằng nhau, tách màu và ply. Outcome calibration không đồng nghĩa giá trị optimal-play. Bootstrap theo opening-family clusters, giữ các ván đổi màu, mọi vị trí trong ván và các checkpoint đi cùng nhau. Thiếu cặp hoặc số clusters quá ít thì inconclusive.
- Pilot nhỏ đã đăng ký trước chỉ để đo tính khả thi và thời gian, ví dụ 10 cặp opening mỗi stratum và 20 thế mỗi loại; không dùng kết quả pilot làm bằng chứng promotion. Tính cỡ mẫu xác nhận từ pilot/power trước khi xin phép tournament lớn. `scripts/tournament.py` có paired games nhưng seed phụ thuộc agent label; phải khóa lịch chung. `az/elo.py` hiện bootstrap theo opening pair, chưa triển khai bootstrap theo family nêu trên.

## Thí nghiệm self-play/training đề xuất

1. **Kiểm định dữ liệu trước:** so entropy policy, mass trên tactical defenses, outcome theo màu/ply và tần suất bỏ chặn trên game giữ riêng theo seed, trước/sau từng checkpoint. Gắn hash model thực tế theo worker, hash replay và lineage game; kiểm tra distribution thay vì coi số ván là chất lượng.
2. **Thử một can thiệp chiến thuật có control, không tự thay thuật toán:** trên nhánh thí nghiệm được duyệt, so target/search có chứng cứ oracle với control cùng parent weights, games, optimizer updates, opening seeds, luật, 200 simulations và tài nguyên. Đo khả năng tổng quát hóa trên tactical/game holdout độc lập; không dùng hai canary development làm nghiệm thu. Tách thay đổi tầng search với chất lượng network.
3. **Kiểm tra LR và replay bằng thí nghiệm nhỏ trước khi tiếp tục:** đăng ký trước dữ liệu, LR horizon, replay provenance, steps và checkpoint SHA; so cùng số updates/compute, kiểm tra calibration và paired score, không lấy train loss làm promotion. Không thay hyperparameters trong sản phẩm theo trực giác.

Các thí nghiệm và protocol mới chỉ là thiết kế, chưa được chạy hoặc chứng minh tăng strength. Mọi thay đổi thuật toán/recipe và benchmark lớn cần phê duyệt theo SPEC.
