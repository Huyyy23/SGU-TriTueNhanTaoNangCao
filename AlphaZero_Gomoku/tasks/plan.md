# Implementation Plan: Gomoku 11x11 + AlphaZero

## Overview

Dự án xanh, chưa có code. AGENTS.md là spec duy nhất. Kế hoạch này chia việc thành 26 task theo 11 phase, mỗi task là một lát dọc (một đường chạy hoàn chỉnh có thể kiểm chứng), theo build order trong AGENTS.md: game core → CLI → MCTSPure → AlphaZero → arena → **Colab** → **self-play phân tán** → **nhật kí thí nghiệm** → train 11x11 → **đánh giá 1000 ván** → **tài liệu nguồn** → UI → Renju → **báo cáo**.

Bàn chơi **cố định 11x11**, không làm phase validate 7x7/9x9. Mọi chạy đều trên 11x11. Ngân sách train đã chốt: **20 iteration × 50 ván**.

Code phải chạy được giống nhau ở local (RTX 4060) và trên Google Colab. Không phụ thuộc API riêng của Colab trong code chính. Lần train đầu chạy **local**; Colab dùng cho các vòng chạy dài và cho việc chạy song song.

Chi tiết từng task (acceptance criteria + verification) nằm ở `todo.md` cùng thư mục. Task list index ở mục "Task List" bên dưới.

## Environment (đã kiểm tra)

| Mục | Kết quả |
|---|---|
| Python | 3.10.6 |
| GPU | NVIDIA RTX 4060 Laptop, 8 GB VRAM, driver 610.88 (sm_89) |
| CPU / RAM | 16 luồng logic / 15.8 GB |
| Đã có | numpy 2.2.6, PyYAML 6.0.3 |
| Cần cài | torch (cu124), pytest, pygame (UI), pyyaml đã có |
| Mạng | pip OK |
| Git | chưa init repo; cần tạo remote GitHub để Colab clone |
| Colab free | GPU T4 (hoặc L4), **2 vCPU**, ~12.7 GB RAM, session tối đa 12h, disk tạm bị xóa khi session chết; Drive 15 GB |
| Colab Pro | GPU A100/V100, nhiều core hơn, session dài hơn |

**Sự thật về tốc độ:** self-play là phần tốn CPU (MCTS chạy trong Python), không phải phần chạy mạng. Colab free chỉ cho 2 core so với 16 luồng ở máy bạn, và T4 chậm hơn RTX 4060. Nên một instance Colab free **chậm hơn máy bạn**, không phải nhanh hơn. Colab đổi lại: một máy chạy liên tục nhiều giờ không tốn laptop.

**Hình hài Colab free tối ưu (đã chốt):** tài khoản free chỉ chạy được **1 session GPU tại một thời điểm**. Instance CPU-only không tính vào giới hạn đó. Bạn chọn chạy **1 instance CPU-only** cho an toàn:
- 1 instance GPU (notebook T15): lo train, tự sinh ván nếu rảnh
- 1 instance CPU-only: chỉ sinh shard self-play, ghi npz lên Drive (T17)
- Instance GPU gộp shard theo thứ tự `model_iter` rồi train tiếp (T18)

Tổng lưu lượng vẫn kém máy local, nhưng chạy song song và không tốn laptop. Nhiều instance GPU cùng lúc đòi nhiều tài khoản — nằm ngoài phần plan này.

**Cách code lên Colab (đã chốt):** `git clone` từ GitHub. Tên repo bạn bổ sung sau, nên `colab/README.md` (T15) là bản hướng dẫn để bạn điền: tạo repo, push code, tạo Colab Secret chứa token, dán `REPO_URL` vào cell đầu notebook. Repo private thì token lấy từ Colab Secret, không hardcode trong notebook. `scripts/sync_drive.py` (T16) giữ làm đường dự phòng khi không muốn dùng GitHub.

## Architecture Decisions

1. **Bàn 11x11 cố định.** `gomoku/board.py` đặt hằng số `SIZE = 11` làm nguồn sự thật duy nhất; features `(4, 11, 11)`, policy head 121, không tham số hóa kích thước xuyên suốt các module. Lý do: bạn chọn 11x11, code ít hơn, ít chỗ sai hơn. Đánh đổi: mất mốc kiểm chứng pipeline trên bàn nhỏ — bù lại bằng smoke test 2 iteration trước khi chạy full (xem Checkpoint D).
2. **State lưu trong numpy int8 1D phẳng** (121 phần tử cho 11x11), index `(r, c)` → `r * size + c`. `legal_moves` đọc bằng `np.flatnonzero(grid == 0)`; `is_win` dùng so sánh 4 hướng trên mảng đã shift. Không loop Python trên 121 ô (hard rule 6). Không dùng numba: numpy đủ nhanh cho bàn 11x11, numba thêm dependency + compile time.
3. **Rules là interface duy nhất** (`gomoku/rules/base.py`): `legal_moves`, `is_win`, `is_terminal`, `forbidden_moves` (mặc định rỗng). MCTS, players, UI chỉ gọi qua interface này (hard rule 1).
4. **Features cố định 4 plane cho cả Normal lẫn Renju**: `[my_stones, opp_stones, forbidden, color_to_move]`. Plane `forbidden` bằng 0 trong Normal. Lý do: một kiến trúc mạng duy nhất cho cả hai rule set, không phải sửa lại model khi đổi rule set. (Quyết định đã chốt: Renju **train từ đầu**, không fine-tune từ Normal — nên không cần chia sẻ trọng số, nhưng vẫn giữ 4 plane để code và checkpoint đồng nhất.)
5. **Đánh giá vị trí đúng theo AlphaZero** (đã sửa theo góp ý): mỗi lượt mô phỏng đi tới một lá, lá được đánh giá bằng đúng **một** lượt suy luận; lượt suy luận đó trả về xác suất cho cả 121 ô (làm tiên đoán cho mọi nước đi từ lá) và giá trị của lá. Không gom các nước con của một node để suy luận riêng cho từng con: cách đó tốn gấp nhiều lần tính toán mạng và không phải cách làm chuẩn.
5b. **Gom lô ở cấp chương trình, không phải cấp cây.** Lô thật là gom nhiều lá đang chờ đánh giá, đến từ nhiều ván self-play chạy song song, rồi gửi một lệnh suy luận. Đây là cách dùng GPU: mỗi lá vẫn tốn đúng một lượt suy luận, nhưng nhiều lá dùng chung một lệnh.
6. **Torch CUDA cu124** cho sm_89. CPU fallback vẫn chạy được để test nhanh.
7. **Game runner đặt ở `gomoku/runner.py`**, chỉ gọi `player.select_move(board)` — không import players, không import torch. Giữ `gomoku/` sạch torch (hard rule 7).
8. **Replay buffer gắn nhãn rule set**, trainer chỉ lấy mẫu của rule set đang train (hard rule 7).
9. **UI dùng pygame, làm sau khi train xong 11x11** (đã chốt). Web UI để lại.
10. **`git init` + `.gitignore`** loại `checkpoints/`, `logs/`, `__pycache__/`, `*.pt`.
11. **Code không phụ thuộc Colab.** Trainer nhận cờ `--device`, `--workers`, `--time-budget-hours`, `--ckpt-every`, `--seed`, `--out-dir`, `--resume`. Colab chỉ là nơi chạy lệnh đó. Lý do: giữ một đường chạy duy nhất, không có nhánh code riêng cho cloud, và chạy local được ngay.
12. **Buffer replay không ghi ra Drive.** Drive I/O chậm, buffer 50k mẫu sẽ làm nghẽn mỗi iteration. Chỉ checkpoint (~vài chục MB) chạy ra Drive.
13. **Đặt time budget ngắn hơn giới hạn session.** Colab free giết session ở 12h, còn disconnect khi không tương tác. Trainer tự dừng và lưu `last.pt` trước khi bị giết.
14. **Mọi ván để lại log JSONL, một dòng = một ván** (T19). Schema cố định: `end_reason` phân biệt thắng 5 quân / bị cấm nước / hòa / hết giờ. Ghi từng nước là tùy chọn `--log-moves` vì tốn I/O trong self-play. Lý do: mọi câu hỏi phân tích ("thua vì sao", "ai đi trước thắng hay thua") đều trả lời từ log, không từ trí nhớ.
15. **Đánh giá luôn xen kẽ lượt đầu** (T21). Ván chẵn A đi trước, ván lẻ B đi trước. Gomoku cho lợi thế lớn cho người đi trước, nên đánh giá chỉ có nghĩa nếu cân bằng.
16. **Self-play chia shard theo checkpoint.** Instance CPU-only sinh `shard_<model_iter>_<id>.npz`; gộp theo `model_iter` tăng dần để không trộn policy của hai thế hệ. Features lưu uint8 để shard nhỏ (~30 KB mỗi ván).
17. **Lần train đầu chạy local.** Ngân sách chốt 20 iteration × 50 ván. Colab dùng cho vòng chạy dài sau đó.
18. **Ngân sách ván nằm trong cấu hình, không hardcode.** `n_iter × games_per_iter` = 1000 mặc định; nâng lên 1200 (24 × 50) hay 1500 (30 × 50) chỉ cần sửa YAML, và phải nối tiếp được từ checkpoint cũ chứ không chạy lại từ đầu (T20).
19. **Siêu tham số đủ để tái lập thí nghiệm.** AGENTS.md thiếu learning rate, optimizer, weight decay, trọng số mất mát và lịch giảm learning rate; bổ sung vào `DEFAULT_CONFIG` (T2). `n_simulations` quyết định cả độ mạnh lẫn thời gian chạy nên phải cấu hình được, không hardcode.
20. **MCTS thuần dùng cùng ngân sách mô phỏng với agent** khi so sánh (E2): 200 lượt mỗi nước, để phép so sánh công bằng.

## Dependency Graph

```
gomoku/board.py
     │
     ├── gomoku/rules/base.py ──► normal.py ──► renju.py
     │        │                      │            │
     │        │                      └────┬───────┘
     │        │                           │
     │        └──► gomoku/features.py ◄────┘   (+ symmetry 8-fold)
     │                    │
     │                    ├──► az/model.py
     │                    │        │
     │                    ├──► az/mcts.py ◄────┐
     │                    │        │           │
     │                    ├──► az/selfplay.py ┤
     │                    │        │           │
     │                    └──► az/replay_buffer.py
     │                             │           │
     │                             └──► az/trainer.py
     │                                          │
     ├── players/base.py ─► random.py           │
     │                   └► human.py            │
     │                   └► mcts_pure.py ◄──────┤
     │                   └► az_player.py ◄──────┘ (dùng az/mcts.py + az/model.py)
     │
     └── gomoku/runner.py ─► scripts/play.py ─► ui/ (pygame)

scripts/train.py ◄── az/trainer.py + az/selfplay.py + az/evaluator.py
scripts/arena.py ◄── az/evaluator.py
colab/train.ipynb ─► scripts/train.py (+ --device/--workers/--time-budget-hours/--resume) ─► checkpoints/ trên Drive
```

Thứ tự implement đi ngược lên graph: board → rules → features → model → mcts → selfplay/buffer → trainer → evaluator → players/ui.

## Task List

### Phase 0 — Nền tảng
- [ ] T1: Skeleton repo, dependency, pytest, .gitignore
- [ ] T2: Config loader + configs/normal.yaml + configs/renju.yaml

### Checkpoint A — Nền tảng
- [ ] `pytest` chạy được, torch CUDA available, git repo sạch

### Phase 1 — Game core (Normal)
- [ ] T3: `gomoku/board.py` — state, play/undo, history
- [ ] T4: `gomoku/rules/base.py` + `normal.py` — interface + luật Normal
- [ ] T5: `gomoku/features.py` — 4 plane + 8-fold symmetry
- [ ] T6: `players/base.py|random.py|human.py` + `gomoku/runner.py` + `scripts/play.py`

### Checkpoint B — Game core chạy được
- [ ] Toàn bộ test pass, người chơi đánh hết ván với random qua CLI

### Phase 2 — Baseline
- [ ] T7: `players/mcts_pure.py` — rollout baseline

### Checkpoint C — Baseline mạnh
- [ ] MCTSPure thắng random >60% trên 11x11

### Phase 3 — AlphaZero core (11x11)
- [ ] T8: `az/model.py` — ResNet + policy/value head
- [ ] T9: `az/mcts.py` — PUCT, node-batched eval, temperature, noise
- [ ] T10: `az/replay_buffer.py` — lưu mẫu + 8-fold augmentation
- [ ] T11: `az/selfplay.py` — sinh ván + multiprocessing
- [ ] T12: `az/trainer.py` + `scripts/train.py`

### Checkpoint D — Pipeline 11x11 end-to-end
- [ ] Smoke test: 2 iteration train chạy xong trên 11x11, loss giảm (bắt lỗi pipeline trước khi chạy dài)
- [ ] Train 20 iteration trên 11x11, loss giảm, agent thắng random

### Phase 4 — Arena
- [ ] T13: `az/evaluator.py` + `scripts/arena.py` + cổng promote

### Phase 5 — Hạ tầng chạy dài và chạy song song
- [ ] T14: `scripts/train.py` — device, workers, time budget, ckpt định kỳ, seed, resume, log thời gian
- [ ] T15: `colab/train.ipynb` + `configs/colab.yaml` — clone code từ GitHub
- [ ] T16: `scripts/sync_drive.py` — đường dự phòng không dùng GitHub
- [ ] T17: `scripts/selfplay_shard.py` — sinh shard npz, chạy được không cần GPU
- [ ] T18: `scripts/merge_shards.py` — gộp shard + `train.py --buffer-in`

### Checkpoint E — Colab
- [ ] Arena chạy, cổng promote đúng ngưỡng 0.55
- [ ] `git clone $REPO_URL` ra đủ `scripts/train.py`, `configs/colab.yaml`, `az/`
- [ ] Notebook chạy 1 iteration, checkpoint nằm trong Drive
- [ ] Kill session rồi chạy lại → nối tiếp đúng chỗ

### Checkpoint F — Chạy song song
- [ ] Shard gộp giữ đúng thứ tự model, shard hỏng bị bỏ qua
- [ ] `train.py --buffer-in` nạp buffer đã gộp và train tiếp

### Phase 6 — Nhật kí thí nghiệm
- [ ] T19: `az/game_log.py` — schema ván (JSONL) + schema nước đi, dùng chung cho runner/self-play/đánh giá

### Phase 7 — Train 11x11
- [ ] T20: Chạy train 11x11 theo configs/normal.yaml (local trước, sau đó có thể Colab)

### Checkpoint G — 11x11 có học
- [ ] Arena win rate tăng qua các iteration, thắng được random và MCTSPure
- [ ] Log ván của lần train đầy đủ, đọc ra được ai thắng và vì sao

### Phase 8 — Đánh giá 1000 ván + tài liệu nguồn
- [ ] T21: `scripts/simulate.py` — 1000 ván, đổi lượt đầu xen kẽ, `summary.json`
- [ ] T22: `docs/provenance.md` — nguồn từng module + danh mục tài liệu tham khảo + `docs/README.md` hướng dẫn cài đặt và chạy

### Checkpoint H — Đánh giá và truy xuất nguồn
- [ ] 1000 ván xong, đúng 500 ván mỗi bên đi trước, `summary.json` đủ số liệu
- [ ] Mọi module có dòng nguồn trong `docs/provenance.md`

### Phase 9 — UI
- [ ] T23: `ui/pygame_app.py` — human vs AI, AI vs AI

### Checkpoint I — UI chạy
- [ ] Click để đánh, AI đáp lại, ván kết thúc có kết quả

### Phase 10 — Renju
- [ ] T24: `gomoku/rules/renju.py` — 3-3, 4-4, overline, ngoại lệ đúng 5
- [ ] T25: Tích hợp Renju (features, MCTS mask, buffer riêng) + train từ đầu

### Checkpoint J — Hoàn tất
- [ ] Hai rule set đều train được, buffer tách biệt, mọi test pass

### Phase 11 — Báo cáo tổng kết
- [ ] T26: `docs/report.md` — phương pháp, kết quả từ log, phân tích nguyên nhân thua

### Checkpoint K — Báo cáo
- [ ] Báo cáo đủ mục, mọi con số tra được trong log

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Sai dấu value khi backup (child value là góc nhìn đối thủ) | Cao, model học sai | Test riêng cho T9, kiểm tra dấu trong mọi vòng backup |
| MCTS chậm, không train nổi 11x11 | Cao | Gom lô ở cấp chương trình giữa các ván song song để tận dụng GPU; đo số lượt gọi NN mỗi ván trước khi scale; giảm `n_simulations` nếu cần |
| Renju tính forbidden move quá chậm trong MCTS | Trung bình | Tính mask 1 lần mỗi nước rồi cache theo position, invalidate khi `play()` |
| Torch CUDA lỗi driver / wheel | Trung bình | Dùng cu124 cho sm_89; có fallback CPU |
| Windows multiprocessing dùng spawn, nhân bản model | Trung bình | `if __name__ == "__main__"` guard, `torch.set_num_threads(1)` trong worker |
| 11x11 quá khó: agent học chậm, điểm arena ảo | Trung bình | Không có mốc 7x7 để bắt lỗi sớm → smoke test 2 iteration trước Checkpoint D; đo loss + arena win rate từng iteration |
| Train 11x11 lâu hơn dự kiến | Trung bình | `n_simulations` cấu hình được, giảm 200 → 100 nếu 1 vòng > 20 phút; tăng `games_per_iter` thay vì giảm chất lượng tìm kiếm |
| Tăng ngân sách giữa chừng làm mất dữ liệu đã train | Trung bình | `--resume` nối tiếp từ checkpoint, buffer được lưu kèm checkpoint; không chạy lại từ vòng 0 |
| Board 11x11 cho lợi thế quá lớn cho người đen | Thấp | Đọc arena win rate có cân bằng màu (đổi lượt đầu xen kẽ) |
| Buffer quá lớn → dữ liệu cũ chi phối | Thấp | `buffer_size` 50k trước augmentation, theo AGENTS.md |
| Colab giết session giữa chừng | Cao nếu không xử lý | `--time-budget-hours` ngắn hơn giới hạn 12h, `--ckpt-every 1`, `--resume` nối tiếp |
| Colab free chỉ 2 core → self-play chậm hơn local | Trung bình | Chạy overnight trên Colab, giảm `n_simulations` trong `configs/colab.yaml`; nếu cần tốc hơn thì chạy nhiều instance song song |
| Ghi buffer ra Drive làm nghẽn | Trung bình | Buffer chỉ ở RAM, chỉ checkpoint ra Drive |
| Torch trên Colab khác version local | Thấp | Không cài lại torch; `state_dict` load được qua version khác |
| Drive đầy (15 GB free) | Thấp | Chỉ giữ `best.pt` + `last.pt`, xóa checkpoint cũ |
| Token GitHub lộ trong notebook | Cao nếu repo private | Token đọc từ Colab Secret (`google.colab.userdata`), không ghi vào file; repo public thì clone không cần token |
| Instance Colab chạy lệch `model_iter` | Trung bình | Shard mang theo `model_iter`; merge sort theo `model_iter`; dữ liệu cũ hơn vài vòng vẫn dùng được |
| Code trên GitHub là public | Trung bình nếu chưa muốn | Repo private + token; hoặc dùng đường dự phòng `sync_drive.py` |

## Decisions Confirmed

1. Renju **train từ đầu**, không fine-tune từ model Normal (T24).
2. Đánh giá vị trí theo cách làm chuẩn của AlphaZero: **một lượt suy luận cho mỗi lá**, xác suất 121 ô làm tiên đoán cho cả nhánh con (T9). Gom lô ở cấp chương trình giữa nhiều ván song song, không gom lô ở cấp cây.
3. UI pygame **làm sau** mốc 11x11 (T23 sau T20).
4. Bàn **11x11 cố định**, bỏ phase validate 7x7/9x9. Đánh đổi: mất mốc an toàn nhất của pipeline, bù bằng smoke test 2 iteration ở Checkpoint D.
5. **Dùng Colab để train dài**, nhưng code chính không phụ thuộc Colab (T14, T15). Không cam kết Colab nhanh hơn local: nó chạy lâu hơn và không tốn laptop.
6. **Lần train đầu chạy local** (RTX 4060). Colab dùng cho vòng chạy dài sau đó.
7. **Ngân sách 11x11 mặc định 20 iteration × 50 ván = 1000 ván mỗi bộ luật**, nằm trong cấu hình. Tăng lên 1200 hay 1500 ván chỉ cần đổi YAML, không sửa mã.
8. **Code lên Colab qua GitHub** (`git clone $REPO_URL`), token đọc từ Colab Secret khi repo private. `scripts/sync_drive.py` (T16) giữ làm đường dự phòng, không phải đường chính.
9. **Chạy song song theo hình hài 1 GPU + 1 CPU-only.** Instance CPU-only chỉ sinh shard npz; instance GPU gộp rồi train. Không cần nhiều tài khoản Google.
10. **Số instance CPU-only: 1.** An toàn trong giới hạn tài khoản free, đủ để tách phần tốn CPU khỏi instance GPU.
11. **`git init` trong T1.** User tạo repo GitHub và push sau, theo hướng dẫn `colab/README.md` (T15).
12. **Proposal nhắm giảng viên bộ môn AI, không có mẫu.** Cấu trúc dựng trong `tasks/proposal.md` theo 15 mục checklist. Bản nháp trong `tasks/proposal.md`.
13. **Mục tiêu proposal: tạo game chơi được, có agent ra quyết định.** Không phải nghiên cứu khoa học thuần; vì vậy proposal bám theo hướng sản phẩm có số liệu đo được.
14. **Câu hỏi nghiên cứu đã chốt**: "Với ngân sách 1000 ván tự chơi, agent Gomoku 11x11 đạt được trình độ nào, và win rate tăng thế nào theo từng 100 ván?" Hai hướng mở rộng (một model cho hai luật, gom lô ở cấp chương trình) ghi ở proposal mục 3.2 dưới dạng hướng sau khi đạt mục tiêu.
15. **Proposal không nhắc Colab.** Hạ tầng đám mây là chi tiết triển khai, không đưa vào tài liệu trình bày. Mục tài nguyên chỉ nêu máy cá nhân có GPU.
16. **Proposal không gán lịch theo tuần.** Bảng giai đoạn giữ nguyên, mốc thời gian gán sau khi chốt hạn nộp với giảng viên.
17. **Nhóm 3 người, chia theo vai trò A/B/C** (chốt 2026-10-21, cập nhật 2026-10-23): A = lõi game, luật, người chơi, UI; B = tầng AlphaZero; C = hạ tầng, đánh giá, tài liệu, báo cáo. Tên: A = Khang, B = Huy, C = Hường (chốt 2026-10-24). Luật sở hữu file: người sở hữu file dẫn dắt task đụng nhiều file. Bản 2026-10-23: T20 chạy train chuyển từ C sang B, T22 chia cho A và B viết dòng nguồn module mình. Chi tiết ở `todo.md` mục "Phân công A/B/C", bản trình bày ở `proposal.md` §7.2.
18. **Danh mục tài liệu tham khảo đã chốt** (7 mục, proposal mục 9).
19. **Góp ý đã sửa trong proposal**: mô tả đánh giá vị trí theo cách làm chuẩn của AlphaZero (mục 5.3), bổ sung bảng siêu tham số (mục 5.5), E2 dùng cùng số mô phỏng 200 để so sánh công bằng, ngân sách ván nằm trong cấu hình nên mở rộng 1000 → 1200/1500 được.

## Open Questions

Không còn blocker cho T1–T25. Bản nháp proposal nằm ở `tasks/proposal.md`. Còn lại:

1. **Gán tên cho A/B/C**: phạm vi đã chốt, chỉ còn tên người sẽ điền sau.
2. **Hạn nộp thật**: proposal cố ý không gán lịch theo tuần, sẽ điền sau khi biết hạn.
3. **Repo GitHub**: bạn bổ sung sau, làm theo `colab/README.md` (T15).
4. **Đường dẫn Drive for Desktop** — chỉ cần nếu dùng đường dự phòng T16.
