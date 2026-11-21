# SPEC: Gomoku AlphaZero 11×11

> **Trạng thái tài liệu:** baseline kỹ thuật và đặc tả mục tiêu nghiên cứu, tổng hợp từ source, tests, configs và báo cáo hiện có. Không thay thế bằng chứng runtime của từng thay đổi mới.
>
> **Nhãn bằng chứng:** **Verified** = hành vi thấy rõ trong source và được test/report có liên quan xác nhận; **Implemented / Unverified** = có implementation nhưng mức kiểm chứng chưa đủ cho kết luận; **Planned** = mục tiêu/thiết kế chưa hiện diện; **Unknown** = chưa đủ dữ liệu để xác định. Các test status dưới đây là kết quả đã ghi trong tài liệu/reports của repo, không phải test chạy trong lần soạn SPEC này.

## Mục lục

1. [Mục tiêu và câu hỏi nghiên cứu](#1-mục-tiêu-và-câu-hỏi-nghiên-cứu)
2. [Phạm vi, yêu cầu và tiêu chí thành công](#2-phạm-vi-yêu-cầu-và-tiêu-chí-thành-công)
3. [Tech stack, cấu trúc và code style](#3-tech-stack-cấu-trúc-và-code-style)
4. [Kiến trúc và execution paths hiện tại](#4-kiến-trúc-và-execution-paths-hiện-tại)
5. [Hợp đồng game engine và luật](#5-hợp-đồng-game-engine-và-luật)
6. [Policy-value network](#6-policy-value-network)
7. [MCTS / PUCT](#7-mcts--puct)
8. [Self-play, replay và training](#8-self-play-replay-và-training)
9. [Python/C++ dual backend](#9-pythonc-dual-backend)
10. [Tournament, Elo và đánh giá sức chơi](#10-tournament-elo-và-đánh-giá-sức-chơi)
11. [Performance và benchmark](#11-performance-và-benchmark)
12. [Reproducibility và persistence](#12-reproducibility-và-persistence)
13. [CLI, UI và commands](#13-cli-ui-và-commands)
14. [Testing strategy và traceability](#14-testing-strategy-và-traceability)
15. [Code style](#15-code-style)
16. [Boundaries](#16-boundaries)
17. [Current state, gaps và roadmap](#17-current-state-gaps-và-roadmap)
18. [Open questions và giới hạn bằng chứng](#18-open-questions-và-giới-hạn-bằng-chứng)

## 1. Mục tiêu và câu hỏi nghiên cứu

### 1.1 Đối tượng và problem

Hệ thống phục vụ AI coding agents, người bảo trì/nghiên cứu và giảng viên. Bài toán là xây dựng agent Gomoku trên bàn 11×11, học policy và value bằng self-play kiểu AlphaZero, rồi đánh giá sức chơi bằng đối kháng có kiểm soát. Đây vừa là phần mềm chơi game, vừa là nền tảng thí nghiệm; một pipeline chạy xong hoặc loss giảm không đồng nghĩa chơi tốt hơn.

### 1.2 Research questions

- **RQ1:** Với cùng luật, checkpoint, ngân sách tìm kiếm, opening và phân bố màu, checkpoint AlphaZero có score/rating cao hơn Heuristic hay không?
- **RQ2:** Các mốc số ván self-play và optimizer steps ảnh hưởng thế nào đến playing strength, không chỉ loss?
- **RQ3:** Native C++ có tăng throughput trên workload đại diện mà vẫn giữ semantics, dữ liệu huấn luyện và playing strength trong sai số định trước không?
- **RQ4:** Normal và Renju tạo ra khác biệt gì khi so sánh trong rating pool và protocol riêng?

### 1.3 Mục tiêu và giới hạn

**Research objective, chưa đạt:** AlphaZero thắng Heuristic với bằng chứng đủ tin cậy trong một rating pool/protocol chung. **Scale objective:** hỗ trợ nghiên cứu 5.000–10.000+ ván. **Optimization objective:** tăng tốc không đánh đổi correctness hoặc data/quality. Số ván và speedup chỉ là operational metrics, không phải chỉ số chất lượng mô hình.

Phạm vi hiện tại là board 11×11, Normal và Renju như code mô tả; không tuyên bố tương đương đầy đủ luật Renju quốc tế. Không có chứng cứ rằng mọi mục tiêu hiệu năng, GPU soak dài hạn, parity thống kê hoặc Elo đã đạt.

## 2. Phạm vi, yêu cầu và tiêu chí thành công

Trạng thái của yêu cầu được đánh dấu theo phần đầu tài liệu. Ưu tiên: P0 correctness/data integrity; P1 reproducible evaluation và backend parity; P2 scale/performance.

| ID | Yêu cầu / nghiệm thu | Hiện trạng |
|---|---|---|
| RQ-GAME-01 | Board 11×11, lượt luân phiên, nước hợp lệ và terminal state thống nhất qua `Rules` API | Verified: `gomoku/board.py`, `gomoku/rules/`; unit tests tồn tại |
| RQ-RULE-02 | Normal: ≥5 liên tiếp theo 4 hướng thắng; bàn đầy không thắng là hòa | Verified trong implementation/tests |
| RQ-RULE-03 | Renju: Black exact-five thắng và được ưu tiên hơn forbidden checks; overline, double-four, double-three bị cấm; White ≥5 thắng | Implemented; tests Renju có, nhưng chuẩn luật quốc tế/edge coverage chưa được chứng minh đầy đủ |
| RQ-AZ-01 | Policy/value input/output shapes, value perspective và checkpoint contract đúng | Verified qua source; tests `test_model.py` |
| RQ-MCTS-01 | Terminal không gọi network; legal masking không expand nước cấm; value backup đổi dấu mỗi ply; win-in-1 test | Verified qua source và `test_mcts.py` |
| RQ-SELFPLAY-01 | Mỗi trạng thái trước nước đi tạo đúng một mẫu; `z` là kết quả cuối ván theo player-to-move; buffer tách luật | Verified qua source và `test_selfplay.py` / `test_replay_buffer.py` |
| RQ-TRAIN-01 | Resume khôi phục model, optimizer, iteration/global step, RNG và replay; duy trì latest/best | Implemented / Unverified: Normal dùng generation chính thức `checkpoints/normal/latest.json`; quality gate độc lập với việc commit latest |
| RQ-NATIVE-01 | C++ MCTS giữ hành vi Python theo các differential criteria và không fallback âm thầm | Differential tests có; trạng thái experimental, cần mở rộng trajectory/quality gates |
| RQ-EVAL-01 | Tournament cân bằng màu, ghi kết quả/provenance; promotion dùng Elo có uncertainty | Implemented: pairwise Davidson + paired opening-cluster bootstrap cho Normal az_v0; full leaderboard Elo vẫn Planned |
| RQ-QUALITY-01 | Kết luận AZ > Heuristic chỉ khi protocol matched, log đầy đủ và uncertainty loại trừ tie theo tiêu chí đã định trước | Chưa đạt: báo cáo T35 ghi Heuristic thắng 100/100 AZ v0, threat và AZ Renju trong các pool được khảo sát |
| RQ-SCALE-01 | Self-play liên tục có monitoring, quality checkpoints và điều kiện dừng | Pipeline gate có; chưa có bằng chứng strength sau migration hoặc soak safety |
| RQ-REPRO-01 | Có thể dựng lại run từ config/seed/code/checkpoint identity/environment/log; replayable tournament records | Tournament khá đầy đủ; training provenance còn thiếu model init seed/code hash và lịch sử checkpoint nguồn |

**Success criteria của nghiên cứu:** (1) luật và training target correctness qua tests; (2) tournament matched-condition với checkpoint SHA-256; (3) theo Elo v1.0, đạt mục tiêu khi `ΔElo > 0` và lower 95% CI `>0`, mục tiêu mạnh khi `ΔElo ≥ +100` và lower 95% CI `>0`; (4) kết quả lặp lại trên nhiều seed/opening và đổi màu; (5) claims C++ quality/parity tách khỏi throughput. Các ngưỡng và cỡ mẫu 100/400/600/800 là đề xuất protocol, chưa được xác nhận bằng pilot/power analysis; phải chốt/version trước nghiên cứu chính thức, không sửa sau khi thấy kết quả.

## 3. Tech stack, cấu trúc và code style

### 3.1 Stack hiện có

- Python với NumPy, PyTorch, PyYAML; pytest cho tests; pygame cho UI.
- Optional native extensions C++17 qua pybind11, compile bằng compiler được `scripts/build_gomoku_native.py` gọi. Trên Windows còn phụ thuộc Python ABI, linker/compiler và thư viện Python tương thích.
- `../requirements.txt` là requirements dùng chung nhiều lab, không pin phiên bản và ghi Python 3.12; các report Gomoku ghi môi trường thí nghiệm cụ thể Python 3.10.6 / torch 2.14.1+cu126. Đây là khác biệt môi trường cần xử lý khi tái lập, không nên suy ra môi trường chuẩn duy nhất.

### 3.2 Cấu trúc thực tế (file tiêu biểu)

```text
gomoku/       Board, rules, game/runner, features, patterns
az/           Model, MCTS, self-play, replay, trainer, evaluator/logging
players/      Random, human, heuristic, alpha-beta, pure MCTS, AZ wrappers
native/       C++ patterns và C++ MCTS tree
scripts/      train, play, arena, tournament, benchmark, native build
configs/      Normal, Renju, 5k training YAML
tests/        Unit, integration, CLI, backend differential tests
ui/           Pygame interface
docs/         Reports, performance studies, tournament/repro guides
logs/, data/*.sqlite*, checkpoints/  Runtime artifacts; ignored by Git, local-only
```

Một số README mô tả cấu trúc dự kiến; SPEC ưu tiên file/code thực có trong working tree. Entry points đã xác nhận: `scripts/play.py`, `scripts/train.py`, `scripts/arena.py`, `scripts/tournament.py`, `python -m ui.pygame_app`; native build: `scripts/build_gomoku_native.py`.

## 4. Kiến trúc và execution paths hiện tại

```mermaid
flowchart TD
    CLI[CLI / Pygame UI] --> Runner[gomoku.runner / Game]
    Runner --> Board[Board state]
    Runner --> Rules[Rules interface]
    Runner --> Players[Player factory]
    Players --> Classical[Random / Heuristic / AlphaBeta / MCTS Pure]
    Players --> AZPlayer[AZ player and tactical wrappers]
    AZPlayer --> MCTS[MCTS: Python or native tree]
    MCTS --> Rules
    MCTS --> Features[board_to_planes + legal mask]
    Features --> Net[AZNet policy/value]
    Selfplay[az.selfplay] --> MCTS
    Selfplay --> Buffer[ReplayBuffer]
    Buffer --> Trainer[Trainer]
    Trainer --> Net
    Trainer --> Checkpoint[latest.pt + best.pt + buffer_latest.npz]
    Trainer --> SelfplayDB[data/selfplay.sqlite: runs, games, moves, MCTS roots, training metrics, events]
    Tournament[scripts/tournament.py] --> Runner
    Tournament --> TournamentDB[data/tournament.sqlite: experiments, agents, matches, games, moves, Elo, events]
    TournamentDB --> Analytics[resume, Elo replay, SQL research queries]
    NativePatterns[native/gomoku_native.cpp] -. optional patterns .-> Patterns[gomoku.patterns]
```

### 4.1 Responsibilities and data flow

1. `Board` owns flat `int8` 121-cell grid, current player, move history; `play/undo/copy` không phụ thuộc luật.
2. `Rules` cung cấp `legal_moves`, `forbidden_moves`, `is_win`, `is_terminal`. Players và UI dùng interface này.
3. `board_to_planes` tạo `(4,11,11)` `uint8`, model trả `(B,121)` policy probabilities và `(B,)` value. MCTS nhận policy/value leaf request, mask policy priors bằng legal moves rồi mở child nodes.
4. AZ player thực hiện MCTS, lấy move theo visit distribution/temperature. Self-play giữ `(planes, pi, player)` trước nước thật và gán `z` sau terminal.
5. ReplayBuffer lưu dữ liệu compact, augmentation theo đối xứng lúc sample; Trainer tối ưu model, checkpoint và buffer; `scripts/train.py` ghi run metadata, self-play games/moves, root telemetry và training metrics vào `data/selfplay.sqlite`; checkpoint/replay vẫn giữ định dạng và đường dẫn riêng.
6. Tournament dùng player factory, chạy paired color-swapped games, ghi experiment/game/move records vào `data/tournament.sqlite`; resume dùng DB với `--resume --run-id`, analytics và Elo đọc kết quả từ cùng database, không sinh JSONL/CSV/JSON mặc định.

**Dependency boundaries:** `gomoku/` không phụ thuộc `torch`; rules không nằm trong players/MCTS; board không ra quyết định luật. Native MCTS chỉ sở hữu tree/search mechanics; Python giữ Board, Rules, RNG, feature generation và model inference. Native Patterns tăng tốc `find_patterns`, được tiêu thụ bởi heuristic/tactical pattern consumers; AZ v0 self-play không gọi patterns.
Normal inference and tournament alias `az` now resolves to `az_v1_threat` using the official AZ Normal checkpoint; explicit `az_v0` remains available. This is search-layer routing only: it does not change checkpoint weights or the training self-play path, which remains AZ v0 unless separately integrated and evaluated.

## 5. Hợp đồng game engine và luật

### 5.1 Board contract

`SIZE=11`, `EMPTY=0`, `BLACK=1`, `WHITE=2`; flat action `a=11r+c`; initial side Black. `play(r,c)` từ chối tọa độ ngoài bàn/ô chiếm, ghi history rồi đổi player; không tự kiểm tra legality theo rules. `undo()` khôi phục ô cuối và current player. Caller phải kiểm tra luật trước `play`.

### 5.2 Luật được xác minh

| Điều kiện | Normal | Renju |
|---|---|---|
| Black line | ≥5 thắng | đúng 5 thắng; overline không thắng và nước tạo overline bị cấm |
| White line | ≥5 thắng | ≥5 thắng |
| Black forbidden | không có | double-three, double-four, overline; exact-five exception |
| Draw | không còn ô trống và không thắng | không còn legal move và không thắng |
| Forbidden mask | rỗng | chỉ xét lượt Black; White trả mask rỗng |

Renju code xét exact-five trước các điều kiện forbidden khác. Candidate scan chỉ xét ô trong phạm vi 4 bước theo bốn hướng từ Black stone; open-three dựa trên extension tạo open-four và đệ quy kiểm tra hợp lệ. Cache dùng board bytes/current player. Đây là semantics code; không chứng minh toàn bộ quy tắc quốc tế về precedence, định nghĩa pattern phức tạp, foul/game adjudication hoặc tình huống bàn đã có trạng thái thắng.

### 5.3 Acceptance / edge cases

- `AC-RULE-01`: thắng horizontal, vertical, hai diagonal, cắt biên, exact 5 và overline theo ruleset.
- `AC-RULE-02`: full board không line thắng => terminal draw; Renju không còn legal move do forbidden cells => terminal.
- `AC-RULE-03`: occupied/out-of-range move bị từ chối; undo rỗng bị từ chối; play/undo round-trip giữ grid/player/history.
- `AC-RULE-04`: Renju Black exact five vẫn legal kể cả đồng thời hình thành forbidden threats; Black overline/3-3/4-4 fixture bị loại; White không bị mask.
- `AC-RULE-05`: mỗi consumer chỉ gọi Rules API; không tạo luật thứ hai ngoài Rules.

Test locations: `tests/test_board.py`, `test_rules_normal.py`, `test_rules_renju.py`, `test_renju_integration.py`, `test_runner.py`. Còn cần differential/reference fixtures hoặc expert review cho biên Renju mà hiện tests không bao phủ.

## 6. Policy-value network

### 6.1 Input/output và kiến trúc

State input `(B,4,11,11)`; channels: own stones, opponent stones, forbidden squares, player color (all ones for Black, zeros for White). Không có last-move/history plane. Cùng bốn channels cho cả hai rulesets. Augmentation dùng 8 dihedral transforms với cùng transform cho feature và policy.

`AZNet(n_blocks=6,n_filters=64)`: stem conv 3×3, BN, ReLU; mỗi residual block hai conv 3×3 + BN + skip + ReLU. Policy head conv 1×1 (2 filters), BN/ReLU, flatten, FC→121 logits. Value head conv 1×1, BN/ReLU, flatten, FC→64→ReLU→1→tanh. `predict` nhận numpy batch, eval/no_grad, trả softmax probabilities float32 và values float32 thuộc `[-1,1]`, giá trị theo current player.

### 6.2 Training objective và defaults

Với logits `l`, policy target `π`, prediction value `v`, terminal target `z`:

$$L_\pi=-\frac1B\sum_i\sum_a\pi_{ia}\log\operatorname{softmax}(l_i)_a,\qquad L_v=\frac1B\sum_i(v_i-z_i)^2,$$
$$L=w_\pi L_\pi+w_vL_v.$$

Code dùng Adam, `learning_rate=1e-3`, `weight_decay=1e-4`, cosine schedule theo global step; `weight_decay` đi qua optimizer, không cộng khoản L2 explicit vào `_total_loss`. `DEFAULT_CONFIG` giữ giá trị khởi đầu 20 iterations × 50 games, 200 train steps/iteration; `configs/normal.yaml` đặt mô hình liên tục `az_v0` với 200 × 50 games và 1000 steps/iteration. Kiến trúc 6 blocks × 64 filters, `w_policy=w_value=1`, `batch_size=128`, `buffer_size=50000`, `eval_fixed_batch_size=512`.

Checkpoint training `.pt` gồm `model_state_dict`, optional `optimizer_state_dict`, metadata `format_version=1`, architecture, `rule_set/config`, `n_iter`, `global_step`, `promotion_eval_games` và Trainer RNG state. Inference cũng nhận `.pt` chỉ chứa `state_dict`; loader dùng kiến trúc mặc định AZNet (6 blocks × 64 filters), xem weights-only là Normal và không coi chúng là checkpoint resume. Replay buffer cùng frozen eval batch lưu ở NPZ bên cạnh checkpoint đầy đủ. `load_checkpoint` map_location CPU/CUDA theo khả dụng. Backward compatibility có defaults cho metadata thiếu; không có schema migration versioning rộng.

**Rủi ro metric:** fixed eval batch là batch lấy từ cùng self-play stream, có thể có vị trí cùng ván/training overlap; sampler lấy có hoàn lại; buffer vòng có thể overwrite slot. Đây không phải independent generalization set hay strength metric.

## 7. MCTS / PUCT

### 7.1 Công thức và contract

Node lưu prior `P`, visit count `N`, value sum `W`; `Q=W/N` theo góc nhìn người chơi tại node. Với child `a`:

$$a^*=\arg\max_a\left[-Q(s_a)+c_{puct}P(s,a)\frac{\sqrt{\max(1,N(s))}}{1+N(s_a)}\right].$$

Mặc định `c_puct=1.5`. Ties trong tolerance `1e-9` chọn ngẫu nhiên qua RNG. Root noise chỉ khi bật:

$$P'(s,a)=(1-\epsilon)P(s,a)+\epsilon\eta_a,\quad\eta\sim Dirichlet(\alpha),$$

trên legal moves; defaults training `alpha=0.15`, `epsilon=0.25`. Priors bị mask bằng legal mask rồi normalize; nếu tổng hợp lệ ≤`1e-8`, dùng uniform legal priors. Terminal win ở state sau nước đi nhận `-1` cho player-to-move (đã thua); draw `0`; node terminal không gọi model. Leaf NN value backup đổi dấu mỗi cạnh. Forbidden moves bị xóa khỏi legal mask và không được mở rộng.

**Ngân sách cố định:** Hợp đồng AZ MCTS cố định `n_simulations=200` cho self-play/training, diagnostics, ablation và evaluation; không điều chỉnh, sweep hoặc ablate số simulations. Mỗi lần MCTS thực sự được gọi dùng ngân sách 200. Tactical/threat override có thể chọn nước trước khi gọi MCTS; ghi rõ override và `simulations_run=0`, không coi đó là một budget khác. Budget rollout của MCTS Pure là cấu hình riêng.

### 7.2 Pseudocode tương ứng

```text
reset(root_state)
while simulations_done < 200:
    node = root; path = []
    while node.expanded and not node.terminal:
        child = argmax(-child.Q + c_puct * child.P * sqrt(max(1,node.N))/(1+child.N))
        play(child.move); path.append((node, child)); node = child
    if terminal(node.state):
        backup(result_from_side_to_move(node))  # alternate sign per edge
    else:
        planes = encode(state); legal = rules.legal_moves(state) - forbidden
        policy, value = evaluator(planes)
        prior = normalize(policy masked to legal; uniform legal if zero mass)
        if node is root and root_noise: mix Dirichlet over legal priors
        expand(node, legal, prior); backup(value, alternating_sign)
    undo(path)
return root child visits
```

Implementation uses request/receive API (`az/mcts.py`); self-play interleaves requests from multiple games into inference batch. Root tree resets per actual move; no subtree reuse in `run_games`. Temperature is separate from target temperature: action probability `N^(1/τ)`; `τ=0` chooses max visit (random tie). Self-play uses action temperature 1 for first `temp_moves=15`, thereafter 0; `pi_target_temp=1` independently shapes normalized visit target. Exact default target/action mechanics in `az/selfplay.py`.

### 7.3 Invariants / corner cases / tests

- Pending request must be received before another request; reset before search.
- Every accepted policy/value has valid shape/domain at consumer contract; value finite in `[-1,1]`.
- No legal move => draw/terminal; zero visit counts cannot choose action.
- Terminal branches count a simulation but no NN call; first simulation can expand root without incrementing any child visits, so root child visit sum need not equal budget.
- Root-only noise; no noise for tournament by default.
- For Renju, forbidden move masking must agree with same Rules implementation at every leaf.
- Tests: win-in-1, backup sign, terminal call counts, forbidden mask, root noise scope, temperature zero, exhausted budget in `tests/test_mcts.py`; parity/terminal/RNG/noise coverage in `tests/test_native_mcts.py`.

**Research correctness caveat:** unit fixtures prove these cases, not all board states. Add fuzz/differential states and property invariants before claiming universal parity.

## 8. Self-play, replay và training

### 8.1 Contracts / pipeline

**Implemented / Unverified — canonical AZ migration:** Normal has one official training/inference pointer, `checkpoints/normal/latest.json`, initialized from the audited 60k generation without changing its weights. Generations, recovery state, rollback history and historical evaluation baselines remain distinct artifacts, not additional official models. Training defaults to loading the validated official generation and its replay, optimizer, RNG and counters; missing or invalid state fails closed. A new model requires explicit fresh mode. Successful training commits the official pointer atomically; quality evaluation remains separate and cannot be inferred from this commit. Record parent checkpoint SHA-256, generation, run ID, and distinguish generated games, games actually sampled for training and lineage games; unknown historical counts remain unknown, never include the unrelated `selfplay_20k_cpp_2w` run. Migration preserves historical checkpoint paths used by tournament evidence and does not alter audited 60k weights.


1. `scripts/train.py` loads YAML merged over `DEFAULT_CONFIG`; initializes or resumes Trainer and ReplayBuffer.
2. `run_games` creates game-local Board/MCTS/sample list and interleaves up to `parallel_games`; optionally worker process chunks are generated by `run_games_parallel`, returned samples are stored by parent. `parallel_games` is active game concurrency for batching; `--workers` is process parallelism.
3. With `selfplay_opening_plies=0`, start from an empty board. With `2`, first play two seeded random legal moves as an opening; they are recorded in game provenance, not used as training targets. Two stones cannot form a terminal or immediate five-in-a-row threat under Normal rules. For each subsequent MCTS move, store features, visit-based `pi`, and player identity. End game: `z=+1/-1/0` from the stored player's final outcome; one sample per MCTS move and no terminal-state sample. `moves_count` counts MCTS moves; full game length includes the recorded opening.
4. ReplayBuffer fixed-capacity ring: features `uint8`, pi `float16`, z `float32`, rule ID; overwrite oldest when full. Enforces one rule_set per buffer. Each sampled item gets one of 8 rotations/reflections applied to planes and policy together; `z` unchanged.
5. Trainer samples mini-batches with replacement, optimizes weighted policy CE + value MSE; logs loss components, distinct sample count, throughput and fixed-batch loss.
6. Commit immutable `generations/<id>/checkpoint.pt`, `replay.npz`, and hash manifest before atomically replacing the official `checkpoints/normal/latest.json`; `best.json` identifies an evaluation-approved generation, not a separate official model. Normal training metadata, game/move records, root telemetry and metrics use `data/normal.sqlite`; tournament/Elo remain separate. Historical databases and tournament/evaluation records remain as evidence; deleted continuation generations are identified by SHA-256 and are no longer reproducible.

### 8.2 Reproducibility and recovery contract

Training driver seed default 0; checkpoints preserve Trainer NumPy RNG state. `--resume` resolves and validates the committed generation pointer and both file hashes; STANDARD resume without `--generation` requires the original run ID both in checkpoint metadata (or prior recovery identity) and in the self-play database, including for an initial checkpoint. An older generation missing both identities requires explicit rollback with `--generation` and a new run ID; it is not silently assigned an owner. `--generation` selects a committed ancestor for explicit rollback and, with STANDARD logging, requires a new run ID. Historical artifacts lacking cursor or pairing identity cannot be claimed verified or migrated automatically. Resume restores optimizer/global step/iteration, replay and frozen eval batch; changing `n_iter` changes the cosine LR horizon. `Trainer.from_pretrained` for Normal→Renju transfers weights but starts fresh optimizer/training state and Renju buffer.

Historical Normal bootstrap runs retain their original metadata and remain readable; the active training CLI does not create a separate bootstrap branch. Explicit `--fresh` starts a new model only in an empty output directory. Default Normal resume preserves model, replay, optimizer and recorded schedule; extension of a cosine horizon is refused rather than silently changing the schedule.

Self-play record lưu game ID, seed, SHA-256 checkpoint nguồn và digest tensor thực đo sau worker load; replay lưu game ID theo mẫu để truy vết về record. Digest tensor và SHA file checkpoint là hai loại identity khác nhau. Legacy sample không có game ID không được coi là đã có lineage. Các split độc lập không được suy ra chỉ từ hash hoặc fixed eval batch.

Required future run record: immutable run ID; git commit + dirty diff/hash; config snapshot after CLI overrides; ruleset; initial/final checkpoint SHA-256 and parent checkpoint; seed scheme for initialization/game/worker/MCTS; Python/OS/hardware, NumPy/PyTorch/CUDA/compiler/native ABI; exact command; counts of games/samples/optimizer steps; worker/backend settings; logs and exit status. Do not treat fixed-batch loss or sample volume as strength.

### 8.3 Failure handling and acceptance

- Each generation contains immutable checkpoint, replay and a manifest with SHA-256 for both; the atomic pointer replacement is the commit point. `previous` identifies the checkpoint/replay parent actually loaded. After branching from an older generation, `commit_previous` identifies the formerly latest committed generation so rollback can still access the abandoned branch; legacy manifests use `previous` for both. On process interruption before pointer replacement, the previous committed generation remains reachable; explicit rollback only resolves committed history. No automatic deletion of generations. File fsync is performed; Windows directory-entry durability across power loss is not verified.
- STANDARD training writes a durable, immutable per-run/iteration/source-checkpoint recovery outbox before committing games: replay targets and exact game/move/root payloads plus hashes. Resume reuses those bytes, verifies provenance and strict SQLite identities, and refuses ambiguous legacy games without an outbox. A checkpoint binds the outbox manifest and pre-pointer training metrics; resume completes a missing metrics commit idempotently. Explicit rollback starts with a new run ID and records the selected generation and checkpoint SHA; retry before its generation commit may reuse that ID only for the same source. After commit, resume without `--generation`; reuse with a different source fails closed. Another branch's outbox is never consumed. Orphaned temporary directories and uncommitted generations are retained, not deleted.
- **Implemented / Unverified:** crash-injection tests exercise commit boundaries but are not power-loss tests. Hashes detect accidental changes when the reference remains trustworthy, not an attacker able to rewrite both data and unanchored manifest. Durability depends on filesystem/SQLite guarantees; Windows directory-entry durability and hardware power loss remain unproven. A run/output directory requires a single writer; concurrent writers are not coordinated or supported. Recovery does not reproduce games lost before the outbox is committed, nor guarantee bitwise-identical retraining after an interrupted optimizer step.
- Replay NPZ lưu cursor để resume giữ thứ tự ghi đè mẫu cũ nhất. NPZ legacy đầy không có cursor không thể khôi phục thứ tự tuổi: loader cảnh báo và dùng cursor 0 cho mục đích đọc; `Trainer.load` từ chối resume với cursor không rõ. Checkpoint cũ thiếu replay hash không được resume cùng replay mà chưa có quy trình khôi phục được phê duyệt. NPZ legacy chưa đầy khôi phục cursor từ số mẫu.
- Assert `samples == moves`, pi finite/nonnegative/sums to 1, pi mass only on legal actions, z ∈ {−1,0,+1}; ensure per-game IDs unique and one ruleset per buffer.
- Verify resume optimizer/global_step/iteration/buffer and schedule values against same config; test missing/corrupt/mismatched artifacts explicitly.
- Quality gate: candidate-vs-current-best paired Davidson 95% opening-cluster CI lower bound must be >0 to update `best.pt`; report objective success separately against Heuristic. Loss, score threshold or game count alone never promotes.

## 9. Python/C++ dual backend

### 9.1 Implemented boundary

**Verified implementation presence:** `native/gomoku_mcts_native.cpp` is a C++17 pybind11 arena-backed search tree (node indices, PUCT selection, expansion, backup, terminal values, visits). Python `NativeMCTS` owns board/rules/RNG/features/network inference and passes legal move/prior/value arrays across binding. `native/gomoku_native.cpp` implements pattern detection `find_patterns`; `gomoku/patterns.py` selects it. `az/backend.py` resolves overall preset `--backend python|cpp` plus independent MCTS/Patterns overrides; unavailable selected extension fails closed, no silent fallback. Python remains default.

`gomoku_native` is not a native game/rules/board engine; no C++ neural network, training or complete self-play engine is evidenced. Native patterns and MCTS are distinct components; standard AZ self-play marks Patterns `not_used`.

### 9.2 Parity levels and acceptance design

Never require bitwise equality across RNG implementations or floating point unless deterministic contract makes it possible. Compare by level:

1. **Exact:** same state → same legal moves, forbidden mask, terminal/win result, game result, moves/sample count when trajectory is intentionally fixed.
2. **Numerical tolerance:** same fixed model output and deterministic tie handling → same request planes/legal mask, visit counts and simulation count; root/value outputs `abs(delta)≤ε` declared by test. Existing parity tests compare masks/requests/visits/root value on fixtures.
3. **Statistical:** across paired seeds/openings, compare move/visit distributions, game outcomes/score and training-target distributions with pre-registered confidence intervals/effect tolerance.
4. **Performance:** identical compiled extension/Python ABI, checkpoint hash, rules, search budget, hardware/load, warm-up and workload; report repeated median/tail and memory.

Add tests for Python-vs-C++ exact legal/terminal/forbidden masks across generated fixtures, fixed evaluator parity, full self-play paired per-sample features/pi/z plus winner/end reason across multiple seeds, resume and worker spawn. Compare policies/values with explicit tolerances; test ties separately. For a backend optimization: correctness first, training-target/data-distribution regression second, controlled arena quality third, speedup last. If stochastic outcome equivalence is not established, keep backend experimental even if faster.

### 9.3 Current evidence and limits

Reports document native differential tests, CPU full suite, multiprocessing self-play, checkpoint/resume and tournament smoke. `docs/performance/12-5000-games-readiness.md` later documents a small GPU train/resume smoke (2 iterations, 30 optimizer steps total) and a four-game paired self-play comparison with exact matching samples for that seed. This is stronger than earlier GPU failure report, but not long-run stability or broad stochastic/quality parity. Reports document C++ MCTS no material speedup in small depth-1 tournament, while an optimized self-play workload records improvement; do not blend those workloads into one speedup claim. No sanitizer or long RSS soak is evidenced. Build output is CPython ABI-specific `.pyd`/`.so`; compiler, headers, OS and ABI must match.

## 10. Tournament, Elo và đánh giá sức chơi

### 10.1 Tournament hiện có

`scripts/tournament.py` schedules round-robin pairs; requires even games ≥2 and alternates colors. Paired games share opening and per-side seeds; optional random opening. Budgets can be `sims:N` or `time:S`; they are not cross-agent equivalent (AlphaBeta depth differs from MCTS simulations; time deadlines are soft). Tournament supports Normal/Renju separately, agents random/heuristic/alphabeta/mcts-pure/AZ variants, checkpoint mapping and workers. It preflights agents, hashes checkpoints, and writes experiment/agent/match/game/move/event records to local `data/tournament.sqlite` (or `--db`), which is also the resume and analytics source. Resume requires `--resume --run-id RUN_ID` and validates the stored configuration. STANDARD is the default; NONE is an unsaved measurement baseline, not a research run.

Logging V2 uses `data/selfplay.sqlite` for self-play/training and `data/tournament.sqlite` for evaluation/tournament/Elo. SQLite and sidecars are local-only and ignored by Git. Historical import is additive/idempotent by source identity and content hash; dry-run scans `logs/` and `evidence/`, and import never deletes or changes source files. The experiment exchange bundle is explicit, additive, and rejects conflicts; it never replaces the target database. Existing evidence snapshots are not assumed disposable.

Current primary aggregate is score `(W + 0.5D)/N`, WDL, paired/color splits and Wilson interval on score. Wilson interval with draws-as-half is an approximation, not exact multinomial uncertainty and does not account for dependence/multiple comparisons. Do not interpret interval covering .5 as proof equivalence. Seed-paired tests should be analyzed with paired methods or model-level uncertainty where appropriate.

### 10.2 Elo Evaluation Protocol v1.0 (Planned)

`az.elo.recalculate_elo` remains the sequential leaderboard. `az.elo.paired_davidson_ci` implements pairwise Davidson ΔElo with a paired opening-cluster bootstrap for the automated Normal `az_v0` promotion gate; the full multi-agent Bradley–Terry/Davidson leaderboard and historical production recalculation remain unimplemented.

**Rating pools và định danh.** Normal và Renju là hai pool độc lập, không so Elo chéo luật. Trong mỗi pool, `Heuristic` là anchor cố định `R=1200`; các model khác cùng protocol ước lượng relative rating trên cùng graph đối thủ. `model_id` bất biến gồm ruleset + checkpoint SHA-256 (hoặc code/config identity với agent không có checkpoint) + agent/search config; backend/compiler/extension version thuộc `execution_id` và được log như factor, không làm một checkpoint thành model khác. Cùng checkpoint AZ Python/C++ mặc định cùng model rating; chỉ tách thành paired backend contrast khi nghiên cứu parity. Nếu model graph không nối được tới anchor thì không báo rating tuyệt đối.

**Ước lượng chính: Bradley–Terry.** Với skill `θ_i` theo log-odds và game i-vs-j, `P(i beats j)=σ(θ_i−θ_j)`; đổi sang Elo bằng `R_i=1200+(400/ln 10)(θ_i−θ_Heuristic)` để heuristic neo 1200. Mô hình chính phải xử lý hòa bằng Davidson extension của Bradley–Terry (tie parameter, draws recorded as draws), và điều chỉnh màu nếu pilot cho thấy hiệu ứng màu; báo cả mô hình/identifiability/fit checks. Không loại hòa hoặc mã hóa thành thắng/thua. Bootstrap 95% CI resample **opening pair cluster** (hai game dùng cùng opening/seed, đổi màu), giữ hai ván đi cùng nhau; dùng cùng bootstrap draws để có CI cho `ΔElo = Elo_AZ − 1200`. Nếu mẫu cluster ít, fit lỗi hoặc CI không ổn định: kết quả `Unknown/inconclusive`, không ép ra rating.

**Leaderboard tham khảo (không phải suy luận chính).** Sequential Elo dùng logistic expected score `E_A=1/(1+10^((R_B-R_A)/400))`, `R'_A=R_A+16(S_A-E_A)`, `S_A=1/0.5/0` cho W/D/L; cập nhật đối thủ theo `S_B=1-S_A`. Heuristic giữ cố định ở 1200 làm anchor; trong trận hai model không-anchor, update cả hai theo kết quả đối xứng. Giữ lịch sử theo trận và thứ tự deterministic. Đây chỉ là leaderboard operational vì phụ thuộc thứ tự trận/K và draws; không thay BT estimate/CI. Tests kiểm tra expected score, K=16, anchor bất biến, side-swap invariance, và zero-sum change khi cả hai rating được cập nhật.

**Protocol thi đấu.** Đánh giá Normal/Renju riêng; fixed checkpoint hash, rule/config snapshot, seed derivation, openings và protocol version. Heuristic baseline version/config đóng băng ở rating 1200. AZ: cấu hình MCTS cố định 200 simulations mỗi lần search; action temperature `0` (chọn max visits, tie-break RNG theo seed), Dirichlet/root exploration noise tắt. Tactical/threat override được phép bypass MCTS nếu luật search variant quy định; ghi rõ override và 0 simulations. Không dùng softmax exploration trong evaluation. Tournament đổi màu trên từng opening: 2 games/opening pair; cùng opening và paired seed, chỉ đổi ai cầm Black. Đối thủ có cùng thuật toán nhận đúng budget tương ứng; mọi baseline không cùng đơn vị search phải ghi rõ budget/latency và không diễn giải như compute-equivalent. Freeze environment/backend identity; log seed, opening, ruleset, checkpoint/config SHA-256, MCTS sims, temp/noise, player colors, result/end reason, move count, latency/search telemetry. Fail-fast nếu manifest thiếu/mismatch hoặc checkpoint đổi trong run.

**Các lựa chọn cố định cho protocol v1.0:** dùng `HeuristicPlayer` làm anchor với code/config identity SHA-256 theo ruleset; không có file config riêng, patterns backend cố định `python`. AZ dùng MCTS backend `python`, `n_simulations=200`, temperature `0`, root noise tắt. Theo mặc định hiện có của tournament, `random_opening_plies=0` (bắt đầu từ bàn trống); dùng `paired-crc32`, đăng ký `--seed` trước mỗi run và dùng lại cùng seed từng tác tử cùng opening trong cặp đổi màu. Không đổi opening/seed scheme giữa các tranche; muốn đổi phải tăng protocol version. Kết luận chỉ áp dụng cho protocol bắt đầu từ bàn trống.

**Cỡ mẫu và mở rộng định trước.** Screening: 100 games/pair = 50 paired openings, chỉ quyết định có tiếp tục/kiểm tra lỗi dữ liệu, không tuyên bố đạt mục tiêu. Main: 400 games/pair = 200 paired openings. Trước chạy phải đăng ký danh sách opening/seeds, model comparisons và bootstrap seed. Sau 400, nếu CI 95% của ΔElo chứa 0 hoặc precision chưa đạt, thêm tranche cố định 200 games (100 paired openings), tối đa 800 games/pair; phân tích ở mốc định trước, không xem CI liên tục để dừng tùy ý. Nếu đến 800 vẫn chưa phân định thì báo inconclusive. Nếu CI đã tách 0 tại mốc main, không cần mở rộng trừ khi độ chính xác yêu cầu đã đăng ký chưa đạt. Mốc sample-size là protocol đề xuất, chưa có power analysis cho dự án; chạy pilot 100 trước và kiểm tra CI bootstrap/cluster variance, sau đó chốt hoặc version protocol mới trước main study.

**Báo cáo và quyết định.** Mỗi so sánh trình bày BT Elo, `ΔElo`, 95% paired-cluster bootstrap CI, games, W/D/L, score `(W+0.5D)/N`, màu/opening strata, illegal/forfeit/draw breakdown, latency/search and resource counters, model IDs, protocol and log hash. “Đạt mục tiêu”: `ΔElo > 0` và cận dưới CI 95% `>0`. “Mục tiêu mạnh”: `ΔElo ≥ +100` và cận dưới CI 95% `>0`. Đạt một tiêu chí không khẳng định tổng quát ngoài pool/opponents/protocol đã đo. Với heuristic 1200, chiến thắng/thua tuyệt đối ở từng trận vẫn được ghi W/L và ảnh hưởng đúng vào BT/leaderboard; trận illegal move/forfeit là loss cho bên vi phạm, không loại khỏi mẫu; hòa là draw trong likelihood và 0.5 score/Elo. Không hoàn tất/aborted game không là W/D/L; phải resume cùng identity hoặc loại theo rule đã đăng ký và báo số loại.
#### Automated `az_v0` screening gate v1.3 (Implemented / Unverified)

The continuous Normal pipeline checks every 5,000 newly generated self-play games, requiring an iteration size that reaches the gate exactly. The screening gate plays exactly 100 games against frozen Heuristic using Python MCTS, 200 simulations, temperature 0, root noise off, four randomized opening plies and paired color-swapped games; it also records a fixed tactical holdout evaluation. The cluster bootstrap seed and W/D/L stay in SQLite. If no best exists, a paired 95% ΔElo CI lower bound >0 against Heuristic permits promotion. If a best exists, a Heuristic-only gate cannot establish superiority to that best: retain it and do not promote without a separate best comparison. Tactical accuracy and the 100-game screening do not by themselves establish the research objective. Historical 25k exposure to the tactical positions remains unknown.

Historical v1.2 records retain their original protocol: 100 games versus the current champion when present, otherwise Heuristic, without this tactical screening.


**Log schema tối thiểu / acceptance.** Mỗi game cần `protocol_id`, `ruleset`, `game_id`, `pair_id/opening_id`, `seed/opening`, `model_a_id/model_b_id`, checkpoint/config SHA-256, backend/compiler/extension version trong `execution_id`, `black_model/white_model`, MCTS budget, temperature/noise, `winner` (Black/White/draw), `end_reason`, illegal/forfeit actor, moves, runtime/search telemetry. Rating artifact phải có BT/Davidson fit version, anchor, parameter/K settings, bootstrap seed/cluster count, CI, exclusions, input-log SHA-256 và sequential leaderboard trail. `AT-ELO-01`: expected-score, K=16 update cho win/draw/loss, anchor cố định 1200 và zero-sum chỉ khi hai non-anchor ratings cùng được cập nhật. `AT-ELO-02`: deterministic replay logs tái tạo W/D/L, score, paired clusters và sequential ratings. `AT-ELO-03`: toy synthetic symmetric pool ước lượng quanh anchor; mạnh hơn/yếu hơn fixtures cho đúng thứ tự; draw-only/one-sided/unconnected pools trả warning/inconclusive, không rating bịa. `AT-ELO-04`: bootstrap resample cặp opening, giữ đổi màu, trả CI hữu hạn và tái lập seed. `AT-ELO-05`: hash/config/ruleset mismatch, duplicate/missing game IDs và illegal outcomes bị phát hiện; Normal/Renju không gộp. `AT-ELO-06`: screening 100, main 400, staged extension 600/800 và stopping rule được test bằng pure scheduling/decision logic, không chạy tournament tốn kém trong unit tests.

Leaderboard dưới đây là schema, không phải kết quả:

| Model | Rules | Elo ± 95% CI | ΔElo vs Heuristic | Games | W/D/L | Score | Mean/p95 ms/move |
|---|---|---:|---:|---:|---:|---:|---:|
| AlphaZero checkpoint | Normal | TBD | TBD | TBD | TBD | TBD | TBD |
| Heuristic (anchor) | Normal | 1200 | 0 | TBD | TBD | TBD | TBD |
| Alpha-Beta | Normal | TBD | TBD | TBD | TBD | TBD | TBD |
| Random | Normal | TBD | TBD | TBD | TBD | TBD | TBD |

Vẽ learning curve với X là số self-play games (cùng definition ở mọi run), Y là BT Elo; show CI/error bars, AZ−Heuristic ΔElo, checkpoint hash và protocol ID. Dựng panel riêng cho Normal/Renju; không nối rating khác protocol như cùng một chuỗi.


### 10.3 Existing strength evidence

`docs/report.md` T35 records 2,400 completed games across separate Normal fixed, Renju fixed, and Normal time-budget groups. In its fixed Normal pool, Heuristic beat AZ v0 and AZ threat each 100–0; Renju Heuristic beat AZ Renju 100–0. The time-budget AZ threat vs AlphaBeta result does not compare against Heuristic and is a different protocol. Audit of the frozen archive verified all 107 recorded file SHA-256 values, 24 matchups × 100 games, sequential IDs, color balance and W/D/L summaries against raw game records. T35 fixed AZ searches use 200 simulations; 1,133 tactical/threat override moves bypassed MCTS and logged 0 simulations. Its legacy seed scheme did not reuse each agent's seed across color-swapped games; manifests lack Heuristic source hash, explicit backend, evaluation temperature/noise and opening-plies metadata. The exact historical Heuristic identity and v1.0 equivalence are therefore Unknown. T35 is historical evidence, not Elo, not proof about every checkpoint/config, and not evidence that AZ met its objective. Historical initial report also records 1,000 Normal self-play games and 500 Renju fine-tune games; run losses do not prove playing strength. 5,000-game config is not a completed 5,000-game training result.

### 10.4 Kế hoạch chẩn đoán khi AlphaZero thua Heuristic (Planned)

Chạy theo thứ tự, giữ nguyên ruleset/checkpoint/protocol và lưu artifacts; mỗi bước phân biệt lỗi thực thi với giới hạn chất lượng:

1. **Tái lập baseline:** replay các match IDs từ raw logs, đối chiếu checkpoint/config/rules hashes, màu, illegal/forfeit và seed. Nếu không khớp hoặc run cũ thiếu provenance, xem T35 là bằng chứng lịch sử chứ không dùng làm baseline định lượng mới.
2. **Search correctness:** trên fixture tactic (win-in-1, block-in-1, fork, forced loss, Renju forbidden), kiểm tra legal mask, prior sau mask, visit counts, root value/sign, terminal handling, temperature=0 choice và repeatability. Cùng fixed evaluator so Python/native; không thay model để debug MCTS.
3. **Policy/value và targets:** audit các trạng thái self-play đã log: `pi` finite, normalized, legal-only; tính entropy/top-k/agreement với selected move; so target visit distribution với model policy; đối chiếu `z` theo player-to-move và outcome. Đánh giá value calibration/Brier/MSE theo game outcome trên held-out games/positions tách theo game/seed, không dùng fixed training batch làm test.
4. **Tactical error taxonomy:** từ loss games phân loại bỏ lỡ thắng ngay, không chặn thắng, bỏ fork/forced defense, Renju legality, endgame/draw. Replay position với legal-move listing; so MCTS candidate ranking/visits, network prior/value và Heuristic response. Tactical override/threat-space là ablation riêng; không gán hiệu quả cho base network.
5. **Search correctness under fixed budget:** mỗi lần MCTS thực sự chạy dùng đúng 200 simulations trên cùng checkpoint, positions/openings/seeds và no noise; ghi Elo/score, move latency và tactical error rates. Không thay simulation budget để chẩn đoán; tactical override bypass search phải được log riêng và không tính là budget khác.
6. **Checkpoint/training diagnosis:** theo từng checkpoint đo matched Elo, independent outcome/value/policy metrics, replay age/diversity, target entropy, game length/WDL by color, optimizer/global steps/config/hash. Kiểm tra overfit, collapse, data leakage, checkpoint mismatch; loss reduction không phải promotion criterion.

Acceptance `AC-DIAG-01`: mỗi diagnosis có immutable position/game IDs, checkpoint hash, rules/config, hypothesis, metric and pass/fail evidence. `AC-DIAG-02`: search fix chỉ được chấp nhận khi invariant tests qua và paired evaluation không có confirmed quality regression. `AC-DIAG-03`: kết luận nguyên nhân phải có canary/control isolating variable; nếu nhiều nguyên nhân cùng phù hợp dữ liệu thì ghi Unknown, không chọn giải thích hậu nghiệm.

### 10.5 Ablation study plan (Planned)

`scripts/ablation.py` hiện có ablations Alpha-Beta pruning và MCTS Pure rollout budgets; không có AZ Elo ablation framework. AZ studies phải dùng protocol v1.0, cùng frozen evaluation set, heuristic anchor, paired opening/color seeds, fixed checkpoint identity where applicable, no evaluation noise/temperature and same rules/config except biến nghiên cứu. Báo Elo/ΔElo+paired CI, W/D/L/score, compute/search telemetry, training data counts; register comparisons and multiplicity handling before running. Tách screening (pilot) khỏi confirmatory result.

| Factor | Screening levels (đề xuất) | Isolation / fairness rule |
|---|---|---|
| MCTS simulations | 200 (fixed) | Không phải factor ablation; giữ nguyên trong mọi run AZ, kể cả self-play generation và evaluation |
| Network architecture | 3×32, 6×64 baseline, 9×96 | Match training data, optimizer steps, search budget; báo parameter count, peak memory/inference cost; không coi equal steps là equal FLOPs |
| Self-play scale | 1k / 2.5k / 5k / 10k cumulative games | Cùng recipe/seed plan; checkpoints trên learning curve, same Elo protocol; xem games, distinct samples, optimizer steps và compute cùng nhau |
| Backend | Python vs C++ search, same checkpoint; backend self-play retraining separate | First exact/differential parity, then paired multi-seed targets/outcomes, then Elo non-regression; runtime speedup không là quality outcome |

Một run/seed không đủ cho kết luận nghiên cứu. Dùng paired training seeds; số seed cho confirmatory ablation phải được chọn từ variance/power pilot và đăng ký trước. Để giới hạn chi phí, dùng screening nhỏ để loại lỗi/large regressions, chỉ confirm các contrasts có ý nghĩa; không chạy full Cartesian product. Baseline v1 remains 6×64 and 200 simulations for all AZ MCTS runs; simulation count is not a tunable factor or subject to protocol revision.


Acceptance `AC-ABL-01`: manifest chứng minh chỉ factor mục tiêu đổi, các control giữ nguyên. `AC-ABL-02`: mỗi point có checkpoint hash và cùng evaluation protocol/model pool. `AC-ABL-03`: statistical comparison giữ paired openings/training seeds và điều chỉnh multiple comparisons hoặc ghi exploratory. `AC-ABL-04`: report Elo quality alongside compute; không chọn cấu hình theo throughput/loss đơn lẻ. `AC-ABL-05`: backend quality gate không kém control ngoài non-inferiority tolerance đăng ký trước; tolerance hiện Unknown, phải chốt sau pilot.

## 11. Performance và benchmark

Collect move latency mean/p50/p95/p99, MCTS sims/sec and NN leaf calls, games/min, training samples/optimizer steps/sec, wall-time decomposition, CPU/GPU utilization, peak RSS/VRAM, worker scaling and failure rate. Tag workload CPU/search-bound, inference/GPU-bound, process/IPC-bound, storage-bound; report both total run throughput and per-component counters. Node count is not comparable to simulations or neural evaluations.

Benchmark protocol: pin hardware/software/extension/checkpoint hash, rules, opening/state corpus, simulation/time budget, batch/workers/threads, warm-up, repetitions, system load and CUDA synchronization; randomize/interleave backend order; use same positions/seeds; report median and spread, not one run. Separate fixed-work search correctness/throughput from wall-clock policy quality. Any optimization changing scheduling, batching, tie-breaking or RNG must compare self-play distributions and controlled arena outcomes.

Historical numbers belong only to cited workload: `docs/benchmarks.md`, `docs/performance/`, `docs/bao-cao-dual-backend.md`. In particular 129.59 games/min estimate and 38.6 minute projected 5k self-play are linear extrapolation from 50-game benchmark, omit training/evaluation/checkpoint overhead, and do not constitute 5k soak. No figures in this section are asserted as current universal performance.

## 12. Reproducibility và persistence

- Config source: `config.py` defaults + `configs/*.yaml` + CLI overrides. Preserve fully resolved config with logs.
- Training persistence: `.pt` checkpoint and `.npz` replay buffer remain unchanged. SQLite V2 schema v1 is present; the training CLI writes run/game/move/training-step records and compressed root search telemetry in STANDARD mode; NONE skips telemetry collection and database writes. Root fields are captured after each search without changing selection or consuming RNG.
- Tournament persistence: `scripts/tournament.py` writes experiment, agent, match, game and move rows to `data/tournament.sqlite`, while retaining legacy manifests/JSONL/CSV for resume/readers. `--logging-level none` skips telemetry and durable output for transient baselines; it cannot resume and is not for research runs. Basic sequential Elo replay is implemented; historical imports remain unimplemented.
- Experiment exchange: explicit tournament experiment bundles preserve metadata, games/moves, event history and Elo rows; imports validate a canonical content hash and add rows transactionally. Self-play run bundles are not implemented.
- Local data: `.gitignore` ignores `*.sqlite*` and `*.sqlite3*`; no SQLite database was tracked when checked. Import/export is explicit and never overwrites the destination database.
- Logging performance: three-repeat Python/Normal and native-C++/Normal synthetic-evaluator self-play tests at 200 simulations measured 1.49% and 3.24% STANDARD overhead; Python/Renju and native-C++/Renju runtime deltas were −0.09% and 0.34%. Tournament `az_v0`-vs-random tests at 200 simulations measured 1.20% Python-MCTS/Normal, −0.06% C++-MCTS/Python-patterns/Normal, 0.30% Python-MCTS/Renju, and 0.59% C++-MCTS/Python-patterns/Renju. Renju self-play uses a synthetic evaluator; production training workload overhead remains unverified.
- Use content hash, not mutable basename `last.pt`, to identify a model. Store immutable source and parent hashes; do not overwrite evidence.
- Version the ruleset, evaluation protocol and backend implementation; archive dependencies and native build command. Capture actual `python --version`, `torch.__version__`, CUDA availability/device, OS, CPU, GPU, compiler and extension ABI.
- Re-run from a clean, documented environment and frozen config. Document stochastic reproducibility limits, CUDA nondeterminism and process RNG scheme; distinguish repeatability of exact game logs from statistical replication.

Proposed experiment manifest (format proposal; no code change in this spec): `schema_version`, `run_id`, `git_commit`, `dirty_tree_hash`, `command`, `resolved_config`, `ruleset/rules_version`, `seed_root/derivation`, `model_id/parent_hash`, `backend/components/extension_hash`, environment/hardware/compiler, worker/thread settings, game/sample/step counts, start/end/exit status, artifact paths+SHA-256, evaluation protocol ID, tournament raw-log hash and limitations.

### 12.1 Reproducibility and quality acceptance gates (mixed implementation status)

`RG-01 Run identity`: every training/evaluation run persists resolved config, code/dirty-tree identity, all seed streams/derivations, ruleset, environment, backend/ABI and artifact SHA-256; test by loading manifest and verifying every referenced artifact/hash. Missing identity makes result `Unknown`, not “approximately reproducible”.

`RG-02 Seed discipline`: use explicit independent seeds for model initialization, self-play game, worker, action/tie-breaking, opening and bootstrap; derive child seeds deterministically from `(root_seed, purpose, game/pair_id, color)` and persist mapping. Same code/environment/config/seed must reproduce deterministic setup and records where backend/device support deterministic operations; otherwise document exact nondeterministic fields and verify statistical replication.

`RG-03 Checkpoint integrity`: checkpoints immutable/content-addressed in experiment records; resume must match expected parent hash, ruleset, architecture and resolved config; train logs link each promoted checkpoint to parent and iteration/game count. Test mismatch/corruption rejection and round-trip hashes/metadata.

`RG-04 Python/C++ parity`: same rule state produces identical legal moves, terminal outcome and forbidden mask; same fixed evaluator/RNG/tie policy produces equal visits/simulations/root value within declared numeric tolerance; paired self-play compares features, `pi`, `z`, winner and end reason over pre-registered seeds. Any mismatch blocks native promotion until explained and fixed.

`RG-05 Quality regression`: for Normal `az_v0`, the periodic v1.2 gate uses 100 paired games against best, or Heuristic when best does not exist. Promote only when the paired 95% CI for ΔElo vs the selected opponent has lower bound >0. CI crossing zero retains best without extension; CI upper bound below zero vs best rejects the candidate and stops the session. A best-only gate does not evaluate the Heuristic research objective. The 100-game limit increases uncertainty and is not the 400-game research protocol. Other backends and tournament model variants still need their own matched gate.

`RG-06 Quality-regression tests`: unit/integration gates cover rules, MCTS sign/masking/terminal, sample integrity, checkpoint/resume and tournament fairness; a frozen game/position tactical suite catches consumer-visible weakness; every accepted model change has paired matched-protocol Elo evidence. Store regression fixtures with provenance and do not tune against held-out confirmation positions.

`RG-07 Research reproducibility`: independent rerun from manifest reconstructs matchup schedule and all W/D/L from raw logs; rating/CI output reproduces exactly for the same deterministic analysis seed/version. CI may differ only when bootstrap/model code or inputs/version differ, which must be recorded. Report limitations and excluded games.

## 13. CLI, UI và commands

Commands below are source/README-documented interfaces, not executed during this audit. Run in `AlphaZero_Gomoku`; actual Python executable depends on active environment (README references `..\.venv`; native report used a project `.venv`, which itself is an environment discrepancy to resolve).

| Command | Purpose / relevant inputs | Outputs / constraints |
|---|---|---|
| `python scripts/play.py --mode ai-vs-ai --black-agent random --white-agent heuristic --ruleset normal` | CLI game; supports human-vs-AI/AI-vs-AI, Normal/Renju, seeds, agent depth/rollouts/checkpoints | Render result; agent args validated; soft time budget |
| `python -m ui.pygame_app --mode human-vs-ai --agent mcts-pure` | Pygame UI | Interactive; tests in `tests/test_ui.py`; runtime display not verified here |
| `python scripts/train.py --iterations 701` | Resume the official Normal generation through absolute iteration 701; missing checkpoint fails closed; `--fresh` requires an empty output directory | Generation and recovery artifacts under `checkpoints/normal/`, Normal records in `data/normal.sqlite`; no training run is implied by this example |
| `python scripts/arena.py --a heuristic --b az_v0 --games 100 --budget sims:200` | Two-player evaluation | WDL/summary game logs; not Elo |
| `python scripts/tournament.py --agents random,heuristic,alphabeta,az_v0 --rules normal --games 100 --budget sims:200 --seed 1 --db data/tournament.sqlite` | Round robin; even `--games`; options for checkpoints, workers, random openings, pair/resume, backends and output | Run directory and selected experiment/game/move rows in SQLite; legacy JSONL/manifest/CSV/summary remain; no Elo calculation |
| `python scripts/build_gomoku_native.py --module mcts --cxx g++` | Build native MCTS; also `--module patterns` | ABI-specific extension; build requires compatible compiler, Python headers and pybind11 headers (possibly sourced from PyTorch include) |
| `python -m pytest -q` | Test suite | Project-wide tests; exact environment required; full suite was not run for the Logging V2 change (see implementation report). |

Do not infer undocumented CLI support from README command examples alone; confirm flags in script parser. Native selection is explicit; unavailable extension fails with actionable error, not fallback.

## 14. Testing strategy và traceability

Tests use pytest in `tests/`. Run focused tests after implementation; full suite only at integration/release gate, with CPU/GPU/backend matrix where available. Latest performance/native reports document different snapshots/counts (e.g. 260/3 skipped and later 261/2 skipped); do not combine these into one current pass claim. This audit did not execute tests.

| Concern | Existing tests/evidence | Required extension / acceptance |
|---|---|---|
| Board, Normal, Renju | `test_board.py`, `test_rules_normal.py`, `test_rules_renju.py`, `test_renju_integration.py` | Add edge and differential/reference fixtures; Renju exact-five/overline/3-3/4-4 interactions |
| Feature/model contract | `test_features.py`, `test_model.py` | Tensor dimensions/dtype, symmetry feature-policy alignment, finite value/probability and checkpoint roundtrip |
| MCTS | `test_mcts.py` | win-in-1, sign, forbidden mask, terminal calls, root noise, visit policy, no-legal-moves, tie/temperature |
| Replay/self-play | `test_replay_buffer.py`, `test_selfplay.py` | legal-target support, policy sum, one sample per ply, z perspective, ruleset separation, reproducible worker IDs |
| Trainer/resume | `test_trainer.py`, reports | loss math, optimizer/scheduler restore under unchanged config, corrupt/missing buffer and checkpoint behavior |
| Tournament/fairness | `test_tournament.py`, `test_evaluator.py`, reports | color/opening pairing, seed derivation, immutable output/resume/shards, illegal moves, budgets and config/checkpoint hashes |
| Native parity | `test_native_mcts.py`, `test_native_patterns.py`, `test_backend.py` | generated-state exact masks/terminal, multi-seed output/target parity, explicit numerical tolerance, statistical quality gate |
| Elo v1.0 (planned) | Basic sequential Elo calculation/replay tests exist in `az/elo.py` / `tests/test_elo.py`; no advanced protocol evidence | `AT-ELO-01..06`: K=16 research-protocol math, BT/Davidson fit, paired-opening bootstrap, protocol/data validation, staged sample decision |
| CLI/UI/classical | `test_play_cli.py`, `test_ui.py`, `test_alphabeta.py`, `test_mcts_pure.py`, `test_heuristic.py` | smoke actual runtime for changed path; headless UI environment limits documented |
| Performance | scripts/reports, benchmark tests | stable corpus and controlled repeated measurements; no speed acceptance without correctness/quality gates |

**Test invocation:** `python -m pytest -q` (full documented command); focused examples: `python -m pytest -q tests/test_rules_renju.py tests/test_mcts.py`, `python -m pytest -q tests/test_native_mcts.py tests/test_backend.py` after building extension. `CUDA_VISIBLE_DEVICES=-1` can force CPU for native report workflow; use platform-appropriate environment syntax on Windows. No performance gate should run expensive training/tournament without explicit authorization.

## 15. Code style

Follow existing modules: Python 3 type hints, `snake_case` functions/modules, `PascalCase` classes, constants uppercase; narrow responsibilities, NumPy arrays for board/tensors, torch confined to `az/`, rules passed by interface, dataclasses for records. C++ bindings expose narrow contracts and throw Python-visible exceptions on invalid shapes/indices. Tests assert user-visible semantics, not source text. Preserve existing function signatures/contracts unless all callers migrate.

```python
def board_to_planes(board, rules, player: int) -> np.ndarray:
    """Encode the position from the perspective of the player to move."""
    planes = np.zeros((N_PLANES, SIZE, SIZE), dtype=np.uint8)
    planes[0] = (board.grid == player).reshape(SIZE, SIZE)
    planes[1] = (board.grid == opponent).reshape(SIZE, SIZE)
    return planes
```

Style snippet illustrates current compact array-first convention; it omits actual forbidden/color planes for readability and is not replacement code.

## 16. Boundaries

- **Always:** check Graphify before source investigation per repository rules; inspect current worktree and keep user changes; source-over-stale-doc precedence; label verified vs unverified; preserve rule/interface/value-sign/data provenance contracts; use exact model/config/ruleset identities; report commands/tests actually run; keep `gomoku/` free of torch; isolate Normal/Renju replay buffers and evidence.
- **Ask first:** changes beyond this `SPEC.md`; rules interpretation not settled by code/official assignment; changes to training hyperparameters, dependencies, public APIs, checkpoint schema, rating protocol, or benchmark hardware/resource envelope; expensive training, tournament, soak, or artifact migration.
- **Never:** fabricate results or infer strength from loss/game count/speed; claim Python/C++ production parity from a smoke test; silently fallback from requested C++; mix rule sets in one buffer/pool; overwrite/delete checkpoints, logs, shards, or outputs; introduce hardcoded rule logic outside rules interface; commit credentials or generated training artifacts.

This task authorizes the scoped logging implementation and this SPEC/report update. No historical evidence/log/checkpoint/replay data was migrated, renamed, or deleted; benchmark and expensive training/tournament execution remain unperformed. No commit was requested or created.

## 17. Current state, gaps và roadmap

### 17.1 Current state (evidence-backed)

- Board, Normal/Renju rules, feature encoding, AZNet, MCTS, replay/trainer/self-play and classical opponents exist in source; tests cover key game/search/training contracts.
- Tournament and raw telemetry/provenance pipeline exists, with color balancing, paired seeds/openings, resume/sharding protections, checkpoint hashing and separate rule selection.
- Optional native C++ MCTS and pattern extensions exist; Python is default; code explicitly fails closed on unsupported requested native backend.
- Reports document Normal training 1,000 games and Renju fine-tuning 500 games. Separate config targets 5,000 but is not proof it ran.
- Reported T35 sample: Heuristic 100/100 against AZ v0 and AZ threat in fixed Normal pairs, and 100/100 against AZ Renju in fixed Renju pair. Therefore primary objective is not met on these runs.
- Basic sequential Elo recalculation and event persistence exist; leaderboard, uncertainty model, and research-protocol acceptance remain unimplemented.

### 17.2 Gaps and research risks

1. Strength gap is immediate: sampled T35 matches show Heuristic dominance. Training more without diagnosing policy targets, openings, model/checkpoint provenance, MCTS budget and game trajectories may scale weak data rather than improve.
2. Basic Elo does not establish the planned rating/uncertainty protocol; no production historical results were recalculated.
3. Renju semantics need authoritative rule adjudication for international parity beyond current fixtures.
4. Training metrics lack independent game/position holdout; historical logs omit sufficient environment/code/init-seed/parent-checkpoint provenance for bitwise reproduction.
5. Native parity evidence has advanced from initial reports, but broad multiseed/data/quality parity, sanitizer and long-duration soak remain unproven.
6. GPU short smoke and projected 5k throughput do not prove hours-long stability, 10k scaling, or improved playing strength.
7. Requirements mention Python 3.12 while reports use Python 3.10; shared unpinned requirements and environment locations conflict.

### 17.3 Target state and build order

| Order / module id | Target and acceptance | Dependencies |
|---|---|---|
| 1 `game-rules` | Canonicalize Normal/Renju behavior; edge/reference/differential tests and documented rule scope pass | — |
| 2 `ai-search` | Multi-state MCTS/network/self-play invariants; diagnose Heuristic loss with fixed test positions, tactical coverage and training-target audits | game-rules |
| 3 `native-parity` | Broad deterministic differential + multiseed trajectory/target parity; GPU/CPU integration and soak; only then promote backend | game-rules |
| 4 `selfplay-training` | Reliable recoverable runs, complete manifest, independent quality evaluation milestones at 0/500/1k/2.5k/5k/10k; explicit stop/continue rules | ai-search |
| 5 `evaluation` | Immutable protocol/match logs, Elo/Bradley–Terry estimate + uncertainty, frozen baselines, checkpoint learning curve; establish AZ vs Heuristic objective empirically | selfplay-training |
| 6 `reproducibility` | Clean-environment replay of published training/tournament run from manifest and verified artifacts | selfplay-training, evaluation |

Dependency order follows the approved capability map; `native-parity` and `selfplay-training` can proceed after their shared prerequisites without a dependency between them. Ordering is future work, not current implementation structure. Trustworthy logs remain a prerequisite for conclusions.

**Priority decision:** First diagnose and measure quality regression against Heuristic using exact historical protocols and fixed tactical test positions; do not launch 5k/10k run just because config exists. In parallel tighten Renju rule/test contract and training provenance. Then establish evaluation uncertainty, broaden backend parity, and scale self-play with staged resource/quality checkpoints. Optimization only after identifying bottleneck and passing parity/quality gates.

### 17.4 Training roadmap by quality gates (Planned)

Milestones are cumulative iterations in one Normal `az_v0` lineage, not separate model versions or baselines. Migration used the latest valid checkpoint state at iteration 200, global step 184,000, Normal rules, optimizer state, 50,000 replay samples and 512 frozen eval samples. SQLite preserves historical runs but is not asserted to contain every historical game. The former 1,000-game checkpoint is not used as a baseline. Every 5,000 newly generated games, run the v1.2 gate for exactly 100 games against best, or Heuristic when no best exists, using 200 AZ simulations, four opening plies and the Python evaluation backend.

| Cumulative self-play games | Checkpoint/evaluation gate | Decision |
|---|---|---|
| Every +5,000 | Save latest/replay; record hash, iteration and optimizer steps; evaluate against best or, if absent, Heuristic | Promote only if lower 95% ΔElo CI vs selected opponent >0 |

**Stop/continue rule:** data/rule/checkpoint integrity failure stops immediately. A confirmed regression (CI upper <0 vs best) rejects the candidate and stops that training session; an inconclusive result retains best without further games. Self-play continues from latest only after integrity passes; loss and completed-game count never promote. Historical SQLite run IDs and outcomes remain untouched.

Acceptance `AC-TRAIN-MILESTONE-01`: every evaluation checkpoint has hash, iteration/game count, optimizer steps and versioned Elo output or explicit Unknown. `AC-TRAIN-MILESTONE-02`: promotion decision reproduces from stored W/D/L, bootstrap seed and checkpoint hashes. `AC-TRAIN-MILESTONE-03`: reports distinguish self-play games, optimizer steps, Elo CI and hardware/compute; missing stage data is a gap, not interpolated result.

## 18. Open questions và giới hạn bằng chứng

1. Which authoritative Renju ruleset (including opening/forfeit nuances) is the acceptance authority? Current implementation is documented, but international legality is not independently validated.
2. Elo v1.0 fixes the proposed anchor, thresholds and sample sizes, but their suitability is not yet validated: choose the frozen Heuristic implementation/config and use the 100-game pilot to assess paired-cluster variance/power before preregistering a main run. A protocol change requires a new version.
3. The proposed BT/Davidson model, color adjustment and 400/600/800 stopping schedule require synthetic-data tests and pilot fit diagnostics; confirm whether draw frequency or pool connectivity requires modifying the model before official study.
4. Which interpreter/environment is canonical: shared Python 3.12 requirements or report-tested Python 3.10 venv? Resolve before published reruns.
5. Historical game log provenance does not fully bind source commit, initialization seed and parent checkpoint; exact reproduction may be impossible for earlier runs.
6. This document audit read source, tests, configs and selected performance/report artifacts; did not run test suite, native build, GUI, training, or tournament. Check-in report claims are cited as report evidence, not independently revalidated in this turn.
