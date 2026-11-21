# Phân tích thất bại AlphaZero trước Heuristic ở mốc 25.000 ván

## Phạm vi và kết quả

Nguồn chính là `data/tournament.sqlite` (artifact local, không được Git quản lý), experiment `az_v0-15000-continuation-20261119-promotion-500-heuristic-0`. Tên run chứa `15000` vì đây là continuation; mốc 25k là iteration 500 × 50 ván/iteration theo cấu hình được ghi, không được suy ra chỉ từ tên run. Cột `config_json` ghi Normal, `az_v0`, 400 games, `sims:200`, opening 4 ply, seed 5000000, backend MCTS và patterns Python. Đây là phân tích log lịch sử, không chạy tournament, training hay sửa thuật toán.

**Quan sát từ 400 bản ghi:** AZ thắng/hòa/thua **46/0/354** (11,5% score). AZ cầm Đen thắng 35/200, cầm Trắng thắng 11/200; Heuristic thắng lần lượt 165 và 189. Cả 400 kết thúc `five_in_row`. Các bản ghi có tổng 8.978 `raw_moves` sau opening; đây là telemetry đã ghi, không phải MCTS trace đầy đủ. Không xem các nước trong cùng opening pair là độc lập để suy luận CI. Điểm 11,5% là kết quả mô tả, không phải đánh giá Elo hay ước lượng nhân quả.

Checkpoint SHA-256 được lưu trong config **và** agent record là `2384043ae275dfe5eef24cb9ec1f87d771ae0238ccfff01ffa14591baffeb731`. Đường dẫn lịch sử `checkpoints/latest.pt` có thể bị ghi đè: tại lần đọc này SHA-256 file hiện tại là `f7f521c1cc37a5054ed854201b087e484e9fcef3d4d6b1877ab81dff2e51d7e4`, **khác** hash của tournament; không thể dùng file hiện tại để tái chạy checkpoint lịch sử mà không tìm được artifact đúng hash. Metadata ghi commit `914241e5a0168fcdd4a2b60b974229b065eaabc0`, `dirty=true`: commit không xác định đầy đủ code thực thi. Cột `git_commit` của experiment là NULL nhưng config JSON có commit trên.

Tất cả 400 record ghi 4 nước opening; đối chiếu từng cặp `pair_game` 2k/2k+1 được **200/200** cặp có 4 nước đầu giống nhau, opening seed giống nhau và seed Đen/Trắng hoán đổi. Vì vậy không phải 400 opening độc lập. Code hiện tại sinh nước opening ngẫu nhiên đều từ legal moves (`scripts/tournament.py:328-346`) và lập cặp đổi màu/seed (`scripts/tournament.py:437-468`); do historical code dirty, đó là cơ chế hiện tại, còn dữ liệu đã kiểm là opening và seed thực lưu. Opening ngẫu nhiên là tiền tố được áp đặt cho cả hai bên, không phải 4 lựa chọn của AZ.

## Đối chiếu 15k

Experiment `az-15k-vs-heuristic-20261119-az_v0` có 100 ván, AZ thắng/hòa/thua **6/0/94** (Đen 2/50, Trắng 4/50), seed 20261119, Normal, 200 simulations, opening 4 ply; 50/50 cặp có cùng opening và seed đổi màu. Hash checkpoint **`8b775fffc21c4491b959a3af66f829ca7ba9e38ed63879a7143f694fa132cabd`**. Tuy nhiên 15k dùng MCTS/patterns C++ (native MCTS 0.1.0), 25k dùng Python/Python; seed khác, kích thước mẫu khác, lịch sử code dirty. Vì vậy **không phải so sánh matched** và chênh 6% → 11,5% không chứng minh 25k tiến bộ do training. Hash khác chứng minh file checkpoint khác, không đo trực tiếp khác biệt trọng số. Đặc tả hiện hành yêu cầu AZ search 200 simulations và đánh giá paired có identity/protocol (`SPEC.md:193`, `SPEC.md:277-299`); phép so sánh trên không chứng minh parity backend hay tiêu chí AZ > Heuristic.

## Findings ưu tiên

### P1 — Kết quả yếu trước Heuristic trong protocol lịch sử

- **Liên quan:** dữ liệu `games`, `agents`, `experiments` trong `data/tournament.sqlite`; `scripts/tournament.py:361-431` ghi kết quả và telemetry; `scripts/tournament.py:437-468` lập lịch cặp.
- **Tác động:** AZ chỉ đạt 46/400 điểm thắng và thua cả hai màu trong tập quan sát; mục tiêu chứng minh AZ mạnh hơn Heuristic chưa đạt với protocol này.
- **Độ tin cậy:** cao đối với thống kê mô tả trên 400 record; không suy rộng thành năng lực tổng quát hoặc kết luận nhân quả.
- **Tái hiện:** chạy lệnh read-only ở cuối tài liệu, xem `colors/winner`, `end` và `matched pairs`.
- **Đề xuất:** giữ checkpoint, code/config và log bất biến; khi có phép đánh giá mới, dùng protocol định trước, paired openings và uncertainty theo opening cluster. Không tăng hoặc sweep budget 200 simulations.

### P1 — Không thể tái lập checkpoint lịch sử bằng `latest.pt` hiện tại

- **Liên quan:** `experiments.config_json`, `agents.checkpoint_sha256`; `checkpoints/latest.pt`.
- **Tác động:** cùng đường dẫn nhưng khác hash; dùng file hiện tại để tái chạy sẽ đánh giá model khác, không phải checkpoint của 400 ván.
- **Độ tin cậy:** cao tại thời điểm đọc; file `latest.pt` có thể tiếp tục thay đổi. Commit lịch sử có dirty tree nên hash checkpoint đơn lẻ vẫn không xác định toàn bộ môi trường/code.
- **Tái hiện:** lệnh cuối tài liệu in hash từ config, agent và file hiện tại.
- **Đề xuất:** tìm và lưu artifact có đúng SHA-256 cùng dirty diff và metadata môi trường; nếu không có, ghi rõ không thể tái lập chính xác thay vì thay bằng checkpoint hiện tại.

### P2 — Dấu hiệu chiến thuật một ply trong mẫu thua

- **Liên quan:** `games.payload_json` (`moves`, `opening_plies`); luật Normal `SPEC.md:53-55`; phép replay trong lệnh cuối tài liệu.
- **Tác động:** 3/20 ván có lượt AZ không chặn đe dọa thắng ngay duy nhất khi không có nước thắng ngay; 17/20 gặp nhiều ô thắng ngay của đối thủ. Đây là tín hiệu để điều tra vị trí, không phải tỷ lệ trên toàn bộ 354 ván.
- **Độ tin cậy:** cao cho phân loại hình học một ply của mẫu đã chọn, thấp cho suy luận nguyên nhân hoặc khả năng cứu ván. Cả 3 vị trí bỏ chặn có thể đã thua do chiến thuật sâu hơn dù thực hiện nước chặn; không kết luận nguyên nhân gốc hoặc bug P0.
- **Tái hiện:** lệnh cuối tài liệu in `sample` theo game ID và `sample counts`.
- **Đề xuất:** xem lại từng position và các nước dẫn tới nó bằng phân tích chiến thuật sâu hơn trên checkpoint lịch sử đúng hash; không xem cờ một ply như chứng minh lỗi MCTS hay sửa thuật toán từ mẫu này.

### P2 — So sánh 15k và 25k bị nhiễu

- **Liên quan:** `experiments.config_json`, `agents.checkpoint_sha256`, và logic paired scheduling `scripts/tournament.py:437-468`.
- **Tác động:** chênh lệch score giữa hai mốc không tách được ảnh hưởng checkpoint khỏi backend, seed và cỡ mẫu.
- **Độ tin cậy:** cao đối với khác biệt metadata ghi trong DB; chưa biết mức tác động thực tế của từng yếu tố.
- **Tái hiện:** lệnh cuối tài liệu in config và kết quả cả hai experiment.
- **Đề xuất:** nếu cần đánh giá tiến bộ training, đối chiếu hai checkpoint bất biến dưới cùng code/backend, seed/opening và protocol định trước; báo riêng uncertainty theo cặp.

## Replay chiến thuật một ply

Chọn **20 ván thua đầu tiên theo local_game_id số tăng dần** trong 354 ván thua, không chọn ngẫu nhiên; tái dựng từng ô 11×11 từ `moves`, bỏ qua các lượt thuộc bốn ply opening. Ở mỗi lượt AZ, liệt kê tất cả ô trống mà nếu đặt quân hiện tại sẽ tạo đường ≥5 liên tiếp theo 4 hướng (luật Normal, `SPEC.md:53-55`). Nếu AZ không chọn một trong các ô thắng ngay, đánh dấu bỏ lỡ thắng ngay. Nếu AZ không có nước thắng ngay, đối thủ có đúng một ô thắng ngay và AZ không chiếm ô đó, đánh dấu bỏ lỡ chặn đơn. Nếu đối thủ có từ hai ô thắng ngay và AZ không có thắng ngay, đánh dấu đa đe dọa; chặn một ô không đủ chặn mọi nước thắng ngay. Mỗi cờ đếm tối đa một lần/ván, có thể trùng nhau trong tổng quát; không xét sâu hơn một ply.

| Cờ trên 20 ván | Số ván | local_game_id |
|---|---:|---|
| Bỏ lỡ thắng ngay | 0 | — |
| Bỏ lỡ chặn đơn, không có thắng ngay | 3 | 2, 12, 15 |
| Đối thủ có ≥2 ô thắng ngay, AZ không có thắng ngay | 17 | 0, 1, 4–11, 13, 16–19, 21, 22 |

Các cờ mô tả vị trí đã gặp trên trajectory, **không chứng minh** nguyên nhân gốc, thời điểm sai lầm đầu tiên, khả năng cứu ván trước đó hay phân bố của toàn bộ 354 ván. Mẫu đầu theo ID có thể thiên lệch theo seed/màu; không dùng làm tỷ lệ tổng thể. Phép kiểm hình học chỉ áp dụng Normal, không kiểm forbidden Renju; chỉ khẳng định ô không bị chiếm, không kiểm lại terminal/legal của toàn trajectory hoặc search policy/value của checkpoint lịch sử.

## Lệnh tái lập đọc dữ liệu

Chạy từ root repo trong bash; chỉ mở SQLite read-only, không tạo hay sửa artifacts. Cần DB local còn đầy đủ cả WAL nếu có. Lệnh dùng `.venv/Scripts/python.exe` trên workstation Windows; thay interpreter nếu cần. Lệnh này in metadata, W/D/L theo màu, số cặp và phân loại 20 ván đầu; `raw_moves` là số bản ghi telemetry trong payload. File checkpoint là file có thể thay đổi, nên hash in ra phụ thuộc thời điểm đọc.

```bash
.venv/Scripts/python.exe - <<'PY'
import collections, hashlib, json, sqlite3
c = sqlite3.connect('file:data/tournament.sqlite?mode=ro', uri=True)
c.row_factory = sqlite3.Row
ids = ['az_v0-15000-continuation-20261119-promotion-500-heuristic-0',
       'az-15k-vs-heuristic-20261119-az_v0']

def wins(board, side):
    out = []
    for a in range(121):
        if board[a]:
            continue
        r, col = divmod(a, 11)
        for dr, dc in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            n = 1
            for s in (-1, 1):
                rr, cc = r + s*dr, col + s*dc
                while 0 <= rr < 11 and 0 <= cc < 11 and board[rr*11+cc] == side:
                    n += 1
                    rr, cc = rr + s*dr, cc + s*dc
            if n >= 5:
                out.append(a)
                break
    return out

for exp in ids:
    config = json.loads(c.execute('select config_json from experiments where experiment_id=?', (exp,)).fetchone()[0])
    gs = list(c.execute('''select g.local_game_id,g.winner,g.payload_json,
        b.label black,w.label white from games g
        join agents b on g.black_agent_id=b.agent_id
        join agents w on g.white_agent_id=w.agent_id
        where g.experiment_id=?''', (exp,)))
    gs.sort(key=lambda x: int(x['local_game_id']))
    payloads = [json.loads(g['payload_json']) for g in gs]
    print('\n', exp, '\nconfig:', config)
    print('checkpoint agents:', list(map(tuple, c.execute('''select distinct a.label,a.checkpoint_sha256
        from games g join agents a on a.agent_id in (g.black_agent_id,g.white_agent_id)
        where g.experiment_id=?''', (exp,)))))
    print('colors/winner:', collections.Counter((g['black'], g['white'], g['winner']) for g in gs))
    print('end:', collections.Counter(p['end_reason'] for p in payloads),
          'opening:', collections.Counter(p['opening_plies'] for p in payloads),
          'raw_moves:', sum(len(p.get('raw_moves', [])) for p in payloads))
    pairs = {p['pair_game']: p for p in payloads}
    valid = sum(pairs[i]['moves'][:4] == pairs[i+1]['moves'][:4]
        and pairs[i]['seeds']['opening'] == pairs[i+1]['seeds']['opening']
        and pairs[i]['seeds']['black'] == pairs[i+1]['seeds']['white']
        and pairs[i]['seeds']['white'] == pairs[i+1]['seeds']['black']
        for i in range(0, len(payloads), 2) if i in pairs and i+1 in pairs)
    print('matched pairs:', valid)
    losses = [(g, p) for g, p in zip(gs, payloads)
        if ('az_v0' in g['black'] and g['winner'] == 'white')
        or ('az_v0' in g['white'] and g['winner'] == 'black')]
    print('losses:', len(losses))
    if exp != ids[0]:
        continue
    counts = collections.Counter()
    for g, p in losses[:20]:
        board = [0]*121
        az = 1 if 'az_v0' in g['black'] else 2
        flags = set()
        for i, action in enumerate(p['moves']):
            if i % 2 + 1 == az and i >= p['opening_plies']:
                own, opponent = wins(board, az), wins(board, 3-az)
                if own and action not in own:
                    flags.add('missed_win')
                if not own and len(opponent) == 1 and action != opponent[0]:
                    flags.add('missed_single_block')
                if not own and len(opponent) > 1:
                    flags.add('multiple_threats')
            assert not board[action]
            board[action] = i % 2 + 1
        counts.update(flags)
        print('sample', g['local_game_id'], 'black' if az == 1 else 'white', sorted(flags))
    print('sample counts:', counts)
print('current checkpoint sha256:', hashlib.sha256(open('checkpoints/latest.pt','rb').read()).hexdigest())
PY
```

Không có tournament mới, build, lint hoặc full tests trong phân tích này; DB, checkpoint và logs lịch sử không bị thay đổi có chủ ý. Hash/metadata và conclusions tùy theo snapshot local; không suy diễn rằng checkpoint `latest.pt` hiện tại tương ứng experiment.
