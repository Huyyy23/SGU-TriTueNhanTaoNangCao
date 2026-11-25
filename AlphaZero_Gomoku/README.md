# Gomoku 11×11 với AlphaZero

## Clone repository

```powershell
git clone https://github.com/Huyyy23/SGU-TriTueNhanTaoNangCao.git
cd SGU-TriTueNhanTaoNangCao
```

Tạo môi trường ảo và cài dependency từ thư mục gốc repository. Các lệnh còn lại chạy từ `AlphaZero_Gomoku`, dùng `..\.venv\Scripts\python` trên Windows hoặc `../.venv/bin/python` trên macOS và Linux.

Đồ án xây dựng trò Gomoku trên bàn 11×11 với luật Normal và Renju, giao diện Pygame, CLI, các tác tử tìm kiếm cổ điển và mạng AlphaZero. Hướng dẫn này giúp giảng viên cài đặt, mở trò chơi bằng checkpoint có sẵn và chạy các kiểm tra nhỏ mà không huấn luyện lại.

## Cài đặt

Cài Python 3.10. Python 3.10.6 là môi trường đã dùng cho các thí nghiệm của đồ án. Tạo môi trường ảo trong thư mục Gomoku và cài dependency từ `AlphaZero_Gomoku/requirements.txt`:

```powershell
py -3.10 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
cd AlphaZero_Gomoku
```

Trên macOS hoặc Linux, dùng `python3.10 -m venv .venv` tại thư mục gốc repository, sau đó chạy `.venv/bin/python -m pip install -r requirements.txt` và `cd AlphaZero_Gomoku`. PyTorch cài mặc định bản CPU từ PyPI. GPU NVIDIA là tuỳ chọn và cần cài bản PyTorch CUDA tương ứng với máy. Đồ hoạ Pygame cần môi trường desktop có màn hình.


## Chơi trò chơi

Khởi động GUI với người chơi cầm Đen và AlphaZero cầm Trắng theo luật Normal. Tuỳ chọn checkpoint khiến đường dẫn weights được nêu rõ trong lệnh:

```powershell
.venv\Scripts\python -m ui.pygame_app --mode human-vs-ai --human-color black --agent az --ruleset normal --checkpoint AlphaZero_Gomoku/models/az_normal_weights.pt
```

Chọn Renju và tác tử bằng các cờ `--ruleset` và `--agent`. Tác tử có sẵn gồm `random`, `heuristic`, `alphabeta`, `mcts-pure`, `az`, `az_v0`, `az_v1_tactical`, và `az_v1_threat`. AZ dùng checkpoint theo luật đã chọn. GUI không nhận checkpoint thì báo lỗi nếu thiếu weights phù hợp.

Chạy hai máy đấu nhau trong GUI:

```powershell
.venv\Scripts\python -m ui.pygame_app --mode ai-vs-ai --ruleset normal --black-agent az --white-agent heuristic
```

Chơi qua terminal:

```powershell
.venv\Scripts\python scripts/play.py --mode human-vs-ai --human-color black --agent az --ruleset normal
.venv\Scripts\python scripts/play.py --mode ai-vs-ai --black-agent az --white-agent heuristic --ruleset renju
```

Đổi màu người chơi bằng `--human-color black` hoặc `--human-color white`. Dùng `--black-agent` và `--white-agent` để đặt tác tử riêng cho từng bên. Với AlphaBeta, `--depth` đặt độ sâu. Với MCTS Pure, `--rollouts` đặt số rollout. Các nước cấm Renju được kiểm tra qua bộ luật, không phụ thuộc tác tử.

## Checkpoint có sẵn

Repository cung cấp weights Normal tại `models/az_normal_weights.pt` và weights Renju tại `checkpoints/renju/az_renju_weights.pt`. Các file này được dùng khi chơi; bạn không cần huấn luyện lại. Checkpoint đầy đủ để tiếp tục huấn luyện và replay buffer nằm trong `checkpoints/normal/`, nhưng không được đóng gói trong Git clone. Không dùng đường dẫn checkpoint training trong hướng dẫn chơi.

Có thể truyền weights khác cho CLI bằng `--checkpoint PATH`, hoặc đặt riêng bằng `--black-checkpoint PATH` và `--white-checkpoint PATH`. Để kiểm tra hash của weights mặc định:

```powershell
.venv\Scripts\python -c "import hashlib; from pathlib import Path; [print(p, hashlib.sha256(Path(p).read_bytes()).hexdigest()) for p in ['models/az_normal_weights.pt', 'checkpoints/renju/az_renju_weights.pt']]"
```

## Kiểm thử và thử nghiệm nhỏ

Chạy toàn bộ test suite:

```powershell
.venv\Scripts\python -m pytest -q
```

Kiểm tra rules, model, UI và tác tử riêng lẻ:

```powershell
.venv\Scripts\python -m pytest -q tests/test_rules_normal.py tests/test_rules_renju.py tests/test_model.py tests/test_ui.py tests/test_agent_factory.py
```

Chạy self-play thử nghiệm với config tạm, output checkpoint tạm và SQLite riêng. Lệnh huấn luyện ghi các file đó, nên không chạy nó trên dữ liệu thí nghiệm chính:

```powershell
$smoke = [guid]::NewGuid().ToString('N')
.venv\Scripts\python scripts/train.py --config configs/normal.yaml --fresh --iterations 1 --games 2 --simulations 2 --backend python --out-dir "checkpoints/smoke-$smoke" --db "data/smoke-$smoke.sqlite" --run-id "smoke-$smoke"
```

Chạy tournament hai ván trong SQLite riêng:

```powershell
$smoke = [guid]::NewGuid().ToString('N')
.venv\Scripts\python scripts/tournament.py --agents random,heuristic --games 2 --budget sims:1 --backend python --db "data/tournament-smoke-$smoke.sqlite"
```

Mỗi lệnh dùng tên duy nhất để không ghi đè kết quả thí nghiệm đã có.

## Backend C++ tuỳ chọn

Python là backend mặc định an toàn nhất cho cài đặt mới. C++ là tuỳ chọn, yêu cầu compiler và extension được build cho đúng Python ABI. Để build, chạy `.venv\Scripts\python scripts/build_gomoku_native.py --module mcts` hoặc `--module patterns`. Không cần C++ để chạy trò chơi, luật, tests Python, self-play hoặc tournament. Các cờ `--backend python` chọn Python tường minh cho train và tournament.

## Dữ liệu thực nghiệm

`experiment_results.csv` là bảng thống nhất, mỗi dòng mô tả một kết quả hoặc một chỉ số, có cột nguồn, mục đích, kết luận và ghi chú. W/D/L là thắng, hoà, thua của tác tử A. `score_pct` tính `(thắng + 0.5 × hoà) / số ván × 100`; đây không phải win rate. Ô trống nghĩa là số liệu không được xác định hoặc không được công bố.

CSV tổng hợp số liệu đã nêu trong báo cáo và ghi rõ các mục thiếu hoặc chưa kiểm chứng độc lập. Dữ liệu SQLite trong `data/` bị Git ignore, nên có thể không có trong clone mới. Để truy vết, dùng đường dẫn ở `source_file` cùng `report_section`; các file báo cáo chính thức nằm trong `report/` và `docs/`. Các số liệu được chép từ báo cáo không đồng nghĩa với việc cơ sở dữ liệu tương ứng có trong bản clone.

## Tài liệu kỹ thuật

- [SPEC.md](SPEC.md) ghi hợp đồng kỹ thuật, trạng thái kiểm chứng và giới hạn của các kết luận.
- [Báo cáo đồ án](docs/report.md) mô tả thuật toán, kết quả và nguồn checkpoint.
- [Hướng dẫn tournament](docs/tournament_guide.md) mô tả ngân sách, lưu log và tiếp tục tournament.

