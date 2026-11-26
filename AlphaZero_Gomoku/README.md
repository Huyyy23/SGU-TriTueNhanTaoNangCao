# Gomoku 11×11 với AlphaZero

## Clone repository

```powershell
git clone https://github.com/Huyyy23/SGU-TriTueNhanTaoNangCao.git
cd SGU-TriTueNhanTaoNangCao
```

Cài môi trường ảo và dependency trong thư mục gốc repository. Sau đó vào `AlphaZero_Gomoku` để chạy game. Trên Windows, đặt `$python = "..\\.venv\\Scripts\\python.exe"` và gọi `& $python`; trên macOS/Linux dùng `../.venv/bin/python`.

Đồ án xây dựng trò Gomoku trên bàn 11×11 với luật Normal và Renju, giao diện Pygame, CLI, các tác tử tìm kiếm cổ điển và mạng AlphaZero. Hướng dẫn này giúp giảng viên cài đặt, mở trò chơi bằng checkpoint có sẵn và chạy các kiểm tra nhỏ mà không huấn luyện lại.

## Cài đặt

Cài Python 3.10. Python 3.10.6 là môi trường đã dùng cho các thí nghiệm. Tạo môi trường ảo tại thư mục gốc repository và cài dependency của Gomoku:

```powershell
py -3.10 -m venv .venv
.venv\\Scripts\\python -m pip install -r AlphaZero_Gomoku\\requirements.txt
cd AlphaZero_Gomoku
$python = "..\\.venv\\Scripts\\python.exe"
```

Trên macOS hoặc Linux, chạy `python3.10 -m venv .venv` ở thư mục gốc, sau đó `.venv/bin/python -m pip install -r AlphaZero_Gomoku/requirements.txt` và `cd AlphaZero_Gomoku`. Gọi Python bằng `../.venv/bin/python`. PyTorch cài bản CPU từ PyPI. GPU NVIDIA là tuỳ chọn và cần bản PyTorch CUDA tương ứng. GUI cần máy có màn hình.


## Chơi trò chơi

Khởi động GUI với người chơi cầm Đen và AlphaZero cầm Trắng theo luật Normal. GUI tự tìm checkpoint Normal và Renju trong clone:

```powershell
& $python -m ui.pygame_app --mode human-vs-ai --human-color black --agent az --ruleset normal
```

Nếu cần chọn weights tường minh, chạy từ thư mục `AlphaZero_Gomoku`:

```powershell
& $python -m ui.pygame_app --mode human-vs-ai --human-color black --agent az --ruleset renju --checkpoint checkpoints/renju/az_renju_weights.pt
```

Chọn Renju và tác tử bằng các cờ `--ruleset` và `--agent`. Tác tử có sẵn gồm `random`, `heuristic`, `alphabeta`, `mcts-pure`, `az`, `az_v0`, `az_v1_tactical`, và `az_v1_threat`. AZ dùng checkpoint theo luật đã chọn. GUI không nhận checkpoint thì báo lỗi nếu thiếu weights phù hợp.

Chạy hai máy đấu nhau trong GUI:

```powershell
& $python -m ui.pygame_app --mode ai-vs-ai --ruleset normal --black-agent az --white-agent heuristic
```

Chơi qua terminal với người đấu AI theo luật Normal:

```powershell
& $python scripts/play.py --mode human-vs-ai --human-color black --agent az --ruleset normal
```

CLI nhận luật và tác tử riêng cho hai bên. Renju GUI dùng checkpoint Renju; terminal AZ cần truyền weights Renju rõ ràng:

```powershell
& $python scripts/play.py --mode ai-vs-ai --black-agent az --black-checkpoint checkpoints/renju/az_renju_weights.pt --white-agent heuristic --ruleset renju
```

Đổi màu người chơi bằng `--human-color black` hoặc `--human-color white`. Dùng `--black-agent` và `--white-agent` để đặt tác tử riêng cho từng bên. Với AlphaBeta, `--depth` đặt độ sâu. Với MCTS Pure, `--rollouts` đặt số rollout. Các nước cấm Renju được kiểm tra qua bộ luật, không phụ thuộc tác tử.

## Checkpoint có sẵn

Repository cung cấp weights Normal tại `models/az_normal_weights.pt` và weights Renju tại `checkpoints/renju/az_renju_weights.pt`. Các file này được dùng khi chơi; bạn không cần huấn luyện lại. Checkpoint đầy đủ để tiếp tục huấn luyện và replay buffer nằm trong `checkpoints/normal/`, nhưng không được đóng gói trong Git clone. Không dùng đường dẫn checkpoint training trong hướng dẫn chơi.

Có thể truyền weights khác cho CLI bằng `--checkpoint PATH`, hoặc đặt riêng bằng `--black-checkpoint PATH` và `--white-checkpoint PATH`. Để kiểm tra hash của weights mặc định:

```powershell
& $python -c "import hashlib; from pathlib import Path; [print(p, hashlib.sha256(Path(p).read_bytes()).hexdigest()) for p in ['models/az_normal_weights.pt', 'checkpoints/renju/az_renju_weights.pt']]"
```

## Kiểm thử và thử nghiệm nhỏ

Chạy toàn bộ test suite:

```powershell
& $python -m pytest -q
```

Kiểm tra rules, model, UI và tác tử riêng lẻ:

```powershell
& $python -m pytest -q tests/test_rules_normal.py tests/test_rules_renju.py tests/test_model.py tests/test_ui.py tests/test_agent_factory.py
```

Chạy self-play thử nghiệm với config tạm, output checkpoint tạm và SQLite riêng. Lệnh huấn luyện ghi các file đó, nên không chạy nó trên dữ liệu thí nghiệm chính:

```powershell
$smoke = [guid]::NewGuid().ToString('N')
& $python scripts/train.py --config configs/normal.yaml --fresh --iterations 1 --games 2 --simulations 2 --backend python --out-dir "checkpoints/smoke-$smoke" --db "data/smoke-$smoke.sqlite" --run-id "smoke-$smoke"
```

Chạy tournament hai ván trong SQLite riêng:

```powershell
$smoke = [guid]::NewGuid().ToString('N')
& $python scripts/tournament.py --agents random,heuristic --games 2 --budget sims:1 --backend python --db "data/tournament-smoke-$smoke.sqlite"
```

Mỗi lệnh dùng tên duy nhất để không ghi đè kết quả thí nghiệm đã có. Lệnh self-play minh họa huấn luyện một model tạm và cần mạng khởi tạo; nó không phải đường chạy game mặc định.


## Backend C++ tuỳ chọn

Python là backend mặc định an toàn nhất cho cài đặt mới. C++ là tuỳ chọn, yêu cầu compiler và extension được build cho đúng Python ABI. Để build, chạy `& $python scripts/build_gomoku_native.py --module mcts` hoặc `--module patterns`. Không cần C++ để chạy trò chơi, luật, tests Python, self-play hoặc tournament. Các cờ `--backend python` chọn Python tường minh cho train và tournament.

## Dữ liệu thực nghiệm

`experiment_results.csv` là bảng thống nhất, mỗi dòng mô tả một kết quả hoặc chỉ số, có nguồn, mục đích, kết luận và ghi chú. File hiện có 116 dòng dữ liệu mã hóa UTF-8 BOM để mở bằng Excel. W/D/L là thắng, hòa, thua của tác tử A. `score_pct` tính `(thắng + 0.5 × hòa) / số ván × 100`, khác với tỷ lệ thắng. Ô trống nghĩa là số liệu chưa được công bố hoặc không có trong nguồn.

Mỗi dòng trỏ đến `source_file` và `report_section`. T35 chạy lại có đối chiếu trực tiếp W/D/L với từng game trong `data/tournament.sqlite`; chỉ số Normal/Renju được đối chiếu với SQLite và log tương ứng. T35 ban đầu được ghi theo bảng báo cáo vì SQLite local không chứa run ngày 14/11. Các đánh giá Renju và tactical được ghi theo audit khi không có game-level DB. SQLite và audit trong `.audit/` bị Git ignore, nên clone mới có thể không có các nguồn cục bộ đó. Báo cáo chính thức nằm trong `report/` và `docs/`.

## Tài liệu kỹ thuật

- [SPEC.md](SPEC.md) ghi hợp đồng kỹ thuật, trạng thái kiểm chứng và giới hạn của các kết luận.
- [Báo cáo đồ án](docs/report.md) mô tả thuật toán, kết quả và nguồn checkpoint.
- [Hướng dẫn tournament](docs/tournament_guide.md) mô tả ngân sách, lưu log và tiếp tục tournament.

