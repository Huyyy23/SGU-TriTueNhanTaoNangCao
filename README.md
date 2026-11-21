# 📚 Bài tập nhóm môn Trí Tuệ Nhân Tạo Nâng Cao

> **Giảng viên hướng dẫn: Đỗ Như Tài**
>
> **Lớp: DNT1241**

## 👥 Thành viên nhóm

| STT | Họ và tên | MSSV |
|:---:|:----------|:----:|
| 1 | Nguyễn Hoàng Huy | 3124720028 |
| 2 | Mai Thu Hường | 3124720031 |
| 3 | Mai Phúc Khang | 3124720032 |

## 📂 Nội dung repo

| Thư mục | Nội dung | Notebook chính |
|---|---|---|
| `lab01 - Agent` | Tác tử phản xạ, Gridhunt, Lunar Lander | `robot_vacuum.ipynb`, `gridhunt_mini_assignment.ipynb`, `lunar_lander.ipynb` |
| `lab02 - Searching` | BFS, DFS, A* trên mê cung | `01_trees.ipynb`, `04_Maze_BFS_vs_A_Star.ipynb`, `Ex1_Maze.ipynb` |
| `lab03 - local search` | Hill Climbing, Simulated Annealing, N-Queens, TSP | `Maze_Example_LocalSearch.ipynb`, `n_queens.ipynb`, `traveling_salesman_problem.ipynb` |
| `lab04 - csp` | CSP, backtracking, tô màu đồ thị | `CSP_graph_coloring_example.ipynb`, `n_queens.ipynb` |
| `AlphaZero_Gomoku` | Đồ án cuối môn: Gomoku 11x11 tự học bằng AlphaZero | `AlphaZero_Gomoku/README.md` |

Mỗi lab có `README.md` riêng liệt kê bài tập. `lab01/README.md` ghi nguồn tài liệu gốc: Michael Hahsler, giấy phép CC BY-SA 4.0.

## Môi trường Python dùng chung

Toàn bộ labs và `AlphaZero_Gomoku` dùng Python 3.12 và một `.venv` ở thư mục gốc. Tạo môi trường, cài dependency một lần:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Trong PyCharm, chọn interpreter `.venv\Scripts\python.exe` tại thư mục repo. Mở JupyterLab từ thư mục lab tương ứng để giữ đúng đường dẫn dữ liệu:

```powershell
cd "lab02 - Searching"
..\.venv\Scripts\jupyter-lab
```

Chạy Gomoku bằng cùng interpreter:

```powershell
cd AlphaZero_Gomoku
..\.venv\Scripts\python -m pytest
..\.venv\Scripts\python -m ui.pygame_app
```

Các Python khác trên máy không bị gỡ; `.venv` cố định phiên bản dùng trong repo.
Máy có GPU NVIDIA thì cài bản CUDA theo hướng dẫn trong `AlphaZero_Gomoku/requirements.txt`.

## 🚧 Tiến độ đồ án AlphaZero

Cập nhật 2026-10-28. Test hiện tại: `71 passed, 2 skipped`.

| Nhóm | Xong | Còn lại |
|---|---|---|
| Luật + bàn cờ + người chơi (A) | 3/8 | `gomoku/runner.py`, `players/random.py`, `human.py`, `mcts_pure`, UI, luật Renju, train Renju |
| Tầng AlphaZero (B) | 6/7 | gom lô 16 ván, log từng ván cho self-play, chạy 20 × 50 ván |
| Đánh giá + hạ tầng (C) | 2/11 | `az/evaluator.py`, `scripts/arena.py`, Colab, shard, `simulate.py`, tài liệu, báo cáo |

Năm giai đoạn của `az/`: GĐ1 mạng 6×64, GĐ2 PUCT, GĐ3 self-play, GĐ4 trainer + `--resume` + `AZPlayer` — xong, có test. GĐ5 (gom lô, log ván, chạy 1000 ván) chưa làm.

Ba chỗ đang chặn nghiệm thu:

1. `gomoku/runner.py` và `players/random.py` chưa có, nên `AZPlayer` mới được kiểm bằng test cục bộ.
2. `gomoku/rules/renju.py` chưa có, nên `scripts/train.py` dừng khi `rule_set: renju`.
3. `az/evaluator.py` chưa có, nên cổng arena (thắng `best` ≥ 55%) chưa chạy được.

Cổng GĐ4 (2 vòng × 50 ván × 200 lượt) chưa chạy. Số đo hiện có là 4 vòng × 10 ván × 100 lượt, ghi ở `AlphaZero_Gomoku/docs/benchmarks.md`.
