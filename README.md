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

Các lab dùng Python 3.12 và một `.venv` ở thư mục gốc. Tạo môi trường, cài dependency một lần:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Trong PyCharm, chọn interpreter `.venv\Scripts\python.exe` tại thư mục repo. Mở JupyterLab từ thư mục lab tương ứng để giữ đúng đường dẫn dữ liệu:

```powershell
cd "lab02 - Searching"
..\.venv\Scripts\jupyter-lab
```

Đồ án Gomoku có hướng dẫn cài đặt và chơi riêng tại [`AlphaZero_Gomoku/README.md`](AlphaZero_Gomoku/README.md). Các thí nghiệm Gomoku đã dùng Python 3.10.6; có thể tạo môi trường 3.10 riêng để không thay đổi môi trường Python 3.12 của các lab.

## Đồ án Gomoku AlphaZero

Đồ án xây dựng Gomoku 11×11 với luật Normal và Renju, giao diện Pygame, các tác tử tìm kiếm cổ điển và AlphaZero học từ self-play.

- [Báo cáo đồ án](AlphaZero_Gomoku/report/report.md) và [đặc tả kỹ thuật](AlphaZero_Gomoku/SPEC.md).
- [Kết quả thực nghiệm](AlphaZero_Gomoku/experiment_results.csv): bảng tóm tắt dùng trong báo cáo.
- [Chi tiết thực nghiệm](AlphaZero_Gomoku/working_details.csv): số liệu theo run, nguồn và cảnh báo bản sao. Hai bảng dùng UTF-8 BOM; tỷ lệ được ghi theo thang 0–100%.

Trong lần chạy lại T35 Renju với ngân sách cố định, Heuristic và AZ Renju đồng hạng 83,33% điểm, mỗi tác tử đạt 250/300 điểm trong pool. Đây là kết quả của giao thức đó, không phải thứ hạng chung cho mọi điều kiện.

Các SQLite, log và checkpoint huấn luyện được lưu cục bộ, không có trong Git clone. Các báo cáo trong repository giữ kết quả và mô tả nguồn; một số thí nghiệm lịch sử không còn SQLite để kiểm tra từng ván. Không cộng các bản sao hoặc các nhánh huấn luyện độc lập.
