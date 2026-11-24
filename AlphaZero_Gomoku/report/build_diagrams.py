from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).resolve().parent
COLORS = {
    "interface": "#E8F0FE",
    "search": "#E8F5E9",
    "data": "#FFF3E0",
    "optional": "#F3E5F5",
    "edge": "#40536B",
    "text": "#172B4D",
}


def box(ax, xy, wh, text, color="interface", fontsize=10, linestyle="solid"):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.025,rounding_size=0.12",
        linewidth=1.25,
        edgecolor=COLORS["edge"],
        facecolor=COLORS[color],
        linestyle=linestyle,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=COLORS["text"], fontsize=fontsize, linespacing=1.25, wrap=True)


def arrow(ax, start, end, label=None, *, dashed=False, curve=0.0, label_offset=(0, 0)):
    patch = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=12, linewidth=1.25,
        color=COLORS["edge"], linestyle="dashed" if dashed else "solid",
        connectionstyle=f"arc3,rad={curve}", shrinkA=4, shrinkB=4,
    )
    ax.add_patch(patch)
    if label:
        mx = (start[0] + end[0]) / 2 + label_offset[0]
        my = (start[1] + end[1]) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center", fontsize=8,
                color=COLORS["text"], bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5})


def base_figure(title, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8)
    ax.axis("off")
    fig.suptitle(title, fontsize=17, fontweight="bold", color=COLORS["text"], y=0.97)
    return fig, ax


def architecture():
    fig, ax = base_figure("Kiến trúc hệ thống Gomoku AlphaZero", (13, 8))
    # Game layer
    game = [
        (0.35, "Giao diện\nPygame / CLI"),
        (2.9, "Điều phối ván"),
        (5.45, "Luật\nNormal / Renju"),
        (8.0, "Bàn cờ\ntrạng thái + lịch sử"),
        (10.55, "Tác tử\nAZ / heuristic / khác"),
    ]
    for x, label in game:
        box(ax, (x, 6.25), (2.1, 0.9), label, fontsize=9)
    for (x, _), (next_x, _) in zip(game, game[1:]):
        arrow(ax, (x + 2.12, 6.7), (next_x - 0.03, 6.7))

    # AlphaZero search path
    search = [
        (0.55, "board_to_planes\n4 × 11 × 11"),
        (3.55, "AZNet\nchính sách + giá trị"),
        (6.55, "MCTS / PUCT\nmặt nạ · mở rộng · truyền ngược"),
        (9.55, "AZPlayer\nphân phối lượt thăm"),
    ]
    for x, label in search:
        box(ax, (x, 4.35), (2.4, 0.95), label, color="search", fontsize=9)
    for (x, _), (next_x, _) in zip(search, search[1:]):
        arrow(ax, (x + 2.42, 4.82), (next_x - 0.03, 4.82))
    arrow(ax, (9.85, 5.28), (9.85, 6.22), "nước đi")
    ax.text(0.65, 5.55, "Luật và bàn cờ cung cấp trạng thái cùng mặt nạ hợp lệ cho tìm kiếm.",
            fontsize=8.5, color=COLORS["text"])

    # Training and persistence
    training = [
        (0.4, "Tự chơi\nván và mẫu"),
        (3.0, "ReplayBuffer\n(features, π, z)"),
        (5.6, "Trainer\nloss chính sách + giá trị"),
        (8.2, "Checkpoint\ntrọng số + trạng thái"),
        (10.8, "SQLite\ntự chơi / huấn luyện"),
    ]
    for x, label in training:
        box(ax, (x, 1.55), (2.0, 0.9), label, color="data", fontsize=8.5)
    for (x, _), (next_x, _) in zip(training[:4], training[1:4]):
        arrow(ax, (x + 2.02, 2.0), (next_x - 0.03, 2.0))
    arrow(ax, (10.22, 2.0), (10.77, 2.0))
    arrow(ax, (9.2, 1.55), (1.4, 1.55), "checkpoint dùng cho tự chơi", curve=-0.2,
          label_offset=(0, -0.2))

    box(ax, (7.0, 0.35), (2.0, 0.75), "Đấu giải\nso sánh tác tử", color="data", fontsize=8.5)
    box(ax, (10.8, 0.35), (2.0, 0.75), "SQLite\ntournament", color="data", fontsize=8.5)
    arrow(ax, (9.02, 0.72), (10.77, 0.72))
    box(ax, (4.0, 3.0), (2.0, 0.72), "Native MCTS (C++)\nbackend tùy chọn",
        color="optional", fontsize=8, linestyle="dashed")
    arrow(ax, (6.05, 3.36), (6.8, 4.32), "thay cơ chế tìm kiếm", dashed=True)
    ax.text(0.45, 0.55, "C++ chỉ thay tìm kiếm; luật, bàn, mã hóa và suy luận vẫn thuộc Python.",
            fontsize=8, color=COLORS["text"])
    fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.93))
    fig.savefig(OUT / "architecture.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def flow():
    fig, ax = plt.subplots(figsize=(13, 10))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 10)
    ax.axis("off")
    fig.suptitle("Luồng MCTS và sinh dữ liệu self-play", fontsize=17,
                 fontweight="bold", color=COLORS["text"], y=0.97)

    box(ax, (2.8, 8.9), (2.0, 0.8), "Chọn nhánh theo\nPUCT", color="search")
    box(ax, (5.5, 8.9), (2.0, 0.8), "Terminal?", color="search")
    box(ax, (9.2, 8.9), (2.0, 0.8), "Giá trị kết thúc\n(win / draw)", color="search")
    arrow(ax, (4.82, 9.3), (5.47, 9.3))
    arrow(ax, (7.52, 9.3), (9.17, 9.3), "có")

    box(ax, (5.5, 7.35), (2.0, 0.8), "Tạo features\n+ legal mask", color="search")
    box(ax, (5.5, 6.2), (2.0, 0.8), "Inference\npolicy + value", color="search")
    box(ax, (5.5, 5.05), (2.0, 0.8), "Mask policy\nnước hợp lệ", color="search")
    box(ax, (5.5, 3.9), (2.0, 0.8), "Mở rộng node\nvới prior", color="search")
    box(ax, (5.5, 2.75), (2.0, 0.8), "Backup giá trị\nđổi dấu mỗi ply", color="search")
    box(ax, (5.5, 1.6), (2.0, 0.8), "Còn lượt\nmô phỏng?", color="search")
    arrow(ax, (6.5, 8.87), (6.5, 8.17), "không")
    arrow(ax, (6.5, 7.32), (6.5, 7.02))
    arrow(ax, (6.5, 6.17), (6.5, 5.87))
    arrow(ax, (6.5, 5.02), (6.5, 4.72))
    arrow(ax, (6.5, 3.87), (6.5, 3.57))
    arrow(ax, (6.5, 2.72), (6.5, 2.42))

    arrow(ax, (10.2, 8.87), (12.3, 8.87))
    arrow(ax, (12.3, 8.87), (12.3, 3.15))
    arrow(ax, (12.27, 3.15), (7.52, 3.15), "terminal")

    box(ax, (9.2, 1.6), (2.0, 0.8), "Phân phối lượt thăm\n→ chọn nước", color="search")
    arrow(ax, (7.52, 2.0), (9.17, 2.0), "hết")
    arrow(ax, (5.47, 2.0), (1.0, 2.0), "còn")
    arrow(ax, (1.0, 2.0), (1.0, 9.3))
    arrow(ax, (1.0, 9.3), (2.77, 9.3))

    box(ax, (0.4, 0.35), (2.0, 0.8), "Lưu vị trí trước nước\n(features, π, player)", color="data")
    box(ax, (3.3, 0.35), (2.0, 0.8), "Kết thúc ván\n(win / draw)", color="data")
    box(ax, (6.2, 0.35), (2.0, 0.8), "Gán z theo kết quả\n& người chơi đã lưu", color="data")
    box(ax, (9.1, 0.35), (2.0, 0.8), "ReplayBuffer\n(lưu mẫu huấn luyện)", color="data")
    arrow(ax, (2.42, 0.75), (3.27, 0.75))
    arrow(ax, (5.32, 0.75), (6.17, 0.75))
    arrow(ax, (8.22, 0.75), (9.07, 0.75))
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.93))
    fig.savefig(OUT / "flow.png", dpi=180, bbox_inches="tight")
    plt.close(fig)



def save(fig, name):
    fig.savefig(OUT / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_board(ax, black=(), white=(), candidate=None, title=None):
    ax.set_xlim(-0.6, 10.6)
    ax.set_ylim(10.6, -0.6)
    ax.set_aspect("equal")
    ax.set_xticks(range(11))
    ax.set_yticks(range(11))
    ax.grid(color="#75551E", linewidth=0.75)
    ax.set_facecolor("#E9B448")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for x in (2, 5, 8):
        for y in (2, 5, 8):
            ax.scatter(x, y, s=13, color="#75551E", zorder=2)
    if black:
        ax.scatter([c for r, c in black], [r for r, c in black], s=145,
                   color="#171717", edgecolors="black", linewidths=0.7, zorder=3)
    if white:
        ax.scatter([c for r, c in white], [r for r, c in white], s=145,
                   color="white", edgecolors="#6D6D6D", linewidths=0.8, zorder=3)
    if candidate:
        r, c = candidate
        ax.scatter(c, r, s=220, marker="x", color="#C53030", linewidths=2.5, zorder=4)
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", pad=7, color=COLORS["text"])
    for spine in ax.spines.values():
        spine.set_visible(False)

def gameplay_board():
    screenshot = plt.imread(OUT / "gomoku_gameplay.png")
    return screenshot[85:760, 12:690]


def tensor_planes():
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.4))
    fig.suptitle("Mã hóa một trạng thái ván thực tế thành tensor đầu vào (4 × 11 × 11)",
                 fontsize=15, fontweight="bold", color=COLORS["text"])
    board = gameplay_board()
    axes[0].imshow(board)
    axes[0].axis("off")
    axes[0].set_title("Bàn cờ trong ván thực tế", fontsize=9, fontweight="bold",
                      color=COLORS["text"], pad=7)
    black = ((4, 4),)
    white = ((5, 5),)
    channels = [
        ("Mặt phẳng 1 · bên đi", black),
        ("Mặt phẳng 2 · đối thủ", white),
        ("Mặt phẳng 3 · ô cấm", ()),
        ("Mặt phẳng 4 · màu lượt", None),
    ]
    for i, (title, stones) in enumerate(channels, 1):
        ax = axes[i]
        ax.set_xlim(-0.5, 10.5)
        ax.set_ylim(10.5, -0.5)
        ax.set_aspect("equal")
        ax.set_xticks(range(11))
        ax.set_yticks(range(11))
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.grid(color="#9A9A9A", linewidth=0.55)
        ax.set_facecolor("#FAFAFA")
        if stones:
            ax.scatter([c for r, c in stones], [r for r, c in stones],
                       s=92, color="#263238", zorder=3)
        elif stones is None:
            ax.imshow([[1] * 11] * 11, cmap="Blues", vmin=0, vmax=1, alpha=0.38,
                      extent=(-0.5, 10.5, 10.5, -0.5), zorder=0)
        value = "Đen = 1" if i in (1, 4) else "Trắng = 1" if i == 2 else "Toàn 0"
        ax.set_title(f"{title}\n{value}", fontsize=9, fontweight="bold",
                     color=COLORS["text"], pad=7)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.tight_layout(rect=(0.01, 0.02, 0.99, 0.88), w_pad=1.0)
    save(fig, "tensor_planes.png")


def augmentation_board():
    board = gameplay_board()
    transforms = [
        ("Gốc", lambda image: image),
        ("Quay 90°", lambda image: np.rot90(image)),
        ("Lật ngang", lambda image: np.fliplr(image)),
        ("Quay 90° rồi lật", lambda image: np.fliplr(np.rot90(image))),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
    fig.suptitle("Tăng cường dữ liệu: biến đổi đồng bộ bàn cờ và chính sách",
                 fontsize=14, fontweight="bold", color=COLORS["text"])
    for ax, (title, transform) in zip(axes, transforms):
        ax.imshow(transform(board))
        ax.axis("off")
        ax.set_title(title, fontsize=10, fontweight="bold", color=COLORS["text"])
    fig.text(0.5, 0.03,
             "Một trong 8 phép đối xứng D₄ được áp dụng đồng thời lên bàn cờ và π; nhãn kết quả z không đổi.",
             ha="center", fontsize=9, color=COLORS["text"])
    fig.tight_layout(rect=(0.01, 0.12, 0.99, 0.88), w_pad=1.0)
    save(fig, "augmentation_board.png")


def resnet_architecture():
    fig, ax = base_figure("AZNet: ResNet 6 khối, 64 bộ lọc", (13, 4.7))
    ax.set_ylim(0, 8)
    box(ax, (0.2, 3.15), (1.8, 1.15), "Tensor\n4 × 11 × 11", color="interface", fontsize=10)
    box(ax, (2.6, 3.15), (1.8, 1.15), "Stem\nConv 3×3 · 64", color="search", fontsize=10)
    box(ax, (5.0, 2.9), (2.3, 1.65), "6 khối phần dư\nmỗi khối: Conv–BN–ReLU\nConv–BN + nhánh tắt", color="search", fontsize=9)
    box(ax, (8.05, 4.45), (2.0, 1.0), "Policy head\n121 logits", color="data", fontsize=10)
    box(ax, (8.05, 1.8), (2.0, 1.0), "Value head\n1 giá trị tanh", color="data", fontsize=10)
    box(ax, (10.75, 4.45), (2.0, 1.0), "Chấm điểm\n121 ô", color="interface", fontsize=10)
    box(ax, (10.75, 1.8), (2.0, 1.0), "Giá trị\n[−1, +1]", color="interface", fontsize=10)
    arrow(ax, (2.02, 3.72), (2.57, 3.72))
    arrow(ax, (4.42, 3.72), (4.97, 3.72))
    arrow(ax, (7.32, 3.9), (8.02, 4.88))
    arrow(ax, (7.32, 3.45), (8.02, 2.3))
    arrow(ax, (10.08, 4.95), (10.72, 4.95))
    arrow(ax, (10.08, 2.3), (10.72, 2.3))
    ax.text(6.15, 1.05, "Mỗi khối học phần dư F(x); đầu ra = ReLU(F(x) + x).",
            ha="center", fontsize=9, color=COLORS["text"])
    fig.tight_layout(rect=(0.01, 0.05, 0.99, 0.92))
    save(fig, "resnet6x64.png")


def mcts_puct():
    fig, ax = base_figure("Một lượt mô phỏng MCTS dùng PUCT", (13, 4.8))
    stages = [
        (0.25, "Chọn nhánh\nPUCT"),
        (2.8, "Mở rộng\nnước hợp lệ"),
        (5.35, "Đánh giá lá\npolicy + value"),
        (7.9, "Backup\nđổi dấu"),
        (10.45, "Cập nhật lượt\nthăm gốc"),
    ]
    for x, label in stages:
        box(ax, (x, 3.55), (2.05, 1.2), label, color="search", fontsize=10)
    for (x, _), (nx, _) in zip(stages, stages[1:]):
        arrow(ax, (x + 2.08, 4.15), (nx - 0.03, 4.15))
    ax.text(6.5, 2.15,
            "score(s,a) = −Q(sₐ) + cₚᵤcₜ · P(s,a) · √max(1,N(s)) / (1 + N(sₐ))",
            ha="center", fontsize=11, color=COLORS["text"])
    arrow(ax, (11.5, 3.45), (1.15, 3.45), "lặp đến đủ ngân sách mô phỏng",
          curve=-0.05, label_offset=(0, -0.35))
    ax.text(6.5, 0.95,
            "Nút kết thúc được backup trực tiếp; giá trị nút con đổi dấu khi quy về góc nhìn nút cha.",
            ha="center", fontsize=9, color=COLORS["text"])
    fig.tight_layout(rect=(0.01, 0.03, 0.99, 0.92))
    save(fig, "mcts_puct.png")


def renju_board_rules():
    examples = [
        ("Đúng 5: thắng và hợp lệ", tuple((5, c) for c in range(2, 6)),
         (5, 6), "renju_exact_five.png"),
        ("Quá 5: 6 quân liên tiếp, nước bị cấm", tuple((5, c) for c in range(0, 5)),
         (5, 5), "renju_overline.png"),
        ("3–3: tạo hai bộ ba mở, nước bị cấm",
         ((5, 4), (5, 6), (4, 5), (6, 5)), (5, 5), "renju_double_three.png"),
        ("4–4: tạo hai bộ bốn mở, nước bị cấm",
         tuple((5, c) for c in (2, 3, 4)) + tuple((r, 5) for r in (2, 3, 4)),
         (5, 5), "renju_double_four.png"),
    ]
    for title, black, candidate, filename in examples:
        fig, ax = plt.subplots(figsize=(6.0, 6.2))
        fig.suptitle(f"Minh họa luật Renju: {title}", fontsize=13,
                     fontweight="bold", color=COLORS["text"], y=0.97)
        draw_board(ax, black, candidate=candidate)
        fig.text(0.5, 0.03, "Dấu × là nước đang xét; sơ đồ minh họa một thế riêng lẻ.",
                 ha="center", fontsize=9, color=COLORS["text"])
        fig.tight_layout(rect=(0.03, 0.07, 0.97, 0.92))
        save(fig, filename)


def main():
    architecture()
    flow()
    tensor_planes()
    augmentation_board()
    resnet_architecture()
    mcts_puct()
    renju_board_rules()


if __name__ == "__main__":
    main()
