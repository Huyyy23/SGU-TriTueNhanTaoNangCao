from pathlib import Path

import matplotlib.pyplot as plt
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
        (2.9, "Runner\nđiều phối ván"),
        (5.45, "Rules\nNormal / Renju"),
        (8.0, "Board\ntrạng thái + lịch sử"),
        (10.55, "Player\nAZ / heuristic / khác"),
    ]
    for x, label in game:
        box(ax, (x, 6.25), (2.1, 0.9), label, fontsize=9)
    for (x, _), (next_x, _) in zip(game, game[1:]):
        arrow(ax, (x + 2.12, 6.7), (next_x - 0.03, 6.7))

    # AlphaZero search path
    search = [
        (0.55, "board_to_planes\n4 × 11 × 11"),
        (3.55, "AZNet\npolicy + value"),
        (6.55, "MCTS / PUCT\nmask · expand · backup"),
        (9.55, "AZPlayer\nphân phối lượt thăm"),
    ]
    for x, label in search:
        box(ax, (x, 4.35), (2.4, 0.95), label, color="search", fontsize=9)
    for (x, _), (next_x, _) in zip(search, search[1:]):
        arrow(ax, (x + 2.42, 4.82), (next_x - 0.03, 4.82))
    arrow(ax, (9.85, 5.28), (9.85, 6.22), "nước đi")
    ax.text(0.65, 5.55, "Board/Rules cung cấp trạng thái và mặt nạ hợp lệ cho tìm kiếm.",
            fontsize=8.5, color=COLORS["text"])

    # Training and persistence
    training = [
        (0.4, "Self-play\nván và mẫu"),
        (3.0, "ReplayBuffer\n(features, π, z)"),
        (5.6, "Trainer\npolicy + value loss"),
        (8.2, "Checkpoint\nmodel + trạng thái"),
        (10.8, "SQLite\nself-play / train"),
    ]
    for x, label in training:
        box(ax, (x, 1.55), (2.0, 0.9), label, color="data", fontsize=8.5)
    for (x, _), (next_x, _) in zip(training[:4], training[1:4]):
        arrow(ax, (x + 2.02, 2.0), (next_x - 0.03, 2.0))
    arrow(ax, (9.2, 1.55), (1.4, 1.55), "checkpoint dùng cho self-play", curve=-0.2,
          label_offset=(0, -0.2))
    arrow(ax, (2.4, 1.7), (10.75, 1.7), "metadata / metrics", curve=0.12,
          label_offset=(0, 0.2))

    box(ax, (7.0, 0.35), (2.0, 0.75), "Tournament\nso sánh agents", color="data", fontsize=8.5)
    box(ax, (10.8, 0.35), (2.0, 0.75), "SQLite\ntournament", color="data", fontsize=8.5)
    arrow(ax, (9.02, 0.72), (10.77, 0.72))
    box(ax, (4.0, 3.0), (2.0, 0.72), "Native MCTS (C++)\nbackend tùy chọn",
        color="optional", fontsize=8, linestyle="dashed")
    arrow(ax, (6.05, 3.36), (6.8, 4.32), "thay tree", dashed=True)
    ax.text(0.45, 0.55, "Native chỉ thay cơ chế tìm kiếm; luật, bàn, mã hóa và suy luận thuộc Python.",
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




def main():
    architecture()
    flow()


if __name__ == "__main__":
    main()
