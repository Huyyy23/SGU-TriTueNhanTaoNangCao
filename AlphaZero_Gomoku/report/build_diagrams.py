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




def architecture_detail():
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Các thành phần cốt lõi của AlphaZero Gomoku", fontsize=17,
                 fontweight="bold", color=COLORS["text"])
    for ax in axes.flat:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 5)
        ax.axis("off")

    ax = axes[0, 0]
    ax.set_title("(a) Biểu diễn và ResNet 6×64", loc="left", fontsize=12, fontweight="bold")
    box(ax, (0.25, 2.45), (2.15, 1.0), "4 plane đầu vào\nTa · đối thủ · cấm · màu lượt", fontsize=8)
    box(ax, (3.2, 2.45), (1.45, 1.0), "Stem\nConv 3×3", color="search", fontsize=8)
    box(ax, (5.25, 2.25), (1.8, 1.4), "6 residual blocks\n64 filters", color="search", fontsize=8)
    box(ax, (8.0, 3.4), (1.6, 0.8), "Policy\n121 logits", color="data", fontsize=8)
    box(ax, (8.0, 1.45), (1.6, 0.8), "Value\n[-1, 1]", color="data", fontsize=8)
    arrow(ax, (2.43, 2.95), (3.17, 2.95))
    arrow(ax, (4.67, 2.95), (5.22, 2.95))
    arrow(ax, (7.08, 3.2), (7.97, 3.78))
    arrow(ax, (7.08, 2.7), (7.97, 1.88))

    ax = axes[0, 1]
    ax.set_title("(b) Một vòng MCTS dùng PUCT", loc="left", fontsize=12, fontweight="bold")
    nodes = [
        ((0.25, 2.5), "Chọn nhánh\nPUCT"),
        ((2.7, 2.5), "Mở rộng\nnước hợp lệ"),
        ((5.1, 2.5), "Đánh giá lá\npolicy + value"),
        ((7.55, 2.5), "Backup\nđổi dấu"),
    ]
    for (x, y), label in nodes:
        box(ax, (x, y), (1.85, 1.05), label, color="search", fontsize=8)
    for i in range(len(nodes) - 1):
        arrow(ax, (nodes[i][0][0] + 1.88, 3.02), (nodes[i + 1][0][0] - 0.03, 3.02))
    ax.text(1.0, 1.45, "−Q(child) + cₚᵤcₜ · P · √N / (1 + Nₐ)",
            ha="center", fontsize=9, color=COLORS["text"])
    arrow(ax, (8.4, 2.42), (1.15, 2.42), "lặp đến đủ lượt mô phỏng", curve=-0.25,
          label_offset=(0, -0.6))

    ax = axes[1, 0]
    ax.set_title("(c) Tăng cường dữ liệu theo 8 đối xứng", loc="left", fontsize=12, fontweight="bold")
    box(ax, (0.35, 2.4), (2.15, 1.0), "4-plane tensor\n+ policy (121)", color="data", fontsize=8)
    box(ax, (3.9, 2.4), (2.15, 1.0), "8 đối xứng\nD₄: quay / lật", color="search", fontsize=8)
    box(ax, (7.45, 2.4), (2.15, 1.0), "tensor′ + policy′\nz giữ nguyên", color="data", fontsize=8)
    arrow(ax, (2.53, 2.9), (3.86, 2.9))
    arrow(ax, (6.08, 2.9), (7.42, 2.9))
    ax.text(5.0, 1.45, "Cùng phép biến đổi cho bàn và phân phối hành động",
            ha="center", fontsize=8, color=COLORS["text"])

    ax = axes[1, 1]
    ax.set_title("(d) Sơ đồ luật Renju theo implementation", loc="left", fontsize=12, fontweight="bold")
    box(ax, (0.25, 2.45), (1.45, 0.9), "Màu quân?", color="interface", fontsize=8)
    box(ax, (2.2, 3.45), (2.25, 0.8), "Trắng: ≥5 thắng;\nkhác: nước hợp lệ", color="data", fontsize=7.5)
    box(ax, (2.2, 1.35), (2.25, 0.95), "Đen: đúng 5 thắng;\nưu tiên trước kiểm tra cấm", color="search", fontsize=7.5)
    box(ax, (5.15, 1.35), (1.9, 0.95), "Overline hoặc\n3–3 / 4–4?", color="search", fontsize=8)
    box(ax, (8.0, 2.65), (1.75, 0.8), "Có: nước cấm", color="optional", fontsize=8)
    box(ax, (8.0, 0.75), (1.75, 0.8), "Không: hợp lệ", color="data", fontsize=8)
    arrow(ax, (1.73, 3.0), (2.17, 3.85))
    arrow(ax, (1.73, 2.8), (2.17, 1.82))
    arrow(ax, (4.48, 1.82), (5.12, 1.82))
    arrow(ax, (7.08, 2.0), (7.97, 3.05), "có")
    arrow(ax, (7.08, 1.65), (7.97, 1.15), "không")

    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.94))
    fig.savefig(OUT / "architecture_detail.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    architecture()
    flow()
    architecture_detail()

if __name__ == "__main__":
    main()
