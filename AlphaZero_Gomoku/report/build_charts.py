import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
RUNS = (
    (
        "Huấn luyện luật Normal",
        ROOT / "logs" / "train_20261028-063106.jsonl",
        ROOT / "report" / "normal_loss.png",
    ),
    (
        "Tinh chỉnh luật Renju",
        ROOT / "logs" / "renju_finetune_20261112-181734.jsonl",
        ROOT / "report" / "renju_loss.png",
    ),
)
METRICS = (
    ("loss", "Loss huấn luyện"),
    ("policy_loss", "Loss policy"),
    ("value_loss", "Loss value"),
    ("loss_eval_on_fixed_batch", "Loss batch cố định"),
)


def read_log(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    required = {"iteration", *(key for key, _ in METRICS)}
    if not rows or any(not required.issubset(row) for row in rows):
        raise ValueError(f"{path} must contain records with {', '.join(sorted(required))}")
    return rows


def evidence_dashboard():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8))
    fig.suptitle("Tổng hợp một số phép đo lịch sử", fontsize=15, fontweight="bold")

    ax = axes[0]
    labels = ["AZ v0–\nHeuristic", "AZ v0–\nAlpha-Beta", "AZ v0–\nMCTS Pure", "AZ Renju–\nHeuristic"]
    scores = [0, 50, 1, 0]
    bars = ax.bar(labels, scores, color=["#9e3d3d", "#4c78a8", "#4c78a8", "#9e3d3d"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Điểm của AZ (%)")
    ax.set_title("Đối kháng T35 · fixed budget")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, max(value + 2, 3), f"{value}%", ha="center", fontsize=9)
    ax.text(0.02, -0.31, "100 ván/cặp; score=(W+0,5D)/N.\nNguồn: docs/report.md, §§10.2–10.3.",
            transform=ax.transAxes, fontsize=7, va="top")

    ax = axes[1]
    normal = [60786 / 75000 * 100, 14208 / 75000 * 100, 6 / 75000 * 100]
    renju = [100, 0, 0]
    colors = ["#30343b", "#e7e7e7", "#79a86b"]
    for i, (label, values) in enumerate((("Normal\nn=75.000", normal), ("Renju\nn=1.500", renju))):
        ax.bar(label, values[0], color=colors[0], label="Đen thắng" if i == 0 else None)
        ax.bar(label, values[1], bottom=values[0], color=colors[1], label="Trắng thắng" if i == 0 else None)
        ax.bar(label, values[2], bottom=values[0] + values[1], color=colors[2], label="Hòa" if i == 0 else None)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Tỷ lệ ván theo người thắng (%)")
    ax.set_title("Phân bố kết quả trong DB")
    ax.legend(fontsize=7, loc="lower right")
    ax.text(0.02, -0.31, "Normal: 60.786 / 14.208 / 6; Renju: 1.500 Đen thắng.\nNguồn: data/normal.sqlite; data/renju_finetune_1500.sqlite.",
            transform=ax.transAxes, fontsize=7, va="top")

    ax = axes[2]
    workers = [1, 2, 4]
    py = [22.91, 54.08, 80.05]
    cpp = [86.51, 129.59, 124.66]
    x = range(len(workers))
    width = 0.36
    ax.bar([v - width / 2 for v in x], py, width, label="Python", color="#9aa7b5")
    ax.bar([v + width / 2 for v in x], cpp, width, label="C++", color="#4c78a8")
    ax.set_xticks(list(x), [str(v) for v in workers])
    ax.set_xlabel("Số worker")
    ax.set_ylabel("Games/min")
    ax.set_title("Self-play batch 50 games")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    ax.text(0.02, -0.31, "Median; 3 lần/cấu hình; cùng máy RTX 4060.\nNguồn: docs/performance/11-production-benchmark.md, §§6–7.",
            transform=ax.transAxes, fontsize=7, va="top")

    fig.tight_layout(rect=(0, 0.12, 1, 0.92))
    fig.savefig(Path(__file__).resolve().parent / "results_dashboard.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    for run_name, source, destination in RUNS:
        rows = read_log(source)
        iterations = [row["iteration"] for row in rows]
        fig, ax = plt.subplots(figsize=(10, 6))
        for key, label in METRICS:
            ax.plot(iterations, [row[key] for row in rows], label=label, linewidth=1.8)
        ax.set(title=run_name, xlabel="Vòng lặp", ylabel="Loss")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.text(
            0.5,
            0.01,
            f"Nguồn: {source.relative_to(ROOT)}. Mỗi hình là một run, không so sánh trực tiếp hai luật.",
            ha="center",
            fontsize=8,
        )
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(destination, dpi=160)
        plt.close(fig)
    evidence_dashboard()


if __name__ == "__main__":
    main()
