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


if __name__ == "__main__":
    main()
