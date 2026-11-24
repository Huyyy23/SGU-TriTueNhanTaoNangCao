from __future__ import annotations

import json
import math
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "tournament.sqlite"
OLD = ROOT / "evidence" / "colab_return_20261114T085015Z" / "t35_analysis.json"
OUT_MD = ROOT / "docs" / "t35-rerun-20261124.md"
RUNS = {
    "normal_fixed": "t35-rerun-20261124-normal-fixed",
    "renju_fixed": "t35-rerun-20261124-renju-fixed",
    "normal_time": "t35-rerun-20261124-normal-time",
}
EXPECTED = {"normal_fixed": 15, "renju_fixed": 6, "normal_time": 3}
BOOTSTRAPS = 2000
SEED = 20261113


def elo_from_game_rows(rows: list[dict], a: str, b: str, seed: int) -> dict:
    clusters: dict[int, list[int]] = defaultdict(lambda: [0, 0, 0])
    colors: dict[int, list[str]] = defaultdict(list)
    for game in rows:
        black, white = game["first_player"], game["second_player"]
        winner = game["winner"]
        cluster = int(game["game_id"]) // 2
        if winner == "draw":
            result = 1
        else:
            winner_agent = black if winner == "black" else white
            result = 0 if winner_agent == a else 2 if winner_agent == b else -1
            if result < 0:
                raise ValueError(f"Unexpected winner {winner_agent!r} for {a} vs {b}")
        clusters[cluster][result] += 1
        colors[cluster].append("black" if black == a else "white")
    counts = np.asarray(list(clusters.values()), dtype=np.int64)
    if len(counts) != 50 or np.any(counts.sum(axis=1) != 2):
        raise ValueError(f"Expected 50 complete paired openings for {a} vs {b}; got {len(counts)}")
    if any(sorted(value) != ["black", "white"] for value in colors.values()):
        raise ValueError(f"Color swap missing for {a} vs {b}")

    def fit(sample: np.ndarray) -> float:
        wins, _draws, losses = sample.sum(axis=0).astype(float)
        return 400.0 * math.log((wins + 0.5) / (losses + 0.5)) / math.log(10.0)

    rng = np.random.default_rng(seed)
    boots = np.asarray([fit(counts[rng.integers(0, len(counts), size=len(counts))])
                        for _ in range(BOOTSTRAPS)])
    wins, draws, losses = counts.sum(axis=0).tolist()
    return {"wins": int(wins), "draws": int(draws), "losses": int(losses),
            "elo": fit(counts), "ci": np.quantile(boots, [0.025, 0.975]).tolist()}


def load_old() -> tuple[dict, dict]:
    old = json.loads(OLD.read_text(encoding="utf-8"))
    rows = {}
    for group, items in old["groups"].items():
        rows[group] = {}
        for item in items:
            key = tuple(sorted((item["a"], item["b"])))
            path = ROOT / Path(item["path"]) / "games.jsonl"
            games = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
            if len(games) != 100:
                raise ValueError(f"Historical pair {key} has {len(games)} games, expected 100")
            elo = elo_from_game_rows(games, item["a"], item["b"], SEED)
            if (elo["wins"], elo["draws"], elo["losses"]) != (item["W"], item["D"], item["L"]):
                raise ValueError(f"Archived W/D/L disagrees with raw games for {group}/{key}")
            rows[group][key] = {**item, "elo": elo, "games_raw": games}
    return old, rows


def load_new(old_rows: dict) -> tuple[dict, dict, dict]:
    pairs, elo, summaries = {}, {}, {}
    rankings = {}
    total_games = 0
    with sqlite3.connect(DB.as_uri() + "?mode=ro", uri=True) as con:
        con.row_factory = sqlite3.Row
        for group, run_id in RUNS.items():
            exp = con.execute("SELECT started_at FROM experiments WHERE experiment_id=?", (run_id,)).fetchone()
            if exp is None:
                raise ValueError(f"Missing experiment {run_id}")
            game_rows = con.execute("SELECT payload_json FROM games WHERE experiment_id=?", (run_id,)).fetchall()
            games = [json.loads(row["payload_json"]) for row in game_rows]
            if len(games) != EXPECTED[group] * 100:
                raise ValueError(f"{run_id}: expected {EXPECTED[group] * 100} games, got {len(games)}")
            total_games += len(games)
            grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
            for game in games:
                pair = tuple(game.get("pair", ()))
                if len(pair) != 2:
                    raise ValueError(f"Game in {run_id} lacks pair labels")
                grouped[pair].append(game)

            pairs[group] = {}
            agent_totals: dict[str, Counter] = defaultdict(Counter)
            for pair, rows in grouped.items():
                a, b = pair
                wins = draws = losses = 0
                colors = Counter()
                for game in rows:
                    black, white = game["first_player"], game["second_player"]
                    if game["winner"] == "draw":
                        draws += 1
                    else:
                        winner = black if game["winner"] == "black" else white
                        colors["black_wins" if game["winner"] == "black" else "white_wins"] += 1
                        if winner == a:
                            wins += 1
                            colors["a_black_wins" if black == a else "a_white_wins"] += 1
                        else:
                            losses += 1
                count = len(rows)
                score = (wins + draws / 2) / count
                z = 1.96
                center = (score + z*z/(2*count)) / (1 + z*z/count)
                half = z * math.sqrt(score*(1-score)/count + z*z/(4*count*count)) / (1 + z*z/count)
                key = tuple(sorted(pair))
                target_a = old_rows[group][key]["a"]
                item = {
                    "a": a, "b": b, "games": count, "W": wins, "D": draws, "L": losses,
                    "score": score, "wilson95": [center-half, center+half],
                    "colors": dict(colors),
                    "mean_game_sec": sum(game["duration_sec"] for game in rows) / count,
                }
                if a != target_a:
                    item = {
                        **item, "a": target_a, "b": a, "W": losses, "L": wins,
                        "score": 1 - score, "wilson95": [1 - (center+half), 1 - (center-half)],
                        "colors": {
                            **colors,
                            "a_black_wins": colors.get("black_wins", 0) - colors.get("a_black_wins", 0),
                            "a_white_wins": colors.get("white_wins", 0) - colors.get("a_white_wins", 0),
                        },
                    }
                pairs[group][key] = item
                for agent, outcome in ((a, (wins, draws, losses)), (b, (losses, draws, wins))):
                    agent_totals[agent].update(W=outcome[0], D=outcome[1], L=outcome[2])

            if len(pairs[group]) != EXPECTED[group] or any(item["games"] != 100 for item in pairs[group].values()):
                raise ValueError(f"{group}: pair matrix does not match T35")
            rankings[group] = [
                {"agent": agent, **dict(counts),
                 "score": (counts["W"] + counts["D"] / 2) / sum(counts.values())}
                for agent, counts in sorted(
                    agent_totals.items(),
                    key=lambda entry: -(entry[1]["W"] + entry[1]["D"] / 2) / sum(entry[1].values()),
                )
            ]

            for index, (key, item) in enumerate(pairs[group].items()):
                clusters: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
                candidate_colors: dict[str, list[str]] = defaultdict(list)
                for row in con.execute(
                    "SELECT m.pair_id, g.winner, ba.label AS black_label, wa.label AS white_label "
                    "FROM games g JOIN matches m ON m.match_id=g.match_id "
                    "JOIN agents ba ON ba.agent_id=g.black_agent_id "
                    "JOIN agents wa ON wa.agent_id=g.white_agent_id "
                    "WHERE g.experiment_id=? ORDER BY m.pair_id, CAST(g.local_game_id AS INTEGER)",
                    (run_id,),
                ):
                    if {row["black_label"], row["white_label"]} != {item["a"], item["b"]}:
                        continue
                    cluster = row["pair_id"]
                    black, white = row["black_label"], row["white_label"]
                    if row["winner"] == "draw":
                        result = 1
                    elif row["winner"] == ("black" if black == item["a"] else "white"):
                        result = 0
                    else:
                        result = 2
                    clusters[cluster][result] += 1
                    candidate_colors[cluster].append("black" if black == item["a"] else "white")
                counts = np.asarray(list(clusters.values()), dtype=np.int64)
                if len(counts) != 50 or np.any(counts.sum(axis=1) != 2):
                    raise ValueError(f"{group}/{key}: expected 50 paired openings")
                if any(sorted(value) != ["black", "white"] for value in candidate_colors.values()):
                    raise ValueError(f"{group}/{key}: color swap missing")

                def fit(sample: np.ndarray) -> float:
                    w, _d, l = sample.sum(axis=0).astype(float)
                    return 400 * math.log((w + 0.5) / (l + 0.5)) / math.log(10)

                rng = np.random.default_rng(SEED + index)
                boots = np.asarray([
                    fit(counts[rng.integers(0, len(counts), size=len(counts))])
                    for _ in range(BOOTSTRAPS)
                ])
                w, d, l = counts.sum(axis=0).tolist()
                elo[(group, key)] = {
                    "wins": int(w), "draws": int(d), "losses": int(l),
                    "delta_elo": fit(counts),
                    "ci95": np.quantile(boots, [0.025, 0.975]).tolist(),
                }

            event = con.execute(
                "SELECT payload_json FROM events WHERE experiment_id=? AND event='tournament_summary'",
                (run_id,),
            ).fetchone()
            if event is None:
                raise ValueError(f"Missing completed summary for {run_id}")
            summary = json.loads(event["payload_json"])
            summary["total_elapsed_sec"] = (
                datetime.fromisoformat(summary["finished_at"]) - datetime.fromisoformat(exp["started_at"])
            ).total_seconds()
            summaries[group] = summary
    if total_games != 2400:
        raise ValueError(f"Expected 2,400 new games, got {total_games}")
    return {"games": total_games, "groups": pairs, "rankings": rankings}, pairs, {"elo": elo, "runs": summaries}


def percent(value: float) -> str:
    return f"{100*value:.1f}%"


def fmt_ci(values: list[float]) -> str:
    return f"[{values[0]:.3f}; {values[1]:.3f}]"


def make_charts(old_rows: dict, new_pairs: dict) -> tuple[Path, Path]:
    colors = {"old": "#7b8794", "new": "#1677b8"}
    fig, axes = plt.subplots(3, 1, figsize=(11, 13), constrained_layout=True)
    titles = {"normal_fixed": "Normal · 200 mô phỏng", "renju_fixed": "Renju · 200 mô phỏng",
              "normal_time": "Normal · 1 giây/nước"}
    for ax, group in zip(axes, RUNS):
        keys = list(old_rows[group])
        y = np.arange(len(keys))
        old_scores = [old_rows[group][key]["score"] for key in keys]
        new_scores = [new_pairs[group][key]["score"] for key in keys]
        for yi, old_score, new_score in zip(y, old_scores, new_scores):
            ax.plot([old_score, new_score], [yi, yi], color="#c4cbd2", linewidth=1.3, zorder=1)
        ax.scatter(old_scores, y - 0.12, label="T35 cũ", color=colors["old"], s=24, zorder=2)
        ax.scatter(new_scores, y + 0.12, label="Chạy lại", color=colors["new"], s=28, zorder=3)
        ax.set_yticks(y, [f"{old_rows[group][key]['a']} – {old_rows[group][key]['b']}" for key in keys], fontsize=8)
        ax.set_xlim(-0.04, 1.04)
        ax.set_xticks(np.linspace(0, 1, 6), [f"{x:.0%}" for x in np.linspace(0, 1, 6)])
        ax.grid(axis="x", alpha=0.25)
        ax.set_title(titles[group], loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel("Điểm của A (thắng + ½ hòa) / số ván")
    axes[0].legend(loc="lower right", ncol=2, frameon=False)
    score_path = ROOT / "docs" / "t35-rerun-20261124-score.png"
    fig.savefig(score_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    groups = list(RUNS)
    old_mean, new_mean = [], []
    for group in groups:
        old_games = [g for key in old_rows[group] for g in old_rows[group][key]["games_raw"]]
        old_mean.append(np.mean([g["duration_sec"] for g in old_games]))
        new_mean.append(np.mean([new_pairs[group][k]["mean_game_sec"] for k in new_pairs[group]]))
    x = np.arange(3)
    ax.bar(x - 0.18, old_mean, 0.36, label="T35 cũ", color=colors["old"])
    ax.bar(x + 0.18, new_mean, 0.36, label="Chạy lại", color=colors["new"])
    ax.set_xticks(x, [titles[g] for g in groups])
    ax.set_ylabel("Thời gian trung bình mỗi ván (giây)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    time_path = ROOT / "docs" / "t35-rerun-20261124-runtime.png"
    fig.savefig(time_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return score_path, time_path


def build_report(old: dict, old_rows: dict, new: dict, new_rows: dict, stats: dict) -> None:
    lines = [
        "# T35 chạy lại — kết quả cuối (2026-11-24)", "",
        "> Phạm vi: chạy lại đúng 24 cặp T35, 100 ván/cặp, 50 màu Đen và 50 màu Trắng; không huấn luyện và không thay đổi luật. Dữ liệu gốc T35 được giữ nguyên.", "",
        "## Thiết lập và định danh", "",
        "| Nhóm | Run ID SQLite mới | Cặp × ván | Backend / tiến trình | Thời gian từ bắt đầu đến hoàn tất (giây) |", "|---|---|---:|---|---:|",
    ]
    for group, pairs in new_rows.items():
        summary = stats["runs"][group]
        config = summary
        workers = "4→2 (resume)" if group == "normal_fixed" else str(config["workers"])
        lines.append(f"| {group} | `{RUNS[group]}` | {len(pairs)} × 100 | {config['backends']['mcts']} / {workers} × {config['torch_threads_per_game']} thread | {config['total_elapsed_sec']:.1f} |")
    lines += [
        "", "T35 cũ gồm Normal cố định (15 cặp), Renju cố định (6 cặp) và Normal 1 giây/nước (3 cặp). Chạy lại giữ nguyên tập tác tử, luật, ngân sách, độ sâu Alpha-Beta cố định 1, seed gốc `20261113`, seed scheme lịch sử `legacy-v1`, bàn trống và đổi màu theo từng cặp. Mỗi nhóm được ghi vào `data/tournament.sqlite` bằng run ID riêng; không ghi đè dữ liệu cũ.",
        "", "Checkpoint Normal dùng con trỏ best hiện hành `checkpoints/normal/best.json`, generation `7c43f06544a045be8ef1292486a06c15`, SHA-256 `af6337abf4fe733e9bc0dc202f2048f7f2575411bc03d5b102268782771a223e` (Normal, 6×64). Renju dùng checkpoint 1.500 ván được đánh giá tốt nhất trong các tạo tác hiện còn, `checkpoints/renju_finetune_1500/latest.json`, SHA-256 `5e2e45d5a952bdf88c8cce783b882d65d6d5ebd7128549984cd8be21c24af9b5` (Renju, 6×64). Renju không có con trỏ `best.json`; đây là checkpoint được thử nghiệm tốt nhất còn truy xuất được, không phải mô hình đạt ngưỡng sức chơi.",
        "", "Các nhóm ngân sách mô phỏng dùng MCTS và nhận diện mẫu C++ (`gomoku_native` 0.1.0), 1 luồng Torch mỗi tiến trình; AZ tự chọn CUDA khi khả dụng. Normal cố định bắt đầu với 4 tiến trình, sau đó tiếp tục với 2 tiến trình sau lỗi cấp phát bộ nhớ native; Renju sẽ dùng 2 tiến trình để giảm áp lực bộ nhớ. Nhóm ngân sách thời gian dùng Python và 3 tiến trình, giữ giới hạn 1 giây/nước; Alpha-Beta dùng iterative deepening. Bộ chạy hiện không hỗ trợ batching suy luận liên ván. 55 kiểm thử Python/C++ tập trung đã qua; tính tương đương kết quả trên quỹ đạo ngẫu nhiên chưa được xác lập.",
        "", "So sánh với T35 cũ giữ nguyên cặp, màu, luật, ngân sách và seed; checkpoint được cập nhật theo yêu cầu. Backend và một phần môi trường T35 cũ khác/không đồng nhất, vì vậy chênh lệch là mô tả giữa hai snapshot, không quy kết nhân quả cho checkpoint hoặc backend.",
        "", "## Kết quả tổng hợp", "", "| Nhóm | Ván | T35 cũ: tác tử đứng đầu (score) | Chạy lại: tác tử đứng đầu (score) | Hòa | Illegal |",
        "|---|---:|---|---|---:|---:|",
    ]
    for group in RUNS:
        old_rank = old["rankings"][group][0]
        new_rank = new["rankings"][group][0]
        summary = stats["runs"][group]
        draws = sum(item["D"] for item in new_rows[group].values())
        lines.append(f"| {group} | {sum(x['games'] for x in new_rows[group].values())} | {old_rank['agent']} ({percent(old_rank['score'])}) | {new_rank['agent']} ({percent(new_rank['score'])}) | {draws} | {summary['illegal_moves_total']} |")
    lines += ["", "## Kết quả theo cặp", "", "Tỷ lệ thắng = W / N; điểm = (W + 0,5D) / N. Khoảng điểm là Wilson 95% (hòa tính nửa điểm). Elo là chênh lệch pairwise Davidson so với đối thủ, với bootstrap 2.000 lần theo cụm hai ván đổi màu; đây không phải leaderboard Elo tuyệt đối.", ""]
    labels = {"normal_fixed": "Normal — 200 mô phỏng", "renju_fixed": "Renju — 200 mô phỏng", "normal_time": "Normal — 1 giây/nước"}
    for group in RUNS:
        lines += [f"### {labels[group]}", "", "| A | B | T35 cũ W/D/L | Chạy lại W/D/L | A thắng Đen/Trắng | Tỷ lệ thắng | Điểm [Wilson 95%] | ΔElo A−B, cũ → mới [CI 95%] | Giây/ván (cũ → mới) |", "|---|---|---:|---:|---:|---:|---|---|---:|"]
        for key, old_item in old_rows[group].items():
            item = new_rows[group][key]
            a, b = old_item["a"], old_item["b"]
            elo = stats["elo"][(group, key)]
            colors = item["colors"]
            a_black = colors.get("a_black_wins", 0)
            a_white = colors.get("a_white_wins", 0)
            win_rate = item["W"] / item["games"]
            old_elo = old_item["elo"]
            lines.append(f"| {a} | {b} | {old_item['W']}/{old_item['D']}/{old_item['L']} | {item['W']}/{item['D']}/{item['L']} | {a_black}/{a_white} | {percent(win_rate)} | {percent(item['score'])} {fmt_ci(item['wilson95'])} | {old_elo['elo']:+.1f} [{old_elo['ci'][0]:+.1f}; {old_elo['ci'][1]:+.1f}] → {elo['delta_elo']:+.1f} [{elo['ci95'][0]:+.1f}; {elo['ci95'][1]:+.1f}] | {old_item['mean_game_sec']:.2f} → {item['mean_game_sec']:.2f} |")
    lines += ["", "## Biểu đồ", "", "![So sánh điểm T35 cũ và lần chạy lại theo từng cặp](t35-rerun-20261124-score.png)", "", "![Thời gian trung bình mỗi ván theo nhóm](t35-rerun-20261124-runtime.png)", "", "## Diễn giải và giới hạn", "", "- T35 cũ ghi 2.400 ván; lần chạy lại phải có đủ 1.500 Normal cố định, 600 Renju cố định và 300 Normal thời gian. Các kết quả được giữ tách theo luật/ngân sách.",
        "- Bàn trống và các tác tử tất định có thể tạo lại cùng quỹ đạo dù có 50 cặp seed. Do đó khoảng Wilson/cluster-bootstrap là mô tả danh nghĩa; số quỹ đạo độc lập thấp hơn số ván và không hỗ trợ suy rộng ngoài pool này.",
        "- Thời gian cũ ghép từ nhiều môi trường trong nhóm fixed; thời gian mới đo trên Windows, RTX 4060 Laptop, PyTorch 2.14.1+cu126, 16 CPU logic. So sánh wall-clock không cô lập tác động của checkpoint/backend. Thời lượng mới là chênh lệch giữa thời điểm bắt đầu experiment và `finished_at` trong SQLite, gồm cả khoảng gián đoạn khi resume.",
        "- Dữ liệu T35 cũ được giữ nguyên, không có experiment ID tương ứng trong SQLite hiện tại. Bộ tạo báo cáo đã đối chiếu W/D/L của đủ 2.400 ván JSONL với bảng tổng hợp và xác nhận 50 cặp ván đổi màu cho mỗi trận đấu.",
        "- Khi khôi phục run sau lỗi, cấu hình SQLite giữ số tiến trình khởi tạo; số tiến trình thực tế trong lần tiếp tục được ghi tại `logs/tournament/t35-rerun-20261124-normal-fixed-resume.json`.",
        "", "## Tái lập và nguồn", "", f"- Cơ sở dữ liệu: `data/tournament.sqlite`; run IDs: `{', '.join(RUNS.values())}`.",
        "- Baseline: `evidence/colab_return_20261114T085015Z/t35_analysis.json` và các `games.jsonl`/`manifest.json` được trỏ trực tiếp từ tệp phân tích.",
        "- Phân tích/biểu đồ: `report/build_t35_rerun.py`; W/D/L, điểm và thời gian đọc từ SQLite; Elo là ước lượng Davidson theo cụm cặp màu, cùng công thức trong `az/elo.py` và bootstrap 2.000 mẫu.",
        "- T35 lịch sử: `docs/report.md`, mục 10; giao thức: `docs/benchmarks.md`, mục T35; mô tả luật/benchmark: `SPEC.md`, mục 10–11.", ""]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    old, old_rows = load_old()
    new, new_rows, stats = load_new(old_rows)
    score_path, runtime_path = make_charts(old_rows, new_rows)
    build_report(old, old_rows, new, new_rows, stats)
    print(f"Wrote {OUT_MD}")
    print(f"Score chart: {score_path}")
    print(f"Runtime chart: {runtime_path}")
    print("Verified 24 pairings, 2,400 new games, 100 games and 50 paired color-swapped clusters per pair.")


if __name__ == "__main__":
    main()
