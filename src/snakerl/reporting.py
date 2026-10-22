"""Export real logged training and matched-seed evaluation as static figures."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np


def plot_run(run_dir: str | Path, output: str | Path | None = None) -> Path:
    os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = Path(run_dir)
    with (folder / "metrics.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    train = [r for r in rows if r["kind"] == "train"]
    if not train:
        raise ValueError("No completed training episodes in metrics.csv.")
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    curriculum = [r for r in train if r.get("curriculum_start", "0") in ("1", "True", "true")]
    standard = [r for r in train if r.get("curriculum_start", "0") not in ("1", "True", "true")]
    primary = standard or train
    window = min(50, len(primary))
    xs = np.array([float(r["timesteps"]) for r in primary])
    for ax, key, label in zip(axes, ["coverage", "reward"], ["Board coverage", "Episode return"]):
        values = np.array([float(r[key]) for r in primary])
        if curriculum:
            ax.scatter([float(r["timesteps"]) for r in curriculum], [float(r[key]) for r in curriculum],
                       s=3, alpha=.15, color="#8993a2", label="Own-state curriculum starts")
        ax.scatter(xs, values, s=4, alpha=.18, color="#63dfbb", label="Standard starts" if standard else "Real episodes")
        rolling = np.convolve(values, np.ones(window) / window, mode="valid")
        ax.plot(xs[window - 1:], rolling, color="#63dfbb", label=f"Standard {window}-episode average" if standard else f"{window}-episode average")
        ax.set(xlabel="Environment transitions", ylabel=label)
        ax.grid(alpha=.15)
        ax.legend(fontsize=8)
    axes[0].set_ylim(0, 1.02)
    fig.suptitle(f"SnakeRL | {folder.name} | actual training logs")
    target = Path(output) if output else folder / "training.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=150)
    plt.close(fig)
    return target


def plot_comparison(comparison_file: str | Path, output: str | Path | None = None) -> Path:
    os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(comparison_file)
    data = json.loads(path.read_text(encoding="utf-8"))
    labels = list(data["results"])
    values = list(data["results"].values())
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    colors = ["#76859d", "#e0b96a", "#65dcb5", "#729fff", "#df91dd", "#fa9b8d"]
    for ax, key, title in zip(axes, ["success_rate", "mean_coverage"], ["Full-board success rate", "Mean board coverage"]):
        ax.bar(labels, [v[key] for v in values], color=colors[:len(labels)])
        ax.set(ylim=(0, 1.12), title=title, ylabel="Fraction")
        ax.tick_params(axis="x", rotation=15)
        for i, value in enumerate(values):
            ax.text(i, value[key] + .025, f'{value[key]:.1%}', ha="center", fontsize=9)
    fig.suptitle(f'SnakeRL | {data["board_size"]}x{data["board_size"]} | {data["episodes"]} matched-seed episodes')
    target = Path(output) if output else path.with_suffix(".png")
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=150)
    plt.close(fig)
    return target
