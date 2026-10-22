"""Matched-seed evaluation, with no expert or action masking for neural policies."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from snakerl.baselines import GreedyPolicy, HamiltonianPolicy, RandomPolicy
from snakerl.env import SnakeEnv


def make_policy(name: str, seed: int = 0, checkpoint: str | Path | None = None):
    if name == "random":
        return RandomPolicy(seed=seed)
    if name == "greedy":
        return GreedyPolicy()
    if name == "hamiltonian":
        return HamiltonianPolicy()
    if name == "neural":
        if checkpoint is None:
            raise ValueError("A neural policy requires --checkpoint.")
        from snakerl.model import NeuralPolicy

        return NeuralPolicy(checkpoint)
    raise ValueError(f"Unknown policy: {name}")


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def episode(
    policy_name: str,
    board_size: int,
    seed: int,
    checkpoint: str | Path | None = None,
    *,
    record: bool = False,
    deterministic: bool = True,
) -> tuple[dict, dict | None]:
    policy = make_policy(policy_name, seed, checkpoint)
    env = SnakeEnv(board_size=board_size, observation_version=getattr(policy, "observation_version", 1))
    env.reset(seed=seed)
    action_rng = np.random.default_rng(np.random.SeedSequence([seed, 101]))
    frames = [{"state": env.state(), "action": None, "probabilities": None, "reward": None}]
    total_return = 0.0
    while True:
        probabilities = policy.probabilities(env).tolist() if hasattr(policy, "probabilities") else None
        if probabilities is not None:
            normalized = np.asarray(probabilities, dtype=np.float64)
            normalized /= normalized.sum()
            action = int(np.argmax(probabilities)) if deterministic else int(action_rng.choice(3, p=normalized))
        else:
            action = int(policy.act(env))
        _, reward, terminated, truncated, info = env.step(action)
        total_return += float(reward)
        if record:
            frames.append({"state": env.state(), "action": action, "probabilities": probabilities, "reward": float(reward)})
        if terminated or truncated:
            break
    result = {"policy": policy_name, "seed": seed, **info, "total_return": total_return}
    replay = None
    if record:
        replay = {"version": 1, "policy": policy_name, "checkpoint": str(checkpoint) if checkpoint else None,
                  "board_size": board_size, "seed": seed, "result": result, "frames": frames,
                  "action_selection": "deterministic" if deterministic else "seeded_policy_sampling"}
        replay["observation_version"] = env.observation_version
    env.close()
    return result, replay


def _neural_batch(board_size: int, seeds: list[int], checkpoint: str | Path, batch_size: int, deterministic: bool = True) -> list[dict]:
    from snakerl.model import NeuralPolicy

    policy = NeuralPolicy(checkpoint)
    results = []
    for offset in range(0, len(seeds), batch_size):
        group = seeds[offset:offset + batch_size]
        envs = [SnakeEnv(board_size=board_size, observation_version=policy.observation_version) for _ in group]
        observations = [env.reset(seed=seed)[0] for env, seed in zip(envs, group)]
        totals = [0.0] * len(group)
        action_rngs = [np.random.default_rng(np.random.SeedSequence([seed, 101])) for seed in group]
        active = list(range(len(group)))
        while active:
            # The input is only the observation tensor; baseline policies never enter this branch.
            actions, probabilities = policy.predict_batch(np.stack([observations[i] for i in active]))
            if not deterministic:
                actions = [action_rngs[i].choice(3, p=row.astype(np.float64) / row.sum(dtype=np.float64))
                           for i, row in zip(active, probabilities)]
            next_active = []
            for i, action in zip(active, actions):
                observations[i], reward, terminated, truncated, info = envs[i].step(int(action))
                totals[i] += float(reward)
                if terminated or truncated:
                    results.append({"policy": "neural", "seed": group[i], **info, "total_return": totals[i]})
                else:
                    next_active.append(i)
            active = next_active
        for env in envs:
            env.close()
    return sorted(results, key=lambda r: r["seed"])


def wilson_interval(successes: int, count: int) -> list[float]:
    """95% binomial Wilson interval; even 100/100 is not a proof of certainty."""
    z = 1.959963984540054
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def summarize(results: list[dict]) -> dict:
    if not results:
        raise ValueError("At least one episode is required.")
    successes = sum(bool(r["success"]) for r in results)
    successful_steps = [r["steps"] for r in results if r["success"]]
    return {
        "episodes": len(results), "successes": successes,
        "success_rate": successes / len(results),
        "success_rate_ci95": wilson_interval(successes, len(results)),
        "mean_coverage": float(np.mean([r["coverage"] for r in results])),
        "min_coverage": float(min(r["coverage"] for r in results)),
        "max_coverage": float(max(r["coverage"] for r in results)),
        "mean_steps": float(np.mean([r["steps"] for r in results])),
        "mean_success_steps": float(np.mean(successful_steps)) if successful_steps else None,
        "mean_return": float(np.mean([r["total_return"] for r in results])),
        "termination_counts": {reason: sum(r["termination_reason"] == reason for r in results)
                               for reason in sorted({str(r["termination_reason"]) for r in results})},
    }


def evaluate(
    policy_name: str = "hamiltonian", board_size: int = 8, episodes: int = 100,
    seed_start: int = 10000, checkpoint: str | Path | None = None,
    output_dir: str | Path | None = None, batch_size: int = 32,
    save_replays: bool = True,
    deterministic: bool = True,
) -> dict:
    if episodes < 1 or batch_size < 1:
        raise ValueError("episodes and batch_size must be positive.")
    seeds = list(range(seed_start, seed_start + episodes))
    if policy_name == "neural":
        if checkpoint is None:
            raise ValueError("A neural policy requires a checkpoint.")
        results = _neural_batch(board_size, seeds, checkpoint, batch_size, deterministic)
    else:
        results = [episode(policy_name, board_size, seed)[0] for seed in seeds]
    report = {"version": 1, "policy": policy_name, "checkpoint": str(checkpoint) if checkpoint else None,
              "board_size": board_size, "initial_length": min(3, board_size),
              "seed_start": seed_start, "seeds": seeds,
              "action_selection": "seeded_uniform" if policy_name == "random" else ("deterministic_argmax" if deterministic else "seeded_policy_sampling") if policy_name == "neural" else "deterministic_algorithm",
              "environment": {"distance_reward": 0.02, "max_no_food_steps": 4 * board_size**2,
                              "max_steps": 4 * board_size**4, "initial_state": "fixed_top_row_facing_east"},
              "neural_action_masking": False, "summary": summarize(results), "episodes": results}
    if policy_name == "neural":
        from snakerl.model import NeuralPolicy
        report["environment"]["observation_version"] = NeuralPolicy(checkpoint).observation_version
    else:
        report["environment"]["observation_version"] = 1
    report["environment"].update({"reward_profile": "legacy", "self_curriculum_probability": 0.0})
    if output_dir is not None:
        folder = Path(output_dir)
        write_json(folder / "evaluation.json", report)
        # Reusing an output folder must not retain a replay from an older report.
        for label in ("success", "failure"):
            (folder / f"replay_{label}.json").unlink(missing_ok=True)
        with (folder / "episodes.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        if save_replays:
            for success, label in [(True, "success"), (False, "failure")]:
                chosen = next((r for r in results if bool(r["success"]) is success), None)
                if chosen:
                    _, replay = episode(policy_name, board_size, chosen["seed"], checkpoint, record=True, deterministic=deterministic)
                    write_json(folder / f"replay_{label}.json", replay)
    return report


def compare(board_size: int, episodes: int, seed_start: int, models: dict[str, str | Path], output_dir: str | Path) -> dict:
    folder = Path(output_dir)
    reports = {}
    for label in ["random", "greedy", "hamiltonian", *models]:
        print(f"Evaluating {label}: {episodes} episodes on {board_size}x{board_size}", flush=True)
        reports[label] = evaluate("neural" if label in models else label, board_size, episodes, seed_start,
                                  models.get(label), folder / label)
    comparison = {"version": 1, "board_size": board_size, "seed_start": seed_start, "episodes": episodes,
                  "seeds": list(range(seed_start, seed_start + episodes)),
                  "results": {label: report["summary"] for label, report in reports.items()},
                  "checkpoints": {k: str(v) for k, v in models.items()}}
    write_json(folder / "comparison.json", comparison)
    rows = [f"# {board_size}×{board_size} 同种子比赛", "",
            f"每种策略 {episodes} 局；测试种子 {seed_start}–{seed_start + episodes - 1}。神经网络为确定性推理，无动作屏蔽或路径算法。", "",
            "| 策略 | 全覆盖成功 | 平均覆盖率 | 成功局平均步数 |", "|---|---:|---:|---:|"]
    for label, summary in comparison["results"].items():
        steps = f'{summary["mean_success_steps"]:.1f}' if summary["mean_success_steps"] is not None else "—"
        rows.append(f'| {label} | {summary["successes"]}/{episodes} | {summary["mean_coverage"]:.1%} | {steps} |')
    rows += ["", "固定初始蛇身；种子随机化食物位置。模型来源请结合各训练目录的 config.json 与 lineage.json 核对。",
             "详细逐局数据和失败/成功回放保存在各策略子目录；95% Wilson 区间见 JSON。", ""]
    (folder / "comparison.md").write_text("\n".join(rows), encoding="utf-8")
    return comparison
