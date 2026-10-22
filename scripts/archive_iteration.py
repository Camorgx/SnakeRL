"""Archive measured PPO results, provenance and representative traces (no training)."""

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from snakerl.evaluation import write_json
from snakerl.training import checkpoint_lineage
from snakerl.env import SnakeEnv
from snakerl.model import NeuralPolicy


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs" / "ppo_iteration"
DEST = ROOT / "reports" / "ppo_iteration"
DATA = DEST / "data"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def copy(path, name):
    shutil.copy2(path, DATA / name)


def paired(neural, baseline):
    reference = {row["seed"]: row for row in baseline["episodes"]}
    successful = [row for row in neural["episodes"] if row["success"]]
    differences = np.array([reference[row["seed"]]["steps"] - row["steps"] for row in successful])
    rng = np.random.default_rng(20261014)
    means = differences[rng.integers(0, len(differences), (5000, len(differences)))].mean(axis=1)
    reference_mean = float(np.mean([reference[row["seed"]]["steps"] for row in successful]))
    return {"successful_pairs": len(successful), "neural_success_mean_steps": float(np.mean([row["steps"] for row in successful])),
            "matched_hamiltonian_mean_steps": reference_mean, "mean_steps_saved": float(differences.mean()),
            "fraction_steps_saved": float(differences.mean() / reference_mean),
            "paired_bootstrap_ci95": np.quantile(means, [.025, .975]).tolist(),
            "bootstrap_replicates": 5000, "bootstrap_seed": 20261014,
            "condition": "Neural successful episodes only; not an unconditional superiority claim. Food coordinates can differ under matched environment seeds."}


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    names = ["stable8", "curriculum4", "curriculum6", "coverage8", "self_curriculum8", "separate8",
             "exploration8", "explicit_body8", "endgame8", "final8"]
    runs = []
    for name in names:
        folder = BASE / name
        progress = read(folder / "progress.json")
        assert progress["status"] == "completed", name
        config = read(folder / "config.json")
        ancestry = checkpoint_lineage(folder / "checkpoints" / "best.zip")
        assert ancestry["known"] and not ancestry["expert_demonstrations_used"], name
        runs.append({"name": name, "steps_collected": progress["actual_timesteps"], "best_validation": read(folder / "best_eval.json"),
                     "observation_version": config.get("observation_version", 1), "reward_profile": config.get("reward_profile", "legacy")})
        for filename in ("config.json", "best_eval.json", "eval_history.json", "runtime.json", "lineage.json"):
            if (folder / filename).exists():
                copy(folder / filename, f"{name}_{filename}")
    reports = {"4": read(BASE / "confirm4_1000" / "evaluation.json"),
               "6": read(BASE / "confirm6" / "pure_ppo" / "evaluation.json"),
               "8": read(BASE / "confirm8_1000" / "evaluation.json")}
    hamiltonian = {"4": read(BASE / "hamiltonian4_1000" / "evaluation.json"),
                   "6": read(BASE / "confirm6" / "hamiltonian" / "evaluation.json"),
                   "8": read(BASE / "hamiltonian8_1000" / "evaluation.json")}
    for side in reports:
        assert all(row.get("initial_length", 3) == 3 and not row.get("curriculum_start", False) for row in reports[side]["episodes"])
        write_json(DATA / f"pure_ppo_{side}_evaluation.json", reports[side])
        write_json(DATA / f"hamiltonian_{side}_evaluation.json", hamiltonian[side])
    manifest = []
    for name in ("curriculum4", "curriculum6", "final8"):
        model = BASE / name / "checkpoints" / "best.zip"
        manifest.append({"path": str(model.relative_to(ROOT)), "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                         "lineage": checkpoint_lineage(model)})
    write_json(DATA / "model_manifest.json", manifest)
    for name, source in (("comparison4.json", BASE / "confirm4" / "comparison.json"),
                         ("comparison6.json", BASE / "confirm6" / "comparison.json"),
                         ("comparison8.json", BASE / "confirm8_final" / "comparison.json"),
                         ("trajectory4.json", BASE / "adaptive4_confirm" / "trajectory.json"),
                         ("trajectory6.json", BASE / "adaptive6_confirm" / "trajectory.json"),
                         ("trajectory8.json", BASE / "adaptive8_confirm" / "trajectory.json")):
        copy(source, name)
    for side, folder in (("4", "confirm4"), ("6", "confirm6"), ("8", "confirm8_final")):
        shutil.copy2(BASE / folder / "comparison.png", DEST / f"comparison{side}.png")
    shutil.copy2(BASE / "final8" / "training.png", DEST / "training8.png")
    shutil.copy2(BASE / "curriculum4" / "training.png", DEST / "training4.png")
    shutil.copy2(BASE / "final8" / "metrics.csv", DATA / "final8_metrics.csv")
    shutil.copy2(BASE / "adaptive8_confirm" / "replay_adaptive_seed90000.json", DEST / "adaptive8_replay.json")
    replay = read(DEST / "adaptive8_replay.json")
    policy = NeuralPolicy(BASE / "final8" / "checkpoints" / "best.zip")
    env = SnakeEnv(8, observation_version=policy.observation_version)
    env.reset(seed=replay["seed"])
    assert env.state() == replay["frames"][0]["state"]
    for frame in replay["frames"][1:]:
        probabilities = policy.probabilities(env)
        np.testing.assert_allclose(probabilities, frame["probabilities"], atol=1e-6)
        assert int(probabilities.argmax()) == frame["action"]
        _, reward, terminated, truncated, _ = env.step(frame["action"])
        assert reward == frame["reward"] and env.state() == frame["state"]
    assert terminated and not truncated and env.success and len(env.snake) == 64
    env.close()
    summary = {"date": "2026-10-14", "total_new_steps_collected": sum(run["steps_collected"] for run in runs),
               "runs": runs, "evaluation": {side: reports[side]["summary"] for side in reports},
               "successful_seed_efficiency": {side: paired(reports[side], hamiltonian[side]) for side in reports},
               "trajectory_analysis": {side: {key: value for key, value in read(DATA / f"trajectory{side}.json").items() if key != "records"} for side in reports}}
    write_json(DATA / "summary.json", summary)
    print(json.dumps({"evaluation": summary["evaluation"], "efficiency": summary["successful_seed_efficiency"]}, indent=2))


if __name__ == "__main__":
    main()
