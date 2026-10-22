"""Seeded PPO training, held-out evaluation, and optional expert imitation.

The Hamiltonian expert is imported only by the demonstration collector. Every
training evaluation calls the neural actor directly with observations.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from snakerl.env import SnakeEnv
from snakerl.model import SnakeCNN


@dataclass
class TrainConfig:
    board_size: int = 8
    total_timesteps: int = 100_000
    n_envs: int = 8
    seed: int = 0
    output_dir: str = "runs/ppo"
    device: str = "cpu"
    threads: int = 2
    n_steps: int = 128
    batch_size: int = 256
    n_epochs: int = 4
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    mode: str = "ppo"
    vec_env: str = "dummy"
    max_no_food_steps: int | None = None
    max_steps: int | None = None
    initial_length: int = 3
    distance_reward: float = 0.02
    eval_episodes: int = 10
    eval_freq: int = 25_000
    checkpoint_freq: int = 25_000
    init_checkpoint: str | None = None
    bc_samples: int = 20_000
    bc_epochs: int = 20
    bc_batch_size: int = 256
    bc_learning_rate: float = 1e-3
    bc_validation_fraction: float = 0.2
    verbose: int = 1
    final_learning_rate: float | None = None
    final_ent_coef: float | None = None
    target_kl: float | None = None
    observation_version: int = 1
    reward_profile: str = "legacy"
    random_start: bool = False
    require_pure_ppo: bool = False
    self_curriculum_probability: float = 0.0
    curriculum_min_coverage: float = 0.7
    share_features_extractor: bool = True
    initial_actor_temperature: float = 1.0

    def validate(self) -> None:
        if not np.isfinite(self.initial_actor_temperature) or self.initial_actor_temperature <= 0:
            raise ValueError("initial_actor_temperature must be finite and positive")
        if not np.isfinite(self.self_curriculum_probability) or not 0 <= self.self_curriculum_probability <= 1:
            raise ValueError("self_curriculum_probability must be in [0,1]")
        if not np.isfinite(self.curriculum_min_coverage) or not 0 < self.curriculum_min_coverage < 1:
            raise ValueError("curriculum_min_coverage must be in (0,1)")
        if isinstance(self.observation_version, bool) or self.observation_version not in (1, 2, 3) or self.reward_profile not in ("legacy", "coverage"):
            raise ValueError("Invalid observation version or reward profile")
        if self.require_pure_ppo and self.mode != "ppo":
            raise ValueError("Pure PPO cannot use imitation training")
        if self.mode == "imitation_ppo" and self.random_start:
            raise ValueError("The demonstration expert requires the canonical start")
        if self.final_learning_rate is not None and (not np.isfinite(self.final_learning_rate) or self.final_learning_rate <= 0):
            raise ValueError("final_learning_rate must be finite and positive")
        for name in ("final_ent_coef", "target_kl"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value < 0 or (name == "target_kl" and value == 0)):
                raise ValueError(f"Invalid {name}")
        for name, minimum in (("board_size", 4), ("total_timesteps", 0), ("n_envs", 1),
                              ("n_steps", 2), ("batch_size", 2), ("n_epochs", 1),
                              ("threads", 1), ("seed", 0), ("initial_length", 1),
                              ("eval_episodes", 1), ("eval_freq", 1), ("checkpoint_freq", 1)):
            SnakeEnv._positive_integer(getattr(self, name), name, minimum)
        for name in ("max_steps", "max_no_food_steps"):
            if getattr(self, name) is not None:
                SnakeEnv._positive_integer(getattr(self, name), name)
        if not all(np.isfinite(value) for value in (self.learning_rate, self.gamma, self.gae_lambda,
                                                   self.clip_range, self.ent_coef)):
            raise ValueError("PPO numeric settings must be finite")
        if not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1 or self.clip_range <= 0 or self.ent_coef < 0:
            raise ValueError("Invalid PPO discount, advantage, clip, or entropy setting")
        if self.board_size < 4:
            raise ValueError("Training board_size must be at least 4")
        if self.initial_length < 1 or self.initial_length > self.board_size:
            raise ValueError("initial_length must be between 1 and board_size")
        if not np.isfinite(self.distance_reward) or self.distance_reward < 0:
            raise ValueError("distance_reward must be finite and nonnegative")
        if self.total_timesteps < 0 or self.n_envs < 1 or self.n_steps < 2:
            raise ValueError("total_timesteps >= 0, n_envs >= 1, and n_steps >= 2 are required")
        rollout_size = self.n_steps * self.n_envs
        if self.batch_size < 2 or self.batch_size > rollout_size or rollout_size % self.batch_size:
            raise ValueError("batch_size must divide n_steps * n_envs and be at least 2")
        if self.n_epochs < 1 or self.threads < 1 or self.learning_rate <= 0:
            raise ValueError("n_epochs, threads, and learning_rate must be positive")
        if self.seed < 0 or self.eval_episodes < 1 or self.eval_freq < 1 or self.checkpoint_freq < 1:
            raise ValueError("seed must be nonnegative; evaluation/checkpoint settings must be positive")
        if self.mode not in {"ppo", "imitation_ppo"}:
            raise ValueError("mode must be ppo or imitation_ppo")
        if self.vec_env not in {"dummy", "subproc"}:
            raise ValueError("vec_env must be dummy or subproc")
        if self.mode == "imitation_ppo":
            for name, minimum in (("bc_samples", 100), ("bc_epochs", 1), ("bc_batch_size", 1)):
                SnakeEnv._positive_integer(getattr(self, name), name, minimum)
            if self.board_size % 2:
                raise ValueError("Hamiltonian demonstrations require an even square board")
            if self.bc_samples < 100 or self.bc_epochs < 1 or self.bc_batch_size < 1:
                raise ValueError("Imitation requires bc_samples >= 100 and positive epochs/batch size")
            if not np.isfinite(self.bc_learning_rate) or self.bc_learning_rate <= 0 or not 0 < self.bc_validation_fraction < 0.5:
                raise ValueError("Invalid imitation learning rate or validation fraction")


def _write_json(path: Path, data: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def runtime_metadata(config: TrainConfig | None = None) -> dict[str, Any]:
    """Record the execution stack without including secrets or machine paths."""
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "dependencies": {
            name: version(name) for name in ("torch", "gymnasium", "stable-baselines3", "numpy", "tensorboard")
        },
        "torch_build": torch.__version__,
        "device": config.device if config is not None else "cpu",
        "threads": config.threads if config is not None else torch.get_num_threads(),
        "seed": config.seed if config is not None else None,
        "cudnn_deterministic": True if config is not None else bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": False if config is not None else bool(torch.backends.cudnn.benchmark),
        "note": "Fixed seeds aid repeatability; different dependency versions/devices may change numerical results.",
    }


def checkpoint_lineage(checkpoint: str | Path, seen: set[Path] | None = None) -> dict[str, Any]:
    """Audit local run configs transitively; unknown copied weights are not certified pure."""
    path = Path(checkpoint).resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError("Checkpoint ancestry contains a cycle")
    seen.add(path)
    folder = path.parent.parent if path.parent.name == "checkpoints" else path.parent
    config_path = folder / "config.json"
    if not config_path.is_file():
        return {"known": False, "expert_demonstrations_used": None, "ancestors": [str(path)]}
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expert = config.get("mode") == "imitation_ppo"
    known = config.get("mode") in ("ppo", "imitation_ppo")
    ancestors = [str(path)]
    if config.get("init_checkpoint"):
        parent = Path(config["init_checkpoint"])
        if not parent.is_absolute():
            candidates = [Path.cwd() / parent, *[ancestor / parent for ancestor in folder.parents]]
            parent = next((candidate for candidate in candidates if candidate.is_file()), folder / parent)
        source = checkpoint_lineage(parent, seen)
        known = known and source["known"]
        expert = expert or bool(source["expert_demonstrations_used"])
        ancestors.extend(source["ancestors"])
    return {"known": known, "expert_demonstrations_used": expert if known else (True if expert else None), "ancestors": ancestors}


@dataclass
class _EnvironmentFactory:
    config: TrainConfig
    rank: int
    output_dir: Path

    def __call__(self) -> Monitor:
        env = SnakeEnv(
            board_size=self.config.board_size,
            max_no_food_steps=self.config.max_no_food_steps,
            max_steps=self.config.max_steps,
            initial_length=self.config.initial_length,
            distance_reward=self.config.distance_reward,
            observation_version=self.config.observation_version,
            reward_profile=self.config.reward_profile,
            random_start=self.config.random_start,
            self_curriculum_probability=self.config.self_curriculum_probability,
            curriculum_min_coverage=self.config.curriculum_min_coverage,
        )
        env.action_space.seed(self.config.seed + self.rank)
        return Monitor(
            env,
            filename=str(self.output_dir / "episodes" / f"worker_{self.rank}"),
            info_keywords=("length", "coverage", "steps", "foods_eaten", "success", "termination_reason"),
        )


def _make_vector_env(config: TrainConfig, output_dir: Path) -> VecEnv:
    factories = [_EnvironmentFactory(config, rank, output_dir) for rank in range(config.n_envs)]
    if config.vec_env == "subproc":
        vector = SubprocVecEnv(factories, start_method="spawn")
    else:
        # Batched collection in one process is faster for these tiny environments.
        # Use subproc explicitly when actual worker processes are desirable.
        vector = DummyVecEnv(factories)
    vector.seed(config.seed)
    return vector


def _create_model(config: TrainConfig, vector: VecEnv, output_dir: Path) -> PPO:
    model = PPO(
        "CnnPolicy",
        vector,
        policy_kwargs={
            "features_extractor_class": SnakeCNN,
            "features_extractor_kwargs": {"features_dim": 128},
            "normalize_images": False,
            "net_arch": {"pi": [128], "vf": [128]},
            "activation_fn": torch.nn.ReLU,
            "share_features_extractor": config.share_features_extractor,
        },
        learning_rate=LinearLearningRate(config.learning_rate, config.final_learning_rate) if config.final_learning_rate is not None else config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        target_kl=config.target_kl,
        tensorboard_log=str(output_dir / "tensorboard"),
        seed=config.seed,
        device=config.device,
        verbose=config.verbose,
    )
    if config.init_checkpoint:
        prior = PPO.load(config.init_checkpoint, device=config.device)
        # A fresh optimizer and observation space belong to the new stage. The
        # CNN and actor/critic weights have dimensions independent of board size.
        try:
            weights = prior.policy.state_dict()
            if prior.observation_space.shape[0] < vector.observation_space.shape[0]:
                target = model.policy.state_dict()
                for key, tensor in list(weights.items()):
                    if tensor.shape == target[key].shape:
                        continue
                    if key.endswith("cnn.0.weight"):
                        expanded = torch.zeros_like(target[key])
                        expanded[:, :tensor.shape[1]] = tensor
                    elif key.endswith("projection.0.weight"):
                        expanded = torch.zeros_like(target[key])
                        expanded[:, :tensor.shape[1]] = tensor
                    else:
                        raise ValueError(f"Unsupported observation migration for {key}")
                    weights[key] = expanded
            model.policy.load_state_dict(weights, strict=True)
        except RuntimeError as error:
            raise ValueError("Checkpoint architecture cannot transfer into SnakeCNN") from error
        _write_json(
            output_dir / "transfer.json",
            {
                "checkpoint": str(Path(config.init_checkpoint).resolve()),
                "source_observation_shape": list(prior.observation_space.shape),
                "target_observation_shape": list(model.observation_space.shape),
                "optimizer_reset": True,
                "actor_critic_encoders_shared": config.share_features_extractor,
                "observation_migration": "new observation inputs initialized with zero weights" if prior.observation_space.shape[0] < vector.observation_space.shape[0] else None,
                "note": "Weight transfer is an experiment; it does not demonstrate success on the new board.",
            },
        )
    if config.initial_actor_temperature != 1.0:
        with torch.no_grad():
            model.policy.action_net.weight.div_(config.initial_actor_temperature)
            model.policy.action_net.bias.div_(config.initial_actor_temperature)
    return model


@dataclass
class LinearLearningRate:
    initial: float
    final: float

    def __call__(self, progress_remaining: float) -> float:
        return self.final + (self.initial - self.final) * max(0.0, min(1.0, progress_remaining))


_METRIC_FIELDS = (
    "kind", "timesteps", "episode", "seed", "reward", "length", "coverage",
    "steps", "foods_eaten", "success", "termination_reason", "elapsed_seconds", "curriculum_start", "initial_length",
)


class _TrainingCallback(BaseCallback):
    def __init__(self, config: TrainConfig, output_dir: Path):
        super().__init__(verbose=0)
        self.config = config
        self.output_dir = output_dir
        self.started = time.monotonic()
        self.episodes = 0
        self.next_evaluation = config.eval_freq
        self.next_checkpoint = config.checkpoint_freq
        self.last_progress = 0.0
        self.latest_eval: dict[str, Any] | None = None
        self.latest_checkpoint: str | None = None
        self.best_score: tuple[float, float, float] | None = None
        self.eval_history: list[dict[str, Any]] = []
        self.stream = (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.stream, fieldnames=_METRIC_FIELDS)
        self.writer.writeheader()
        self.stream.flush()

    def progress(self, status: str, **extra: Any) -> None:
        timesteps = int(self.model.num_timesteps) if hasattr(self, "model") else 0
        _write_json(
            self.output_dir / "progress.json",
            {
                "status": status,
                "mode": self.config.mode,
                "board_size": self.config.board_size,
                "timesteps": timesteps,
                "target_timesteps": self.config.total_timesteps,
                "episodes": self.episodes,
                "elapsed_seconds": round(time.monotonic() - self.started, 3),
                "latest_eval": self.latest_eval,
                "checkpoint": self.latest_checkpoint,
                **extra,
            },
        )
        self.last_progress = time.monotonic()

    def _on_training_start(self) -> None:
        self.progress("training")

    def _on_rollout_start(self) -> None:
        if self.config.final_ent_coef is not None:
            fraction = min(1.0, self.num_timesteps / max(1, self.config.total_timesteps))
            self.model.ent_coef = self.config.ent_coef + fraction * (self.config.final_ent_coef - self.config.ent_coef)

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if not done:
                continue
            self.episodes += 1
            episode = info.get("episode", {})
            record = {
                "kind": "train",
                "timesteps": self.num_timesteps,
                "episode": self.episodes,
                # Monitor supplies the actual reset seed if the env includes it.
                "seed": info.get("seed", ""),
                "reward": float(episode.get("r", 0)),
                "length": int(info["length"]),
                "coverage": float(info["coverage"]),
                "steps": int(info["steps"]),
                "foods_eaten": int(info["foods_eaten"]),
                "success": int(bool(info["success"])),
                "termination_reason": info["termination_reason"],
                "curriculum_start": int(bool(info.get("curriculum_start", False))),
                "initial_length": info.get("initial_length", self.config.initial_length),
                "elapsed_seconds": round(time.monotonic() - self.started, 3),
            }
            self.writer.writerow(record)
            self.logger.record_mean("snake/coverage", record["coverage"])
            self.logger.record_mean("snake/success", record["success"])
            self.logger.record_mean("snake/foods_eaten", record["foods_eaten"])
            cohort = "curriculum" if record["curriculum_start"] else "standard"
            self.logger.record_mean(f"snake/{cohort}_coverage", record["coverage"])
            self.logger.record_mean(f"snake/{cohort}_success", record["success"])
            self.stream.flush()
        if self.num_timesteps >= self.next_checkpoint:
            checkpoint = self.output_dir / "checkpoints" / f"step_{self.num_timesteps}.zip"
            self.model.save(checkpoint)
            self.latest_checkpoint = str(checkpoint)
            self.next_checkpoint = self.num_timesteps + self.config.checkpoint_freq
        if self.num_timesteps >= self.next_evaluation:
            self.evaluate("periodic")
            self.next_evaluation = self.num_timesteps + self.config.eval_freq
        if time.monotonic() - self.last_progress >= 2:
            self.progress("training")
        return True

    def evaluate(self, phase: str) -> dict[str, Any]:
        env = SnakeEnv(
            board_size=self.config.board_size,
            max_no_food_steps=self.config.max_no_food_steps,
            max_steps=self.config.max_steps,
            initial_length=self.config.initial_length,
            distance_reward=self.config.distance_reward,
            observation_version=self.config.observation_version,
            reward_profile=self.config.reward_profile,
        )
        records = []
        previous_training_mode = self.model.policy.training
        self.model.policy.set_training_mode(False)
        try:
            seeds = [1_000_000 + self.config.seed * 10_000 + index for index in range(self.config.eval_episodes)]
            environments = [SnakeEnv(board_size=self.config.board_size, max_no_food_steps=self.config.max_no_food_steps,
                                     max_steps=self.config.max_steps, initial_length=self.config.initial_length,
                                     distance_reward=self.config.distance_reward, observation_version=self.config.observation_version,
                                     reward_profile=self.config.reward_profile) for _ in seeds]
            observations = [candidate.reset(seed=seed)[0] for candidate, seed in zip(environments, seeds)]
            totals = [0.0] * len(seeds)
            final_infos = [None] * len(seeds)
            active = list(range(len(seeds)))
            while active:
                actions, _ = self.model.predict(np.stack([observations[index] for index in active]), deterministic=True)
                still_active = []
                for index, action in zip(active, actions):
                    observations[index], reward, terminated, truncated, info = environments[index].step(int(action))
                    totals[index] += float(reward)
                    if terminated or truncated:
                        final_infos[index] = info
                    else:
                        still_active.append(index)
                active = still_active
            for index, (seed, total_reward, info) in enumerate(zip(seeds, totals, final_infos)):
                record = {
                    "kind": "eval", "timesteps": int(self.model.num_timesteps),
                    "episode": index + 1, "seed": seed, "reward": total_reward,
                    "length": int(info["length"]), "coverage": float(info["coverage"]),
                    "steps": int(info["steps"]), "foods_eaten": int(info["foods_eaten"]),
                    "success": int(bool(info["success"])), "termination_reason": info["termination_reason"],
                    "curriculum_start": 0, "initial_length": self.config.initial_length,
                    "elapsed_seconds": round(time.monotonic() - self.started, 3),
                }
                records.append(record)
                self.writer.writerow(record)
        finally:
            self.model.policy.set_training_mode(previous_training_mode)
            env.close()
            for candidate in locals().get("environments", []):
                candidate.close()
        self.stream.flush()
        summary = {
            "phase": phase,
            "timesteps": int(self.model.num_timesteps),
            "episodes": len(records),
            "successes": sum(record["success"] for record in records),
            "success_rate": float(np.mean([record["success"] for record in records])),
            "mean_coverage": float(np.mean([record["coverage"] for record in records])),
            "mean_reward": float(np.mean([record["reward"] for record in records])),
            "mean_steps": float(np.mean([record["steps"] for record in records])),
            "mean_success_steps": float(np.mean([record["steps"] for record in records if record["success"]])) if any(record["success"] for record in records) else None,
            "seeds": [record["seed"] for record in records],
        }
        self.latest_eval = summary
        self.eval_history.append(summary)
        _write_json(self.output_dir / "eval_history.json", self.eval_history)
        score = (summary["success_rate"], -summary["mean_success_steps"] if summary["mean_success_steps"] is not None else summary["mean_coverage"], summary["mean_reward"])
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.model.save(self.output_dir / "checkpoints" / "best")
            _write_json(self.output_dir / "best_eval.json", summary)
        # logger exists after _setup_learn; initial evaluation precedes that.
        if hasattr(self.model, "_logger"):
            self.model.logger.record("eval/success_rate", summary["success_rate"])
            self.model.logger.record("eval/coverage", summary["mean_coverage"])
            self.model.logger.record("eval/mean_steps", summary["mean_steps"])
        if self.config.verbose:
            print(
                f"Evaluation {phase}: {summary['successes']}/{summary['episodes']} full boards, "
                f"mean coverage {summary['mean_coverage']:.1%} at {summary['timesteps']} steps",
                flush=True,
            )
        return summary

    def close(self) -> None:
        self.stream.close()


def _collect_demonstrations(config: TrainConfig, output_dir: Path) -> tuple[TensorDataset, TensorDataset]:
    # This is the only training code allowed to consult the traditional expert.
    from snakerl.baselines import HamiltonianPolicy

    rng = np.random.default_rng(config.seed)
    env = SnakeEnv(
        board_size=config.board_size,
        max_no_food_steps=2 * config.board_size**2,
        max_steps=2 * config.board_size**4,
        initial_length=config.initial_length,
        distance_reward=config.distance_reward,
        observation_version=config.observation_version,
        reward_profile=config.reward_profile,
    )
    expert = HamiltonianPolicy()
    episodes: list[dict[str, Any]] = []
    count = 0
    expert_steps = 0
    sample_limit = math.ceil(config.bc_samples / 5)
    try:
        # Collect at least five distinct episodes so validation is genuinely
        # separated by episode seed even for a very small requested dataset.
        while count < config.bc_samples or len(episodes) < 5:
            seed = 2_000_000 + config.seed * 10_000 + len(episodes)
            observation, _ = env.reset(seed=seed)
            observations, actions = [], []
            steps_seen = 0
            while True:
                action = expert.act(env)
                steps_seen += 1
                if len(actions) < sample_limit:
                    observations.append(observation.copy())
                    actions.append(action)
                else:
                    # Reservoir sampling bounds memory on large boards while
                    # retaining demonstrations from all stages of an episode.
                    replace_index = int(rng.integers(steps_seen))
                    if replace_index < sample_limit:
                        observations[replace_index] = observation.copy()
                        actions[replace_index] = action
                observation, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            if not info["success"]:
                raise RuntimeError(
                    f"Hamiltonian demonstration failed for seed {seed}: {info['termination_reason']}"
                )
            episodes.append({"seed": seed, "observations": observations, "actions": actions})
            count += len(actions)
            expert_steps += steps_seen
    finally:
        env.close()
    shuffled = rng.permutation(len(episodes))
    validation_count = max(1, round(len(episodes) * config.bc_validation_fraction))
    validation_indices = set(int(index) for index in shuffled[:validation_count])
    datasets = []
    provenance = {
        "expert": "hamiltonian",
        "requested_samples": config.bc_samples,
        "collected_samples": count,
        "expert_steps": expert_steps,
        "sampling": "uniform reservoir per episode when episode exceeds sample limit",
        "per_episode_sample_limit": sample_limit,
        "split_unit": "episode_seed",
        "expert_used_at_inference": False,
    }
    for split, is_validation in (("train", False), ("validation", True)):
        selected = [episode for index, episode in enumerate(episodes) if (index in validation_indices) == is_validation]
        observations = np.asarray([observation for episode in selected for observation in episode["observations"]], dtype=np.float32)
        actions = np.asarray([action for episode in selected for action in episode["actions"]], dtype=np.int64)
        datasets.append(TensorDataset(torch.from_numpy(observations), torch.from_numpy(actions)))
        provenance[f"{split}_samples"] = len(actions)
        provenance[f"{split}_seeds"] = [episode["seed"] for episode in selected]
    _write_json(output_dir / "bc_dataset.json", provenance)
    return datasets[0], datasets[1]


def _behavior_cloning(model: PPO, config: TrainConfig, output_dir: Path, callback: _TrainingCallback) -> None:
    callback.progress("collecting_demonstrations")
    train_data, validation_data = _collect_demonstrations(config, output_dir)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(train_data, batch_size=config.bc_batch_size, shuffle=True, generator=generator)
    validation_loader = DataLoader(validation_data, batch_size=config.bc_batch_size, shuffle=False)
    policy = model.policy
    actor_parameters = (
        list(policy.features_extractor.parameters())
        + list(policy.mlp_extractor.policy_net.parameters())
        + list(policy.action_net.parameters())
    )
    optimizer = torch.optim.Adam(actor_parameters, lr=config.bc_learning_rate)
    best_accuracy = -1.0
    best_loss = float("inf")
    best_weights: dict[str, torch.Tensor] | None = None
    with (output_dir / "bc_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("epoch", "train_loss", "train_accuracy", "validation_loss", "validation_accuracy"))
        writer.writeheader()
        for epoch in range(1, config.bc_epochs + 1):
            policy.set_training_mode(True)
            train_loss, train_correct, train_count = 0.0, 0, 0
            for observations, actions in train_loader:
                observations, actions = observations.to(model.device), actions.to(model.device)
                logits = policy.get_distribution(observations).distribution.logits
                loss = F.cross_entropy(logits, actions)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor_parameters, 1.0)
                optimizer.step()
                train_loss += float(loss.detach()) * len(actions)
                train_correct += int((logits.argmax(dim=1) == actions).sum())
                train_count += len(actions)
            policy.set_training_mode(False)
            validation_loss, validation_correct, validation_count = 0.0, 0, 0
            with torch.no_grad():
                for observations, actions in validation_loader:
                    observations, actions = observations.to(model.device), actions.to(model.device)
                    logits = policy.get_distribution(observations).distribution.logits
                    validation_loss += float(F.cross_entropy(logits, actions)) * len(actions)
                    validation_correct += int((logits.argmax(dim=1) == actions).sum())
                    validation_count += len(actions)
            row = {
                "epoch": epoch,
                "train_loss": train_loss / train_count,
                "train_accuracy": train_correct / train_count,
                "validation_loss": validation_loss / validation_count,
                "validation_accuracy": validation_correct / validation_count,
            }
            writer.writerow(row)
            stream.flush()
            if row["validation_accuracy"] > best_accuracy or (
                row["validation_accuracy"] == best_accuracy and row["validation_loss"] < best_loss
            ):
                best_accuracy, best_loss = row["validation_accuracy"], row["validation_loss"]
                best_weights = {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()}
            callback.progress("imitation", bc_epoch=epoch, bc_epochs=config.bc_epochs, bc_latest=row)
            if config.verbose:
                print(f"BC epoch {epoch}/{config.bc_epochs}: validation accuracy {row['validation_accuracy']:.2%}", flush=True)
    assert best_weights is not None
    policy.load_state_dict(best_weights)
    policy.set_training_mode(False)
    model.save(output_dir / "checkpoints" / "bc")
    _write_json(output_dir / "bc_summary.json", {
        "best_validation_accuracy": best_accuracy,
        "best_validation_loss": best_loss,
        "epochs": config.bc_epochs,
        "note": "Expert action accuracy is not full-board success; see held-out neural evaluation.",
    })


def train(config: TrainConfig) -> Path:
    """Train a neural policy and return its final saved PPO checkpoint.

    PPO collects complete vector rollouts, so actual timesteps may exceed the
    requested budget by less than n_envs * n_steps. With total_timesteps=0,
    imitation-only or initialization experiments still save/evaluate a model.
    """
    config.validate()
    if config.init_checkpoint:
        input_path = Path(config.init_checkpoint)
        if not input_path.is_file() and Path(str(input_path) + ".zip").is_file():
            input_path = Path(str(input_path) + ".zip")
        if not input_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {config.init_checkpoint}")
        config.init_checkpoint = str(input_path.resolve())
    lineage = checkpoint_lineage(config.init_checkpoint) if config.init_checkpoint else {"known": True, "expert_demonstrations_used": False, "ancestors": []}
    if config.mode == "imitation_ppo":
        lineage["expert_demonstrations_used"] = True
    if config.require_pure_ppo and (not lineage["known"] or lineage["expert_demonstrations_used"]):
        raise ValueError("Pure PPO requires known expert-free checkpoint ancestry")
    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "config.json").exists():
        raise FileExistsError(f"Run already exists at {output_dir}; choose a new output_dir")
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    (output_dir / "episodes").mkdir(exist_ok=True)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    _write_json(output_dir / "config.json", asdict(config))
    _write_json(output_dir / "runtime.json", runtime_metadata(config))
    _write_json(output_dir / "lineage.json", {**lineage, "algorithm": "PPO", "action_masking": False,
                                               "planner_at_inference": False,
                                               "initial_checkpoint_sha256": hashlib.sha256(Path(config.init_checkpoint).read_bytes()).hexdigest() if config.init_checkpoint else None,
                                               "source_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() for name in ("env.py", "model.py", "training.py")}})
    _write_json(output_dir / "environment.json", {
        "width": config.board_size,
        "height": config.board_size,
        "initial_length": config.initial_length,
        "distance_reward": config.distance_reward,
        "max_no_food_steps": config.max_no_food_steps if config.max_no_food_steps is not None else config.board_size**2 * 4,
        "max_steps": config.max_steps if config.max_steps is not None else config.board_size**4 * 4,
        "observation_shape": [{1: 9, 2: 11, 3: 13}[config.observation_version], config.board_size, config.board_size],
        "observation_version": config.observation_version,
        "reward_profile": config.reward_profile,
        "random_start_training": config.random_start,
        "self_curriculum_probability_training_only": config.self_curriculum_probability,
        "curriculum_min_coverage": config.curriculum_min_coverage,
        "actions": ["left", "straight", "right"],
        "neural_inference": "observation only; no expert, lookahead, or action mask",
    })
    vector = _make_vector_env(config, output_dir)
    callback = _TrainingCallback(config, output_dir)
    try:
        model = _create_model(config, vector, output_dir)
        # Set the callback model before imitation/initial evaluation; learn()
        # initializes its logger and environment-specific state later.
        callback.model = model
        model.save(output_dir / "checkpoints" / "initial")
        if config.mode == "imitation_ppo":
            _behavior_cloning(model, config, output_dir, callback)
        model.save(output_dir / "checkpoints" / "pretrain")
        callback.latest_checkpoint = str(output_dir / "checkpoints" / "pretrain.zip")
        callback.evaluate("pretrain")
        if config.total_timesteps:
            model.learn(total_timesteps=config.total_timesteps, callback=callback, tb_log_name=config.mode)
        final_checkpoint = output_dir / "final.zip"
        model.save(final_checkpoint)
        callback.evaluate("final")
        callback.progress("completed", checkpoint=str(final_checkpoint), actual_timesteps=int(model.num_timesteps))
        return final_checkpoint
    except BaseException as error:
        callback.progress("failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        callback.close()
        vector.close()
