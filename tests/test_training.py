import csv
import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from snakerl import baselines
from snakerl.env import SnakeEnv
from snakerl.model import NeuralPolicy
from snakerl.training import TrainConfig, train


@pytest.mark.parametrize("invalid", [
    {"max_steps": 0}, {"max_no_food_steps": -1}, {"board_size": 4.5},
    {"learning_rate": float("nan")}, {"gamma": 1.1}, {"ent_coef": -0.1},
])
def test_invalid_training_settings_fail_before_creating_run(tmp_path, invalid):
    output = tmp_path / "invalid_run"
    with pytest.raises(ValueError):
        train(TrainConfig(output_dir=str(output), **invalid))
    assert not output.exists()


def _forbidden_baseline(*args, **kwargs):
    raise AssertionError("Neural training/inference consulted a traditional baseline")


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory):
    folder = tmp_path_factory.mktemp("tiny_ppo")
    config = TrainConfig(
        board_size=4,
        total_timesteps=128,
        n_envs=2,
        n_steps=32,
        batch_size=32,
        n_epochs=1,
        threads=1,
        seed=19,
        output_dir=str(folder),
        max_steps=32,
        max_no_food_steps=16,
        eval_episodes=2,
        eval_freq=128,
        checkpoint_freq=128,
        verbose=0,
    )
    # Pure PPO must remain runnable when all traditional policies are disabled.
    with pytest.MonkeyPatch.context() as patch:
        for policy in (baselines.HamiltonianPolicy, baselines.GreedyPolicy, baselines.RandomPolicy):
            patch.setattr(policy, "act", _forbidden_baseline)
        patch.setattr(baselines, "hamiltonian_cycle", _forbidden_baseline)
        checkpoint = train(config)
    return config, folder, checkpoint


class ObservationOnly:
    """Any inference access to game internals fails, while observe() still works."""

    def __init__(self, observation):
        object.__setattr__(self, "_array", observation.copy())

    def __getattribute__(self, name):
        if name != "observe":
            raise AssertionError(f"Inference accessed forbidden environment attribute: {name}")
        return object.__getattribute__(self, name)

    def observe(self):
        return object.__getattribute__(self, "_array").copy()


def test_tiny_ppo_changes_actor_and_cnn_weights(trained_run):
    _, folder, checkpoint = trained_run
    before = PPO.load(folder / "checkpoints" / "pretrain.zip", device="cpu")
    after = PPO.load(checkpoint, device="cpu")
    assert before.num_timesteps == 0
    assert after.num_timesteps == 128
    initial = before.policy.state_dict()
    learned = after.policy.state_dict()
    assert not torch.equal(initial["action_net.weight"], learned["action_net.weight"])
    assert not torch.equal(initial["features_extractor.cnn.0.weight"], learned["features_extractor.cnn.0.weight"])
    assert all(torch.isfinite(value).all() for value in learned.values())


def test_checkpoint_probabilities_and_actions_match_single_and_batch(trained_run):
    _, _, checkpoint = trained_run
    policy = NeuralPolicy(checkpoint, threads=1)
    observations = []
    for seed in (3, 5, 7):
        env = SnakeEnv(4)
        observation, _ = env.reset(seed=seed)
        observations.append(observation)
    actions, batch_probabilities = policy.predict_batch(np.stack(observations))
    assert batch_probabilities.shape == (3, 3)
    assert np.isfinite(batch_probabilities).all()
    assert (batch_probabilities >= 0).all()
    np.testing.assert_allclose(batch_probabilities.sum(axis=1), 1, atol=1e-6)
    for index, observation in enumerate(observations):
        proxy = ObservationOnly(observation)
        probabilities = policy.probabilities(proxy)
        np.testing.assert_allclose(probabilities, batch_probabilities[index], rtol=1e-5, atol=1e-7)
        assert policy.act(proxy) == int(actions[index]) == int(probabilities.argmax())
    reloaded = NeuralPolicy(checkpoint, threads=1)
    actions_again, probabilities_again = reloaded.predict_batch(np.stack(observations))
    np.testing.assert_array_equal(actions, actions_again)
    np.testing.assert_array_equal(batch_probabilities, probabilities_again)


def test_neural_inference_uses_only_observation_and_does_not_mask_wall_action(trained_run, monkeypatch):
    _, _, checkpoint = trained_run
    for baseline in (baselines.HamiltonianPolicy, baselines.GreedyPolicy, baselines.RandomPolicy):
        monkeypatch.setattr(baseline, "act", _forbidden_baseline)
    monkeypatch.setattr(baselines, "hamiltonian_cycle", _forbidden_baseline)
    policy = NeuralPolicy(checkpoint, threads=1)
    env = SnakeEnv(4)
    observation, _ = env.reset(seed=0)
    # Force the learned actor to select left at the top wall. An action mask or
    # expert fallback would override this illegal action and fail the test.
    with torch.no_grad():
        policy.model.policy.action_net.weight.zero_()
        policy.model.policy.action_net.bias.copy_(torch.tensor([20.0, -20.0, -20.0]))
    proxy = ObservationOnly(observation)
    assert policy.probabilities(proxy)[0] > 0.999
    assert policy.act(proxy) == 0
    _, _, terminated, truncated, info = env.step(policy.act(proxy))
    assert terminated and not truncated
    assert info["termination_reason"] == "wall_collision"


def test_checkpoint_rejects_wrong_board_and_nonbatched_observation(trained_run):
    _, _, checkpoint = trained_run
    policy = NeuralPolicy(checkpoint, threads=1)
    env = SnakeEnv(6)
    observation, _ = env.reset(seed=0)
    with pytest.raises(ValueError, match="Checkpoint expects board 4x4"):
        policy.act(ObservationOnly(observation))
    with pytest.raises(ValueError, match="batched observations"):
        policy.predict_batch(observation)
    with pytest.raises(ValueError, match="batched observations"):
        policy.predict_batch(observation[None])


def test_curriculum_transfer_preserves_weights_and_updates_observation_shape(trained_run, tmp_path):
    config, _, source_checkpoint = trained_run
    target_config = replace(
        config,
        board_size=6,
        total_timesteps=0,
        output_dir=str(tmp_path / "transfer"),
        init_checkpoint=str(source_checkpoint),
    )
    checkpoint = train(target_config)
    source = PPO.load(source_checkpoint, device="cpu")
    target = PPO.load(checkpoint, device="cpu")
    assert source.observation_space.shape == (9, 4, 4)
    assert target.observation_space.shape == (9, 6, 6)
    assert target.num_timesteps == 0
    for name, tensor in source.policy.state_dict().items():
        assert torch.equal(tensor, target.policy.state_dict()[name]), name
    transfer = json.loads((checkpoint.parent / "transfer.json").read_text(encoding="utf-8"))
    assert transfer["source_observation_shape"] == [9, 4, 4]
    assert transfer["target_observation_shape"] == [9, 6, 6]
    assert transfer["optimizer_reset"] is True
    env = SnakeEnv(6)
    env.reset(seed=0)
    assert env.action_space.contains(NeuralPolicy(checkpoint, threads=1).act(env))
    progress = json.loads((checkpoint.parent / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "completed" and progress["actual_timesteps"] == 0


def test_training_artifacts_report_real_completed_run(trained_run):
    config, folder, checkpoint = trained_run
    progress = json.loads((folder / "progress.json").read_text(encoding="utf-8"))
    saved_config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    environment = json.loads((folder / "environment.json").read_text(encoding="utf-8"))
    runtime = json.loads((folder / "runtime.json").read_text(encoding="utf-8"))
    evaluations = json.loads((folder / "eval_history.json").read_text(encoding="utf-8"))
    assert progress["status"] == "completed"
    assert progress["timesteps"] == progress["actual_timesteps"] == config.total_timesteps
    assert progress["checkpoint"] == str(checkpoint)
    assert progress["episodes"] > 0
    assert saved_config["mode"] == "ppo" and saved_config["seed"] == config.seed
    assert environment["observation_shape"] == [9, 4, 4]
    assert runtime["dependencies"]["torch"]
    assert [evaluation["phase"] for evaluation in evaluations] == ["pretrain", "periodic", "final"]
    assert all(evaluation["seeds"] == [1_190_000, 1_190_001] for evaluation in evaluations)
    assert (folder / "checkpoints" / "best.zip").is_file()
    assert (folder / "checkpoints" / "step_128.zip").is_file()
    assert list((folder / "tensorboard").rglob("events.out.tfevents.*"))
    with (folder / "metrics.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert any(row["kind"] == "train" for row in rows)
    assert sum(row["kind"] == "eval" for row in rows) == 6


def test_training_refuses_to_overwrite_existing_run(trained_run):
    config, _, _ = trained_run
    with pytest.raises(FileExistsError, match="Run already exists"):
        train(config)
