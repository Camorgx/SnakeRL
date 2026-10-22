import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from snakerl.env import SnakeEnv
from snakerl.evaluation import episode, evaluate
from snakerl.model import NeuralPolicy
from snakerl.training import TrainConfig, checkpoint_lineage, train
from snakerl.trajectory import analyze_replay
from snakerl.cli import main


def test_markov_observation_exposes_clocks_and_resets_hunger_after_food():
    env = SnakeEnv(4, observation_version=2)
    observation, _ = env.reset(seed=3)
    assert observation.shape == (11, 4, 4)
    assert not observation[9:].any()
    env.food = (3, 1)
    observation, _, _, _, _ = env.step(1)
    assert np.all(observation[9] == 1 / 64)
    assert np.all(observation[10] == 1 / 1024)
    observation, _, _, _, _ = env.step(2)
    assert not observation[9].any()
    assert np.all(observation[10] == 2 / 1024)
    assert env.observation_space.contains(observation)


def test_time_counters_distinguish_identical_geometry():
    env = SnakeEnv(4, observation_version=2)
    first, _ = env.reset(seed=3)
    env.steps, env.no_food_steps = 20, 12
    second = env.observe()
    np.testing.assert_array_equal(first[:9], second[:9])
    assert not np.array_equal(first[9:], second[9:])


def test_coverage_reward_grows_with_food_and_preserves_game_rules():
    env = SnakeEnv(4, reward_profile="coverage", distance_reward=0)
    env.reset(seed=1)
    env.food = (3, 0)
    _, reward, terminated, truncated, info = env.step(1)
    assert reward == pytest.approx(1 + 2 * (4 / 16)**2 - 0.003)
    assert info["length"] == 4 and not terminated and not truncated
    _, reward, terminated, truncated, _ = env.step(1)
    assert reward == pytest.approx(-(1 + 4 * (4 / 16)**2))
    assert terminated and not truncated


def test_random_straight_starts_are_seeded_legal_and_diverse():
    env = SnakeEnv(6, random_start=True)
    starts = set()
    for seed in range(20):
        observation, _ = env.reset(seed=seed)
        state = env.state()
        repeated, _ = env.reset(seed=seed)
        assert env.state() == state
        np.testing.assert_array_equal(observation, repeated)
        assert len(set(env.snake)) == 3 and env.food not in env.snake
        assert all(0 <= x < 6 and 0 <= y < 6 for x, y in env.snake)
        starts.add((tuple(env.snake[0]), env.direction))
    assert len(starts) > 10


def test_fixed_hamiltonian_replay_is_not_mislabeled_adaptive():
    _, replay = episode("hamiltonian", 4, 32, record=True)
    analysis = analyze_replay(replay)
    assert analysis["success"]
    assert analysis["branching_positions"] == 0
    assert not analysis["incompatible_with_any_fixed_position_cycle"]
    assert analysis["canonical_cycle_edge_agreement"] == 1
    assert analysis["consumed_food_manhattan_lower_bound"] <= analysis["steps"]


def test_branching_metric_checks_actual_position_successors():
    env = SnakeEnv(4)
    env.reset(seed=0)
    env.food = (0, 3)
    frames = [{"state": env.state()}]
    for action in (2, 1, 0, 0, 1, 0, 1):
        _, _, terminated, truncated, _ = env.step(action)
        assert not terminated and not truncated
        frames.append({"state": env.state()})
    result = analyze_replay({"frames": frames, "seed": 0, "result": env.state()})
    assert result["incompatible_with_any_fixed_position_cycle"]


def test_self_curriculum_uses_own_reachable_states_without_inflating_test_initial_length():
    env = SnakeEnv(4, self_curriculum_probability=1, curriculum_min_coverage=0.25, observation_version=2)
    env.reset(seed=0)
    env.food = (3, 0)
    env.step(1)
    recorded_body = env.snake.copy()
    env.step(2)
    observation, info = env.reset(seed=42)
    assert env.snake == recorded_body
    assert info["curriculum_start"] and info["initial_length"] == 4
    assert info["foods_eaten"] == 0 and info["steps"] == 0
    assert not observation[9:].any()
    assert env.food not in env.snake
    standard = SnakeEnv(4, observation_version=2)
    _, info = standard.reset(seed=42)
    assert not info["curriculum_start"] and info["initial_length"] == 3
    assert not standard._self_states


def test_pure_gate_rejects_unknown_and_expert_checkpoint_before_writing(tmp_path):
    unknown = tmp_path / "copied.zip"
    unknown.touch()
    assert not checkpoint_lineage(unknown)["known"]
    for index, mode in enumerate((None, "imitation_ppo")):
        folder = tmp_path / f"source{index}"
        folder.mkdir()
        model = folder / "final.zip"
        model.touch()
        if mode:
            (folder / "config.json").write_text(json.dumps({"mode": mode}))
        target = tmp_path / f"target{index}"
        with pytest.raises(ValueError, match="expert-free"):
            train(TrainConfig(init_checkpoint=str(model), require_pure_ppo=True, output_dir=str(target)))
        assert not target.exists()


@pytest.mark.parametrize("version", [2, 3])
def test_expanded_observation_transfer_preserves_actor_with_zero_new_weights(tmp_path, version):
    config = TrainConfig(board_size=4, total_timesteps=0, output_dir=str(tmp_path / "source"),
                         eval_episodes=1, threads=1, verbose=0, max_steps=16)
    source = train(config)
    target = train(replace(config, observation_version=version, init_checkpoint=str(source),
                           require_pure_ppo=True, output_dir=str(tmp_path / "target")))
    first, second = NeuralPolicy(source, threads=1), NeuralPolicy(target, threads=1)
    env = SnakeEnv(4, observation_version=version)
    observation, _ = env.reset(seed=7)
    observation[9:] = 0.8
    _, old_probs = first.predict_batch(observation[None, :9])
    _, new_probs = second.predict_batch(observation[None])
    np.testing.assert_allclose(old_probs, new_probs, atol=1e-7)
    lineage = json.loads((target.parent / "lineage.json").read_text())
    assert lineage["known"] and not lineage["expert_demonstrations_used"]


def test_explicit_occupancy_and_tail_are_physical_state_and_update_on_movement():
    env = SnakeEnv(8, observation_version=3)
    observation, _ = env.reset(seed=0)
    assert observation.shape == (13, 8, 8)
    assert observation[11].sum() == 3 and observation[12].sum() == 1
    assert observation[12, 0, 0] == 1
    env.food = (7, 7)
    observation, _, _, _, _ = env.step(1)
    assert observation[11].sum() == 3 and observation[12, 0, 1] == 1
    assert observation[11, 0, 0] == 0
    assert env.observation_space.contains(observation)


def test_separate_critic_backward_does_not_modify_actor_encoder(tmp_path):
    checkpoint = train(TrainConfig(board_size=4, total_timesteps=0, output_dir=str(tmp_path / "separate"),
                                  share_features_extractor=False, eval_episodes=1, threads=1, verbose=0, max_steps=8))
    model = PPO.load(checkpoint)
    actor, critic = model.policy.pi_features_extractor, model.policy.vf_features_extractor
    assert actor is not critic
    env = SnakeEnv(4)
    observation, _ = env.reset(seed=7)
    model.policy.predict_values(torch.from_numpy(observation[None])).sum().backward()
    assert actor.cnn[0].weight.grad is None
    assert critic.cnn[0].weight.grad is not None


def test_initial_temperature_preserves_argmax_and_increases_sampling_entropy(tmp_path):
    config = TrainConfig(board_size=4, total_timesteps=0, output_dir=str(tmp_path / "source_temp"),
                         eval_episodes=1, threads=1, verbose=0, max_steps=8)
    source = train(config)
    model = PPO.load(source)
    with torch.no_grad():
        model.policy.action_net.bias.copy_(torch.tensor([2.0, 1.0, -1.0]))
    model.save(source)
    target = train(replace(config, initial_actor_temperature=2, init_checkpoint=str(source),
                           output_dir=str(tmp_path / "target_temp")))
    env = SnakeEnv(4)
    env.reset(seed=7)
    first = NeuralPolicy(source, threads=1).probabilities(env)
    second = NeuralPolicy(target, threads=1).probabilities(env)
    assert first.argmax() == second.argmax()
    assert -(second * np.log(second)).sum() > -(first * np.log(first)).sum()


def test_seeded_policy_sampling_is_reproducible_across_batch_sizes(tmp_path):
    checkpoint = train(TrainConfig(board_size=4, total_timesteps=0, output_dir=str(tmp_path / "sampling"),
                                  eval_episodes=1, threads=1, verbose=0, max_steps=8))
    first = evaluate("neural", 4, 5, 19, checkpoint, batch_size=1, deterministic=False)
    second = evaluate("neural", 4, 5, 19, checkpoint, batch_size=5, deterministic=False)
    assert first["action_selection"] == "seeded_policy_sampling"
    assert first["episodes"] == second["episodes"]
    for result in first["episodes"]:
        single, _ = episode("neural", 4, result["seed"], checkpoint, deterministic=False)
        assert single == result


def test_pure_flag_rejects_combined_expert_experiment_before_any_run(tmp_path):
    output = tmp_path / "forbidden_combined"
    with pytest.raises(SystemExit, match="expert imitation"):
        main(["experiment", "--pure-ppo", "--out", str(output)])
    assert not output.exists()
