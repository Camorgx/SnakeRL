import json

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from snakerl.env import SnakeEnv


def test_gymnasium_contract_and_render():
    env = SnakeEnv(board_size=4, render_mode="rgb_array")
    check_env(env, skip_render_check=False)
    frame = env.render()
    assert frame.shape == (96, 96, 3)
    assert frame.dtype == np.uint8


def test_observation_encodes_order_direction_food_and_coordinates():
    env = SnakeEnv(width=5, height=4)
    observation, info = env.reset(seed=8)
    assert observation.shape == (9, 4, 5)
    assert observation.dtype == np.float32
    assert env.observation_space.contains(observation)
    assert observation[0, 0, 2] == 1
    assert observation[1, 0, 2] == pytest.approx(3 / 20)
    assert observation[1, 0, 1] == pytest.approx(2 / 20)
    assert observation[1, 0, 0] == pytest.approx(1 / 20)
    assert observation[2].sum() == 1
    assert np.all(observation[4] == 1)
    assert not observation[[3, 5, 6]].any()
    assert np.all(observation[7, :, -1] == 1)
    assert np.all(observation[8, -1] == 1)
    assert info["coverage"] == 3 / 20
    assert info["termination_reason"] is None
    assert np.array_equal(observation, env.observe())
    json.dumps(env.state())


def test_seeded_resets_and_transitions_are_reproducible():
    first, second = SnakeEnv(6), SnakeEnv(6)
    a, ia = first.reset(seed=314)
    b, ib = second.reset(seed=314)
    assert np.array_equal(a, b)
    assert ia == ib
    for action in [1, 2, 1, 1, 2, 1, 1, 2]:
        result_a, result_b = first.step(action), second.step(action)
        assert np.array_equal(result_a[0], result_b[0])
        assert result_a[1:] == result_b[1:]
        assert first.state() == second.state()
        if result_a[2] or result_a[3]:
            break
    initial_foods = set()
    for seed in range(20):
        first.reset(seed=seed)
        initial_foods.add(first.food)
    assert len(initial_foods) > 1


def test_food_is_uniformly_selected_from_free_cells_and_never_on_body():
    env = SnakeEnv(2)
    foods = set()
    for seed in range(100):
        env.reset(seed=seed)
        assert env.food not in env.snake
        foods.add(env.food)
    assert foods == {(0, 1), (1, 1)}


def test_food_increases_length_and_does_not_remove_tail():
    env = SnakeEnv(4)
    env.reset(seed=0)
    original_tail = env.snake[-1]
    env.food = (3, 0)
    _, reward, terminated, truncated, info = env.step(1)
    assert env.snake[0] == (3, 0)
    assert env.snake[-1] == original_tail
    assert info["length"] == 4
    assert info["foods_eaten"] == 1
    assert reward > 0
    assert env.food not in env.snake
    assert not terminated and not truncated
    assert env.no_food_steps == 0


def test_wall_collision_terminates_and_requires_reset():
    env = SnakeEnv(4)
    env.reset(seed=0)
    _, reward, terminated, truncated, info = env.step(0)
    assert reward == -1
    assert terminated and not truncated
    assert info["termination_reason"] == "wall_collision"
    assert info["steps"] == 1
    assert not info["success"]
    with pytest.raises(RuntimeError, match="reset"):
        env.step(1)


def test_body_collision_terminates():
    env = SnakeEnv(3)
    env.reset(seed=0)
    env.snake = [(1, 1), (1, 2), (0, 2), (0, 1), (0, 0), (1, 0)]
    env.direction = 0
    env.food = (2, 2)
    _, _, terminated, truncated, info = env.step(0)
    assert terminated and not truncated
    assert info["termination_reason"] == "body_collision"


def test_vacating_tail_is_legal_but_eating_into_tail_is_not():
    env = SnakeEnv(3)
    env.reset(seed=0)
    env.snake = [(1, 1), (1, 2), (0, 2), (0, 1)]
    env.direction = 0
    env.food = (2, 2)
    _, _, terminated, truncated, _ = env.step(0)
    assert not terminated and not truncated
    assert env.snake == [(0, 1), (1, 1), (1, 2), (0, 2)]
    env.reset(seed=0)
    env.snake = [(1, 1), (1, 2), (0, 2), (0, 1)]
    env.direction = 0
    env.food = (0, 1)  # Deliberately malformed placement checks collision priority.
    assert env.step(0)[2]


def test_full_board_success_terminates_before_time_limits():
    env = SnakeEnv(2, max_steps=1, max_no_food_steps=1)
    env.reset(seed=0)
    env.snake = [(1, 0), (0, 0), (0, 1)]
    env.food = (1, 1)
    _, reward, terminated, truncated, info = env.step(2)
    assert terminated and not truncated
    assert info["success"] and info["coverage"] == 1
    assert info["length"] == 4
    assert info["termination_reason"] == "full_board"
    assert env.food is None
    assert reward > 10


@pytest.mark.parametrize(
    "options, expected",
    [({"max_no_food_steps": 1}, "no_food_timeout"), ({"max_steps": 1}, "step_limit")],
)
def test_time_limits_truncate_instead_of_terminate(options, expected):
    env = SnakeEnv(4, **options)
    env.reset(seed=0)
    env.food = (0, 3)
    _, _, terminated, truncated, info = env.step(1)
    assert not terminated and truncated
    assert info["termination_reason"] == expected
    assert not info["success"]
    with pytest.raises(RuntimeError):
        env.step(1)


def test_relative_actions_and_configurable_distance_reward():
    env = SnakeEnv(4, distance_reward=0)
    env.reset(seed=0)
    env.food = (0, 3)
    _, reward, _, _, _ = env.step(2)
    assert env.direction == 2
    assert env.snake[0] == (2, 1)
    assert reward == -0.001
    env.step(0)
    assert env.direction == 1
    assert env.snake[0] == (3, 1)


@pytest.mark.parametrize(
    "options",
    [{"board_size": 1}, {"height": 0}, {"initial_length": 0}, {"max_steps": 0}, {"distance_reward": -1}],
)
def test_invalid_environment_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        SnakeEnv(**options)


def test_small_board_caps_initial_length_and_invalid_actions_are_rejected():
    env = SnakeEnv(2)
    env.reset(seed=1)
    assert len(env.snake) == 2
    with pytest.raises(ValueError, match="action"):
        env.step(3)


def test_step_before_reset_is_rejected():
    with pytest.raises(RuntimeError, match="reset"):
        SnakeEnv().step(1)
