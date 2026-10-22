import pytest

from snakerl.baselines import GreedyPolicy, HamiltonianPolicy, RandomPolicy, hamiltonian_cycle
from snakerl.env import SnakeEnv


@pytest.mark.parametrize("width,height", [(2, 2), (3, 2), (2, 3), (4, 5), (5, 4), (8, 8), (12, 12), (20, 20)])
def test_hamiltonian_cycle_visits_every_cell_and_closes(width, height):
    cycle = hamiltonian_cycle(width, height)
    assert len(cycle) == width * height
    assert len(set(cycle)) == width * height
    assert set(cycle) == {(x, y) for y in range(height) for x in range(width)}
    assert cycle[0] == (0, 0) and cycle[1] == (1, 0)
    for start, end in zip(cycle, (*cycle[1:], cycle[0])):
        assert abs(start[0] - end[0]) + abs(start[1] - end[1]) == 1


@pytest.mark.parametrize("width,height", [(3, 3), (5, 7), (1, 4)])
def test_unsupported_cycles_are_rejected(width, height):
    with pytest.raises(ValueError):
        hamiltonian_cycle(width, height)


@pytest.mark.parametrize("width,height", [(2, 2), (4, 4), (4, 5), (5, 4), (8, 8)])
@pytest.mark.parametrize("seed", [0, 1, 42])
def test_hamiltonian_policy_fills_board_for_seeded_food(width, height, seed):
    env = SnakeEnv(width=width, height=height)
    env.reset(seed=seed)
    policy = HamiltonianPolicy()
    while True:
        _, _, terminated, truncated, info = env.step(policy.act(env))
        if terminated or truncated:
            break
    assert terminated and not truncated
    assert info["success"]
    assert info["coverage"] == 1
    assert info["foods_eaten"] == width * height - env.initial_length
    assert info["termination_reason"] == "full_board"


def test_hamiltonian_policy_rejects_body_outside_its_proven_cycle():
    env = SnakeEnv(4)
    env.reset(seed=0)
    env.snake = [(2, 1), (1, 1), (0, 1)]
    with pytest.raises(ValueError, match="aligned"):
        HamiltonianPolicy().act(env)


def test_random_policy_is_seeded_and_does_not_mask_collisions():
    first, second = RandomPolicy(seed=5), RandomPolicy(seed=5)
    env = SnakeEnv(4)
    env.reset(seed=0)
    actions = [first.act(env) for _ in range(30)]
    assert actions == [second.act(env) for _ in range(30)]
    assert set(actions) == {0, 1, 2}  # Left hits the top wall in this state.


def test_greedy_policy_chooses_adjacent_food():
    env = SnakeEnv(4)
    env.reset(seed=0)
    env.food = (2, 1)
    action = GreedyPolicy().act(env)
    assert action == 2
    assert env.step(action)[4]["foods_eaten"] == 1


def test_greedy_policy_can_move_into_vacating_tail_when_other_moves_blocked():
    env = SnakeEnv(3)
    env.reset(seed=0)
    env.snake = [(1, 1), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0), (0, 0), (0, 1)]
    env.direction = 0
    env.food = (0, 2)
    action = GreedyPolicy().act(env)
    assert action == 0
    assert not env.step(action)[2]


def test_greedy_returns_valid_action_when_food_has_no_static_path():
    env = SnakeEnv(5)
    env.reset(seed=0)
    env.snake = [(2, 2), (2, 3), (3, 3), (3, 2), (3, 1), (2, 1), (1, 1), (1, 2)]
    env.direction = 0
    env.food = (4, 4)
    action = GreedyPolicy().act(env)
    assert action == 0  # The only surviving move enters the vacating tail.
    assert not env.step(action)[2]
