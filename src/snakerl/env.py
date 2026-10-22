"""A seeded Snake environment with relative actions and complete board observations."""

from __future__ import annotations

from typing import Any
from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces


Position = tuple[int, int]
DELTAS: tuple[Position, ...] = ((0, -1), (1, 0), (0, 1), (-1, 0))


class SnakeEnv(gym.Env[np.ndarray, int]):
    """Snake on a rectangular board; actions are left, straight, and right.

    The original nine channels contain head, ordered body, food, direction,
    and coordinates. Version 2 adds clocks; version 3 adds occupancy and tail.
    No search algorithm or action masking participates in the transition.
    Moving into the current tail is legal when it vacates.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 12}

    def __init__(
        self,
        board_size: int = 8,
        *,
        width: int | None = None,
        height: int | None = None,
        initial_length: int = 3,
        render_mode: str | None = None,
        max_no_food_steps: int | None = None,
        max_steps: int | None = None,
        distance_reward: float = 0.02,
        observation_version: int = 1,
        reward_profile: str = "legacy",
        random_start: bool = False,
        self_curriculum_probability: float = 0.0,
        curriculum_min_coverage: float = 0.7,
    ) -> None:
        super().__init__()
        self.width = self._positive_integer(width if width is not None else board_size, "width", minimum=2)
        self.height = self._positive_integer(height if height is not None else board_size, "height", minimum=2)
        self.board_size = board_size
        self.area = self.width * self.height
        self.initial_length = min(self._positive_integer(initial_length, "initial_length"), self.width)
        self.max_no_food_steps = self._positive_integer(
            self.area * 4 if max_no_food_steps is None else max_no_food_steps,
            "max_no_food_steps",
        )
        self.max_steps = self._positive_integer(
            self.area * self.area * 4 if max_steps is None else max_steps,
            "max_steps",
        )
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or 'rgb_array'")
        if not np.isfinite(distance_reward) or distance_reward < 0:
            raise ValueError("distance_reward must be a finite nonnegative number")
        self.distance_reward = float(distance_reward)
        if observation_version not in (1, 2, 3) or isinstance(observation_version, bool):
            raise ValueError("observation_version must be 1, 2 or 3")
        if reward_profile not in ("legacy", "coverage"):
            raise ValueError("reward_profile must be legacy or coverage")
        self.observation_version = observation_version
        self.reward_profile = reward_profile
        self.random_start = bool(random_start)
        if not np.isfinite(self_curriculum_probability) or not 0 <= self_curriculum_probability <= 1:
            raise ValueError("self_curriculum_probability must be in [0,1]")
        if not np.isfinite(curriculum_min_coverage) or not 0 < curriculum_min_coverage < 1:
            raise ValueError("curriculum_min_coverage must be in (0,1)")
        self.self_curriculum_probability = float(self_curriculum_probability)
        self.curriculum_min_coverage = float(curriculum_min_coverage)
        self._self_states = deque(maxlen=128)
        self.curriculum_start = False
        self.episode_initial_length = self.initial_length
        self.step_cost = 0.001 if reward_profile == "legacy" else 0.003
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(0.0, 1.0, ({1: 9, 2: 11, 3: 13}[observation_version], self.height, self.width), np.float32)
        x, y = np.meshgrid(
            np.linspace(0, 1, self.width, dtype=np.float32),
            np.linspace(0, 1, self.height, dtype=np.float32),
        )
        self._coordinates = np.stack((x, y))
        self._starts = [(x, y, direction) for y in range(self.height) for x in range(self.width)
                        for direction, (dx, dy) in enumerate(DELTAS)
                        if 0 <= x - (self.initial_length - 1) * dx < self.width
                        and 0 <= y - (self.initial_length - 1) * dy < self.height]
        self.snake: list[Position] = []
        self.direction = 1
        self.food: Position | None = None
        self.steps = 0
        self.foods_eaten = 0
        self.no_food_steps = 0
        self._episode_done = True
        self._success = False
        self._termination_reason: str | None = None

    @staticmethod
    def _positive_integer(value: Any, name: str, minimum: int = 1) -> int:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
        return int(value)

    @property
    def coverage(self) -> float:
        return len(self.snake) / self.area

    @property
    def success(self) -> bool:
        return self._success

    @property
    def termination_reason(self) -> str | None:
        return self._termination_reason

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.snake = [(x, 0) for x in range(self.initial_length - 1, -1, -1)]
        self.direction = 1
        if self.random_start:
            x, y, self.direction = self._starts[int(self.np_random.integers(len(self._starts)))]
            dx, dy = DELTAS[self.direction]
            self.snake = [(x - index * dx, y - index * dy) for index in range(self.initial_length)]
        self.curriculum_start = False
        if self.self_curriculum_probability > 0 and self._self_states and self.np_random.random() < self.self_curriculum_probability:
            body, self.direction = self._self_states[int(self.np_random.integers(len(self._self_states)))]
            self.snake = body.copy()
            self.curriculum_start = True
        self.episode_initial_length = len(self.snake)
        self.steps = 0
        self.foods_eaten = 0
        self.no_food_steps = 0
        self._episode_done = False
        self._success = False
        self._termination_reason = None
        self.food = self._sample_food()
        return self._observation(), self._info()

    def _sample_food(self) -> Position | None:
        occupied = set(self.snake)
        free = [(x, y) for y in range(self.height) for x in range(self.width) if (x, y) not in occupied]
        if not free:
            return None
        return free[int(self.np_random.integers(len(free)))]

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._episode_done:
            raise RuntimeError("Call reset() before stepping a new or finished episode")
        if not self.action_space.contains(action):
            raise ValueError("action must be 0 (left), 1 (straight), or 2 (right)")
        self.direction = (self.direction + int(action) - 1) % 4
        dx, dy = DELTAS[self.direction]
        head = self.snake[0]
        next_head = (head[0] + dx, head[1] + dy)
        eating = next_head == self.food
        self.steps += 1
        self.no_food_steps += 1
        inside = 0 <= next_head[0] < self.width and 0 <= next_head[1] < self.height
        blocked = self.snake if eating else self.snake[:-1]
        if not inside or next_head in blocked:
            self._episode_done = True
            self._termination_reason = "wall_collision" if not inside else "body_collision"
            penalty = 1.0 if self.reward_profile == "legacy" else 1.0 + 4.0 * self.coverage**2
            return self._observation(), -penalty, True, False, self._info()

        reward = -self.step_cost
        if eating:
            self.snake.insert(0, next_head)
            self.foods_eaten += 1
            self.no_food_steps = 0
            reward += 1.0 if self.reward_profile == "legacy" else 1.0 + 2.0 * self.coverage**2
            if len(self.snake) == self.area:
                self.food = None
                self._success = True
                self._episode_done = True
                self._termination_reason = "full_board"
                return self._observation(), reward + (10.0 if self.reward_profile == "legacy" else 20.0), True, False, self._info()
            self.food = self._sample_food()
            if self.self_curriculum_probability > 0 and self.coverage >= self.curriculum_min_coverage:
                self._self_states.append((self.snake.copy(), self.direction))
        else:
            self.snake = [next_head, *self.snake[:-1]]
            if self.food is not None:
                old_distance = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])
                new_distance = abs(next_head[0] - self.food[0]) + abs(next_head[1] - self.food[1])
                reward += self.distance_reward * (old_distance - new_distance) / (self.width + self.height - 2)

        truncated = False
        if self.no_food_steps >= self.max_no_food_steps:
            truncated = True
            self._termination_reason = "no_food_timeout"
        elif self.steps >= self.max_steps:
            truncated = True
            self._termination_reason = "step_limit"
        if truncated:
            self._episode_done = True
            reward -= 0.25 if self.reward_profile == "legacy" else 1.0
        return self._observation(), float(reward), False, truncated, self._info()

    def _observation(self) -> np.ndarray:
        observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        if self.snake:
            head_x, head_y = self.snake[0]
            observation[0, head_y, head_x] = 1.0
        for index, (x, y) in enumerate(self.snake):
            observation[1, y, x] = (len(self.snake) - index) / self.area
            if self.observation_version == 3:
                observation[11, y, x] = 1
        if self.food is not None:
            food_x, food_y = self.food
            observation[2, food_y, food_x] = 1.0
        observation[3 + self.direction].fill(1.0)
        observation[7:9] = self._coordinates
        if self.observation_version >= 2:
            observation[9].fill(min(1.0, self.no_food_steps / self.max_no_food_steps))
            observation[10].fill(min(1.0, self.steps / self.max_steps))
        if self.observation_version == 3 and self.snake:
            tail_x, tail_y = self.snake[-1]
            observation[12, tail_y, tail_x] = 1
        return observation

    def observe(self) -> np.ndarray:
        """Return the current observation without advancing the environment."""
        return self._observation()

    def _info(self) -> dict[str, Any]:
        return {
            "length": len(self.snake),
            "coverage": self.coverage,
            "steps": self.steps,
            "foods_eaten": self.foods_eaten,
            "success": self._success,
            "termination_reason": self._termination_reason,
            "curriculum_start": self.curriculum_start,
            "initial_length": self.episode_initial_length,
        }

    def state(self) -> dict[str, Any]:
        """Return a JSON-friendly snapshot for viewers and episode recording."""
        return {
            "width": self.width,
            "height": self.height,
            "snake": [list(position) for position in self.snake],
            "food": list(self.food) if self.food is not None else None,
            "direction": self.direction,
            **self._info(),
        }

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        cell = 24
        frame = np.full((self.height * cell, self.width * cell, 3), (16, 24, 36), dtype=np.uint8)
        for y in range(self.height):
            for x in range(self.width):
                frame[y * cell, x * cell : (x + 1) * cell] = (28, 39, 53)
                frame[y * cell : (y + 1) * cell, x * cell] = (28, 39, 53)
        for index, (x, y) in enumerate(reversed(self.snake)):
            intensity = (index + 1) / max(len(self.snake), 1)
            color = (int(28 + 18 * intensity), int(114 + 98 * intensity), int(112 + 54 * intensity))
            frame[y * cell + 2 : (y + 1) * cell - 1, x * cell + 2 : (x + 1) * cell - 1] = color
        if self.snake:
            x, y = self.snake[0]
            frame[y * cell + 4 : (y + 1) * cell - 3, x * cell + 4 : (x + 1) * cell - 3] = (106, 255, 185)
        if self.food is not None:
            x, y = self.food
            frame[y * cell + 5 : (y + 1) * cell - 4, x * cell + 5 : (x + 1) * cell - 4] = (255, 110, 105)
        return frame
