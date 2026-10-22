"""Traditional reference policies. These are never called by neural policies."""

from __future__ import annotations

from collections import deque
from functools import lru_cache

import numpy as np

from snakerl.env import DELTAS, Position, SnakeEnv


class RandomPolicy:
    """Uniform, unmasked random relative actions from an independent RNG."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, env: SnakeEnv) -> int:
        return int(self.rng.integers(3))


class GreedyPolicy:
    """BFS toward food, using reachable free space to resolve unsafe paths."""

    @staticmethod
    def _candidate(env: SnakeEnv, action: int) -> Position:
        direction = (env.direction + action - 1) % 4
        dx, dy = DELTAS[direction]
        x, y = env.snake[0]
        return x + dx, y + dy

    @staticmethod
    def _neighbors(position: Position, env: SnakeEnv) -> list[Position]:
        x, y = position
        return [
            (x + dx, y + dy)
            for dx, dy in DELTAS
            if 0 <= x + dx < env.width and 0 <= y + dy < env.height
        ]

    def _distance(self, start: Position, target: Position | None, blocked: set[Position], env: SnakeEnv) -> int | None:
        if target is None:
            return None
        queue = deque([(start, 0)])
        seen = {start}
        while queue:
            position, distance = queue.popleft()
            if position == target:
                return distance
            for neighbor in self._neighbors(position, env):
                if neighbor not in seen and neighbor not in blocked:
                    seen.add(neighbor)
                    queue.append((neighbor, distance + 1))
        return None

    def _reachable_area(self, start: Position, blocked: set[Position], env: SnakeEnv) -> int:
        queue = deque([start])
        seen = {start}
        while queue:
            for neighbor in self._neighbors(queue.popleft(), env):
                if neighbor not in seen and neighbor not in blocked:
                    seen.add(neighbor)
                    queue.append(neighbor)
        return len(seen)

    def act(self, env: SnakeEnv) -> int:
        candidates: list[tuple[int, int | None, int]] = []
        # Prefer straight movement when equally useful.
        for action in (1, 0, 2):
            next_head = self._candidate(env, action)
            x, y = next_head
            eating = next_head == env.food
            if not (0 <= x < env.width and 0 <= y < env.height):
                continue
            occupied = set(env.snake if eating else env.snake[:-1])
            if next_head in occupied:
                continue
            # Project the one-step body first, so the old head is an obstacle.
            projected_body = [*env.snake] if eating else env.snake[:-1]
            blocked = set(projected_body)
            if not eating and projected_body:
                # The projected tail may vacate on the following move.
                blocked.discard(projected_body[-1])
            distance = self._distance(next_head, env.food, blocked, env)
            area = self._reachable_area(next_head, blocked, env)
            candidates.append((action, distance, area))
        if not candidates:
            return 1  # Collision is unavoidable; preserve the relative-action API.
        with_path = [candidate for candidate in candidates if candidate[1] is not None]
        if with_path:
            return min(with_path, key=lambda candidate: (candidate[1], -candidate[2]))[0]
        return max(candidates, key=lambda candidate: candidate[2])[0]


@lru_cache(maxsize=32)
def hamiltonian_cycle(width: int, height: int) -> tuple[Position, ...]:
    """Construct an East-starting Hamiltonian cycle for an even-sided rectangle.

    Both dimensions must be at least two. A cycle on two odd dimensions does
    not exist in the rectangular grid graph and is explicitly rejected.
    """
    width = SnakeEnv._positive_integer(width, "width", minimum=2)
    height = SnakeEnv._positive_integer(height, "height", minimum=2)
    if width % 2 and height % 2:
        raise ValueError("A rectangular Hamiltonian cycle requires at least one even side")
    if height % 2:
        transposed = tuple((y, x) for x, y in hamiltonian_cycle(height, width))
        return (transposed[0], *reversed(transposed[1:]))
    cycle: list[Position] = [(x, 0) for x in range(width)]
    for y in range(1, height):
        row = range(width - 1, 0, -1) if y % 2 else range(1, width)
        cycle.extend((x, y) for x in row)
    cycle.extend((0, y) for y in range(height - 1, 0, -1))
    return tuple(cycle)


class HamiltonianPolicy:
    """Follow a proven cycle, requiring the current body to be cycle-aligned."""

    def __init__(self) -> None:
        self._shape: tuple[int, int] | None = None
        self._cycle: tuple[Position, ...] = ()
        self._indices: dict[Position, int] = {}

    def act(self, env: SnakeEnv) -> int:
        shape = env.width, env.height
        if shape != self._shape:
            self._cycle = hamiltonian_cycle(*shape)
            self._indices = {position: index for index, position in enumerate(self._cycle)}
            self._shape = shape
        head_index = self._indices[env.snake[0]]
        area = len(self._cycle)
        if any(position != self._cycle[(head_index - index) % area] for index, position in enumerate(env.snake)):
            raise ValueError("HamiltonianPolicy requires a body aligned with its cycle; use the canonical reset")
        next_head = self._cycle[(head_index + 1) % area]
        head = env.snake[0]
        delta = next_head[0] - head[0], next_head[1] - head[1]
        absolute_direction = DELTAS.index(delta)
        turn = (absolute_direction - env.direction) % 4
        if turn == 2:
            raise ValueError("Current direction is opposite to the Hamiltonian cycle")
        return {3: 0, 0: 1, 1: 2}[turn]
