"""PyTorch policy features and observation-only neural inference.

The extractor never reads a SnakeEnv object. Its fixed output size permits weight
transfer between curriculum boards, while each saved PPO checkpoint retains the
observation shape for which it was trained.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

if TYPE_CHECKING:
    from snakerl.env import SnakeEnv


class SnakeCNN(BaseFeaturesExtractor):
    """A small spatial CNN with global context and features at the snake's head."""

    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        if len(observation_space.shape) != 3 or observation_space.shape[0] not in (9, 11, 13):
            raise ValueError("SnakeCNN requires a channels-first (9, 11 or 13, H, W) observation")
        super().__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(observation_space.shape[0], 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 24, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.projection = nn.Sequential(
            nn.Linear(24 * 4 * 4 + 24 + observation_space.shape[0], features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        spatial = self.cnn(observations)
        head = observations[:, :1]
        # The head mask comes from the observation, not an external rule or path.
        head_features = (spatial * head).sum(dim=(-2, -1))
        head_channels = (observations * head).sum(dim=(-2, -1))
        global_features = self.pool(spatial).flatten(start_dim=1)
        return self.projection(torch.cat((global_features, head_features, head_channels), dim=1))


class NeuralPolicy:
    """A saved PPO actor. No expert, lookahead, or action mask is used here."""

    name = "neural"

    def __init__(self, checkpoint: str | Path, device: str = "cpu", threads: int = 2):
        self.checkpoint = Path(checkpoint)
        if threads < 1:
            raise ValueError("threads must be positive")
        torch.set_num_threads(threads)
        self.model = PPO.load(str(self.checkpoint), device=device)
        shape = self.model.observation_space.shape
        if len(shape) != 3 or shape[0] not in (9, 11, 13):
            raise ValueError(f"Incompatible checkpoint observation shape: {shape}")
        self.height, self.width = int(shape[1]), int(shape[2])
        self.observation_version = {9: 1, 11: 2, 13: 3}[shape[0]]
        self.board_size = self.width if self.width == self.height else (self.width, self.height)
        self.model.policy.set_training_mode(False)

    def _observation(self, env: SnakeEnv) -> np.ndarray:
        observation = np.asarray(env.observe(), dtype=np.float32)
        expected = self.model.observation_space.shape
        if observation.shape != expected:
            raise ValueError(
                f"Checkpoint expects board {self.width}x{self.height} ({expected}); "
                f"environment observation is {observation.shape}. "
                "Transfer weights through training before using another board size."
            )
        return observation

    def act(self, env: SnakeEnv) -> int:
        action, _ = self.model.predict(self._observation(env), deterministic=True)
        return int(np.asarray(action).item())

    def probabilities(self, env: SnakeEnv) -> np.ndarray:
        _, probabilities = self.predict_batch(self._observation(env)[None])
        return probabilities[0]

    def predict_batch(self, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return deterministic actions and probabilities for a batch of boards."""
        observations = np.asarray(observations, dtype=np.float32)
        expected = self.model.observation_space.shape
        if observations.ndim != 4 or observations.shape[1:] != expected:
            raise ValueError(f"Expected batched observations (N, {expected}), got {observations.shape}")
        with torch.no_grad():
            tensor, _ = self.model.policy.obs_to_tensor(observations)
            distribution = self.model.policy.get_distribution(tensor)
            probabilities = distribution.distribution.probs.cpu().numpy().copy()
            return probabilities.argmax(axis=1), probabilities
