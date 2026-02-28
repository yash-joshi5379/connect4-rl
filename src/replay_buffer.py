# src/replay_buffer.py
import random
import numpy as np
from collections import deque
from src.config import Config


class ReplayBuffer:
    def __init__(self):
        self.buffer = deque(maxlen=Config.BUFFER_CAPACITY)

    def push(self, state, policy, z):
        """
        Store a single training sample.

        Args:
            state:  np.array of shape (C, H, W)
            policy: np.array of shape (board_size^2,) — MCTS visit distribution
            z:      float — game outcome (+1 or -1) from this player's perspective
        """
        self.buffer.append((state, policy, z))

    def sample(self, batch_size=None):
        """
        Sample a random batch.

        Returns:
            states:   np.array of shape (B, C, H, W)
            policies: np.array of shape (B, board_size^2)
            zs:       np.array of shape (B,)
        """
        if batch_size is None:
            batch_size = Config.BATCH_SIZE

        batch = random.sample(self.buffer, batch_size)
        states, policies, zs = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(policies, dtype=np.float32),
            np.array(zs, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)
