# src/buffer.py
import random
from collections import deque
import numpy as np
from src.config import Config


class ReplayBuffer:
    def __init__(self):
        self.buffer = deque(maxlen=Config.BUFFER_CAPACITY)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Handle None next_states (terminal states)
        next_states_array = []
        for ns in next_states:
            if ns is None:
                # Use zeros for terminal state (doesn't matter, done=1 zeros it out)
                next_states_array.append(np.zeros_like(states[0]))
            else:
                next_states_array.append(ns)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states_array),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)
