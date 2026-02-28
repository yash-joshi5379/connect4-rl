# src/symmetry.py
import numpy as np
from src.config import Config


def _flip_action_horizontal(action, board_size):
    """Flip an action (flattened index) horizontally (left-right mirror)."""
    row = action // board_size
    col = action % board_size
    new_col = board_size - 1 - col
    return row * board_size + new_col


def _transform_state_and_action(state, action, k_rot, flip, board_size):
    """
    Apply a single symmetry transformation to a state and action.

    Args:
        state:      np.array of shape (C, H, W)
        action:     int in [0, board_size^2 - 1]
        k_rot:      number of 90-degree CCW rotations (0, 1, 2, 3)
        flip:       bool, whether to flip horizontally after rotation
        board_size: int

    Returns:
        (transformed_state, transformed_action)
    """
    transformed_state = np.rot90(state, k=k_rot, axes=(1, 2)).copy()

    a = action
    for _ in range(k_rot):
        row = a // board_size
        col = a % board_size
        new_row = board_size - 1 - col
        new_col = row
        a = new_row * board_size + new_col

    if flip:
        transformed_state = np.flip(transformed_state, axis=2).copy()
        a = _flip_action_horizontal(a, board_size)

    return transformed_state, a


def _transform_policy(policy, k_rot, flip, board_size):
    """
    Apply a symmetry transformation to a flattened policy vector.
    Reshapes to 2D, applies rotation/flip, flattens back.
    """
    policy_2d = policy.reshape(board_size, board_size)
    policy_2d = np.rot90(policy_2d, k=k_rot)
    if flip:
        policy_2d = np.flip(policy_2d, axis=1)
    return policy_2d.copy().flatten()


def get_symmetric_samples(state, policy, z):
    """
    Generate all 8 symmetry variants of an AlphaZero training sample.

    Args:
        state:  np.array of shape (C, H, W)
        policy: np.array of shape (board_size^2,) — MCTS visit distribution
        z:      float — game outcome from this player's perspective

    Returns:
        List of 8 tuples: (sym_state, sym_policy, z)
    """
    board_size = Config.BOARD_SIZE
    samples = []

    for k_rot in range(4):
        for flip in (False, True):
            sym_state = np.rot90(state, k=k_rot, axes=(1, 2))
            if flip:
                sym_state = np.flip(sym_state, axis=2)
            sym_state = sym_state.copy()

            sym_policy = _transform_policy(policy, k_rot, flip, board_size)

            samples.append((sym_state, sym_policy, z))

    return samples
