# train.py
import os
import random
import numpy as np
import torch
import torch.optim as optim
from collections import deque
from enum import Enum, auto
from tqdm import trange

from src.config import Config
from src.game import Game, GameResult, Color
from src.network import PolicyValueNetwork
from src.mcts import MCTS
from src.replay_buffer import ReplayBuffer
from src.symmetry import get_symmetric_samples
from src.heuristic import HeuristicAgent


class Stage(Enum):
    HEURISTIC = auto()
    SELFPLAY = auto()


def set_seeds():
    torch.manual_seed(Config.RANDOM_SEED)
    random.seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)


def play_game(mcts_current, mcts_opponent, temperature_threshold=None):
    """
    Play one full game using MCTS for both sides.

    Args:
        mcts_current:  MCTS instance for the training agent
        mcts_opponent: MCTS instance for the opponent (heuristic uses None)
        temperature_threshold: move number after which we switch to greedy

    Returns:
        samples: list of (state, policy, player_color) before outcome is known
        result:  GameResult
        num_moves: int
    """
    if temperature_threshold is None:
        temperature_threshold = Config.TEMPERATURE_THRESHOLD

    game = Game()
    game.reset()

    # Randomly assign who plays black
    current_is_black = random.random() < 0.5
    current_color = Color.BLACK if current_is_black else Color.WHITE
    opponent_color = Color.WHITE if current_is_black else Color.BLACK

    samples = []  # (state, policy, color_of_player_who_moved)
    move_count = 0

    while game.result == GameResult.ONGOING:
        is_current_turn = game.current_player == current_color

        if is_current_turn:
            # Get state from current player's perspective
            state = game.get_state_for_network(perspective_color=current_color)

            # Run MCTS
            policy = mcts_current.search(game, add_noise=True)

            # Temperature-based move selection
            if move_count < temperature_threshold:
                action_int = sample_from_policy(policy, temperature=1.0)
            else:
                action_int = np.argmax(policy)

            samples.append((state, policy, current_color))

            action = game.int_to_action(action_int)
            game.step(action)

        else:
            if mcts_opponent is not None:
                # Opponent uses MCTS too (self-play)
                state = game.get_state_for_network(perspective_color=opponent_color)
                policy = mcts_opponent.search(game, add_noise=True)

                if move_count < temperature_threshold:
                    action_int = sample_from_policy(policy, temperature=1.0)
                else:
                    action_int = np.argmax(policy)

                samples.append((state, policy, opponent_color))

                action = game.int_to_action(action_int)
                game.step(action)
            else:
                # This shouldn't happen — opponent is always provided
                raise ValueError("No opponent MCTS provided")

        move_count += 1

    return samples, game.result, move_count


def play_game_vs_heuristic(mcts_current, heuristic_agent):
    """
    Play one game: MCTS agent vs heuristic opponent.
    Only collect training samples from the MCTS agent's moves.
    """
    game = Game()
    game.reset()

    current_is_black = random.random() < 0.5
    current_color = Color.BLACK if current_is_black else Color.WHITE

    samples = []
    move_count = 0

    while game.result == GameResult.ONGOING:
        is_current_turn = game.current_player == current_color

        if is_current_turn:
            state = game.get_state_for_network(perspective_color=current_color)
            policy = mcts_current.search(game, add_noise=True)

            if move_count < Config.TEMPERATURE_THRESHOLD:
                action_int = sample_from_policy(policy, temperature=1.0)
            else:
                action_int = np.argmax(policy)

            samples.append((state, policy, current_color))

            action = game.int_to_action(action_int)
            game.step(action)

        else:
            action = heuristic_agent.select_action(game)
            game.step(action)

        move_count += 1

    return samples, game.result, move_count


def sample_from_policy(policy, temperature=1.0):
    """Sample an action from the policy with temperature."""
    if temperature < 1e-6:
        return np.argmax(policy)

    # Apply temperature
    policy = policy ** (1.0 / temperature)
    total = policy.sum()
    if total > 0:
        policy = policy / total
    else:
        # Fallback to uniform
        policy = np.ones_like(policy) / len(policy)

    return np.random.choice(len(policy), p=policy)


def label_samples(samples, result):
    """
    Label each sample with the game outcome from that player's perspective.

    Args:
        samples: list of (state, policy, player_color)
        result:  GameResult

    Returns:
        labeled: list of (state, policy, z)
    """
    labeled = []

    for state, policy, color in samples:
        if result == GameResult.DRAW:
            z = 0.0
        elif result == GameResult.BLACK_WIN:
            z = 1.0 if color == Color.BLACK else -1.0
        else:  # WHITE_WIN
            z = 1.0 if color == Color.WHITE else -1.0

        labeled.append((state, policy, z))

    return labeled


def train_step(network, optimizer, replay_buffer):
    """
    One training step: sample batch, compute loss, update weights.

    Returns:
        (total_loss, policy_loss, value_loss) or None if buffer too small
    """
    if len(replay_buffer) < Config.BATCH_SIZE:
        return None

    states, policies, zs = replay_buffer.sample()

    states_t = torch.FloatTensor(states)
    policies_t = torch.FloatTensor(policies)
    zs_t = torch.FloatTensor(zs).unsqueeze(1)

    network.train()
    log_policy, value = network(states_t)

    # Policy loss: cross-entropy with MCTS policy target
    # log_policy is log_softmax output, policies_t is the target distribution
    policy_loss = -torch.mean(torch.sum(policies_t * log_policy, dim=1))

    # Value loss: MSE between predicted value and game outcome
    value_loss = torch.mean((value - zs_t) ** 2)

    # Combined loss
    total_loss = Config.POLICY_LOSS_WEIGHT * policy_loss + Config.VALUE_LOSS_WEIGHT * value_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss.item(), policy_loss.item(), value_loss.item()


def save_checkpoint(network, path):
    torch.save(network.state_dict(), path)


def load_checkpoint(path):
    network = PolicyValueNetwork()
    network.load_state_dict(torch.load(path, weights_only=True))
    network.eval()
    return network


def train():
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    set_seeds()

    # Core components
    network = PolicyValueNetwork()
    optimizer = optim.Adam(
        network.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.L2_REG,
    )
    mcts = MCTS(network)
    replay_buffer = ReplayBuffer()
    heuristic = HeuristicAgent()

    # Self-play opponent pool
    snapshot_pool = []
    snapshot_cache = {}  # path -> network

    # Tracking
    stage = Stage.HEURISTIC
    recent_outcomes = deque(maxlen=100)
    recent_policy_loss = deque(maxlen=100)
    recent_value_loss = deque(maxlen=100)

    for episode in trange(Config.TOTAL_EPISODES, desc="Training"):

        # --- Stage transition ---
        if stage == Stage.HEURISTIC and episode >= Config.HEURISTIC_EPISODES:
            stage = Stage.SELFPLAY
            recent_outcomes.clear()
            path = f"{Config.MODEL_DIR}/checkpoint_0.pth"
            save_checkpoint(network, path)
            snapshot_pool.append(path)
            print(f"\n[SELFPLAY] Starting self-play at episode {episode}")
            print(f"[SELFPLAY] Saved initial checkpoint to pool")

        # --- Play one game ---
        if stage == Stage.HEURISTIC:
            samples, result, num_moves = play_game_vs_heuristic(mcts, heuristic)
        else:
            # Pick opponent from snapshot pool
            if len(snapshot_pool) > 1 and random.random() < Config.OLD_OPPONENT_CHANCE:
                opp_path = random.choice(snapshot_pool[:-1])
            else:
                opp_path = snapshot_pool[-1]

            if opp_path not in snapshot_cache:
                snapshot_cache[opp_path] = load_checkpoint(opp_path)
                # Trim cache if pool exceeded
                if len(snapshot_cache) > Config.SNAPSHOT_POOL_SIZE + 2:
                    oldest = list(snapshot_cache.keys())[0]
                    del snapshot_cache[oldest]

            opp_network = snapshot_cache[opp_path]
            opp_mcts = MCTS(opp_network)
            samples, result, num_moves = play_game(mcts, opp_mcts)

        # --- Label samples with game outcome ---
        labeled = label_samples(samples, result)

        # --- Store with 8-fold symmetry augmentation ---
        for state, policy, z in labeled:
            for sym_state, sym_policy, sym_z in get_symmetric_samples(state, policy, z):
                replay_buffer.push(sym_state, sym_policy, sym_z)

        # --- Training steps ---
        loss_result = train_step(network, optimizer, replay_buffer)

        # --- Self-play checkpointing ---
        if stage == Stage.SELFPLAY:
            selfplay_ep = episode - Config.HEURISTIC_EPISODES
            if (selfplay_ep + 1) % Config.CHECKPOINT_INTERVAL == 0:
                idx = len(snapshot_pool)
                path = f"{Config.MODEL_DIR}/checkpoint_{idx}.pth"
                save_checkpoint(network, path)
                snapshot_pool.append(path)

                # Trim pool to max size
                if len(snapshot_pool) > Config.SNAPSHOT_POOL_SIZE:
                    removed = snapshot_pool.pop(0)
                    snapshot_cache.pop(removed, None)

                print(f"\n[SELFPLAY] Checkpoint {idx} saved (pool: {len(snapshot_pool)})")

        # --- Tracking ---
        # Determine outcome from training agent's perspective
        # In heuristic games, samples only contain current player's moves
        # In self-play, both sides contribute samples
        if result == GameResult.DRAW:
            recent_outcomes.append("D")
        elif len(labeled) > 0:
            # Check last sample's z to determine if agent won
            last_z = labeled[0][2]  # first sample's outcome
            recent_outcomes.append("W" if last_z > 0 else "L")

        if loss_result is not None:
            _, pl, vl = loss_result
            recent_policy_loss.append(pl)
            recent_value_loss.append(vl)

        # --- Printing ---
        if (episode + 1) % Config.PRINT_FREQUENCY == 0:
            wr = recent_outcomes.count("W") / max(len(recent_outcomes), 1)
            lr = recent_outcomes.count("L") / max(len(recent_outcomes), 1)
            dr = recent_outcomes.count("D") / max(len(recent_outcomes), 1)
            avg_pl = np.mean(recent_policy_loss) if recent_policy_loss else 0.0
            avg_vl = np.mean(recent_value_loss) if recent_value_loss else 0.0

            print(
                f"\nEp {episode + 1} [{stage.name}] "
                f"W/L/D: {wr:.2f}/{lr:.2f}/{dr:.2f} | "
                f"P_loss: {avg_pl:.4f} | V_loss: {avg_vl:.4f} | "
                f"Buffer: {len(replay_buffer)} | Moves: {num_moves}"
            )

    # Save final model
    save_checkpoint(network, f"{Config.MODEL_DIR}/final.pth")
    print("\nTraining complete. Final model saved.")


if __name__ == "__main__":
    train()
