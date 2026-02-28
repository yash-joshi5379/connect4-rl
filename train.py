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
    RANDOM = auto()
    HEURISTIC = auto()
    SELFPLAY = auto()


class RandomAgent:
    def select_action(self, game):
        return random.choice(game.get_legal_actions())


def set_seeds():
    torch.manual_seed(Config.RANDOM_SEED)
    random.seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)


def sample_from_policy(policy, temperature=1.0):
    """Sample an action from the policy with temperature."""
    if temperature < 1e-6:
        return np.argmax(policy)

    policy = policy ** (1.0 / temperature)
    total = policy.sum()
    if total > 0:
        policy = policy / total
    else:
        policy = np.ones_like(policy) / len(policy)

    return np.random.choice(len(policy), p=policy)


def play_game_vs_rule_agent(mcts_current, rule_agent):
    """
    Play one game: MCTS agent vs a rule-based opponent (random or heuristic).
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
            action = rule_agent.select_action(game)
            game.step(action)

        move_count += 1

    return samples, game.result, move_count


def play_game_selfplay(mcts_current, mcts_opponent):
    """
    Play one self-play game. Collect training samples from both sides.
    """
    game = Game()
    game.reset()

    current_is_black = random.random() < 0.5
    current_color = Color.BLACK if current_is_black else Color.WHITE
    opponent_color = Color.WHITE if current_is_black else Color.BLACK

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
            state = game.get_state_for_network(perspective_color=opponent_color)
            policy = mcts_opponent.search(game, add_noise=True)

            if move_count < Config.TEMPERATURE_THRESHOLD:
                action_int = sample_from_policy(policy, temperature=1.0)
            else:
                action_int = np.argmax(policy)

            samples.append((state, policy, opponent_color))

            action = game.int_to_action(action_int)
            game.step(action)

        move_count += 1

    return samples, game.result, move_count


def label_samples(samples, result):
    """Label each sample with the game outcome from that player's perspective."""
    labeled = []

    for state, policy, color in samples:
        if result == GameResult.DRAW:
            z = 0.0
        elif result == GameResult.BLACK_WIN:
            z = 1.0 if color == Color.BLACK else -1.0
        else:
            z = 1.0 if color == Color.WHITE else -1.0

        labeled.append((state, policy, z))

    return labeled


def train_step(network, optimizer, replay_buffer):
    """One training step. Returns (total_loss, policy_loss, value_loss) or None."""
    if len(replay_buffer) < Config.BATCH_SIZE:
        return None

    states, policies, zs = replay_buffer.sample()

    states_t = torch.FloatTensor(states).to(Config.DEVICE)
    policies_t = torch.FloatTensor(policies).to(Config.DEVICE)
    zs_t = torch.FloatTensor(zs).unsqueeze(1).to(Config.DEVICE)

    network.train()
    log_policy, value = network(states_t)

    policy_loss = -torch.mean(torch.sum(policies_t * log_policy, dim=1))
    value_loss = torch.mean((value - zs_t) ** 2)

    total_loss = Config.POLICY_LOSS_WEIGHT * policy_loss + Config.VALUE_LOSS_WEIGHT * value_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss.item(), policy_loss.item(), value_loss.item()


def save_checkpoint(network, path):
    torch.save(network.state_dict(), path)


def load_checkpoint(path):
    network = PolicyValueNetwork().to(Config.DEVICE)
    network.load_state_dict(torch.load(path, weights_only=True, map_location=Config.DEVICE))
    network.eval()
    return network


def train():
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    set_seeds()

    print(f"Using device: {Config.DEVICE}")

    # Core components
    network = PolicyValueNetwork().to(Config.DEVICE)
    optimizer = optim.Adam(
        network.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.L2_REG,
    )
    mcts = MCTS(network)
    replay_buffer = ReplayBuffer()

    # Opponents
    random_agent = RandomAgent()
    heuristic_agent = HeuristicAgent(
        win_rate=0.9,
        block_win_rate=0.5,
        make_open4_rate=0.2,
        block_open3_rate=0.1,
        make_half4_rate=0.1,
        block_half3_rate=0.05,
        attack_2_rate=0.15,
        attack_1_rate=0.1,
    )

    # Self-play pool
    snapshot_pool = []
    snapshot_cache = {}

    # Tracking
    stage = Stage.RANDOM
    recent_outcomes = deque(maxlen=100)
    recent_policy_loss = deque(maxlen=100)
    recent_value_loss = deque(maxlen=100)

    heuristic_start = Config.RANDOM_EPISODES
    selfplay_start = Config.RANDOM_EPISODES + Config.HEURISTIC_EPISODES

    for episode in trange(Config.TOTAL_EPISODES, desc="Training"):

        # --- Stage transitions ---
        if stage == Stage.RANDOM and episode >= heuristic_start:
            stage = Stage.HEURISTIC
            recent_outcomes.clear()
            print(f"\n[HEURISTIC] Switching to heuristic opponent at episode {episode}")

        if stage == Stage.HEURISTIC and episode >= selfplay_start:
            stage = Stage.SELFPLAY
            recent_outcomes.clear()
            path = f"{Config.MODEL_DIR}/checkpoint_0.pth"
            save_checkpoint(network, path)
            snapshot_pool.append(path)
            print(f"\n[SELFPLAY] Switching to self-play at episode {episode}")

        # --- Play one game ---
        if stage == Stage.RANDOM:
            samples, result, num_moves = play_game_vs_rule_agent(mcts, random_agent)

        elif stage == Stage.HEURISTIC:
            samples, result, num_moves = play_game_vs_rule_agent(mcts, heuristic_agent)

        else:
            # Pick opponent from snapshot pool
            if len(snapshot_pool) > 1 and random.random() < Config.OLD_OPPONENT_CHANCE:
                opp_path = random.choice(snapshot_pool[:-1])
            else:
                opp_path = snapshot_pool[-1]

            if opp_path not in snapshot_cache:
                snapshot_cache[opp_path] = load_checkpoint(opp_path)
                if len(snapshot_cache) > Config.SNAPSHOT_POOL_SIZE + 2:
                    oldest = list(snapshot_cache.keys())[0]
                    del snapshot_cache[oldest]

            opp_network = snapshot_cache[opp_path]
            opp_mcts = MCTS(opp_network)
            samples, result, num_moves = play_game_selfplay(mcts, opp_mcts)

        # --- Label samples with game outcome ---
        labeled = label_samples(samples, result)

        # --- Store with 8-fold symmetry ---
        for state, policy, z in labeled:
            for sym_state, sym_policy, sym_z in get_symmetric_samples(state, policy, z):
                replay_buffer.push(sym_state, sym_policy, sym_z)

        # --- Multiple training steps per game ---
        for _ in range(Config.TRAIN_STEPS_PER_GAME):
            loss_result = train_step(network, optimizer, replay_buffer)

        # --- Self-play checkpointing ---
        if stage == Stage.SELFPLAY:
            selfplay_ep = episode - selfplay_start
            if (selfplay_ep + 1) % Config.CHECKPOINT_INTERVAL == 0:
                idx = len(snapshot_pool)
                path = f"{Config.MODEL_DIR}/checkpoint_{idx}.pth"
                save_checkpoint(network, path)
                snapshot_pool.append(path)

                if len(snapshot_pool) > Config.SNAPSHOT_POOL_SIZE:
                    removed = snapshot_pool.pop(0)
                    snapshot_cache.pop(removed, None)

                print(f"\n[SELFPLAY] Checkpoint {idx} saved (pool: {len(snapshot_pool)})")

        # --- Tracking ---
        if result == GameResult.DRAW:
            recent_outcomes.append("D")
        elif len(labeled) > 0:
            last_z = labeled[0][2]
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
