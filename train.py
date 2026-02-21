# train.py
import torch
import os
from collections import deque
from enum import Enum, auto
from src.game import Game, GameResult, Color
from src.network import DQNAgent
from src.logger import Logger
from src.config import Config
from src.rewards import calculate_shaped_reward
from src.symmetry import get_symmetric_transitions
import random
import numpy as np
from tqdm import trange


class Stage(Enum):
    RANDOM = auto()
    SELFPLAY = auto()


class RandomAgent:
    def select_action(self, game):
        legal_actions = game.get_legal_actions()
        return random.choice(legal_actions)


def load_pool_opponent(checkpoint_path):
    """Load a frozen opponent from a checkpoint path and force its epsilon."""
    opponent = DQNAgent()
    opponent.load_model(checkpoint_path)
    opponent.epsilon = Config.OPPONENT_EPSILON
    return opponent


def select_opponent(pool, opponent_cache):
    if len(pool) == 1 or random.random() > Config.OLD_OPPONENT_CHANCE:
        path = pool[-1]
    else:
        path = random.choice(pool[:-1])

    if path not in opponent_cache:
        opponent_cache[path] = load_pool_opponent(path)

    return opponent_cache[path]


def flush_nstep_buffer(nstep_buffer, gamma, next_state, done):
    """
    Collapse the n-step buffer into a single transition.

    nstep_buffer contains: [(s0, a0, r0), (s1, a1, r1), ..., (s_{n-1}, a_{n-1}, r_{n-1})]

    Returns: (s0, a0, R_nstep, next_state, done)
    where R_nstep = r0 + γ*r1 + γ²*r2 + ... + γ^{n-1}*r_{n-1}
    """
    state_0, action_0, _ = nstep_buffer[0]

    # Compute discounted n-step return
    R = 0.0
    for i in reversed(range(len(nstep_buffer))):
        _, _, r_i = nstep_buffer[i]
        R = r_i + gamma * R

    return (state_0, action_0, R, next_state, done)


def play_episode(player, opponent):
    game = Game()
    game.reset()

    agent_is_black = random.random() < 0.5
    agent_color = Color.BLACK if agent_is_black else Color.WHITE
    opponent_color = Color.WHITE if agent_is_black else Color.BLACK

    gamma = Config.GAMMA
    n = Config.N_STEP

    # n-step buffer holds (state, action_int, reward) for recent agent steps
    nstep_buffer = deque(maxlen=n)
    completed_transitions = []

    while game.result == GameResult.ONGOING:
        is_agent_turn = (game.current_player == Color.BLACK and agent_is_black) or (
            game.current_player == Color.WHITE and not agent_is_black
        )

        if is_agent_turn:
            current_state = game.get_state_for_network(perspective_color=agent_color)

            # If buffer is full, flush the oldest transition
            # The "next_state" for that transition is the current state (before we move)
            if len(nstep_buffer) == n:
                transition = flush_nstep_buffer(nstep_buffer, gamma, current_state, False)
                completed_transitions.append(transition)
                nstep_buffer.popleft()

            action = player.select_action(game)
            action_int = game.action_to_int(action)

            game.step(action)

            step_reward = calculate_shaped_reward(
                game, action, agent_color.value, opponent_color.value
            )
            step_reward += Config.STEP_PENALTY

            if game.result != GameResult.ONGOING:
                # Terminal after agent's move (agent won or draw)
                final_reward = (
                    Config.WIN_REWARD if game.result != GameResult.DRAW else Config.DRAW_REWARD
                )
                # Replace step_reward with terminal reward for this step
                nstep_buffer.append((current_state, action_int, final_reward))

                # Flush all remaining transitions in the buffer as terminal
                while len(nstep_buffer) > 0:
                    transition = flush_nstep_buffer(nstep_buffer, gamma, None, True)
                    completed_transitions.append(transition)
                    nstep_buffer.popleft()
            else:
                nstep_buffer.append((current_state, action_int, step_reward))

        else:
            # Opponent's turn
            action = opponent.select_action(game)
            game.step(action)

            if game.result != GameResult.ONGOING:
                # Terminal after opponent's move (agent lost or draw)
                final_reward = (
                    Config.LOSS_REWARD if game.result != GameResult.DRAW else Config.DRAW_REWARD
                )

                if len(nstep_buffer) > 0:
                    # Overwrite the reward of the most recent agent step
                    # That was the last move the agent made before losing
                    last_state, last_action, _ = nstep_buffer[-1]
                    nstep_buffer[-1] = (last_state, last_action, final_reward)

                    # Flush all remaining as terminal
                    while len(nstep_buffer) > 0:
                        transition = flush_nstep_buffer(nstep_buffer, gamma, None, True)
                        completed_transitions.append(transition)
                        nstep_buffer.popleft()

    return completed_transitions, game.result, len(game.move_history), agent_is_black


def train():
    os.makedirs(Config.MODEL_DIR, exist_ok=True)

    torch.manual_seed(Config.RANDOM_SEED)
    torch.cuda.manual_seed(Config.RANDOM_SEED)
    random.seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)

    player = DQNAgent()
    random_opponent = RandomAgent()
    logger = Logger()
    recent_outcomes = deque(maxlen=Config.OUTCOMES_MAXLEN)
    recent_rewards = deque(maxlen=Config.REWARDS_MAXLEN)
    recent_moves = deque(maxlen=Config.OUTCOMES_MAXLEN)

    stage = Stage.RANDOM
    opponent_pool = []
    opponent_cache = {}

    for episode in trange(Config.TOTAL_EPISODES):

        # Stage switch
        if stage == Stage.RANDOM and episode >= Config.RANDOM_EPISODES:
            stage = Stage.SELFPLAY
            checkpoint_path = f"{Config.MODEL_DIR}/checkpoint_0.pth"
            player.save_model(checkpoint_path)
            opponent_pool.append(checkpoint_path)
            print(f"\n[{stage.name}] Switching to self-play at episode {episode}")

        # Opponent selection
        if stage == Stage.RANDOM:
            opponent = random_opponent
        else:
            opponent = select_opponent(opponent_pool, opponent_cache)

        transitions, result, num_moves, agent_is_black = play_episode(player, opponent)

        # Store all transitions with 8-fold symmetry
        for state, action, reward, next_state, done in transitions:
            for sym in get_symmetric_transitions(state, action, reward, next_state, done):
                player.store_transition(*sym)

        # Multiple training steps per episode
        losses = []
        for _ in range(Config.TRAIN_STEPS_PER_EPISODE):
            loss = player.train_step()
            if loss is not None:
                losses.append(loss)

        avg_loss = np.mean(losses) if losses else 0.0

        player.decay_epsilon()

        # Stage 2 checkpointing
        if (
            stage == Stage.SELFPLAY
            and (episode + 1 - Config.RANDOM_EPISODES) % Config.CHECKPOINT_INTERVAL == 0
        ):
            checkpoint_idx = len(opponent_pool)
            checkpoint_path = f"{Config.MODEL_DIR}/checkpoint_{checkpoint_idx}.pth"
            player.save_model(checkpoint_path)
            opponent_pool.append(checkpoint_path)
            print(
                f"\n[{stage.name}] Saved checkpoint_{checkpoint_idx}.pth (pool size: {len(opponent_pool)})"
            )

        # Process outcome
        won = (result == GameResult.BLACK_WIN and agent_is_black) or (
            result == GameResult.WHITE_WIN and not agent_is_black
        )
        outcome = "D"
        if result != GameResult.DRAW:
            outcome = "W" if won else "L"

        logger.log_episode(
            {
                "episode": episode + 1,
                "outcome": outcome,
                "reward": round(sum(t[2] for t in transitions), 3),
                "loss": round(avg_loss, 3),
                "epsilon": round(player.epsilon, 3),
                "buffer": len(player.buffer),
                "moves": num_moves,
            }
        )

        recent_outcomes.append(outcome)
        recent_rewards.append(sum(t[2] for t in transitions))
        recent_moves.append(num_moves)

        if (episode + 1) % Config.PRINT_FREQUENCY == 0:
            wr = recent_outcomes.count("W") / len(recent_outcomes)
            lr = recent_outcomes.count("L") / len(recent_outcomes)
            dr = recent_outcomes.count("D") / len(recent_outcomes)
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
            avg_moves = np.mean(recent_moves) if recent_moves else 0.0

            print(f"\nEpisode {episode + 1} [{stage.name}], W L D: {wr:.2f} - {lr:.2f} - {dr:.2f}")
            print(
                f"Reward: {avg_reward:.3f}, Epsilon: {player.epsilon:.3f}, Loss: {avg_loss:.3f}, Buffer: {len(player.buffer)}, Moves: {avg_moves:.1f}"
            )
            logger.save()

    logger.save()
    print("\nTraining complete")


if __name__ == "__main__":
    train()
