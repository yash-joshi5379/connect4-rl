# train.py
import torch
import os
from collections import deque
from src.game import Game, GameResult, Color
from src.network import DQNAgent
from src.logger import Logger
from src.config import Config
from src.rewards import calculate_shaped_reward
import random
import numpy as np
from tqdm import trange


class RandomAgent:
    def select_action(self, game):
        legal_actions = game.get_legal_actions()
        return random.choice(legal_actions)


def play_episode(player, opponent):
    game = Game()
    game.reset()

    # Randomize which color the agent plays
    agent_is_black = random.random() < 0.5
    agent_color = Color.BLACK if agent_is_black else Color.WHITE
    opponent_color = Color.WHITE if agent_is_black else Color.BLACK

    episode_transitions = []

    last_agent_state = None
    last_agent_action = None
    last_agent_reward = 0.0

    # these are buffer variables to track the agent while we wait for the opponent to move

    while game.result == GameResult.ONGOING:
        is_agent_turn = (game.current_player == Color.BLACK and agent_is_black) or (
            game.current_player == Color.WHITE and not agent_is_black
        )

        if is_agent_turn:
            # state = game.get_state_for_network()

            current_state = game.get_state_for_network(
                perspective_color=agent_color
            )  # get state from agent's perspective

            if last_agent_state is not None:
                episode_transitions.append(
                    (last_agent_state, last_agent_action, last_agent_reward, current_state, False)
                )

            action = player.select_action(game)
            action_int = game.action_to_int(action)

            state_before_move = current_state

            game.step(action)

            # Calculate reward including shaped rewards

            step_reward = calculate_shaped_reward(
                game, action, agent_color.value, opponent_color.value
            )

            if game.result != GameResult.ONGOING:
                final_reward = (
                    Config.WIN_REWARD if game.result != GameResult.DRAW else Config.DRAW_REWARD
                )

                episode_transitions.append(
                    (state_before_move, action_int, final_reward, None, True)
                )
            else:
                last_agent_state = state_before_move
                last_agent_action = action_int
                last_agent_reward = step_reward

        else:
            action = opponent.select_action(game)
            game.step(action)
            if game.result != GameResult.ONGOING:

                outcome_reward = (
                    Config.LOSS_REWARD if game.result != GameResult.DRAW else Config.DRAW_REWARD
                )
                final_reward = last_agent_reward + outcome_reward

                episode_transitions.append(
                    (last_agent_state, last_agent_action, final_reward, None, True)
                )

    return episode_transitions, game.result, len(game.move_history), agent_is_black


def train():
    os.makedirs(Config.MODEL_DIR, exist_ok=True)

    # Set seeds for reproducibility
    torch.manual_seed(Config.RANDOM_SEED)
    torch.cuda.manual_seed(Config.RANDOM_SEED)
    random.seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)

    player = DQNAgent()
    random_opponent = RandomAgent()
    logger = Logger()
    recent_outcomes = deque(maxlen=Config.OUTCOMES_MAXLEN)
    recent_rewards = deque(maxlen=Config.REWARDS_MAXLEN)

    for episode in trange(Config.TOTAL_EPISODES):
        opponent = random_opponent

        transitions, result, _, agent_is_black = play_episode(player, opponent)

        # Store all transitions
        for state, action, reward, next_state, done in transitions:
            player.store_transition(state, action, reward, next_state, done)

        # Multiple training steps per episode
        losses = []
        for _ in range(Config.TRAIN_STEPS_PER_EPISODE):
            loss = player.train_step()
            if loss is not None:
                losses.append(loss)

        avg_loss = np.mean(losses) if losses else 0.0

        player.decay_epsilon()

        # Process outcome
        won = (result == GameResult.BLACK_WIN and agent_is_black) or (
            result == GameResult.WHITE_WIN and not agent_is_black
        )

        outcome = "D"
        if result != GameResult.DRAW:
            outcome = "W" if won else "L"

        # Log to the csv
        logger.log_episode(
            {
                "episode": episode + 1,
                "outcome": outcome,
                "reward": round(sum(t[2] for t in transitions), 3),
                "loss": round(avg_loss, 3),
                "epsilon": round(player.epsilon, 3),
                "buffer": len(player.buffer),
            }
        )

        recent_outcomes.append(outcome)  # Win Loss or Draw
        recent_rewards.append(sum(t[2] for t in transitions))  # t[2] is the reward, S A R S' done

        # Print the moving averages every PRINT_FREQUENCY episodes
        if (episode + 1) % Config.PRINT_FREQUENCY == 0:
            wr = recent_outcomes.count("W") / len(recent_outcomes)
            lr = recent_outcomes.count("L") / len(recent_outcomes)
            dr = recent_outcomes.count("D") / len(recent_outcomes)
            reward = np.mean(recent_rewards) if recent_rewards else 0.0

            print(f"\nEpisode {episode + 1}, W L D: {wr:.2f} - {lr:.2f} - {dr:.2f}")
            print(
                f"Reward: {reward:.3f}, Epsilon: {player.epsilon:.3f}, Loss: {avg_loss:.3f}, Buffer: {len(player.buffer)}"
            )
            logger.save()

    player.save_model(f"{Config.MODEL_DIR}/checkpoint_random.pth")
    logger.save()
    print("\nTraining complete")


if __name__ == "__main__":
    train()
