# train.py
import torch
import os
from src.game import GomokuGame, GameResult, Color
from src.network import DQNAgent
from src.logger import GameLogger
from src.config import Config
import random
import numpy as np
from tqdm import trange
from collections import deque


class RandomAgent:
    def select_action(self, game):
        legal_actions = game.get_legal_actions()
        return random.choice(legal_actions)


def count_line(game, row, col, dr, dc, color):
    """Count consecutive stones of given color in one direction from (row, col)"""
    count = 0
    r, c = row + dr, col + dc
    while 0 <= r < Config.BOARD_SIZE and 0 <= c < Config.BOARD_SIZE and game.board[r, c] == color:
        count += 1
        r += dr
        c += dc
    return count


def get_pattern_length(game, row, col, color):
    """Get maximum line length for a stone of given color at (row, col)"""
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    max_length = 0

    for dr, dc in directions:
        # Count in both directions and add 1 for the stone itself
        length = (
            1
            + count_line(game, row, col, dr, dc, color)
            + count_line(game, row, col, -dr, -dc, color)
        )
        max_length = max(max_length, length)

    return max_length


def check_blocks_opponent(game, row, col, opponent_color):
    """Check if placing a stone at (row, col) blocks an opponent threat"""
    # Temporarily check what opponent would have had at this position
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    max_blocked_length = 0

    for dr, dc in directions:
        # Check what the opponent line would be through this empty position
        length = (
            1
            + count_line(game, row, col, dr, dc, opponent_color)
            + count_line(game, row, col, -dr, -dc, opponent_color)
        )
        max_blocked_length = max(max_blocked_length, length)

    return max_blocked_length


def calculate_shaped_reward(game, action, agent_color, opponent_color):
    """Calculate intermediate rewards for threats and blocks"""
    row, col = action

    # Threat reward: what pattern did we create?
    threat_length = get_pattern_length(game, row, col, agent_color)
    threat_reward = 0.0
    if threat_length == 4:
        threat_reward = Config.THREAT_REWARD_4
    elif threat_length == 3:
        threat_reward = Config.THREAT_REWARD_3
    elif threat_length == 2:
        threat_reward = Config.THREAT_REWARD_2

    # Block reward: what opponent pattern did we block?
    blocked_length = check_blocks_opponent(game, row, col, opponent_color)
    block_reward = 0.0
    if blocked_length >= 4:
        block_reward = Config.BLOCK_REWARD_4
    elif blocked_length == 3:
        block_reward = Config.BLOCK_REWARD_3

    return threat_reward + block_reward


def get_reward(game_result, agent_is_black, game, action, agent_color, opponent_color):
    """Calculate total reward including terminal and shaped rewards"""
    # Terminal reward
    if game_result != GameResult.ONGOING:
        if game_result == GameResult.DRAW:
            return Config.DRAW_REWARD

        agent_won = (game_result == GameResult.BLACK_WIN and agent_is_black) or (
            game_result == GameResult.WHITE_WIN and not agent_is_black
        )

        return Config.WIN_REWARD if agent_won else Config.LOSS_REWARD

    # Calculate shaped rewards (intermediate smaller rewards for creating threats or blocking opponent threats)
    shaped_reward = calculate_shaped_reward(game, action, agent_color, opponent_color)

    return shaped_reward


def play_episode(player, opponent):
    game = GomokuGame()
    game.reset()

    # Randomize which color the agent plays
    agent_is_black = random.random() < 0.5
    agent_color = Color.BLACK if agent_is_black else Color.WHITE
    opponent_color = Color.WHITE if agent_is_black else Color.BLACK

    episode_transitions = []

    while game.result == GameResult.ONGOING:
        is_agent_turn = (game.current_player == Color.BLACK and agent_is_black) or (
            game.current_player == Color.WHITE and not agent_is_black
        )

        if is_agent_turn:
            state = game.get_state_for_network()
            action = player.select_action(game)
            action_int = game.action_to_int(action)

            game.step(action)

            # Calculate reward including shaped rewards
            reward = get_reward(
                game.result, agent_is_black, game, action, agent_color.value, opponent_color.value
            )

            next_state = game.get_state_for_network() if game.result == GameResult.ONGOING else None
            done = game.result != GameResult.ONGOING

            episode_transitions.append((state, action_int, reward, next_state, done))
        else:
            action = opponent.select_action(game)
            game.step(action)

    return episode_transitions, game.result, len(game.move_history), agent_is_black


def train():
    os.makedirs(Config.MODEL_DIR, exist_ok=True)

    player = DQNAgent()
    random_opponent = RandomAgent()
    logger = GameLogger()

    win_count = 0
    loss_count = 0
    draw_count = 0

    # Rolling win rate tracking
    rolling_window = deque(maxlen=Config.ROLLING_WINDOW_SIZE)
    best_win_rate = 0.0

    for episode in trange(Config.TOTAL_EPISODES):
        opponent = random_opponent

        transitions, result, num_moves, agent_is_black = play_episode(player, opponent)

        # Store all transitions
        for state, action, reward, next_state, done in transitions:
            player.store_transition(state, action, reward, next_state, done)

        # Multiple training steps per episode
        losses = []
        for _ in range(Config.TRAIN_STEPS_PER_EPISODE):
            loss = player.train_step()
            if loss is not None:
                losses.append(loss)

        avg_loss = np.mean(losses) if losses else None

        player.decay_epsilon()

        # Track win rate
        agent_won = (result == GameResult.BLACK_WIN and agent_is_black) or (
            result == GameResult.WHITE_WIN and not agent_is_black
        )
        if result == GameResult.DRAW:
            draw_count += 1
            rolling_window.append(0)
        elif agent_won:
            win_count += 1
            rolling_window.append(1)
        else:
            loss_count += 1
            rolling_window.append(0)

        # Calculate rolling win rate
        rolling_win_rate = (
            sum(rolling_window) / len(rolling_window) if len(rolling_window) > 0 else 0.0
        )

        # Save best model based on rolling win rate
        if len(rolling_window) == Config.ROLLING_WINDOW_SIZE and rolling_win_rate > best_win_rate:
            best_win_rate = rolling_win_rate
            player.save_model(f"{Config.MODEL_DIR}/player_best.pth")

        logger.log_game(
            result,
            num_moves,
            metadata={"episode": episode, "agent_color": "BLACK" if agent_is_black else "WHITE"},
        )

        if avg_loss is not None:
            logger.log_episode(
                episode,
                {
                    "loss": avg_loss,
                    "epsilon": player.epsilon,
                    "win_rate": win_count / (episode + 1),
                    "rolling_win_rate": rolling_win_rate,
                    "buffer_size": len(player.buffer),
                },
            )

        if (episode + 1) % Config.SAVE_FREQ == 0:
            total_games = episode + 1
            win_rate = win_count / total_games
            loss_str = f"{avg_loss:.4f}" if avg_loss is not None else "N/A"
            print(f"Episode {episode + 1}/{Config.TOTAL_EPISODES}")
            print(f"  Win Rate: {win_rate:.3f} ({win_count}W-{loss_count}L-{draw_count}D)")
            print(f"  Rolling Win Rate (last {Config.ROLLING_WINDOW_SIZE}): {rolling_win_rate:.3f}")
            print(f"  Best Rolling Win Rate: {best_win_rate:.3f}")
            print(
                f"  Epsilon: {player.epsilon:.3f}, Loss: {loss_str}, Buffer: {len(player.buffer)}"
            )
            logger.save()
            player.save_model(f"{Config.MODEL_DIR}/player_ep{episode+1}.pth")

    player.save_model(f"{Config.MODEL_DIR}/player_final.pth")
    logger.save()
    print("\nTraining complete")
    print(f"Final Win Rate: {win_count / Config.TOTAL_EPISODES:.3f}")
    print(f"Best Rolling Win Rate: {best_win_rate:.3f}")


if __name__ == "__main__":
    answer = input("Run training? (y/n): ").strip().lower()
    if answer == "y":
        train()
    else:
        print("Training cancelled to avoid overwriting existing models")
