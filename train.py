# train.py
import torch
import os
from src.game import GomokuGame, GameResult, Player as GamePlayer
from src.network import Player
from src.logger import GameLogger
from src.config import Config
import random
import numpy as np
from tqdm import trange


class RandomAgent:
    def select_action(self, game):
        legal_actions = game.get_legal_actions()
        return random.choice(legal_actions)


def calculate_shaped_reward(game, action, agent_color):
    """Calculate intermediate rewards for threats and blocks"""
    row, col = action

    # Check if this move creates a threat (3 or 4 in a row)
    threat_reward = 0.0
    block_reward = 0.0

    # Temporarily check what this move creates
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

    for dr, dc in directions:
        count = 1
        # Count in positive direction
        r, c = row + dr, col + dc
        while (
            0 <= r < Config.BOARD_SIZE
            and 0 <= c < Config.BOARD_SIZE
            and game.board[r, c] == agent_color
        ):
            count += 1
            r += dr
            c += dc

        # Count in negative direction
        r, c = row - dr, col - dc
        while (
            0 <= r < Config.BOARD_SIZE
            and 0 <= c < Config.BOARD_SIZE
            and game.board[r, c] == agent_color
        ):
            count += 1
            r -= dr
            c -= dc

        # Reward for creating threats
        if count == 4:
            threat_reward = 0.05
        elif count == 3:
            threat_reward = max(threat_reward, 0.03)
        elif count == 2:
            threat_reward = max(threat_reward, 0.01)

    return threat_reward


def check_opponent_threat(game, agent_color):
    """Check if opponent has a threatening position that should be blocked"""
    opponent_color = (
        GamePlayer.WHITE.value if agent_color == GamePlayer.BLACK.value else GamePlayer.BLACK.value
    )

    for row in range(Config.BOARD_SIZE):
        for col in range(Config.BOARD_SIZE):
            if game.board[row, col] == opponent_color:
                directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
                for dr, dc in directions:
                    count = 1
                    r, c = row + dr, col + dc
                    while (
                        0 <= r < Config.BOARD_SIZE
                        and 0 <= c < Config.BOARD_SIZE
                        and game.board[r, c] == opponent_color
                    ):
                        count += 1
                        r += dr
                        c += dc

                    r, c = row - dr, col - dc
                    while (
                        0 <= r < Config.BOARD_SIZE
                        and 0 <= c < Config.BOARD_SIZE
                        and game.board[r, c] == opponent_color
                    ):
                        count += 1
                        r -= dr
                        c -= dc

                    if count >= 3:
                        return True
    return False


def get_reward(game_result, is_agent_black, game, action, agent_color):
    """Calculate total reward including terminal and shaped rewards"""
    # Terminal reward
    if game_result != GameResult.ONGOING:
        if game_result == GameResult.DRAW:
            return 0.0

        agent_won = (game_result == GameResult.BLACK_WIN and is_agent_black) or (
            game_result == GameResult.WHITE_WIN and not is_agent_black
        )

        return 1.0 if agent_won else -1.0

    # Intermediate shaped rewards
    threat_reward = calculate_shaped_reward(game, action, agent_color)

    return threat_reward


def play_episode(player, opponent):
    game = GomokuGame()
    game.reset()

    # Randomize which color the agent plays
    agent_is_black = random.random() < 0.5
    agent_color = GamePlayer.BLACK if agent_is_black else GamePlayer.WHITE

    episode_transitions = []

    while game.result == GameResult.ONGOING:
        is_agent_turn = (game.current_player == GamePlayer.BLACK and agent_is_black) or (
            game.current_player == GamePlayer.WHITE and not agent_is_black
        )

        if is_agent_turn:
            state = game.get_state_for_network()
            action = player.select_action(game)
            action_int = game.action_to_int(action)

            game.step(action)

            # Calculate reward including shaped rewards
            reward = get_reward(game.result, agent_is_black, game, action, agent_color.value)

            next_state = game.get_state_for_network() if game.result == GameResult.ONGOING else None
            done = game.result != GameResult.ONGOING

            episode_transitions.append((state, action_int, reward, next_state, done))
        else:
            action = opponent.select_action(game)
            game.step(action)

    return episode_transitions, game.result, len(game.move_history), agent_is_black


def train():
    os.makedirs(Config.MODEL_DIR, exist_ok=True)

    player = Player()
    random_opponent = RandomAgent()
    logger = GameLogger()

    win_count = 0
    loss_count = 0
    draw_count = 0

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
        elif agent_won:
            win_count += 1
        else:
            loss_count += 1

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
                    "buffer_size": len(player.buffer),
                },
            )

        if (episode + 1) % Config.SAVE_FREQ == 0:
            total_games = episode + 1
            win_rate = win_count / total_games
            loss_str = f"{avg_loss:.4f}" if avg_loss is not None else "N/A"
            print(f"Episode {episode + 1}/{Config.TOTAL_EPISODES}")
            print(f"  Win Rate: {win_rate:.3f} ({win_count}W-{loss_count}L-{draw_count}D)")
            print(
                f"  Epsilon: {player.epsilon:.3f}, Loss: {loss_str}, Buffer: {len(player.buffer)}"
            )
            logger.save()
            player.save_model(f"{Config.MODEL_DIR}/player_ep{episode+1}.pth")

    player.save_model(f"{Config.MODEL_DIR}/player_final.pth")
    logger.save()
    print("\nTraining complete!")
    print(f"Final Win Rate: {win_count / Config.TOTAL_EPISODES:.3f}")


if __name__ == "__main__":
    train()
