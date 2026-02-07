# train.py
import torch
import os
from src.game import GomokuGame, GameResult, Player as GamePlayer
from src.network import Player
from src.logger import GameLogger
from src.config import Config
import random


class RandomAgent:
    def select_action(self, game):
        legal_actions = game.get_legal_actions()
        return random.choice(legal_actions)


def get_reward(game_result, is_agent_black):
    if game_result == GameResult.DRAW:
        return 0.0
    
    agent_won = (game_result == GameResult.BLACK_WIN and is_agent_black) or \
                (game_result == GameResult.WHITE_WIN and not is_agent_black)
    
    return 1.0 if agent_won else -1.0


def play_episode(player, opponent):
    game = GomokuGame()
    game.reset()
    
    # Agent plays as BLACK
    agent_is_black = True
    
    episode_transitions = []
    
    while game.result == GameResult.ONGOING:
        current_agent = player if game.current_player == GamePlayer.BLACK else opponent
        
        if current_agent == player:
            state = game.get_state_for_network()
            action = player.select_action(game)
            action_int = game.action_to_int(action)
            
            game.step(action)
            
            next_state = game.get_state_for_network() if game.result == GameResult.ONGOING else None
            done = game.result != GameResult.ONGOING
            reward = get_reward(game.result, agent_is_black) if done else 0.0
            
            episode_transitions.append((state, action_int, reward, next_state, done))
        else:
            action = current_agent.select_action(game)
            game.step(action)
    
    return episode_transitions, game.result, len(game.move_history)


def train():
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    
    player = Player()
    random_opponent = RandomAgent()
    logger = GameLogger()
    
    for episode in range(Config.TOTAL_EPISODES):
        opponent = random_opponent
        
        transitions, result, num_moves = play_episode(player, opponent)
        
        for state, action, reward, next_state, done in transitions:
            player.store_transition(state, action, reward, next_state, done)
        
        loss = player.train_step()
        player.decay_epsilon()
        
        logger.log_game(result, num_moves, metadata={"episode": episode})
        
        if loss is not None:
            logger.log_episode(episode, {"loss": loss, "epsilon": player.epsilon})
        
        if (episode + 1) % Config.SAVE_FREQ == 0:
            loss_str = f"{loss:.4f}" if loss is not None else "N/A"
            print(f"Episode {episode + 1}/{Config.TOTAL_EPISODES}, Epsilon: {player.epsilon:.3f}, Loss: {loss_str}")
            logger.save()
            player.save_model(f"{Config.MODEL_DIR}/player_ep{episode+1}.pth")
    
    player.save_model(f"{Config.MODEL_DIR}/player_final.pth")
    logger.save()
    print("Training complete!")


if __name__ == "__main__":
    train()