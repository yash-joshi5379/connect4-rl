# evaluate.py
from src.game import GomokuGame, GameResult, Color
from src.network import DQNAgent
from src.config import Config
import random
from tqdm import trange


class RandomAgent:
    def select_action(self, game):
        legal_actions = game.get_legal_actions()
        return random.choice(legal_actions)


def evaluate(player, opponent, num_games):
    player.epsilon = 0.0

    wins = 0
    losses = 0
    draws = 0

    for i in trange(num_games):
        game = GomokuGame()
        game.reset()

        player_color = Color.BLACK if i % 2 == 0 else Color.WHITE

        while game.result == GameResult.ONGOING:
            if game.current_player == player_color:
                action = player.select_action(game)
            else:
                action = opponent.select_action(game)
            game.step(action)

        if game.result == GameResult.DRAW:
            draws += 1
        elif (game.result == GameResult.BLACK_WIN and player_color == Color.BLACK) or (
            game.result == GameResult.WHITE_WIN and player_color == Color.WHITE
        ):
            wins += 1
        else:
            losses += 1

    win_rate = wins / num_games
    print(f"Results: {wins} wins, {losses} losses, {draws} draws | Win rate: {win_rate:.2%}")
    return win_rate


if __name__ == "__main__":
    player = DQNAgent()
    player.load_model(f"{Config.MODEL_DIR}/player_best.pth")

    print("Evaluating against random opponent...")
    random_opponent = RandomAgent()
    evaluate(player, random_opponent, 1000)
