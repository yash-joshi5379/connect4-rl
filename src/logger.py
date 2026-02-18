# src/logger.py
import pandas as pd
from pathlib import Path
from datetime import datetime
from src.config import Config


class Logger:
    def __init__(self, log_dir=None):
        if log_dir is None:
            log_dir = Config.LOG_DIR
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.games = []
        self.episodes = []

    def log_game(self, result, num_moves, metadata=None):
        game_data = {
            "timestamp": datetime.now().strftime("%d-%b %H:%M"),
            "result": result.name,
            "num_moves": num_moves,
            "winner": result.value,
        }
        if metadata:
            game_data.update(metadata)
        self.games.append(game_data)

    def log_episode(self, episode, metrics):
        episode_data = {"episode": episode}
        episode_data.update(metrics)
        self.episodes.append(episode_data)

    def save(self):
        if self.games:
            df_games = pd.DataFrame(self.games)
            df_games.to_csv(self.log_dir / "games.csv", index=False)

        if self.episodes:
            df_episodes = pd.DataFrame(self.episodes)
            df_episodes.to_csv(self.log_dir / "training.csv", index=False)

    def get_dataframes(self):
        df_games = pd.DataFrame(self.games) if self.games else pd.DataFrame()
        df_episodes = pd.DataFrame(self.episodes) if self.episodes else pd.DataFrame()
        return df_games, df_episodes
