# src/logger.py
import pandas as pd
from pathlib import Path
from src.config import Config

# tensorboard for live plots
from torch.utils.tensorboard import SummaryWriter

# 0.95 smoothing is nice for the three curves of outcome, reward, and loss


class Logger:
    def __init__(self, log_dir=None):
        self.log_dir = Path(log_dir or Config.LOG_DIR)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.data = []
        # summary writer is what tensorboard uses
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def log_episode(self, stats):
        self.data.append(stats)
        ep = stats["episode"]
        outcome_map = {"W": 1, "D": 0, "L": -1}
        self.writer.add_scalar("outcome", outcome_map[stats["outcome"]], ep)
        self.writer.add_scalar("reward", stats["reward"], ep)
        self.writer.add_scalar("loss", stats["loss"], ep)
        self.writer.flush()

    def save(self):
        self.writer.flush()
        if self.data:
            df = pd.DataFrame(self.data)
            cols = ["episode", "outcome", "reward", "loss", "epsilon", "buffer"]
            existing = [c for c in cols if c in df.columns]
            df[existing].to_csv(self.log_dir / "training.csv", index=False)
