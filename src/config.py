# src/config.py
import torch


class Config:
    # Device
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Board
    BOARD_SIZE: int = 9
    WIN_LENGTH: int = 5
    RANDOM_SEED: int = 42

    # Network
    NUM_FILTERS_1: int = 32
    NUM_FILTERS_2: int = 64
    NUM_FILTERS_3: int = 64
    POLICY_FC_SIZE: int = 81
    VALUE_FC_SIZE: int = 64
    INPUT_CHANNELS: int = 3

    # MCTS
    NUM_SIMULATIONS: int = 25
    C_PUCT: float = 1.5
    DIRICHLET_ALPHA: float = 0.25
    DIRICHLET_WEIGHT: float = 0.25
    TEMPERATURE_THRESHOLD: int = 8

    # Training
    LEARNING_RATE: float = 1e-3
    L2_REG: float = 1e-4
    BATCH_SIZE: int = 256
    BUFFER_CAPACITY: int = 100_000
    POLICY_LOSS_WEIGHT: float = 1.0
    VALUE_LOSS_WEIGHT: float = 1.0
    TRAIN_STEPS_PER_GAME: int = 3

    # Curriculum
    RANDOM_EPISODES: int = 300
    HEURISTIC_EPISODES: int = 0
    SELFPLAY_EPISODES: int = 0
    TOTAL_EPISODES: int = RANDOM_EPISODES + HEURISTIC_EPISODES + SELFPLAY_EPISODES
    CHECKPOINT_INTERVAL: int = 500
    SNAPSHOT_POOL_SIZE: int = 10
    OLD_OPPONENT_CHANCE: float = 0.2

    # Evaluation
    EVAL_SIMULATIONS: int = 200

    # Logging
    PRINT_FREQUENCY: int = 100
    LOG_DIR: str = "./logs"
    MODEL_DIR: str = "./models"
