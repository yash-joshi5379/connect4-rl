# src/config.py
class Config:
    # Settings
    BOARD_SIZE: int = 9
    WIN_LENGTH: int = 5
    RANDOM_SEED: int = 42  # for reproducibility

    # Printing/logging
    PRINT_FREQUENCY: int = 100
    OUTCOMES_MAXLEN: int = 100  # for tracking recent outcomes
    REWARDS_MAXLEN: int = 100  # for tracking recent rewards

    EPSILON_DECAY: float = 0.995
    LEARNING_RATE: float = 1e-4
    TARGET_UPDATE_FREQ: int = 2000
    TRAIN_STEPS_PER_EPISODE: int = 4
    BUFFER_CAPACITY: int = 50_000
    GAMMA: float = 0.99
    EPSILON_END: float = 0.01
    EPSILON_START: float = 1.0
    BATCH_SIZE: int = 64
    # GRAD_CLIP_NORM: float = 1.0

    # Paths
    MODEL_DIR: str = "./models"
    LOG_DIR: str = "./logs"


# Assertions
assert Config.BOARD_SIZE in (9, 15), "BOARD_SIZE must be 9 or 15"
assert Config.WIN_LENGTH == 5, "WIN_LENGTH must be 5"
assert Config.RANDOM_SEED == 42, "RANDOM_SEED should be 42 for reproducibility"