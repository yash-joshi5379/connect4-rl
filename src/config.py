# src/config.py
class Config:
    # Board settings
    BOARD_SIZE: int = 9
    WIN_LENGTH: int = 5

    # Episodes and printing
    TOTAL_EPISODES: int = 2000
    PRINT_FREQUENCY: int = 100

    # Tunable hyperparameters (most -> least impactful)
    EPSILON_DECAY: float = 0.995  # 1. proven highest impact
    LEARNING_RATE: float = 5e-5  # 2. controls update magnitude
    TRAIN_STEPS_PER_EPISODE: int = 4  # 3. controls learning density
    TARGET_UPDATE_FREQ: int = 1000  # 4. stability of target

    # Sort of fixed hyperparameters
    GAMMA: float = 0.99
    EPSILON_END: float = 0.1
    EPSILON_START: float = 1.0
    BATCH_SIZE: int = 64
    BUFFER_CAPACITY: int = 50_000
    # GRAD_CLIP_NORM: float = 1.0

    # Threat rewards (own stones) - open = both ends free, half = one end blocked
    OPEN_TWO: float = 0.01  # 2 in a row, both ends open
    HALF_THREE: float = 0.015  # 3 in a row, one end open
    OPEN_THREE: float = 0.03  # 3 in a row, both ends open
    HALF_FOUR: float = 0.03  # 4 in a row, one end open
    OPEN_FOUR: float = 0.05  # 4 in a row, both ends open

    # Block rewards (opponent threats) - ordered by urgency
    BLOCK_THREE: float = 0.04  # block opponent's open 3
    BLOCK_FOUR: float = 0.10  # block opponent's 4 (open or half) - highest priority

    # Terminal rewards
    WIN_REWARD: float = 1.0
    LOSS_REWARD: float = -1.0
    DRAW_REWARD: float = 0.0

    # Paths
    MODEL_DIR: str = "./models"
    LOG_DIR: str = "./logs"


# Assertions
assert Config.BOARD_SIZE in (9, 15), "BOARD_SIZE must be 9 or 15"
assert Config.WIN_LENGTH == 5, "WIN_LENGTH must be 5"