# src/config.py
class Config:
    # Board settings
    BOARD_SIZE: int = 9
    WIN_LENGTH: int = 5

    # Training
    TOTAL_EPISODES: int = 5000
    SAVE_FREQ: int = 1000

    # DQN hyperparameters
    GAMMA: float = 0.99
    EPSILON_START: float = 1.0
    EPSILON_END: float = 0.1
    EPSILON_DECAY: float = 0.999
    LEARNING_RATE: float = 5e-5
    BATCH_SIZE: int = 64
    BUFFER_CAPACITY: int = 10000
    TARGET_UPDATE_FREQ: int = 10

    # Paths
    MODEL_DIR: str = "./models"
    LOG_DIR: str = "./logs"


# Assertions
assert Config.BOARD_SIZE in (9, 15), "BOARD_SIZE must be 9 or 15"
assert Config.WIN_LENGTH == 5, "WIN_LENGTH must be 5"
