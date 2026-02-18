# src/config.py
class Config:
    # Board settings
    BOARD_SIZE: int = 9
    WIN_LENGTH: int = 5

    # Number of training episodes
    TOTAL_EPISODES: int = 3000

    # How often to check the progress, and save the best rolling model
    SAVE_FREQ: int = 1000
    ROLLING_WINDOW_SIZE: int = 500

    # DQN hyperparameters
    GAMMA: float = 0.99
    EPSILON_START: float = 1.0
    EPSILON_END: float = 0.1
    EPSILON_DECAY: float = 0.999
    LEARNING_RATE: float = 5e-5
    BATCH_SIZE: int = 64
    BUFFER_CAPACITY: int = 100_000
    TARGET_UPDATE_FREQ: int = 1000

    # Training the network for multiple steps per episode can help it learn better long-term strategies, but it also makes training slower
    TRAIN_STEPS_PER_EPISODE: int = 4
    GRAD_CLIP_NORM: float = 1.0

    # Shaped rewards in order of how big they should be (very important for training)
    THREAT_REWARD_2: float = 0.01  # 2 in a row is it starting a threat
    THREAT_REWARD_3: float = 0.03  # 3 in a row is a stronger threat
    BLOCK_REWARD_3: float = 0.04  # prefer blocking a 3 in a row to threatening a 3 in a row
    THREAT_REWARD_4: float = 0.05  # prefer threatening a 4 in a row to blocking a 3 in a row
    BLOCK_REWARD_4: float = 0.10  # block 4 in a row or you lose

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
