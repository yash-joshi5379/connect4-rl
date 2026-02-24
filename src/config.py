# src/config.py
class Config:
    # Settings
    BOARD_SIZE: int = 9
    WIN_LENGTH: int = 5
    RANDOM_SEED: int = 42  # for reproducibility

    # Episodes
    RANDOM_EPISODES: int = 1000
    HEURISTIC_EPISODES: int = 5000
    CHECKPOINT_INTERVAL: int = 3000  # episodes between saving a new pool checkpoint
    SELFPLAY_EPISODES: int = CHECKPOINT_INTERVAL * 0
    TOTAL_EPISODES: int = RANDOM_EPISODES + HEURISTIC_EPISODES + SELFPLAY_EPISODES + 1

    # Self-play opponents
    OPPONENT_EPSILON: float = 0.02  # forced epsilon for all frozen pool opponents
    OLD_OPPONENT_CHANCE: float = 0.20  # probability of picking an old opponent (not N-1)

    # Printing/logging
    PRINT_FREQUENCY: int = 100
    OUTCOMES_MAXLEN: int = 100  # for tracking recent outcomes
    REWARDS_MAXLEN: int = 100  # for tracking recent rewards

    # Tunable hyperparameters (most -> least impactful)
    """optimal parameters found to be:
    - epsilon decay between 0.995 and 0.996
    - learning rate around 1e-4
    - target update frequency >2000
    - train steps per episode >4
    - replay buffer capacity >50k respectively"""
    EPSILON_DECAY: float = 0.995  # 1. proven highest impact
    LEARNING_RATE: float = 1e-4  # 2. controls update magnitude
    TARGET_UPDATE_FREQ: int = 2000  # 3. stability of target
    TRAIN_STEPS_PER_EPISODE: int = 4  # 4. controls learning density
    BUFFER_CAPACITY: int = 50_000  # CRUCIALLY MUST BE 50K EXACTLY

    # Sort of fixed hyperparameters
    GAMMA: float = 0.99
    EPSILON_END: float = 0.01
    EPSILON_START: float = 1.0
    BATCH_SIZE: int = 64
    # GRAD_CLIP_NORM: float = 1.0

    # Offensive - building your own lines
    OPEN_TWO: float = 0.01  # barely a nudge
    HALF_THREE: float = 0.02
    OPEN_THREE: float = 0.05
    HALF_FOUR: float = 0.15
    OPEN_FOUR: float = 0.3

    # Defensive - blocking opponent lines
    BLOCK_THREE: float = 0.05
    BLOCK_FOUR: float = 0.4  # higher than OPEN_FOUR — failing to block a 4 loses the game

    WIN_REWARD: float = 1.0
    LOSS_REWARD: float = -1.0
    DRAW_REWARD: float = 0.0

    STEP_PENALTY: float = 0.0

    # Paths
    MODEL_DIR: str = "./models"
    LOG_DIR: str = "./logs"


# Assertions
assert Config.BOARD_SIZE in (9, 15), "BOARD_SIZE must be 9 or 15"
assert Config.WIN_LENGTH == 5, "WIN_LENGTH must be 5"
assert Config.RANDOM_SEED == 42, "RANDOM_SEED should be 42 for reproducibility"