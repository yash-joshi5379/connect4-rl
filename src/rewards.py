# src/rewards.py
from src.config import Config
from src.game import Color


def count_line(game, row, col, dr, dc, color):
    """Count consecutive stones of given color in one direction from (row, col), not including it."""
    count = 0
    r, c = row + dr, col + dc
    while 0 <= r < Config.BOARD_SIZE and 0 <= c < Config.BOARD_SIZE and game.board[r, c] == color:
        count += 1
        r += dr
        c += dc
    return count


def is_open_end(game, row, col, dr, dc):
    """Check if the cell just beyond the line end in direction (dr, dc) is empty."""
    r, c = row + dr, col + dc
    return (
        0 <= r < Config.BOARD_SIZE
        and 0 <= c < Config.BOARD_SIZE
        and game.board[r, c] == Color.EMPTY.value
    )


def get_threat_score(game, row, col, color):
    """
    Evaluate the threat created by the stone at (row, col) for the given color.
    Returns a reward based on line length and how many ends are open.
    """
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    best_score = 0.0

    for dr, dc in directions:
        fwd_count = count_line(game, row, col, dr, dc, color)
        bwd_count = count_line(game, row, col, -dr, -dc, color)
        length = 1 + fwd_count + bwd_count

        if length >= 5:
            # Win - handled by terminal reward, skip
            continue

        # Find the open ends: step past the line in each direction
        fwd_tip_r = row + (fwd_count + 1) * dr
        fwd_tip_c = col + (fwd_count + 1) * dc
        bwd_tip_r = row + (bwd_count + 1) * (-dr)
        bwd_tip_c = col + (bwd_count + 1) * (-dc)

        fwd_open = (
            0 <= fwd_tip_r < Config.BOARD_SIZE
            and 0 <= fwd_tip_c < Config.BOARD_SIZE
            and game.board[fwd_tip_r, fwd_tip_c] == Color.EMPTY.value
        )
        bwd_open = (
            0 <= bwd_tip_r < Config.BOARD_SIZE
            and 0 <= bwd_tip_c < Config.BOARD_SIZE
            and game.board[bwd_tip_r, bwd_tip_c] == Color.EMPTY.value
        )

        open_ends = int(fwd_open) + int(bwd_open)

        if open_ends == 0:
            # Dead - no reward, can't extend to a win
            score = 0.0
        elif length == 4:
            score = Config.OPEN_FOUR if open_ends == 2 else Config.HALF_FOUR
        elif length == 3:
            score = Config.OPEN_THREE if open_ends == 2 else Config.HALF_THREE
        elif length == 2:
            score = Config.OPEN_TWO if open_ends == 2 else 0.0
        else:
            score = 0.0

        best_score = max(best_score, score)

    return best_score


def get_block_score(game, row, col, opponent_color):
    """
    Evaluate how dangerous the opponent's line that was just blocked at (row, col) was.
    Temporarily treats the cell as the opponent's to measure what it would have been.
    """
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    best_score = 0.0

    for dr, dc in directions:
        fwd_count = count_line(game, row, col, dr, dc, opponent_color)
        bwd_count = count_line(game, row, col, -dr, -dc, opponent_color)
        length = 1 + fwd_count + bwd_count

        if length >= 4:
            best_score = max(best_score, Config.BLOCK_FOUR)
        elif length == 3:
            best_score = max(best_score, Config.BLOCK_THREE)

    return best_score


def calculate_shaped_reward(game, action, agent_color_val, opponent_color_val):
    """
    Shaped reward for placing a stone at action.
    Combines the threat created by the agent and the block applied to the opponent.
    Called after game.step(), so the stone is already on the board.
    """
    row, col = action
    threat_score = get_threat_score(game, row, col, agent_color_val)
    block_score = get_block_score(game, row, col, opponent_color_val)
    return threat_score + block_score
