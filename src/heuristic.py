# src/heuristic.py
import random
from src.game import Color
from src.config import Config


def _scan_threats(game, color_val):
    """
    Scan the board for all empty cells that extend a line of the given color.
    Returns a list of (row, col, length, open_ends).
    length = number of existing stones the move connects (placing here makes length+1).
    """
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    threats = []

    for row in range(Config.BOARD_SIZE):
        for col in range(Config.BOARD_SIZE):
            if game.board[row, col] != Color.EMPTY.value:
                continue

            for dr, dc in directions:
                fwd = 0
                r, c = row + dr, col + dc
                while (
                    0 <= r < Config.BOARD_SIZE
                    and 0 <= c < Config.BOARD_SIZE
                    and game.board[r, c] == color_val
                ):
                    fwd += 1
                    r += dr
                    c += dc

                bwd = 0
                r, c = row - dr, col - dc
                while (
                    0 <= r < Config.BOARD_SIZE
                    and 0 <= c < Config.BOARD_SIZE
                    and game.board[r, c] == color_val
                ):
                    bwd += 1
                    r -= dr
                    c -= dc

                length = fwd + bwd
                if length < 1:
                    continue

                fwd_tip_r = row + (fwd + 1) * dr
                fwd_tip_c = col + (fwd + 1) * dc
                bwd_tip_r = row - (bwd + 1) * dr
                bwd_tip_c = col - (bwd + 1) * dc

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
                threats.append((row, col, length, open_ends))

    return threats


def _best_of(threats, min_length, min_open=0):
    """Pick the best threat matching criteria. Prefers longer lines, then more open ends."""
    candidates = [t for t in threats if t[2] >= min_length and t[3] >= min_open]
    if not candidates:
        return None
    candidates.sort(key=lambda t: (t[2], t[3]), reverse=True)
    return (candidates[0][0], candidates[0][1])


class HeuristicAgent:
    def __init__(
        self,
        win_rate=0.9,
        block_win_rate=0.3,
        make_open4_rate=0.8,
        block_open3_rate=0.3,
        make_half4_rate=0.6,
        block_half3_rate=0.2,
        attack_2_rate=0.5,
        attack_1_rate=0.3,
    ):
        self.win_rate = win_rate  # finish own 5 (length >= 4)
        self.block_win_rate = block_win_rate  # block opponent's 5 (length >= 4)
        self.make_open4_rate = make_open4_rate  # extend open 3 -> open 4
        self.block_open3_rate = block_open3_rate  # block opponent's open 3
        self.make_half4_rate = make_half4_rate  # extend half 3 -> half 4
        self.block_half3_rate = block_half3_rate  # block opponent's half 3
        self.attack_2_rate = attack_2_rate  # extend own 2 -> 3
        self.attack_1_rate = attack_1_rate  # extend own 1 -> 2

    def select_action(self, game):
        my_color = game.current_player
        opp_color = Color.WHITE if my_color == Color.BLACK else Color.BLACK

        my_threats = _scan_threats(game, my_color.value)
        opp_threats = _scan_threats(game, opp_color.value)

        # 1. Win: complete 5 in a row
        if random.random() < self.win_rate:
            move = _best_of(my_threats, min_length=4)
            if move is not None:
                return move

        # 2. Block win: stop opponent completing 5
        if random.random() < self.block_win_rate:
            move = _best_of(opp_threats, min_length=4)
            if move is not None:
                return move

        # 3. Make open 4: extend open 3 (open_ends=2)
        if random.random() < self.make_open4_rate:
            move = _best_of(my_threats, min_length=3, min_open=2)
            if move is not None:
                return move

        # 4. Block open 3: stop opponent's open 3 becoming open 4
        if random.random() < self.block_open3_rate:
            move = _best_of(opp_threats, min_length=3, min_open=2)
            if move is not None:
                return move

        # 5. Make half 4: extend half 3 (open_ends >= 1)
        if random.random() < self.make_half4_rate:
            move = _best_of(my_threats, min_length=3, min_open=1)
            if move is not None:
                return move

        # 6. Block half 3: stop opponent's half 3
        if random.random() < self.block_half3_rate:
            move = _best_of(opp_threats, min_length=3, min_open=1)
            if move is not None:
                return move

        # 7. Extend own 2s: build lines
        if random.random() < self.attack_2_rate:
            move = _best_of(my_threats, min_length=2, min_open=1)
            if move is not None:
                return move

        # 8. Extend own 1s: start lines
        if random.random() < self.attack_1_rate:
            move = _best_of(my_threats, min_length=1, min_open=1)
            if move is not None:
                return move

        # 9. Fallback: random
        return random.choice(game.get_legal_actions())
