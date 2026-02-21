# src/heuristic.py
import random
from src.game import Game, Color, GameResult
from src.config import Config


def _find_best_move(game, color_val, min_length=3):
    """
    Find the best offensive move for the given color.
    Returns (action, length) for the longest extendable line, or (None, 0).
    Prioritises: win (5) > open four > half four > open three > half three
    """
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    best_action = None
    best_score = 0

    for row in range(Config.BOARD_SIZE):
        for col in range(Config.BOARD_SIZE):
            if game.board[row, col] != Color.EMPTY.value:
                continue

            for dr, dc in directions:
                # Count consecutive stones of color in both directions from this empty cell
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

                length = fwd + bwd  # stones already placed that this move connects

                if length < min_length:
                    continue

                # Playing here would make a line of length+1
                if length >= 4:
                    # Winning move
                    return (row, col), 100

                # Check open ends for prioritisation
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

                # Score: length * 10 + open_ends so open lines are preferred
                score = length * 10 + open_ends

                if score > best_score:
                    best_score = score
                    best_action = (row, col)

    return best_action, best_score


class HeuristicAgent:
    def __init__(self, block_rate=1.0, attack_rate=1.0):
        self.block_rate = block_rate
        self.attack_rate = attack_rate

    def select_action(self, game):
        my_color = game.current_player
        opp_color = Color.WHITE if my_color == Color.BLACK else Color.BLACK

        # 1. Always take a winning move
        win_move, win_score = _find_best_move(game, my_color.value, min_length=4)
        if win_move is not None and win_score >= 100:
            return win_move

        # 2. Always block opponent's winning move
        block_win, block_score = _find_best_move(game, opp_color.value, min_length=4)
        if block_win is not None and block_score >= 100:
            return block_win

        # 3. Block: block opponent's best line (with probability)
        block_move, block_score = _find_best_move(game, opp_color.value, min_length=2)
        # 4. Attack: extend own best line (with probability)
        attack_move, attack_score = _find_best_move(game, my_color.value, min_length=2)

        # Prioritise blocking if opponent's threat is more urgent
        if block_move is not None and block_score >= attack_score:
            if random.random() < self.block_rate:
                return block_move

        if attack_move is not None:
            if random.random() < self.attack_rate:
                return attack_move

        # Second chance to block if we didn't attack
        if block_move is not None:
            if random.random() < self.block_rate:
                return block_move

        # 5. Fallback: random
        return random.choice(game.get_legal_actions())