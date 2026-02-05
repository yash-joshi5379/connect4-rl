import numpy as np
from enum import Enum


class Player(Enum):
    EMPTY = 0
    BLACK = 1
    WHITE = 2


class GameResult(Enum):
    ONGOING = 0
    BLACK_WIN = 1
    WHITE_WIN = 2
    DRAW = 3


class GomokuGame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((15, 15), dtype=np.int8)
        self.current_player = Player.BLACK
        self.move_history = []
        self.result = GameResult.ONGOING
        self.last_move = None
        return self.get_board()

    def get_board(self):
        return self.board.copy()

    def get_legal_actions(self):
        if self.result != GameResult.ONGOING:
            return []
        return list(zip(*np.where(self.board == Player.EMPTY.value)))

    def step(self, action):
        row, col = action

        if not self._is_legal(row, col):
            return self.get_board(), self.result

        self.board[row, col] = self.current_player.value
        self.last_move = (row, col)
        self.move_history.append((self.current_player, action))

        if self._check_win(row, col):
            self.result = (
                GameResult.BLACK_WIN
                if self.current_player == Player.BLACK
                else GameResult.WHITE_WIN
            )
        elif len(self.get_legal_actions()) == 0:
            self.result = GameResult.DRAW

        if self.result == GameResult.ONGOING:
            self.current_player = (
                Player.WHITE if self.current_player == Player.BLACK else Player.BLACK
            )

        return self.get_board(), self.result

    def _is_legal(self, row, col):
        return (
            0 <= row < 15
            and 0 <= col < 15
            and self.board[row, col] == Player.EMPTY.value
            and self.result == GameResult.ONGOING
        )

    def _check_win(self, row, col):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        player = self.board[row, col]

        for dr, dc in directions:
            count = 1
            count += self._count_consecutive(row, col, dr, dc, player)
            count += self._count_consecutive(row, col, -dr, -dc, player)

            if count >= 5:
                return True
        return False

    def _count_consecutive(self, row, col, dr, dc, player):
        count = 0
        r, c = row + dr, col + dc
        while 0 <= r < 15 and 0 <= c < 15 and self.board[r, c] == player:
            count += 1
            r += dr
            c += dc
        return count

    def clone(self):
        game = GomokuGame()
        game.board = self.board.copy()
        game.current_player = self.current_player
        game.move_history = self.move_history.copy()
        game.result = self.result
        game.last_move = self.last_move
        return game

    def get_state_for_network(self):
        current = self.current_player.value
        opponent = Player.WHITE.value if current == Player.BLACK.value else Player.BLACK.value

        state = np.zeros((3, 15, 15), dtype=np.float32)
        state[0] = self.board == current
        state[1] = self.board == opponent
        state[2] = 1.0
        return state
