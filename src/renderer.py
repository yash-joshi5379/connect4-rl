import pygame
import numpy as np
from src.game import GomokuGame, Player, GameResult


class GomokuRenderer:
    def __init__(self, game, cell_size=40):
        self.game = game
        self.cell_size = cell_size
        self.margin = 40
        self.width = 15 * cell_size + 2 * self.margin
        self.height = 15 * cell_size + 2 * self.margin + 60

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Gomoku")

        self.bg_color = (220, 179, 92)
        self.line_color = (0, 0, 0)
        self.black_color = (0, 0, 0)
        self.white_color = (255, 255, 255)
        self.last_move_color = (255, 0, 0)

    def render(self):
        self.screen.fill(self.bg_color)
        self._draw_grid()
        self._draw_stones()
        self._draw_info()
        pygame.display.flip()

    def _draw_grid(self):
        for i in range(15):
            start_x = self.margin + i * self.cell_size
            start_y = self.margin
            end_y = self.margin + 14 * self.cell_size
            pygame.draw.line(self.screen, self.line_color, (start_x, start_y), (start_x, end_y), 1)

            start_y = self.margin + i * self.cell_size
            start_x = self.margin
            end_x = self.margin + 14 * self.cell_size
            pygame.draw.line(self.screen, self.line_color, (start_x, start_y), (end_x, start_y), 1)

    def _draw_stones(self):
        for row in range(15):
            for col in range(15):
                if self.game.board[row, col] != Player.EMPTY.value:
                    x = self.margin + col * self.cell_size
                    y = self.margin + row * self.cell_size
                    color = (
                        self.black_color
                        if self.game.board[row, col] == Player.BLACK.value
                        else self.white_color
                    )
                    pygame.draw.circle(self.screen, color, (x, y), self.cell_size // 2 - 2)

                    if self.game.last_move and (row, col) == self.game.last_move:
                        pygame.draw.circle(
                            self.screen,
                            self.last_move_color,
                            (x, y),
                            self.cell_size // 2 - 2,
                            3,
                        )

    def _draw_info(self):
        font = pygame.font.Font(None, 32)

        if self.game.result == GameResult.ONGOING:
            player_text = "Black" if self.game.current_player == Player.BLACK else "White"
            text = font.render(
                f"Current: {player_text} | Moves: {len(self.game.move_history)}",
                True,
                self.line_color,
            )
        elif self.game.result == GameResult.BLACK_WIN:
            text = font.render("Black Wins", True, self.line_color)
        elif self.game.result == GameResult.WHITE_WIN:
            text = font.render("White Wins", True, self.line_color)
        else:
            text = font.render("Draw", True, self.line_color)

        self.screen.blit(text, (self.margin, self.height - 50))

    def save_frame(self, filepath):
        pygame.image.save(self.screen, filepath)

    def close(self):
        pygame.quit()
