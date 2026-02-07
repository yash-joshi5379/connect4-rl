import random
import pygame
from src.game import GomokuGame, GameResult
from src.logger import GameLogger
from src.renderer import GomokuRenderer


def play_random_game():
    game = GomokuGame()
    renderer = GomokuRenderer(game)
    logger = GameLogger("logs")

    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if game.result == GameResult.ONGOING:
            legal_actions = game.get_legal_actions()
            if legal_actions:
                action = random.choice(legal_actions)
                game.step(action)

        renderer.render()
        clock.tick(5)  # 5 moves per second

        if game.result != GameResult.ONGOING:
            pygame.time.wait(2000)  # Wait 2 seconds to show result
            running = False

    logger.log_game(game.result, len(game.move_history))
    logger.save()

    print(f"Game finished: {game.result.name}")
    print(f"Total moves: {len(game.move_history)}")

    renderer.close()


if __name__ == "__main__":
    play_random_game()
