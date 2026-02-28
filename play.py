# play.py
import argparse
import os
import sys
import pygame
from src.game import Game, GameResult, Color
from src.network import PolicyValueNetwork
from src.mcts import MCTS
from src.config import Config
from src.renderer import Renderer
import torch


def load_agent(model_path):
    network = PolicyValueNetwork().to(Config.DEVICE)
    network.load_state_dict(torch.load(model_path, weights_only=True, map_location=Config.DEVICE))
    network.eval()
    return MCTS(network)


def play(model_path, human_color):
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        sys.exit(1)

    print(f"Loading: {model_path}")
    print(f"Device: {Config.DEVICE}")
    print(f"MCTS simulations: {Config.EVAL_SIMULATIONS}")

    ai = load_agent(model_path)

    game = Game()
    game.reset()
    renderer = Renderer(game)

    human_is_black = human_color == "black"
    human_col = Color.BLACK if human_is_black else Color.WHITE
    ai_col = Color.WHITE if human_is_black else Color.BLACK

    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

            # Human move on click
            if (
                event.type == pygame.MOUSEBUTTONDOWN
                and game.result == GameResult.ONGOING
                and game.current_player == human_col
            ):
                cell = renderer.pixel_to_cell(*event.pos)
                if cell is not None and game._is_legal(*cell):
                    game.step(cell)

        # AI move
        if game.result == GameResult.ONGOING and game.current_player == ai_col:
            renderer.render(status="AI thinking...")
            pygame.display.flip()

            # Use MCTS with higher simulation count for evaluation
            policy = ai.search(game, num_simulations=Config.EVAL_SIMULATIONS, add_noise=False)
            action_int = policy.argmax()
            action = game.int_to_action(action_int)
            game.step(action)

        # Render
        if game.result != GameResult.ONGOING:
            if game.result == GameResult.DRAW:
                status = "Draw! Click to exit."
            elif (game.result == GameResult.BLACK_WIN and human_is_black) or (
                game.result == GameResult.WHITE_WIN and not human_is_black
            ):
                status = "You win! Click to exit."
            else:
                status = "AI wins! Click to exit."

            renderer.render(status=status)

            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or event.type == pygame.MOUSEBUTTONDOWN:
                        waiting = False
                        running = False
            break
        else:
            your_turn = game.current_player == human_col
            status = "Your turn" if your_turn else None
            renderer.render(status=status)

        clock.tick(60)

    renderer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play against a Gomoku agent.")
    parser.add_argument(
        "--model",
        type=str,
        default=f"{Config.MODEL_DIR}/final.pth",
        help="Path to model file (default: models/final.pth)",
    )
    parser.add_argument(
        "--color",
        choices=["black", "white"],
        default="black",
        help="Your color (default: black, plays first).",
    )
    args = parser.parse_args()
    play(args.model, args.color)