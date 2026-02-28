# test_phase1.py
from src.game import Game
from src.network import PolicyValueNetwork
from src.mcts import MCTS

game = Game()
game.reset()

network = PolicyValueNetwork()
mcts = MCTS(network)

# Run MCTS from the opening position
policy = mcts.search(game, num_simulations=50)

print(f"Policy shape: {policy.shape}")
print(f"Policy sum: {policy.sum():.4f}")
print(f"Top 5 moves: {policy.argsort()[-5:][::-1]}")
print(f"Top 5 probs: {policy[policy.argsort()[-5:][::-1]]}")

# Verify network directly
state = game.get_state_for_network()
p, v = network.predict(state)
print(f"\nRaw network policy shape: {p.shape}")
print(f"Raw network value: {v:.4f}")