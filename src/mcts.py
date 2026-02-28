# src/mcts.py
import numpy as np
import math
from src.config import Config


class MCTSNode:
    def __init__(self, parent=None, prior=0.0):
        self.parent = parent
        self.prior = prior  # P(s, a) from network policy
        self.children = {}  # action_int -> MCTSNode
        self.visit_count = 0
        self.value_sum = 0.0

    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, parent_visits):
        exploration = Config.C_PUCT * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.q_value() + exploration

    def is_leaf(self):
        return len(self.children) == 0


class MCTS:
    def __init__(self, network):
        self.network = network

    def search(self, game, num_simulations=None, add_noise=True):
        """
        Run MCTS from the given game state.
        Returns a policy vector (81-dim) based on visit counts.
        """
        if num_simulations is None:
            num_simulations = Config.NUM_SIMULATIONS

        root = MCTSNode()

        # Expand root
        self._expand(root, game)

        # Add Dirichlet noise at root for exploration
        if add_noise and root.children:
            self._add_dirichlet_noise(root)

        # Run simulations
        for _ in range(num_simulations):
            node = root
            sim_game = game.clone()

            # SELECT: traverse tree using UCB
            while not node.is_leaf():
                action_int, node = self._select_child(node)
                action = sim_game.int_to_action(action_int)
                sim_game.step(action)

            # If game is over at this node, backpropagate the true result
            if sim_game.result.value != 0:  # not ONGOING
                value = self._game_result_to_value(sim_game, game.current_player)
                self._backpropagate(node, value)
                continue

            # EXPAND: add children using network prediction
            self._expand(node, sim_game)

            # EVALUATE: use network value estimate
            state = sim_game.get_state_for_network()
            _, value = self.network.predict(state)

            # Value is from perspective of current player in sim_game
            # We need it from perspective of the root player
            if sim_game.current_player != game.current_player:
                value = -value

            # BACKPROPAGATE
            self._backpropagate(node, value)

        # Build policy from visit counts
        return self._get_policy(root)

    def _expand(self, node, game):
        """Expand a leaf node by adding children for all legal moves."""
        legal_actions = game.get_legal_actions()
        if not legal_actions:
            return

        state = game.get_state_for_network()
        policy, _ = self.network.predict(state)

        # Mask illegal moves and renormalize
        legal_mask = np.zeros(Config.BOARD_SIZE * Config.BOARD_SIZE, dtype=np.float32)
        for row, col in legal_actions:
            action_int = game.action_to_int((row, col))
            legal_mask[action_int] = 1.0

        policy = policy * legal_mask
        policy_sum = policy.sum()
        if policy_sum > 0:
            policy = policy / policy_sum
        else:
            # Uniform over legal moves if network gives zero everywhere
            policy = legal_mask / legal_mask.sum()

        for row, col in legal_actions:
            action_int = game.action_to_int((row, col))
            node.children[action_int] = MCTSNode(parent=node, prior=policy[action_int])

    def _select_child(self, node):
        """Select child with highest UCB score."""
        best_score = -float("inf")
        best_action = None
        best_child = None

        for action_int, child in node.children.items():
            score = child.ucb_score(node.visit_count)
            if score > best_score:
                best_score = score
                best_action = action_int
                best_child = child

        return best_action, best_child

    def _backpropagate(self, node, value):
        """Backpropagate value up the tree, flipping sign at each level."""
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value  # opponent's perspective
            node = node.parent

    def _add_dirichlet_noise(self, node):
        """Add Dirichlet noise to root node priors for exploration."""
        actions = list(node.children.keys())
        noise = np.random.dirichlet([Config.DIRICHLET_ALPHA] * len(actions))

        for i, action_int in enumerate(actions):
            child = node.children[action_int]
            child.prior = (1 - Config.DIRICHLET_WEIGHT) * child.prior + Config.DIRICHLET_WEIGHT * noise[i]

    def _game_result_to_value(self, game, root_player):
        """Convert a terminal game result to a value from root player's perspective."""
        from src.game import GameResult, Color

        if game.result == GameResult.DRAW:
            return 0.0

        # Who won?
        if game.result == GameResult.BLACK_WIN:
            winner = Color.BLACK
        else:
            winner = Color.WHITE

        return 1.0 if winner == root_player else -1.0

    def _get_policy(self, root):
        """Convert root visit counts to a probability distribution."""
        policy = np.zeros(Config.BOARD_SIZE * Config.BOARD_SIZE, dtype=np.float32)

        for action_int, child in root.children.items():
            policy[action_int] = child.visit_count

        total = policy.sum()
        if total > 0:
            policy = policy / total

        return policy