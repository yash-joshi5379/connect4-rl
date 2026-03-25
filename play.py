"""
Play against a trained AlphaZero Gomoku model with pygame GUI.
Run: python play.py
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
import pygame

# ========== SET THESE ==========
MODEL_PATH = "output/model_final.pt"
NUM_SIMULATIONS = 800   # Max 800 simulations
# ===============================

BOARD_SIZE = 9
WIN_LENGTH = 5
NUM_RES_BLOCKS = 10
NUM_FILTERS = 128
C_PUCT = 1.5
ACTION_SIZE = BOARD_SIZE * BOARD_SIZE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Game
# ---------------------------------------------------------------------------

class GomokuGame:
    __slots__ = ["board", "current_player", "move_count", "last_move", "_winner"]

    def __init__(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.current_player = 1
        self.move_count = 0
        self.last_move = None
        self._winner = 0

    def clone(self):
        g = GomokuGame()
        g.board = self.board.copy()
        g.current_player = self.current_player
        g.move_count = self.move_count
        g.last_move = self.last_move
        g._winner = self._winner
        return g

    def legal_moves(self):
        return np.flatnonzero(self.board.ravel() == 0)

    def play(self, action):
        r, c = divmod(action, BOARD_SIZE)
        assert self.board[r, c] == 0, f"Illegal move at ({r},{c})"
        self.board[r, c] = self.current_player
        self.last_move = action
        self.move_count += 1
        self._winner = self._check_winner_at(r, c)
        self.current_player *= -1

    def _check_winner_at(self, r, c):
        player = self.board[r, c]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for sign in [1, -1]:
                nr, nc = r + sign * dr, c + sign * dc
                while (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE
                       and self.board[nr, nc] == player):
                    count += 1
                    nr += sign * dr
                    nc += sign * dc
            if count >= WIN_LENGTH:
                return player
        return 0

    def winner(self):
        return self._winner

    def is_terminal(self):
        return self._winner != 0 or self.move_count == ACTION_SIZE

    def terminal_value(self):
        w = self._winner
        if w == self.current_player:
            return 1.0
        elif w == -self.current_player:
            return -1.0
        return 0.0

    def get_state_tensor(self):
        state = np.zeros((3, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        state[0] = (self.board == self.current_player)
        state[1] = (self.board == -self.current_player)
        state[2] = 1.0 if self.current_player == 1 else 0.0
        return state


# ---------------------------------------------------------------------------
# Neural network (must match train.py exactly)
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)


class AlphaZeroNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(3, NUM_FILTERS, 3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(NUM_FILTERS)
        self.res_blocks = nn.Sequential(*[ResBlock(NUM_FILTERS) for _ in range(NUM_RES_BLOCKS)])

        self.policy_conv = nn.Conv2d(NUM_FILTERS, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, ACTION_SIZE)

        self.value_conv = nn.Conv2d(NUM_FILTERS, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(BOARD_SIZE * BOARD_SIZE, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.res_blocks(x)

        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        return p, v


# ---------------------------------------------------------------------------
# Inference + MCTS (no Dirichlet noise for play)
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(net, state_np):
    state_tensor = torch.from_numpy(state_np).unsqueeze(0).to(DEVICE)
    net.eval()
    with autocast('cuda'):
        logits, value = net(state_tensor)
    logits = logits.squeeze(0).float().cpu().numpy()
    value = value.item()
    return logits, value


def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


class MCTSNode:
    __slots__ = ["visit_count", "value_sum", "prior", "children"]

    def __init__(self, prior):
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.children = {}

    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def select_child(node):
    total_visits = sum(c.visit_count for c in node.children.values())
    sqrt_total = math.sqrt(total_visits + 1)
    best_score = -float("inf")
    best_action = None
    best_child = None
    for action, child in node.children.items():
        ucb = child.value() + C_PUCT * child.prior * sqrt_total / (1 + child.visit_count)
        if ucb > best_score:
            best_score = ucb
            best_action = action
            best_child = child
    return best_action, best_child


def mcts_search(game, net, num_simulations):
    root = MCTSNode(prior=0.0)

    state = game.get_state_tensor()
    logits, _ = predict(net, state)
    legal = game.legal_moves()

    mask = np.full(ACTION_SIZE, -1e9)
    mask[legal] = logits[legal]
    probs = _softmax(mask)

    for action in legal:
        root.children[action] = MCTSNode(prior=probs[action])

    for _ in range(num_simulations):
        node = root
        sim_game = game.clone()
        search_path = [node]

        while node.children:
            action, node = select_child(node)
            sim_game.play(action)
            search_path.append(node)

        if sim_game.is_terminal():
            value = -sim_game.terminal_value()
        else:
            state = sim_game.get_state_tensor()
            logits, value = predict(net, state)
            legal = sim_game.legal_moves()

            mask = np.full(ACTION_SIZE, -1e9)
            mask[legal] = logits[legal]
            probs = _softmax(mask)

            for a in legal:
                node.children[a] = MCTSNode(prior=probs[a])

            value = -value

        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            value = -value

    visits = np.zeros(ACTION_SIZE, dtype=np.float32)
    for action, child in root.children.items():
        visits[action] = child.visit_count
    return visits


# ---------------------------------------------------------------------------
# Pygame GUI renderer
# ---------------------------------------------------------------------------

class Renderer:
    def __init__(self):
        self.cell_size = 50
        self.margin = 40
        self.extra_height = 60
        self.stone_radius = self.cell_size // 2 - 3

        self.width = (BOARD_SIZE - 1) * self.cell_size + 2 * self.margin
        self.height = (BOARD_SIZE - 1) * self.cell_size + 2 * self.margin + self.extra_height

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("AlphaZero Gomoku")
        self.font = pygame.font.Font(None, 32)
        self.small_font = pygame.font.Font(None, 24)

        self.bg_color = (220, 179, 92)
        self.line_color = (0, 0, 0)
        self.black_color = (30, 30, 30)
        self.white_color = (240, 240, 240)
        self.last_move_color = (220, 50, 50)
        self.text_color = (40, 40, 40)

    def pixel_to_cell(self, x, y):
        col = round((x - self.margin) / self.cell_size)
        row = round((y - self.margin) / self.cell_size)
        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
            return row, col
        return None

    def render(self, game, status=None):
        self.screen.fill(self.bg_color)

        # Grid lines
        for i in range(BOARD_SIZE):
            x = self.margin + i * self.cell_size
            y_start = self.margin
            y_end = self.margin + (BOARD_SIZE - 1) * self.cell_size
            pygame.draw.line(self.screen, self.line_color, (x, y_start), (x, y_end), 1)

            y = self.margin + i * self.cell_size
            x_start = self.margin
            x_end = self.margin + (BOARD_SIZE - 1) * self.cell_size
            pygame.draw.line(self.screen, self.line_color, (x_start, y), (x_end, y), 1)

        # Coordinate labels
        for i in range(BOARD_SIZE):
            label = self.small_font.render(str(i), True, self.text_color)
            lx = self.margin + i * self.cell_size - label.get_width() // 2
            self.screen.blit(label, (lx, self.margin - 25))

            label = self.small_font.render(str(i), True, self.text_color)
            ly = self.margin + i * self.cell_size - label.get_height() // 2
            self.screen.blit(label, (self.margin - 25, ly))

        # Stones
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                piece = game.board[r, c]
                if piece != 0:
                    x = self.margin + c * self.cell_size
                    y = self.margin + r * self.cell_size
                    color = self.black_color if piece == 1 else self.white_color
                    pygame.draw.circle(self.screen, color, (x, y), self.stone_radius)
                    if piece == -1:
                        pygame.draw.circle(self.screen, (150, 150, 150), (x, y), self.stone_radius, 2)

        # Last move indicator
        if game.last_move is not None:
            lr, lc = divmod(game.last_move, BOARD_SIZE)
            lx = self.margin + lc * self.cell_size
            ly = self.margin + lr * self.cell_size
            pygame.draw.circle(self.screen, self.last_move_color, (lx, ly), self.stone_radius, 3)

        # Status bar
        if status:
            text = self.font.render(status, True, self.text_color)
        elif game.is_terminal():
            w = game.winner()
            if w == 1:
                text = self.font.render("Black (X) wins!", True, self.text_color)
            elif w == -1:
                text = self.font.render("White (O) wins!", True, self.text_color)
            else:
                text = self.font.render("Draw!", True, self.text_color)
        else:
            player_name = "Black (X)" if game.current_player == 1 else "White (O)"
            text = self.font.render(f"{player_name} to move  |  Move {game.move_count + 1}", True, self.text_color)

        self.screen.blit(text, (self.margin, self.height - 45))
        pygame.display.flip()

    def close(self):
        pygame.quit()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading model: {MODEL_PATH}")
    net = AlphaZeroNet().to(DEVICE)
    net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    net.eval()
    print(f"Device: {DEVICE}")
    print(f"MCTS sims: {NUM_SIMULATIONS}")

    renderer = Renderer()
    game = GomokuGame()

    # Choose side
    renderer.render(game, status="Click LEFT = play Black  |  Click RIGHT = play White")
    print("Click left half of window to play Black (first)")
    print("Click right half of window to play White (second)")

    human_player = None
    while human_player is None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                renderer.close()
                return
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.pos[0] < renderer.width // 2:
                    human_player = 1
                else:
                    human_player = -1

    ai_player = -human_player
    human_name = "Black (X)" if human_player == 1 else "White (O)"
    ai_name = "Black (X)" if ai_player == 1 else "White (O)"
    print(f"You are {human_name}. AI is {ai_name}.")

    renderer.render(game)

    while not game.is_terminal():
        if game.current_player == human_player:
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        renderer.close()
                        return
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        cell = renderer.pixel_to_cell(*event.pos)
                        if cell is not None:
                            r, c = cell
                            if game.board[r, c] == 0:
                                action = r * BOARD_SIZE + c
                                print(f"Human plays: ({r},{c})")
                                game.play(action)
                                renderer.render(game)
                                waiting = False
                pygame.time.wait(30)
        else:
            renderer.render(game, status=f"AI thinking... ({NUM_SIMULATIONS} sims)")
            pygame.event.pump()

            visits = mcts_search(game, net, NUM_SIMULATIONS)
            action = int(np.argmax(visits))
            r, c = divmod(action, BOARD_SIZE)

            top3 = np.argsort(visits)[-3:][::-1]
            top_str = ", ".join(
                f"({a // BOARD_SIZE},{a % BOARD_SIZE}): {int(visits[a])}"
                for a in top3
            )
            print(f"AI plays: ({r},{c})  [top: {top_str}]")

            game.play(action)
            renderer.render(game)

    # Game over
    renderer.render(game)
    w = game.winner()
    if w == human_player:
        print("You win!")
    elif w == ai_player:
        print("AI wins!")
    else:
        print("Draw!")

    # Wait for close
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False
            if event.type == pygame.KEYDOWN:
                waiting = False
        pygame.time.wait(30)

    renderer.close()


if __name__ == "__main__":
    main()