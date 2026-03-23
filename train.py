"""
AlphaZero for 9x9 Gomoku (Five-in-a-Row)
Based on: Silver et al., "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm" (2017)

Batched parallel MCTS self-play for GPU efficiency.
Single-file implementation for RTX 4090.
Run: pip install torch numpy tqdm → python train.py
"""

import os
import sys
import math
import random
import logging
from collections import deque
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import trange

# ========== TUNE THESE ==========
# NUM_SIMULATIONS = 400  # 200 -> 400
# GAMES_PER_ITERATION = 50  # 15 -> 50
# NUM_ITERATIONS = 200  # 100 -> 200
# LEARNING_RATE = 0.05  # 0.2 -> 0.01 -> 0.05 (0.05 is logarithmic middle so good for the third try)
# REPLAY_BUFFER_SIZE = 1_000_000  # 200_000 -> 1_000_000

# still didnt work 
# NUM_SIMULATIONS = 200
# GAMES_PER_ITERATION = 15
# NUM_ITERATIONS = 100
# LEARNING_RATE = 0.05
# REPLAY_BUFFER_SIZE = 200_000

# SUCCESS 1
# NUM_SIMULATIONS = 400
# GAMES_PER_ITERATION = 20
# NUM_ITERATIONS = 150
# LEARNING_RATE = 0.05
# REPLAY_BUFFER_SIZE = 500_000
# TRAINING_EPOCHS_PER_ITERATION = 20  # 10 -> 20

# POTENTIAL SUCCESS 2 (18 hours)
# NUM_SIMULATIONS = 400
# GAMES_PER_ITERATION = 30
# NUM_ITERATIONS = 180
# LEARNING_RATE = 0.05
# REPLAY_BUFFER_SIZE = 1_000_000
# TRAINING_EPOCHS_PER_ITERATION = 20

# POTENTIAL SUCCESS 2 (40? hours)
NUM_SIMULATIONS = 800
GAMES_PER_ITERATION = 40
NUM_ITERATIONS = 200
LEARNING_RATE = 0.05
REPLAY_BUFFER_SIZE = 1_500_000
TRAINING_EPOCHS_PER_ITERATION = 20

# =================================

# ========== PROBABLY DON'T TOUCH ===========================
BOARD_SIZE = 9
WIN_LENGTH = 5
NUM_RES_BLOCKS = 10
NUM_FILTERS = 128
C_PUCT = 1.5
DIRICHLET_ALPHA = 0.15
DIRICHLET_EPSILON = 0.25
TEMPERATURE_THRESHOLD = 10
BATCH_SIZE = 256
LR_DROP_STEPS = [int(NUM_ITERATIONS * 0.5), int(NUM_ITERATIONS * 0.75)]
WEIGHT_DECAY = 1e-4
EVAL_INTERVAL = 10
EVAL_GAMES = 20
BOARD_DISPLAY_INTERVAL = 10
CHECKPOINT_INTERVAL = 10
OUTPUT_DIR = "output"
# ===========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACTION_SIZE = BOARD_SIZE * BOARD_SIZE

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
 
def setup_logging():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_path = os.path.join(OUTPUT_DIR, "training_log.txt")
    logger = logging.getLogger("alphazero")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger
 
log = setup_logging()

# ---------------------------------------------------------------------------
# Game
# ---------------------------------------------------------------------------
 
class GomokuGame:
    __slots__ = ["board", "current_player", "move_count", "last_move", "_winner"]
 
    def __init__(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.current_player = 1  # 1 = black, -1 = white
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
 
    def display(self):
        symbols = {0: ".", 1: "X", -1: "O"}
        lines = []
        for r in range(BOARD_SIZE):
            row = " ".join(symbols[self.board[r, c]] for c in range(BOARD_SIZE))
            lines.append(row)
        return "\n".join(lines)
 
 
# ---------------------------------------------------------------------------
# Data augmentation (8-fold symmetry)
# ---------------------------------------------------------------------------
 
def augment(state, policy_flat):
    policy_grid = policy_flat.reshape(BOARD_SIZE, BOARD_SIZE)
    augmented = []
    for k in range(4):
        s_rot = np.rot90(state, k, axes=(1, 2)).copy()
        p_rot = np.rot90(policy_grid, k).copy()
        augmented.append((s_rot, p_rot.ravel()))
        s_flip = np.flip(s_rot, axis=2).copy()
        p_flip = np.flip(p_rot, axis=1).copy()
        augmented.append((s_flip, p_flip.ravel()))
    return augmented
 
 
# ---------------------------------------------------------------------------
# Neural network
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
 
 
@torch.no_grad()
def predict_batch(net, states_np):
    """Run network on a batch of states. Returns (logits_batch, values_batch) as numpy."""
    states_tensor = torch.from_numpy(np.stack(states_np)).to(DEVICE)
    net.eval()
    with autocast('cuda'):
        logits, values = net(states_tensor)
    logits = logits.float().cpu().numpy()
    values = values.squeeze(1).float().cpu().numpy()
    return logits, values
 
 
@torch.no_grad()
def predict_single(net, state_np):
    """Run network on a single state. Returns (logits, value_scalar)."""
    logits, values = predict_batch(net, [state_np])
    return logits[0], values[0]
 
 
# ---------------------------------------------------------------------------
# MCTS (batched across multiple games)
# ---------------------------------------------------------------------------
 
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
 
 
def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()
 
 
def _check_immediate_threats(game):
    """
    Check for immediate win/loss threats. Returns:
      - (forced_move,) if there's a winning move, must-block, or open-three block
      - None if no immediate threat
    
    Priority:
      1. Current player can win in 1 move → play it
      2. Opponent can win in 1 move → block it
      3. Opponent has an open three (consecutive or broken, both ends empty) → block it
    
    This is used during training MCTS to produce sharp policy targets
    so the network learns to block/win. Uses game rules only (no heuristics).
    """
    board = game.board
    current = game.current_player
    opponent = -current
    legal = game.legal_moves()
    
    # 1. Check if current player can win immediately
    for action in legal:
        r, c = divmod(action, BOARD_SIZE)
        board[r, c] = current
        if game._check_winner_at(r, c):
            board[r, c] = 0
            return (action,)
        board[r, c] = 0
    
    # 2. Check if opponent can win immediately (must block)
    for action in legal:
        r, c = divmod(action, BOARD_SIZE)
        board[r, c] = opponent
        if game._check_winner_at(r, c):
            board[r, c] = 0
            return (action,)
        board[r, c] = 0
    
    # 3. Check if opponent has an open three → block at either end
    # Only fires 50% of the time so that some games develop threats on sparse boards,
    # producing training data for early-game blocking positions.
    # Parts 1 & 2 (immediate win/block) ALWAYS fire — those are forced.
    if random.random() > 0.5:
        return None
    # Only scan from opponent stones (skip empty/current), and for each stone
    # only check windows that START with empty (the pattern begins with .)
    # This avoids scanning all 81 cells × 4 directions.
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] != opponent:
                continue
            for dr, dc in directions:
                # Check if this stone could be position 1 in a .OOO. / .O.OO. / .OO.O. pattern
                # That means (r-dr, c-dc) must be empty and in bounds
                pr, pc = r - dr, c - dc
                if not (0 <= pr < BOARD_SIZE and 0 <= pc < BOARD_SIZE and board[pr, pc] == 0):
                    continue
                
                # Build 6-cell window starting from (pr, pc)
                window = []
                for i in range(6):
                    nr, nc = pr + i * dr, pc + i * dc
                    if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                        window.append(board[nr, nc])
                    else:
                        window.append(9)
                
                # Consecutive open three: . OOO .
                if window[:5] == [0, opponent, opponent, opponent, 0]:
                    end1_r, end1_c = pr, pc
                    end2_r, end2_c = pr + 4 * dr, pc + 4 * dc
                    dist1 = abs(end1_r - 4) + abs(end1_c - 4)
                    dist2 = abs(end2_r - 4) + abs(end2_c - 4)
                    if dist1 <= dist2:
                        return (end1_r * BOARD_SIZE + end1_c,)
                    else:
                        return (end2_r * BOARD_SIZE + end2_c,)
                
                # Broken open three A: . O . OO .
                if window == [0, opponent, 0, opponent, opponent, 0]:
                    gap_r, gap_c = pr + 2 * dr, pc + 2 * dc
                    return (gap_r * BOARD_SIZE + gap_c,)
                
                # Broken open three B: . OO . O .
                if window == [0, opponent, opponent, 0, opponent, 0]:
                    gap_r, gap_c = pr + 3 * dr, pc + 3 * dc
                    return (gap_r * BOARD_SIZE + gap_c,)
    
    return None
 
 
def _select_child(node):
    """Pick child with highest UCB score."""
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
 
 
def _expand_node(node, logits, legal_moves, add_noise=False, game=None):
    """Create children for a node given network logits and legal moves.
    If game is provided, checks for immediate win/block threats and
    overrides the network prior to concentrate on the forced move."""
    mask = np.full(ACTION_SIZE, -1e9)
    mask[legal_moves] = logits[legal_moves]
    probs = _softmax(mask)
 
    # Override priors if there's an immediate threat
    threat = None
    if game is not None:
        threat = _check_immediate_threats(game)
    
    if threat is not None:
        forced_action = threat[0]
        # Put 90% prior on the forced move, spread 10% on the rest
        for action in legal_moves:
            if action == forced_action:
                probs[action] = 0.9
            else:
                probs[action] = 0.1 / (len(legal_moves) - 1) if len(legal_moves) > 1 else 0.0
 
    if add_noise:
        noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(legal_moves))
        for i, action in enumerate(legal_moves):
            noisy_prior = (1 - DIRICHLET_EPSILON) * probs[action] + DIRICHLET_EPSILON * noise[i]
            node.children[action] = MCTSNode(prior=noisy_prior)
    else:
        for action in legal_moves:
            node.children[action] = MCTSNode(prior=probs[action])
 
 
def _backpropagate(search_path, value):
    """Backprop value up the search path, alternating signs."""
    for node in reversed(search_path):
        node.visit_count += 1
        node.value_sum += value
        value = -value
 
 
def batched_mcts_search(games, net, num_simulations):
    """
    Run MCTS for multiple games in lockstep.
 
    Each simulation step:
      1) SELECT down each tree to a leaf
      2) Collect all leaf states into one batch
      3) One batched GPU inference
      4) EXPAND + BACKPROPAGATE for each tree
 
    Returns list of visit-count arrays, one per game.
    """
    n = len(games)
    roots = [MCTSNode(prior=0.0) for _ in range(n)]
 
    # --- Expand all roots with one batched inference ---
    root_states = [g.get_state_tensor() for g in games]
    root_logits, _ = predict_batch(net, root_states)
 
    for i in range(n):
        legal = games[i].legal_moves()
        _expand_node(roots[i], root_logits[i], legal, add_noise=True, game=games[i])
 
    # --- Run simulations in lockstep ---
    for _ in range(num_simulations):
        sim_games = []
        search_paths = []
        leaf_terminals = []
        leaf_states = []
 
        for i in range(n):
            node = roots[i]
            sim_game = games[i].clone()
            path = [node]
 
            while node.children:
                action, node = _select_child(node)
                sim_game.play(action)
                path.append(node)
 
            sim_games.append(sim_game)
            search_paths.append(path)
 
            if sim_game.is_terminal():
                leaf_terminals.append(True)
            else:
                leaf_terminals.append(False)
                leaf_states.append(sim_game.get_state_tensor())
 
        # Batch inference for all non-terminal leaves
        if leaf_states:
            batch_logits, batch_values = predict_batch(net, leaf_states)
        else:
            batch_logits, batch_values = np.empty((0, ACTION_SIZE)), np.empty(0)
 
        # Expand + backprop
        batch_idx = 0
        for i in range(n):
            if leaf_terminals[i]:
                value = -sim_games[i].terminal_value()
            else:
                logits = batch_logits[batch_idx]
                value = batch_values[batch_idx]
                batch_idx += 1
 
                legal = sim_games[i].legal_moves()
                leaf_node = search_paths[i][-1]
                _expand_node(leaf_node, logits, legal, add_noise=False, game=sim_games[i])
 
                value = -value
 
            _backpropagate(search_paths[i], value)
 
    # Extract visit distributions
    visit_arrays = []
    for i in range(n):
        visits = np.zeros(ACTION_SIZE, dtype=np.float32)
        for action, child in roots[i].children.items():
            visits[action] = child.visit_count
        visit_arrays.append(visits)
 
    return visit_arrays
 
 
# ---------------------------------------------------------------------------
# Self-play (batched)
# ---------------------------------------------------------------------------
 
def play_games_batched(net, num_games):
    """
    Play num_games via batched MCTS self-play.
    All games run in parallel until each finishes.
    Returns list of training examples and list of finished games.
    """
    active_games = [GomokuGame() for _ in range(num_games)]
    histories = [[] for _ in range(num_games)]
 
    finished_examples = []
    finished_games = []
 
    while active_games:
        visit_arrays = batched_mcts_search(active_games, net, NUM_SIMULATIONS)
 
        newly_finished = []
        for i in range(len(active_games)):
            game = active_games[i]
            visits = visit_arrays[i]
 
            if game.move_count < TEMPERATURE_THRESHOLD:
                pi = visits / visits.sum()
            else:
                pi = np.zeros(ACTION_SIZE, dtype=np.float32)
                pi[np.argmax(visits)] = 1.0
 
            histories[i].append((game.get_state_tensor(), pi, game.current_player))
 
            if game.move_count < TEMPERATURE_THRESHOLD:
                action = np.random.choice(ACTION_SIZE, p=pi)
            else:
                action = int(np.argmax(pi))
 
            game.play(action)
 
            if game.is_terminal():
                newly_finished.append(i)
 
        for i in sorted(newly_finished, reverse=True):
            game = active_games[i]
            history = histories[i]
            winner = game.winner()
 
            examples = []
            for state, pi, player in history:
                if winner == 0:
                    z = 0.0
                elif winner == player:
                    z = 1.0
                else:
                    z = -1.0
                for aug_state, aug_pi in augment(state, pi):
                    examples.append((
                        torch.from_numpy(aug_state),
                        torch.from_numpy(aug_pi),
                        z,
                    ))
 
            finished_examples.extend(examples)
            finished_games.append(game)
 
            active_games.pop(i)
            histories.pop(i)
 
    return finished_examples, finished_games
 
 
# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
 
def train_network(net, optimizer, scaler, replay_buffer):
    if len(replay_buffer) < BATCH_SIZE:
        return 0.0, 0.0, 0.0
 
    net.train()
    total_loss = 0.0
    total_v_loss = 0.0
    total_p_loss = 0.0
 
    for _ in range(TRAINING_EPOCHS_PER_ITERATION):
        indices = np.random.choice(len(replay_buffer), BATCH_SIZE, replace=False)
        batch = [replay_buffer[i] for i in indices]
 
        states = torch.stack([b[0] for b in batch]).to(DEVICE)
        target_pis = torch.stack([b[1] for b in batch]).to(DEVICE)
        target_zs = torch.tensor([b[2] for b in batch], dtype=torch.float32).to(DEVICE)
 
        with autocast('cuda'):
            logits, values = net(states)
            values = values.squeeze(1)
            v_loss = F.mse_loss(values, target_zs)
            log_probs = F.log_softmax(logits, dim=1)
            p_loss = -torch.sum(target_pis * log_probs, dim=1).mean()
            loss = v_loss + p_loss
 
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
 
        total_loss += loss.item()
        total_v_loss += v_loss.item()
        total_p_loss += p_loss.item()
 
    n = TRAINING_EPOCHS_PER_ITERATION
    return total_loss / n, total_v_loss / n, total_p_loss / n
 
 
# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
 
def eval_against_random(net, num_games):
    wins = 0
    for i in range(num_games):
        game = GomokuGame()
        az_player = 1 if i % 2 == 0 else -1
 
        while not game.is_terminal():
            if game.current_player == az_player:
                visits = batched_mcts_search([game], net, num_simulations=50)[0]
                action = int(np.argmax(visits))
            else:
                legal = game.legal_moves()
                action = np.random.choice(legal)
            game.play(action)
 
        if game.winner() == az_player:
            wins += 1
 
    return wins / num_games
 
 
def print_empty_board_policy(net):
    game = GomokuGame()
    state = game.get_state_tensor()
    logits, value = predict_single(net, state)
    probs = _softmax(logits)
    top5 = np.argsort(probs)[-5:][::-1]
    moves_str = ", ".join(
        f"({a // BOARD_SIZE},{a % BOARD_SIZE})={probs[a]:.3f}" for a in top5
    )
    log.info(f"  Empty board top 5: {moves_str}  value={value:.3f}")
 
 
# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
 
def main():
    log.info("=" * 60)
    log.info("AlphaZero Gomoku Training (Batched MCTS)")
    log.info(f"Board: {BOARD_SIZE}x{BOARD_SIZE}, Win length: {WIN_LENGTH}")
    log.info(f"Network: {NUM_RES_BLOCKS} res blocks, {NUM_FILTERS} filters")
    log.info(f"MCTS sims: {NUM_SIMULATIONS}")
    log.info(f"Iterations: {NUM_ITERATIONS}, Games/iter: {GAMES_PER_ITERATION}")
    log.info(f"Device: {DEVICE}")
    log.info("=" * 60)
 
    net = AlphaZeroNet().to(DEVICE)
    param_count = sum(p.numel() for p in net.parameters())
    log.info(f"Parameters: {param_count:,}")
 
    # SGD with momentum — from the paper (Section: Configuration)
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=LEARNING_RATE,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY,
    )
    scaler = GradScaler('cuda')
    replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
 
    lr_drop_set = set(LR_DROP_STEPS)
    current_lr = LEARNING_RATE
 
    for iteration in trange(1, NUM_ITERATIONS + 1, desc="Training", ncols=100):
        # --- Learning rate schedule (paper: dropped 3 times during training) ---
        if iteration in lr_drop_set:
            current_lr *= 0.1
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr
            log.info(f"  LR dropped to {current_lr}")
 
        # --- Self-play phase (batched) ---
        net.eval()
        examples, finished = play_games_batched(net, GAMES_PER_ITERATION)
        replay_buffer.extend(examples)
 
        last_game = finished[-1] if finished else None
 
        # --- Training phase ---
        avg_loss, avg_v_loss, avg_p_loss = train_network(net, optimizer, scaler, replay_buffer)
 
        # --- Logging ---
        log.info(
            f"Iter {iteration:>4d} | loss={avg_loss:.4f} (v={avg_v_loss:.4f} p={avg_p_loss:.4f}) "
            f"| buf={len(replay_buffer):>7d} | games={len(finished)} | lr={current_lr}"
        )
 
        # --- Periodic eval & display ---
        if iteration % EVAL_INTERVAL == 0 or iteration == 1:
            win_rate = eval_against_random(net, EVAL_GAMES)
            log.info(f"  Eval vs random: {win_rate * 100:.0f}% win rate ({EVAL_GAMES} games)")
            print_empty_board_policy(net)
 
        if iteration % BOARD_DISPLAY_INTERVAL == 0 or iteration == 1:
            if last_game is not None:
                w = last_game.winner()
                wstr = 'X' if w == 1 else 'O' if w == -1 else 'draw'
                log.info(f"  Last self-play game ({last_game.move_count} moves, winner={wstr}):")
                for line in last_game.display().split("\n"):
                    log.info(f"    {line}")
 
        # --- Checkpoint ---
        if iteration % CHECKPOINT_INTERVAL == 0:
            path = os.path.join(OUTPUT_DIR, f"model_iter_{iteration:04d}.pt")
            torch.save(net.state_dict(), path)
 
    # --- Final save ---
    final_path = os.path.join(OUTPUT_DIR, "model_final.pt")
    torch.save(net.state_dict(), final_path)
    log.info(f"Training complete. Final model saved to {final_path}")
 
    # --- Final eval ---
    win_rate = eval_against_random(net, EVAL_GAMES * 2)
    log.info(f"Final eval vs random: {win_rate * 100:.0f}% win rate ({EVAL_GAMES * 2} games)")
 
 
if __name__ == "__main__":
    main()