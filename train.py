# train.py
import torch
import os
from src.game import GomokuGame, GameResult, Color
from src.network import DQNAgent
from src.network import QNetwork
from src.logger import GameLogger
from src.config import Config
import random
import numpy as np
from tqdm import trange
from collections import deque


class RandomAgent:
    def select_action(self, game):
        legal_actions = game.get_legal_actions()
        return random.choice(legal_actions)

def _line_info(game, row, col, color, dr, dc):
    """For a virtual stone of `color` at (row, col), return (length, open_neg, open_pos).
    (row, col) is assumed empty; we count the line as if we placed color there."""
    n = count_line(game, row, col, -dr, -dc, color)
    p = count_line(game, row, col, dr, dc, color)
    length = 1 + n + p
    # Negative end: cell at (row - (n+1)*dr, col - (n+1)*dc)
    rn, cn = row - (n + 1) * dr, col - (n + 1) * dc
    open_neg = (
        0 <= rn < Config.BOARD_SIZE
        and 0 <= cn < Config.BOARD_SIZE
        and game.board[rn, cn] == Color.EMPTY.value
    )
    rp, cp = row + (p + 1) * dr, col + (p + 1) * dc
    open_pos = (
        0 <= rp < Config.BOARD_SIZE
        and 0 <= cp < Config.BOARD_SIZE
        and game.board[rp, cp] == Color.EMPTY.value
    )
    return length, open_neg, open_pos


class HeuristicAgent:
    # Configurable weights for pattern scoring (easy to tune)
    SCORE_WIN = 1_000_000
    SCORE_BLOCK_WIN = 900_000
    SCORE_OPEN_4 = 5_000
    SCORE_SEMI_OPEN_4 = 2_000
    SCORE_OPEN_3 = 800
    SCORE_SEMI_OPEN_3 = 300
    SCORE_OPEN_2 = 100
    SCORE_SEMI_OPEN_2 = 30
    SCORE_BLOCK_4 = 4_000
    SCORE_BLOCK_OPEN_3 = 1_000
    SCORE_BLOCK_SEMI_OPEN_3 = 400
    SCORE_CENTER_PER_CELL = 15
    SCORE_NEIGHBOR_OWN = 20
    TOP_CANDIDATES_FOR_LOOKAHEAD = 4
    RANDOM_TIE_BAND = 0.95  # pick randomly among moves with score >= best * this

    def __init__(self, color):
        self.color = color
        self.opponent_color = Color.WHITE.value if color == Color.BLACK.value else Color.BLACK.value

    def _pattern_score(self, game, row, col, color):
        """Score for patterns we create by placing our color at (row, col)."""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        total = 0
        open_fours = 0
        semi_open_fours = 0
        open_threes = 0
        semi_open_threes = 0
        open_twos = 0
        semi_open_twos = 0
        for dr, dc in directions:
            length, open_neg, open_pos = _line_info(game, row, col, color, dr, dc)
            if length >= 5:
                return self.SCORE_WIN
            both_open = open_neg and open_pos
            one_open = open_neg or open_pos
            if length == 4:
                if both_open:
                    open_fours += 1
                elif one_open:
                    semi_open_fours += 1
            elif length == 3:
                if both_open:
                    open_threes += 1
                elif one_open:
                    semi_open_threes += 1
            elif length == 2:
                if both_open:
                    open_twos += 1
                elif one_open:
                    semi_open_twos += 1
        total += open_fours * self.SCORE_OPEN_4
        total += semi_open_fours * self.SCORE_SEMI_OPEN_4
        total += open_threes * self.SCORE_OPEN_3
        total += semi_open_threes * self.SCORE_SEMI_OPEN_3
        total += open_twos * self.SCORE_OPEN_2
        total += semi_open_twos * self.SCORE_SEMI_OPEN_2
        return total

    def _center_bias(self, row, col):
        """Prefer center and near-center on 9x9."""
        center = (Config.BOARD_SIZE - 1) / 2.0
        dist = abs(row - center) + abs(col - center)
        max_dist = center * 2
        return self.SCORE_CENTER_PER_CELL * (1.0 - dist / max_dist)

    def _neighbor_bonus(self, game, row, col):
        """Small bonus for playing next to our own stones."""
        count = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                r, c = row + dr, col + dc
                if 0 <= r < Config.BOARD_SIZE and 0 <= c < Config.BOARD_SIZE and game.board[r, c] == self.color:
                    count += 1
        return count * self.SCORE_NEIGHBOR_OWN

    def _score_move_for_color(self, game, row, col, own_color):
        """Score for placing stone of own_color at (row, col). Used for us and for opponent in lookahead."""
        other = self.opponent_color if own_color == self.color else self.color
        off = self._pattern_score(game, row, col, own_color)
        if off >= self.SCORE_WIN:
            return off
        # Block score: how much we block the other color
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        def_ = 0
        for dr, dc in directions:
            length, open_neg, open_pos = _line_info(game, row, col, other, dr, dc)
            if length >= 5:
                return self.SCORE_BLOCK_WIN
            both_open = open_neg and open_pos
            one_open = open_neg or open_pos
            if length == 4:
                def_ += self.SCORE_BLOCK_4
            elif length == 3:
                def_ += self.SCORE_BLOCK_OPEN_3 if both_open else (self.SCORE_BLOCK_SEMI_OPEN_3 if one_open else 0)
        center = self._center_bias(row, col)
        # Neighbor bonus only when evaluating our own color
        neighbor = self._neighbor_bonus(game, row, col) if own_color == self.color else 0
        return off + def_ + center + neighbor

    def score_move(self, game, row, col):
        """Combined static score for placing our stone at (row, col)."""
        return self._score_move_for_color(game, row, col, self.color)

    def _opponent_best_reply_score(self, game):
        """Fast greedy score for opponent: win > block win > best static score."""
        legal = game.get_legal_actions()
        if not legal:
            return 0.0
        best = -1e9
        for r, c in legal:
            our_len = get_pattern_length(game, r, c, self.color)
            opp_len = get_pattern_length(game, r, c, self.opponent_color)
            if our_len >= 5:
                return 1e9
            if opp_len >= 5:
                return -1e9
            s = self.score_move(game, r, c)
            best = max(best, s)
        return best

    def _lookahead_score(self, game, candidate):
        """After we play candidate, opponent replies with their best move; return our score in that position."""
        cloned = game.clone()
        cloned.step(candidate)
        if cloned.result != GameResult.ONGOING:
            return 1e9 if cloned.result == (GameResult.BLACK_WIN if self.color == Color.BLACK.value else GameResult.WHITE_WIN) else -1e9
        opp_legal = cloned.get_legal_actions()
        if not opp_legal:
            return 0.0
        best_opp_score = -1e9
        best_opp_move = None
        for r, c in opp_legal:
            if get_pattern_length(cloned, r, c, self.opponent_color) >= 4:
                return -1e9
            s = self._score_move_for_color(cloned, r, c, self.opponent_color)
            if s > best_opp_score:
                best_opp_score = s
                best_opp_move = (r, c)
        if best_opp_move is None:
            return 0.0
        cloned.step(best_opp_move)
        if cloned.result != GameResult.ONGOING:
            return -1e9
        our_legal = cloned.get_legal_actions()
        if not our_legal:
            return 0.0
        return max(self.score_move(cloned, r, c) for r, c in our_legal)

    def select_action(self, game):
        legal_actions = game.get_legal_actions()
        if not legal_actions:
            return None

        # Immediate win
        for r, c in legal_actions:
            if get_pattern_length(game, r, c, self.color) >= 4:
                return (r, c)

        # Immediate block of opponent win
        for r, c in legal_actions:
            if get_pattern_length(game, r, c, self.opponent_color) >= 4:
                return (r, c)

        # Score all moves
        scored = [(self.score_move(game, r, c), (r, c)) for r, c in legal_actions]
        scored.sort(key=lambda x: -x[0])
        best_score = scored[0][0]

        # Optional 1-ply lookahead on top candidates
        k = min(self.TOP_CANDIDATES_FOR_LOOKAHEAD, len(scored))
        top = scored[:k]
        if k > 1 and best_score < self.SCORE_WIN and best_score < self.SCORE_BLOCK_WIN:
            lookahead_scores = []
            for _, move in top:
                ls = self._lookahead_score(game, move)
                lookahead_scores.append((ls, move))
            lookahead_scores.sort(key=lambda x: -x[0])
            best_la = lookahead_scores[0][0]
            ties = [(s, m) for s, m in lookahead_scores if s >= best_la * self.RANDOM_TIE_BAND]
            return random.choice(ties)[1]

        # Pick among top moves with small randomness
        ties = [m for s, m in scored if s >= best_score * self.RANDOM_TIE_BAND]
        return random.choice(ties)

class SelfPlayOpponent: 
    def __init__(self, model_dir, device): 
        self.model_dir = model_dir
        self.device = device 
        self.network = QNetwork().to(device)
        self.has_model = False
        self.update_model() 
    
    #to load a previously saved model to play as the opponent 
    def update_model(self):
        if not os.path.exists(self.model_dir):
            return
        
        models = [f for f in os.listdir(self.model_dir) if f.startswith("player_ep") and f.endswith(".pth")]
        if models:
            chosen_model = random.choice(models)
            model_path = os.path.join(self.model_dir, chosen_model)
            try:
                self.network.load_state_dict(torch.load(model_path, map_location=self.device))
                self.network.eval()
                self.has_model = True 
            except Exception as e: 
                print(f"Couldnt load self play model {chosen_model}: {e}")

    def select_action(self, game):
        legal_actions = game.get_legal_actions()
        if not legal_actions:
            return None
        
        if not self.has_model:
            return random.choice(legal_actions)
        
        if random.random() < 0.1:
            return random.choice(legal_actions)
        
        state = game.get_state_for_network(perspective_color=game.current_player)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.network(state_tensor).cpu().numpy()[0]

        legal_q_values = [(a, q_values[game.action_to_int(a)]) for a in legal_actions]

        best_action = max(legal_q_values, key=lambda x: x[1])[0]
        
        return best_action



        

def count_line(game, row, col, dr, dc, color):
    """Count consecutive stones of given color in one direction from (row, col)"""
    count = 0
    r, c = row + dr, col + dc
    while 0 <= r < Config.BOARD_SIZE and 0 <= c < Config.BOARD_SIZE and game.board[r, c] == color:
        count += 1
        r += dr
        c += dc
    return count


def get_pattern_length(game, row, col, color):
    """Get maximum line length for a stone of given color at (row, col)"""
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    max_length = 0

    for dr, dc in directions:
        # Count in both directions and add 1 for the stone itself
        length = (
            1
            + count_line(game, row, col, dr, dc, color)
            + count_line(game, row, col, -dr, -dc, color)
        )
        max_length = max(max_length, length)

    return max_length


def check_blocks_opponent(game, row, col, opponent_color):
    """Check if placing a stone at (row, col) blocks an opponent threat"""
    # Temporarily check what opponent would have had at this position
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    max_blocked_length = 0

    for dr, dc in directions:
        # Check what the opponent line would be through this empty position
        length = (
            1
            + count_line(game, row, col, dr, dc, opponent_color)
            + count_line(game, row, col, -dr, -dc, opponent_color)
        )
        max_blocked_length = max(max_blocked_length, length)

    return max_blocked_length


def calculate_shaped_reward(game, action, agent_color, opponent_color):
    """Calculate intermediate rewards for threats and blocks"""
    row, col = action

    # Threat reward: what pattern did we create?
    threat_length = get_pattern_length(game, row, col, agent_color)
    threat_reward = 0.0
    if threat_length == 4:
        threat_reward = Config.THREAT_REWARD_4
    elif threat_length == 3:
        threat_reward = Config.THREAT_REWARD_3
    elif threat_length == 2:
        threat_reward = Config.THREAT_REWARD_2

    # Block reward: what opponent pattern did we block?
    blocked_length = check_blocks_opponent(game, row, col, opponent_color)
    block_reward = 0.0
    if blocked_length >= 4:
        block_reward = Config.BLOCK_REWARD_4
    elif blocked_length == 3:
        block_reward = Config.BLOCK_REWARD_3

    return threat_reward + block_reward


def get_reward(game_result, agent_is_black, game, action, agent_color, opponent_color):
    """Calculate total reward including terminal and shaped rewards"""
    # Terminal reward
    if game_result != GameResult.ONGOING:
        if game_result == GameResult.DRAW:
            return Config.DRAW_REWARD

        agent_won = (game_result == GameResult.BLACK_WIN and agent_is_black) or (
            game_result == GameResult.WHITE_WIN and not agent_is_black
        )

        return Config.WIN_REWARD if agent_won else Config.LOSS_REWARD

    # Calculate shaped rewards (intermediate smaller rewards for creating threats or blocking opponent threats)
    shaped_reward = calculate_shaped_reward(game, action, agent_color, opponent_color)

    return shaped_reward


def play_episode(player, opponent, agent_is_black):
    game = GomokuGame()
    game.reset()

    # Randomize which color the agent plays
    # agent_is_black = random.random() < 0.5



    agent_color = Color.BLACK if agent_is_black else Color.WHITE
    opponent_color = Color.WHITE if agent_is_black else Color.BLACK

    episode_transitions = []

    last_agent_state = None
    last_agent_action = None
    last_agent_reward = 0.0

    #these are buffer variables to track the agent while we wait for the opponent to move

    while game.result == GameResult.ONGOING:
        is_agent_turn = (game.current_player == Color.BLACK and agent_is_black) or (
            game.current_player == Color.WHITE and not agent_is_black
        )

        if is_agent_turn:
            # state = game.get_state_for_network()

            current_state = game.get_state_for_network(perspective_color=agent_color)  # get state from agent's perspective

            if last_agent_state is not None:
                episode_transitions.append(
                    (last_agent_state, last_agent_action, last_agent_reward, current_state, False)
                )

            action = player.select_action(game)
            action_int = game.action_to_int(action)

            state_before_move = current_state

            # PRINT BEFORE STEP
            #print(f"\n=== BEFORE AGENT MOVE ===")
            #print(f"Agent color: {agent_color}")
            #print(f"Current player: {game.current_player}")
            #print(f"State channel 0 (should be agent): sum = {state[0].sum()}")
            #print(f"State channel 1 (should be opponent): sum = {state[1].sum()}")

            game.step(action)

            # PRINT AFTER STEP
            #print(f"\n=== AFTER AGENT MOVE ===")
            #print(f"Current player: {game.current_player}")

            # Calculate reward including shaped rewards

            step_reward = calculate_shaped_reward(
                game, action, agent_color.value, opponent_color.value
            )


            if game.result != GameResult.ONGOING:
                final_reward = Config.WIN_REWARD if game.result != GameResult.DRAW else Config.DRAW_REWARD

                episode_transitions.append(
                    (state_before_move, action_int, final_reward, None, True)
                )
            else:
                last_agent_state = state_before_move 
                last_agent_action = action_int
                last_agent_reward = step_reward
        
        else:
            action = opponent.select_action(game)
            game.step(action)
            if game.result != GameResult.ONGOING:

                outcome_reward = Config.LOSS_REWARD if game.result != GameResult.DRAW else Config.DRAW_REWARD
                final_reward = last_agent_reward + outcome_reward

                episode_transitions.append(
                    (last_agent_state, last_agent_action, final_reward, None, True)
                )
            # reward = get_reward(
            #     game.result, agent_is_black, game, action, agent_color.value, opponent_color.value
            # )

        #     next_state = (
        #         game.get_state_for_network(perspective_color=agent_color)  # crucial to get next state from agent's perspective
        #         if game.result == GameResult.ONGOING
        #         else None
        #     )
        #     done = game.result != GameResult.ONGOING

        #     if next_state is not None:
        #         pass
        #         #print(f"Next state channel 0 sum: {next_state[0].sum()}")
        #         #print(f"Next state channel 1 sum: {next_state[1].sum()}")
        #         #print(f"Does next_state[0] match agent pieces? {(next_state[0].sum() == (game.board == agent_color.value).sum())}")

        #     episode_transitions.append((state, action_int, reward, next_state, done))
        # else:
        #     action = opponent.select_action(game)
        #     game.step(action)

    return episode_transitions, game.result, len(game.move_history), agent_is_black


def train():
    os.makedirs(Config.MODEL_DIR, exist_ok=True)

    player = DQNAgent()
    random_opponent = RandomAgent()
    self_play_opponent = SelfPlayOpponent(Config.MODEL_DIR, player.device)
    logger = GameLogger()

    win_count = 0
    loss_count = 0
    draw_count = 0

    # Rolling win rate tracking
    rolling_window = deque(maxlen=Config.ROLLING_WINDOW_SIZE)
    best_win_rate = 0.0

    for episode in trange(Config.TOTAL_EPISODES):
        #uncomment this for ranndom opponent training 
        # opponent = random_opponent
        
        #ucomment these for heuristic agent training
        progress = episode / Config.TOTAL_EPISODES

        agent_is_black = random.random() < 0.5 
        opponent_color = Color.WHITE.value if agent_is_black else Color.BLACK.value

        '''
        if progress < 0.2:
            opponent = random_opponent
        elif progress < 0.6:
            if random.random() < 0.5:
                opponent = random_opponent
            else:
                opponent = HeuristicAgent(opponent_color)
        else:
            opponent = HeuristicAgent(opponent_color)

        # transitions, result, num_moves, agent_is_black = play_episode(player, opponent)
        transitions, result, num_moves, _ = play_episode(player, opponent, agent_is_black)
        '''
        # Update self-play model every 500 episodes
        if episode > 0 and episode % 500 == 0:
            self_play_opponent.update_model()

        if progress < 0.1:
            # 0% - 10%: Random (Learn basic movement)
            opponent = random_opponent
        elif progress < 0.6:
            # 10% - 60%: Heuristic Agent (Force it to learn defense for 9,000 episodes)
            opponent = HeuristicAgent(opponent_color)
        else:
            # 60% - 100%: Self-Play (Now that it knows defense, learn advanced traps)
            if self_play_opponent.has_model:
                opponent = self_play_opponent
            else:
                opponent = HeuristicAgent(opponent_color)

        transitions, result, num_moves, _ = play_episode(player, opponent, agent_is_black)

        # Store all transitions
        for state, action, reward, next_state, done in transitions:
            player.store_transition(state, action, reward, next_state, done)

        # Multiple training steps per episode
        losses = []
        for _ in range(Config.TRAIN_STEPS_PER_EPISODE):
            loss = player.train_step()
            if loss is not None:
                losses.append(loss)

        avg_loss = np.mean(losses) if losses else None

        player.decay_epsilon()

        # Track win rate
        agent_won = (result == GameResult.BLACK_WIN and agent_is_black) or (
            result == GameResult.WHITE_WIN and not agent_is_black
        )
        if result == GameResult.DRAW:
            draw_count += 1
            rolling_window.append(0)
        elif agent_won:
            win_count += 1
            rolling_window.append(1)
        else:
            loss_count += 1
            rolling_window.append(0)

        # Calculate rolling win rate
        rolling_win_rate = (
            sum(rolling_window) / len(rolling_window) if len(rolling_window) > 0 else 0.0
        )

        # Save best model based on rolling win rate
        if len(rolling_window) == Config.ROLLING_WINDOW_SIZE and rolling_win_rate > best_win_rate:
            best_win_rate = rolling_win_rate
            player.save_model(f"{Config.MODEL_DIR}/player_best.pth")

        logger.log_game(
            result,
            num_moves,
            metadata={"episode": episode, "agent_color": "BLACK" if agent_is_black else "WHITE"},
        )

        if avg_loss is not None:
            logger.log_episode(
                episode,
                {
                    "loss": avg_loss,
                    "epsilon": player.epsilon,
                    "win_rate": win_count / (episode + 1),
                    "rolling_win_rate": rolling_win_rate,
                    "buffer_size": len(player.buffer),
                },
            )

        if (episode + 1) % Config.SAVE_FREQ == 0:
            total_games = episode + 1
            win_rate = win_count / total_games
            loss_str = f"{avg_loss:.4f}" if avg_loss is not None else "N/A"
            print(f"Episode {episode + 1}/{Config.TOTAL_EPISODES}")
            print(f"  Win Rate: {win_rate:.3f} ({win_count}W-{loss_count}L-{draw_count}D)")
            print(f"  Rolling Win Rate (last {Config.ROLLING_WINDOW_SIZE}): {rolling_win_rate:.3f}")
            print(f"  Best Rolling Win Rate: {best_win_rate:.3f}")
            print(
                f"  Epsilon: {player.epsilon:.3f}, Loss: {loss_str}, Buffer: {len(player.buffer)}"
            )
            logger.save()
            player.save_model(f"{Config.MODEL_DIR}/player_ep{episode+1}.pth")

    player.save_model(f"{Config.MODEL_DIR}/player_final.pth")
    logger.save()
    print("\nTraining complete")
    print(f"Final Win Rate: {win_count / Config.TOTAL_EPISODES:.3f}")
    print(f"Best Rolling Win Rate: {best_win_rate:.3f}")


if __name__ == "__main__":
    answer = input("Run training? (y/n): ").strip().lower()
    if answer == "y":
        train()
    else:
        print("Training cancelled to avoid overwriting existing models")
