# src/network.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from src.buffer import ReplayBuffer
from src.config import Config


class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(64 * Config.BOARD_SIZE * Config.BOARD_SIZE, 256)
        self.fc2 = nn.Linear(256, Config.BOARD_SIZE * Config.BOARD_SIZE)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * Config.BOARD_SIZE * Config.BOARD_SIZE)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DQNAgent:
    def __init__(self):
        self.gamma = Config.GAMMA
        self.epsilon = Config.EPSILON_START
        self.epsilon_end = Config.EPSILON_END
        self.epsilon_decay = Config.EPSILON_DECAY
        self.batch_size = Config.BATCH_SIZE
        self.target_update_freq = Config.TARGET_UPDATE_FREQ
        self.learning_rate = Config.LEARNING_RATE

        self.update_counter = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = QNetwork().to(self.device)
        self.target_network = QNetwork().to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.buffer = ReplayBuffer()

    def select_action(self, game, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon

        legal_actions = game.get_legal_actions()

        if random.random() < epsilon:
            return random.choice(legal_actions)

        state = game.get_state_for_network()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.q_network(state_tensor).cpu().numpy()[0]

        legal_q_values = [(a, q_values[game.action_to_int(a)]) for a in legal_actions]
        best_action = max(legal_q_values, key=lambda x: x[1])[0]

        return best_action

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_online = self.q_network(next_states)
            next_actions = next_q_online.argmax(dim=1)
            next_q_target = (
                self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            )
            target_q = rewards + (1 - dones) * self.gamma * next_q_target

        loss = F.smooth_l1_loss(current_q, target_q)
        # loss = F.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save_model(self, filepath):
        torch.save(self.q_network.state_dict(), filepath)

    def load_model(self, filepath):
        self.q_network.load_state_dict(torch.load(filepath, map_location=self.device))
        self.target_network.load_state_dict(self.q_network.state_dict())
