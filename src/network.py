# src/network.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import Config


class PolicyValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        board_size = Config.BOARD_SIZE

        # Shared trunk
        self.conv1 = nn.Conv2d(
            Config.INPUT_CHANNELS, Config.NUM_FILTERS_1, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(Config.NUM_FILTERS_1)

        self.conv2 = nn.Conv2d(Config.NUM_FILTERS_1, Config.NUM_FILTERS_2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(Config.NUM_FILTERS_2)

        self.conv3 = nn.Conv2d(Config.NUM_FILTERS_2, Config.NUM_FILTERS_3, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(Config.NUM_FILTERS_3)

        # Policy head
        self.policy_conv = nn.Conv2d(Config.NUM_FILTERS_3, 16, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(16)
        self.policy_fc = nn.Linear(16 * board_size * board_size, board_size * board_size)

        # Value head
        self.value_conv = nn.Conv2d(Config.NUM_FILTERS_3, 16, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(16)
        self.value_fc1 = nn.Linear(16 * board_size * board_size, Config.VALUE_FC_SIZE)
        self.value_fc2 = nn.Linear(Config.VALUE_FC_SIZE, 1)

    def forward(self, x):
        # Shared trunk
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        p = F.log_softmax(p, dim=1)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v

    def predict(self, state):
        """Single state inference for MCTS. Returns policy (numpy), value (float)."""
        self.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            log_policy, value = self(state_tensor)
            policy = torch.exp(log_policy).squeeze(0).numpy()
            value = value.item()
        return policy, value
