import torch
from torch import nn


class QNet(nn.Module):
    #  Аппроксимирует Q(s, a)
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features=state_dim + action_dim, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, states, actions):
        X = torch.cat((states, actions), 1)
        return self.model(X).squeeze(-1)


class DoubleQNet(nn.Module):
    #  Двойная min-clipped QNet
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.model1 = QNet(state_dim, action_dim)
        self.model2 = QNet(state_dim, action_dim)

    def double_forward(self, states, actions):
        return self.model1(states, actions), self.model2(states, actions)

    def forward(self, states, actions):
        return torch.min(self.model1(states, actions), self.model2(states, actions))


class VNet(nn.Module):
    #  Аппроксимирует V(s)
    def __init__(self, state_dim: int):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features=state_dim, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, states):
        return self.model(states).squeeze(-1)
