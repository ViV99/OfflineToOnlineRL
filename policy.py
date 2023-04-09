import torch
import copy
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal


class EGreedyPolicy(nn.Module):
    #  Попытка реализовать эпсилон жадную политику (не уверен,
    #  насколько она хорошо себя показывает, поэтому ниже будет ещё и другая)
    def __init__(self, state_dim: int, action_dim: int, max_action: float, epsilon: float):
        super().__init__()
        self.max_action = max_action
        self.epsilon = epsilon
        self.model = nn.Sequential(
            nn.Linear(in_features=state_dim, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Tanh(),
        )

    def forward(self, states):
        should_explore = torch.rand(states.shape[0])
        actions = self.model(states)
        rnd_actions = torch.rand(actions.shape)
        idx = should_explore < self.epsilon
        actions[idx] = rnd_actions[idx]
        return actions

    @torch.no_grad()
    def act(self, device, state):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        action = torch.clamp(self(state) * self.max_action, -self.max_action, self.max_action)
        return action.cpu().data.numpy().flatten()


class GaussianPolicy(nn.Module):
    #  Гауссова политика
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super().__init__()
        self.max_action = max_action
        self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))

        self.model = nn.Sequential(
            nn.Linear(in_features=state_dim, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Tanh(),
        )

    def forward(self, states):
        mean = self.model(states)
        std = torch.exp(self.log_std.clamp(-3, 3))
        return MultivariateNormal(mean, scale_tril=torch.diag(std))

    @torch.no_grad()
    def act(self, device, state):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        dist = self(state)
        action = dist.mean if not self.training else dist.sample()
        action = torch.clamp(self.max_action * action, -self.max_action, self.max_action)
        return action.cpu().data.numpy().flatten()
