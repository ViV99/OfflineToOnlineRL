import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal
from utils import soft_update


class QNet(nn.Module):
    #  Аппроксимирует Q(s, a) через V(s')
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


import copy


class IQL:
    def __init__(self, max_action: float, device: str,
                 policy, policy_opt, q_net, q_opt, v_net, v_opt,
                 tau: float = 0.7,
                 beta: float = 3.0,
                 discount: float = 0.99,
                 target_refresh_freq: int = 5000,
                 alpha: float = 0.005,
                 logg_freq: int = 50):
        self.max_action = max_action
        self.q_net = q_net  # Аппроксимация Q
        self.q_opt = q_opt  # Отпимайзер для Q
        self.q_target = copy.deepcopy(self.q_net).requires_grad_(False).to(device)  # Q-target
        self.v_net = v_net  # Аппроксимация V
        self.v_opt = v_opt  # Отпимайзер для V
        self.policy = policy  # Аппроксимация политики
        self.policy_opt = policy_opt  # Отпимайзер для политики
        self.tau = tau  # Коэффициент для экспектильной регрессии
        self.beta = beta  # Коэффициент для advantage-weighted регрессии

        self.discount = discount

        self.target_refresh_freq = target_refresh_freq  # Обновляю таргет каждые FREQ шагов (не используется по дефолту)
        self.alpha = alpha  # Обновляю таргет как (1-alpha)*old + alpha*new (используется по дефолту)
        self.logg_freq = logg_freq  # Насколько часто логируем лоссы и всё остальное
        self.iter = 0
        self.device = device

    @staticmethod
    def _expectile_loss(advantages, tau: float):
        return torch.mean(torch.abs(tau - (advantages < 0).float()) * advantages ** 2)

    @staticmethod
    def _mse_loss(y1, y2):
        return torch.mean((y1 - y2) ** 2, dim=-1)

    def _update_v(self, states, actions, logger):
        with torch.no_grad():
            target = self.q_target(states, actions)

        v = self.v_net(states)
        advantages = target - v
        v_loss = self._expectile_loss(advantages, self.tau)

        if self.iter % self.logg_freq == 0:
            logger['v_loss'] = v_loss.item()

        self.v_opt.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_opt.step()
        return advantages

    def _update_q(self, next_v, states, actions, rewards, is_done, logger):
        is_not_done = 1.0 - is_done.float()
        targets = rewards + is_not_done * self.discount * next_v.squeeze().detach()
        q1, q2 = self.q_net.double_forward(states, actions)
        q_loss = (self._mse_loss(q1, targets) + self._mse_loss(q2, targets)) / 2

        if self.iter % self.logg_freq == 0:
            logger['q_loss'] = q_loss.item()

        self.q_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_opt.step()

        # Обновляем Q-target
        # if self.iter % self.target_refresh_freq == 0:
        #     self.q_target.load_state_dict(self.q_net.state_dict())
        soft_update(self.q_target, self.q_net, self.alpha)

    def _update_policy(self, advantages, states, actions, logger):
        exp_advantages = torch.exp(self.beta * advantages.detach()).clamp(max=100)
        policy_act = self.policy(states)
        if isinstance(policy_act, torch.distributions.Distribution):
            action_losses = -policy_act.log_prob(actions)
        elif torch.is_tensor(policy_act):
            action_losses = self._mse_loss(policy_act, actions)

        policy_loss = torch.mean(exp_advantages * action_losses)

        if self.iter % self.logg_freq == 0:
            logger['policy_loss'] = policy_loss.item()

        self.policy_opt.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_opt.step()

    def train(self, device, states, actions, rewards, next_states, is_done):
        states = torch.tensor(states, device=device, dtype=torch.float32)
        actions = torch.tensor(actions, device=device, dtype=torch.float32)
        rewards = torch.tensor(rewards, device=device, dtype=torch.float32)
        next_states = torch.tensor(next_states, device=device, dtype=torch.float32)
        is_done = torch.tensor(is_done.astype('float32'), device=device, dtype=torch.float32)

        self.iter += 1
        logger = {}

        with torch.no_grad():
            next_v = self.v_net(next_states)
        advantages = self._update_v(states, actions, logger)
        self._update_q(next_v, states, actions, rewards, is_done, logger)
        self._update_policy(advantages, states, actions, logger)
        return logger
