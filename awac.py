import torch
import copy

from utils import soft_update


class AWAC:
    def __init__(self, device: str,
                 q_net, q_opt, policy, policy_opt,
                 beta: float = 3.0,
                 discount: float = 0.99,
                 exp_adv_max: float = 100.0,
                 alpha: float = 0.005,
                 logg_freq: int = 50):
        self.q_net = q_net  # Аппроксимация Q
        self.q_opt = q_opt  # Отпимайзер для Q
        self.q_target = copy.deepcopy(self.q_net).requires_grad_(False).to(device)  # Q-target
        self.policy = policy  # Аппроксимация политики
        self.policy_opt = policy_opt  # Отпимайзер для политики

        self.beta = beta  # Коэффициент для advantage-weighted регрессии
        self.discount = discount
        self.exp_adv_max = exp_adv_max  # Обрезаем экспоненту A(s, a), чтобы вес не был слишком большой
        self.alpha = alpha  # Обновляю таргет как (1-alpha)*old + alpha*new
        self.logg_freq = logg_freq  # Насколько часто логируем лоссы и всё остальное

        self.iter = 0
        self.device = device

    @staticmethod
    def _mse_loss(y1, y2):
        return torch.mean((y1 - y2) ** 2, dim=-1)

    def _policy_loss(self, states, actions):
        policy_act = self.policy(states)

        with torch.no_grad():
            cur_actions = policy_act.rsample()
            v = self.q_net(states, cur_actions)
            q = self.q_net(states, actions)
            advantages = q - v
            exp_advantages = torch.exp(self.beta * advantages).clamp(max=self.exp_adv_max)

        if isinstance(policy_act, torch.distributions.Distribution):
            action_losses = -policy_act.log_prob(actions)
        elif torch.is_tensor(policy_act):
            action_losses = self._mse_loss(policy_act, actions)

        loss = torch.mean(exp_advantages * action_losses)
        return loss

    def _q_loss(self, states, actions, rewards, next_states, is_done):
        is_not_done = 1.0 - is_done.float()

        with torch.no_grad():
            next_actions = self.policy(next_states).rsample().clamp(min=self.policy.min_action,
                                                                    max=self.policy.max_action)
            q_next = self.q_target(next_states, next_actions)
            targets = rewards + self.discount * is_not_done * q_next

        q1, q2 = self.q_net.double_forward(states, actions)
        q_loss = (self._mse_loss(q1, targets) + self._mse_loss(q2, targets)) / 2
        return q_loss

    def _update_q(self, states, actions, rewards, next_states, is_done):
        loss = self._q_loss(states, actions, rewards, next_states, is_done)
        self.q_opt.zero_grad()
        loss.backward()
        self.q_opt.step()
        return loss.item()

    def _update_policy(self, states, actions):
        loss = self._policy_loss(states, actions)
        self.policy_opt.zero_grad()
        loss.backward()
        self.policy_opt.step()
        return loss.item()

    def train(self, device, states, actions, rewards, next_states, is_done):
        states = torch.tensor(states, device=device, dtype=torch.float32)
        actions = torch.tensor(actions, device=device, dtype=torch.float32)
        rewards = torch.tensor(rewards, device=device, dtype=torch.float32)
        next_states = torch.tensor(next_states, device=device, dtype=torch.float32)
        is_done = torch.tensor(is_done.astype('float32'), device=device, dtype=torch.float32)

        self.iter += 1

        q_loss = self._update_q(states, actions, rewards, next_states, is_done)
        policy_loss = self._update_policy(states, actions)
        soft_update(self.q_target, self.q_net, self.alpha)

        if self.iter % self.logg_freq == 0:
            return {'q_loss': q_loss, 'policy_loss': policy_loss}
        else:
            return {}
