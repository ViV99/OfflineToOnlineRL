
import matplotlib.pyplot as plt
import gym
import torch

from tqdm import trange
from IPython.display import clear_output
from replay_buffer import ReplayBuffer

from iql import IQL
from policy import GaussianPolicy, EGreedyPolicy
from networks import DoubleQNet, VNet
from utils import load_dataset_to_replay_buffer, linear_decay, smoothen, evaluate, is_enough_ram, \
    wait_for_keyboard_interrupt


def train_iql_on_env(env_name, device, tau=0.7, beta=3.0, buffer_size=int(2e6), total_steps=int(5e5),
                     batch_size=64, eval_freq=5000, eval_games=10):
    env = gym.make(env_name)
    env.reset()
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    max_action = env.action_space.high[0]

    dataset = env.get_dataset()

    replay = ReplayBuffer(buffer_size)
    load_dataset_to_replay_buffer(dataset, replay)

    policy = GaussianPolicy(state_dim, action_dim, max_action).to(device)
    q_network = DoubleQNet(state_dim, action_dim).to(device)
    v_network = VNet(state_dim).to(device)
    iql = IQL(
        device=device,
        policy=policy,
        policy_opt=torch.optim.Adam(policy.parameters(), lr=1e-4),
        q_net=q_network,
        q_opt=torch.optim.Adam(q_network.parameters(), lr=1e-4),
        v_net=v_network,
        v_opt=torch.optim.Adam(v_network.parameters(), lr=1e-4),
        tau=tau,
        beta=beta,
    )

    mean_rw_history = []
    policy_loss_history = []
    q_loss_history = []
    v_loss_history = []
    initial_state_v_history = []
    step = 0

    with trange(step, total_steps + 1) as progress_bar:
        for step in progress_bar:
            if not is_enough_ram():
                print('less that 100 Mb RAM available, freezing')
                print('make sure everything is ok and use KeyboardInterrupt to continue')
                wait_for_keyboard_interrupt()

            # Для эпсилон жадной политики уменьшаем эпсилон со временем
            if isinstance(iql.policy, EGreedyPolicy):
                iql.policy.epsilon = linear_decay(init_val=1, final_val=0.05, cur_step=step,
                                                  total_steps=total_steps // 2)

            # Семплим из буффера батч для обучения
            states, actions, rewards, next_states, is_done = replay.sample(batch_size)
            logger = iql.train(device, states, actions, rewards, next_states, is_done)

            if logger:
                policy_loss_history.append(logger['policy_loss'])
                q_loss_history.append(logger['q_loss'])
                v_loss_history.append(logger['v_loss'])

            if step % eval_freq == 0:
                e = gym.make(env_name)
                r = evaluate(e, device, iql.policy, seed=step, n_games=eval_games)
                mean_rw_history.append(e.get_normalized_score(r) * 100.0)
                initial_state_v_history.append(iql.v_net(torch.tensor([gym.make(env_name).reset(seed=step)],
                                                                      device=device,
                                                                      dtype=torch.float32)).cpu().data.numpy())

                clear_output(True)

                plt.figure(figsize=[16, 16])

                plt.subplot(2, 3, 1)
                plt.title("Mean reward per life")
                plt.plot(mean_rw_history)
                plt.grid()

                plt.subplot(2, 3, 2)
                plt.title("Policy loss history (smoothened)")
                plt.plot(smoothen(policy_loss_history))
                plt.grid()

                plt.subplot(2, 3, 3)
                plt.title("Q loss history (smoothened)")
                plt.plot(q_loss_history)
                plt.grid()

                plt.subplot(2, 3, 4)
                plt.title("V loss history (smoothened)")
                plt.plot(smoothen(v_loss_history))
                plt.grid()

                plt.subplot(2, 3, 5)
                plt.title("Initial state V")
                plt.plot(initial_state_v_history)
                plt.grid()

                plt.show()

    return iql, mean_rw_history, q_loss_history, v_loss_history, policy_loss_history, initial_state_v_history
