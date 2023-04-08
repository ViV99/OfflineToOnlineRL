import numpy as np
import psutil
import torch
import time
from scipy.signal import fftconvolve, gaussian


def wait_for_keyboard_interrupt():
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


def load_dataset_to_replay_buffer(ds, replay):
    #  Загружает датасет из d4rl в буффер
    for i in range(min(replay.capacity(), len(ds['observations']))):
        replay.add(ds['observations'][i], ds['actions'][i], ds['rewards'][i], ds['next_observations'][i], ds['terminals'][i])


@torch.no_grad()
def evaluate(env, device, policy, seed, n_games=1, t_max=100000):
    #  Оценивает средний ревард агента
    env.seed(seed)
    policy.eval()
    episode_rewards = []
    for _ in range(n_games):
        state = env.reset()
        episode_reward = 0
        for _ in range(t_max):
            action = policy.act(device, state)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            if done:
                break
        episode_rewards.append(episode_reward)

    policy.train()
    return np.array(episode_rewards).mean()


def is_enough_ram(min_available_gb=0.1):
    # Проверяем, что не упадём от нехватки оперативки
    mem = psutil.virtual_memory()
    return mem.available >= min_available_gb * (1024 ** 3)


def linear_decay(init_val, final_val, cur_step, total_steps):
    if cur_step >= total_steps:
        return final_val
    return (init_val * (total_steps - cur_step) +
            final_val * cur_step) / total_steps


def smoothen(values):
    kernel = gaussian(100, std=100)
    kernel = kernel / np.sum(kernel)
    return fftconvolve(values, kernel, 'valid')