import random
import numpy as np


class ReplayBuffer:
    def __init__(self, size):
        """
        Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = [(0, 0, 0, 0, 0)] * size
        self._len = 0
        self._cur = 0
        self._maxsize = size

    def __len__(self):
        return self._len

    def add(self, state, action, reward, next_state, done):

        data = (state, action, reward, next_state, done)

        if self._cur == self._maxsize:
            self._cur = 0

        # add data to storage
        self._storage[self._cur] = data
        self._cur += 1
        self._len = min(self._len + 1, self._maxsize)

    def sample(self, batch_size):
        # collect <s,a,r,s',done> for each index
        sz = min(batch_size, self._len)
        sample = random.choices(self._storage[:self._len], k=sz)
        obs, act, rew, obs1, done = [0] * sz, [0] * sz, [0] * sz, [0] * sz, [0] * sz
        for i in range(len(sample)):
            obs[i] = sample[i][0]
            act[i] = sample[i][1]
            rew[i] = sample[i][2]
            obs1[i] = sample[i][3]
            done[i] = sample[i][4]
        return (
            np.array(obs),
            np.array(act),
            np.array(rew),
            np.array(obs1),
            np.array(done),
        )
