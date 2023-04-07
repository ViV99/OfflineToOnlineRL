import random


class ReplayBuffer(object):
    def __init__(self, size):
        """
        Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = np.empty((size, 5))
        self._len = 0
        self._cur = 0
        self._maxsize = size


    def __len__(self):
        return self._len

    def add(self, obs_t, action, reward, obs_tp1, done):
        
        data = np.array([obs_t, action, reward, obs_tp1, done])
        
        if self._cur == self._maxsize:
            self._cur = 0

        # add data to storage
        self._storage[self._cur] = data
        self._cur += 1
        self._len = min(self._len + 1, self._maxsize)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        # collect <s,a,r,s',done> for each index
        sample = random.choices(self._storage[:self._len], k=batch_size)

        return (
            sample[:, 0],
            sample[:, 1],
            sample[:, 2],
            sample[:, 3],
            sample[:, 4],
        )