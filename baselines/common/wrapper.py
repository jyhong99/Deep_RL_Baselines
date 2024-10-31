import gym
import numpy as np


#https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/running_mean_std.py
class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def copy(self):
        new_object = RunningMeanStd(shape=self.mean.shape)
        new_object.mean = self.mean.copy()
        new_object.var = self.var.copy()
        new_object.count = float(self.count)
        return new_object

    def combine(self, other):
        self.update_from_moments(other.mean, other.var, other.count)

    def update(self, arr):
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

        
class NormalizedEnv(gym.core.Wrapper):
    def __init__(self, env, obs_norm=True, ret_norm=True, gamma=0.99, epsilon=1e-8):
        super(NormalizedEnv, self).__init__(env)
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.obs_rms = RunningMeanStd(shape=self.observation_space.shape) if obs_norm else None
        self.ret_rms = RunningMeanStd(shape=(1,)) if ret_norm else None
        self.ret = np.zeros(())
        self.gamma = gamma
        self.epsilon = epsilon

    def __str__(self):
        return self.env.__str__()
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        obs = self._normalize_obs(obs)
        if self.ret_rms:
            self.ret = self.ret * self.gamma + reward
            self.ret_rms.update(np.array([self.ret].copy()))
            reward = reward / np.sqrt(self.ret_rms.var + self.epsilon)
            self.ret = self.ret * (1. - float(done))

        return obs, reward, done, info

    def reset(self, seed=None):
        self.ret = np.zeros(())
        obs = self.env.reset(seed=seed) if seed is not None else self.env.reset()
        return self._normalize_obs(obs)

    def render(self, mode="human"):
        return self.env.render(mode=mode)

    def close(self):
        return self.env.close()
    
    def _normalize_obs(self, obs):
        if self.obs_rms:
            self.obs_rms.update(obs)
            return (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)
        else:
            return obs