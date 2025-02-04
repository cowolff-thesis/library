from gymnasium.spaces.space import Space
import numpy as np
from collections import deque
from pettingzoo.utils import wrappers
import gym

class ParallelFrameStack(wrappers.BaseWrapper):
    def __init__(self, env, num_stack):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = {agent: deque(maxlen=num_stack) for agent in self.env.agents}
        self.env = env

        self._observation_spaces = {}
        
        # Update observation space to reflect stacked frames
        for agent in self.agents:
            low = np.repeat(self.observation_space(agent).low, num_stack, axis=0)
            high = np.repeat(self.observation_space(agent).high, num_stack, axis=0)
            self.env._observation_spaces[agent] = gym.spaces.Box(low=low, high=high, dtype=self.observation_space(agent).dtype)

    def reset(self, seed=None, options=None):
        observations, infos = self.env.reset()
        for agent, obs in observations.items():
            for _ in range(self.num_stack):
                self.frames[agent].append(obs)
            observations[agent] = np.concatenate(self.frames[agent])
        return observations, infos

    def step(self, actions):
        next_observations, rewards, terminated, truncated, infos = self.env.step(actions)
        for agent, obs in next_observations.items():
            self.frames[agent].append(obs)
            next_observations[agent] = np.concatenate(self.frames[agent])
        return next_observations, rewards, terminated, truncated, infos
    
    def observation_space(self, agent):
        return self.env._observation_spaces[agent]

    def close(self):
        self.env.close()