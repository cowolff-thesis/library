import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from gym.spaces import Box, Discrete, MultiDiscrete, Tuple
from ThesisPackage.RL.Centralized_PPO.action_space_handlers import handle_multidiscrete, handle_discrete, handle_box, handle_tuple

class Agent(nn.Module):
    def __init__(self, env):
        super(Agent, self).__init__()

        self.action_space = env.action_space(env.agents[0])
        if type(self.action_space) == Box:
            self.n = self.action_space.shape[0]
        elif type(self.action_space) == Discrete:
            self.n = self.action_space.n
        elif type(self.action_space) == MultiDiscrete:
            self.n = self.action_space.nvec.sum()
        elif type(self.action_space) == Tuple:
            self.n = 0
            for space in self.action_space:
                if type(space) == Box:
                    self.n += space.shape[0]
                elif type(space) == Discrete:
                    self.n += space.n
                elif type(space) == MultiDiscrete:
                    self.n += space.nvec.sum()
       
        self.critic = nn.Sequential(
            self.layer_init(nn.Linear(np.array(env.observation_space("critic").shape).prod(), 128)),
            nn.Tanh(),
            self.layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            self.layer_init(nn.Linear(np.array(env.observation_space(env.agents[0]).shape).prod(), 128)),
            nn.Tanh(),
            self.layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, self.n), std=0.01),
        )
        if isinstance(self.action_space, Box):
            self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(self.action_space.shape)))
        if isinstance(self.action_space, Tuple):
            number_of_continuous_actions = 0
            for space in self.action_space:
                if isinstance(space, Box):
                    number_of_continuous_actions += space.shape[0]
            self.actor_logstd = nn.Parameter(torch.zeros(1, number_of_continuous_actions))

    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, state, action=None):
        if isinstance(self.action_space, MultiDiscrete):
            return handle_multidiscrete(self.actor, self.critic, self.action_space, x, state, action)
        elif isinstance(self.action_space, Discrete):
            return handle_discrete(self.actor, self.critic, x, state, action)
        elif isinstance(self.action_space, Box):
            return handle_box(self.actor, self.critic, self.actor_logstd, x, state, action)
        elif isinstance(self.action_space, Tuple):
            return handle_tuple(self.actor, self.critic, self.action_space, self.actor_logstd, x, state, action)