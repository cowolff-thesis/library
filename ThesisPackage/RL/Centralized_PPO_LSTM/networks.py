import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from gym.spaces import Box, Discrete, MultiDiscrete, Tuple

class Actor(nn.Module):
    def __init__(self, env) -> None:
        super().__init__()

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

        self.input_layer = nn.Linear(np.array(env.observation_space(env.agents[0]).shape).prod(), 128)
        self.hidden_layer = nn.Linear(128, 64)
        self.lstm_layer = nn.LSTM(64, 64)
        self.output_layer = nn.Linear(64, self.n)

        for name, param in self.lstm_layer.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        if isinstance(self.action_space, Box):
            self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(self.action_space.shape)))
        if isinstance(self.action_space, Tuple):
            number_of_continuous_actions = 0
            for space in self.action_space:
                if isinstance(space, Box):
                    number_of_continuous_actions += space.shape[0]
            self.actor_logstd = nn.Parameter(torch.zeros(1, number_of_continuous_actions))

    def forward(self, x, hidden):
        x = torch.tanh(self.input_layer(x))
        x = torch.tanh(self.hidden_layer(x))
        x, hidden = self.lstm_layer(x, hidden)
        x = self.output_layer(x)
        return x, hidden