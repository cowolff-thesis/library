import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from gym.spaces import Discrete, MultiDiscrete
import numpy as np

class Agent(nn.Module):
    def __init__(self, env):
        super(Agent, self).__init__()
        self.action_space = env.action_space(env.agents[0])
        self.observation_space = env.observation_space(env.agents[0])
        if isinstance(self.action_space, MultiDiscrete):
            self.n = self.action_space.nvec.sum()
        elif isinstance(self.action_space, Discrete):
            self.n = self.action_space.n
        else:
            raise Exception("Action space type is not supported")
        self.actor = nn.Sequential(
            self.layer_init(nn.Linear(self.observation_space.shape[0], 128)),
            nn.ReLU(),
            self.layer_init(nn.Linear(128, 64)),
            nn.ReLU(),
            self.layer_init(nn.Linear(64, self.n), std=0.01)
        )

    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer
    
    def get_action_and_probs(self, x, action=None):
        if isinstance(self.action_space, Discrete):
            logits = self.actor(x)
            categorical = Categorical(logits=logits)
            if action is None:
                action = categorical.sample()
            logprob = categorical.log_prob(action)
            return action, logprob
        elif isinstance(self.action_space, MultiDiscrete):
            logits = self.actor(x)
            split_logits = torch.split(logits, self.action_space.nvec.tolist(), dim=1)
            multi_categoricals = [Categorical(logits=logits) for logits in split_logits]
            if action is None:
                action = torch.stack([categorical.sample() for categorical in multi_categoricals])
            logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
            return action.T, logprob.sum(0)