import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from gym.spaces import Box, Discrete, MultiDiscrete

class Agent(nn.Module):
    def __init__(self, env, vae_latent_dim):
        super(Agent, self).__init__()

        self.action_space = env.action_space(env.agents[0])
        if isinstance(self.action_space, Box):
            self.n = self.action_space.shape[0]
        elif isinstance(self.action_space, Discrete):
            self.n = self.action_space.n
        elif isinstance(self.action_space, MultiDiscrete):
            self.n = self.action_space.nvec.sum()
       
        self.critic = nn.Sequential(
            self.layer_init(nn.Linear(vae_latent_dim, 128)),
            nn.Tanh(),
            self.layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            self.layer_init(nn.Linear(vae_latent_dim, 128)),
            nn.Tanh(),
            self.layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, self.n), std=0.01),
        )
        if isinstance(self.action_space, Box):
            self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(self.action_space.shape)))

    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        if isinstance(self.action_space, MultiDiscrete):
            split_logits = torch.split(logits, self.action_space.nvec.tolist(), dim=1)
            multi_categoricals = [Categorical(logits=logits) for logits in split_logits]
            if action is None:
                action = torch.stack([categorical.sample() for categorical in multi_categoricals])
            logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
            entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
            return action.T, logprob.sum(0), entropy.sum(0), self.critic(x)
        elif isinstance(self.action_space, Discrete):
            categorical = Categorical(logits=logits)
            if action is None:
                action = categorical.sample()
            logprob = categorical.log_prob(action)
            entropy = categorical.entropy()
            return action, logprob, entropy, self.critic(x)
        elif isinstance(self.action_space, Box):
            action_mean = self.actor(x)
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            probs = Normal(action_mean, action_std)
            if action is None:
                action = probs.sample()
            return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
            