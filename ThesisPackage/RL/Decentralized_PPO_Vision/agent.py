import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical

class Agent(nn.Module):
    def __init__(self, env, embedding_space=32):
        super(Agent, self).__init__()
        self.nvec = env.action_space(env.agents[0]).nvec
        self.utterances = env.observation_space(env.agents[0])["utterance"].shape[0]

        self.vision = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            self.layer_init(nn.Linear(32 * 23 * 23, embedding_space))
        )
        self.critic = nn.Sequential(
            self.layer_init(nn.Linear(embedding_space + self.utterances, 128)),
            nn.Tanh(),
            self.layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            self.layer_init(nn.Linear(embedding_space + self.utterances, 128)),
            nn.Tanh(),
            self.layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, self.nvec.sum()), std=0.01),
        )

    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, utterance, image):
        image = image.permute(0, 3, 1, 2)
        hidden = self.vision(image / 255.0)
        hidden = torch.cat((hidden, utterance), dim=1)
        return self.critic(hidden)

    def get_action_and_value(self, utterance, image, action=None):
        image = image.permute(0, 3, 1, 2)
        hidden = self.vision(image / 255.0)
        hidden = torch.cat((hidden, utterance), dim=1)
        logits = self.actor(hidden)
        split_logits = torch.split(logits, self.nvec.tolist(), dim=1)
        multi_categoricals = [Categorical(logits=logits) for logits in split_logits]
        if action is None:
            action = torch.stack([categorical.sample() for categorical in multi_categoricals])
        logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
        entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
        # print(self.critic(state).shape)
        return action.T, logprob.sum(0), entropy.sum(0), self.critic(hidden)