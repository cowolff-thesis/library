import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from gym.spaces import Box, Discrete, MultiDiscrete, Tuple

def handle_multidiscrete(actor, critic, action_space, hiddens, lstm_state, action=None):
    logits = actor(hiddens)
    split_logits = torch.split(logits, action_space.nvec.tolist(), dim=1)
    multi_categoricals = [Categorical(logits=logits) for logits in split_logits]
    if action is None:
        action = torch.stack([categorical.sample() for categorical in multi_categoricals])
    logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
    entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
    return action.T, logprob.sum(0), entropy.sum(0), critic(hiddens), lstm_state

def handle_discrete(actor, critic, hiddens, lstm_state, action=None):
    logits = actor(hiddens)
    categorical = Categorical(logits=logits)
    if action is None:
        action = categorical.sample()
    logprob = categorical.log_prob(action)
    entropy = categorical.entropy()
    return action, logprob, entropy, critic(hiddens), lstm_state


def handle_box(actor, critic, actor_logstd, hiddens, lstm_state, action=None):
    action_mean = actor(hiddens)
    action_logstd = actor_logstd.expand_as(action_mean)
    action_std = torch.exp(action_logstd)
    probs = Normal(action_mean, action_std)
    if action is None:
        action = probs.sample()
    return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), critic(hiddens), lstm_state

def _process_discrete_space(logits, actions, logprobs, entropies, space, action, offset, action_offset):
    categorical = Categorical(logits=logits[:, offset:offset + space.n])
    if action is None:
        cur_action = categorical.sample()
    else:
        cur_action = action[:, action_offset:action_offset + space.n]
        action_offset += space.n
    actions.append(cur_action)
    logprobs.append(categorical.log_prob(cur_action))
    entropies.append(categorical.entropy())
    offset += space.n
    return actions, logprobs, entropies, offset, action_offset

def _process_box_space(actor_logstd, logits, actions, logprobs, entropies, space, action, offset, logstd_offset, action_offset):
    action_mean = logits[:, offset:offset + space.shape[0]]
    action_logstd = actor_logstd[:, logstd_offset:logstd_offset + space.shape[0]]
    action_std = torch.exp(action_logstd)
    probs = Normal(action_mean, action_std)
    if action is None:
        cur_action = probs.sample()
    else:
        cur_action = action[:, action_offset:action_offset + space.shape[0]]
        action_offset += space.shape[0]
    actions.append(cur_action)
    logprobs.append(probs.log_prob(cur_action).sum(1))
    entropies.append(probs.entropy().sum(1))
    offset += space.shape[0]
    logstd_offset += space.shape[0]
    return actions, logprobs, entropies, offset, logstd_offset, action_offset

def _process_multidiscrete_space(logits, actions, logprobs, entropies, space, action, offset, action_offset):
    cur_logits = logits[:, offset:offset + space.nvec.sum()]
    split_logits = torch.split(cur_logits, space.nvec.tolist(), dim=1)
    multi_categoricals = [Categorical(logits=logits) for logits in split_logits]
    if action is None:
        cur_action = torch.stack([categorical.sample() for categorical in multi_categoricals])
    else:
        cur_action = action[:, action_offset:action_offset + space.nvec.sum()].T
        action_offset += space.nvec.sum()
    current_logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(cur_action, multi_categoricals)])
    current_entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
    actions.append(cur_action.T)
    logprobs.append(current_logprob.sum(0))
    entropies.append(current_entropy.sum(0))
    offset += space.nvec.sum()
    return actions, logprobs, entropies, offset, action_offset


def handle_tuple(actor, critic, action_space, actor_logstd, hiddens, action=None):
    logits = actor(hiddens)
    actions = []
    logprobs = []
    entropies = []
    offset = 0
    logstd_offset = 0
    action_offset = 0
    
    for space in action_space:
        if isinstance(space, Discrete):
            actions, logprobs, entropies, offset, action_offset = _process_discrete_space(
                logits, actions, logprobs, entropies, space, action, offset, action_offset
            )
        elif isinstance(space, Box):
            actions, logprobs, entropies, offset, logstd_offset, action_offset = _process_box_space(
                actor_logstd, logits, actions, logprobs, entropies, space, action, offset, logstd_offset, action_offset
            )
        elif isinstance(space, MultiDiscrete):
            actions, logprobs, entropies, offset, action_offset = _process_multidiscrete_space(
                logits, actions, logprobs, entropies, space, action, offset, action_offset
            )
    
    actions = torch.cat(actions, dim=1)
    logprobs = torch.stack(logprobs).sum(0)
    entropies = torch.stack(entropies).sum(0)
    return actions, logprobs, entropies, critic(hiddens)
