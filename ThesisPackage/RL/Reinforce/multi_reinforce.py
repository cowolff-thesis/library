import gym
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from pettingzoo import ParallelEnv
from torch.utils.tensorboard import SummaryWriter
import random
from progressbar import progressbar
import time
try:
    from agent import Agent
    from util import *
except:
    from ThesisPackage.RL.Reinforce.agent import Agent
    from ThesisPackage.RL.Reinforce.util import *

class Multi_Reinforce:

    def __init__(self, env, num_steps=2048,  window_size=100, device="cpu") -> None:
        self.env = env
        self.num_steps = num_steps
        self.window_size = window_size
        self.device = device

        if isinstance(env, list):
            self.num_envs = len(env)
            if isinstance(env[0], ParallelEnv):
                self.agent = Agent(env[0]).to(self.device)
                self.num_agents = len(env[0].agents)
                self.agents = env[0].agents
        elif isinstance(env, ParallelEnv):
            self.num_envs = 1
            self.agent = Agent(env).to(self.device)
            self.num_agents = len(env.agents)
            self.agents = env.agents
        else:
            raise ValueError("Env must be of type ParallelEnv or List[ParallelEnv]")
        
    def save(self, path):
        if "." not in path:
            path = path + ".pt"
        torch.save(self.agent.state_dict(), path)

    def train(self, total_timesteps, learning_rate=1e-4, anneal_lr=True,  tensorboard_folder="results/", exp_name = None, seed = 1, torch_deterministic = True, max_grad_norm=0.5):
        if isinstance(self.env, list):
            if isinstance(self.env[0], ParallelEnv):
                current_env = self.env[0]
        elif isinstance(self.env, ParallelEnv):
            current_env = self.env

        if exp_name is not None:
            run_name = f"Pong__{exp_name}__{seed}__{int(time.time())}"

            writer = SummaryWriter(f"{tensorboard_folder}/{run_name}")

        # TRY NOT TO MODIFY: seeding
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = torch_deterministic

        batch_size = int(self.num_envs * self.num_steps)
        num_updates = total_timesteps // batch_size

        optimizer = optim.Adam(self.agent.parameters(), lr=learning_rate)

        next_obs = [current_env.reset()[0] for current_env in self.env]
        next_obs = np.array(flatten_list(next_obs))
        next_obs = torch.Tensor(next_obs).to(self.device)

        global_step = 0
        start_time = time.time()

        rewards_record = [0]
        lengths_record = [0]

        start_time = time.time()

        rewards_record = []
        lengths_record = []

        for update in progressbar(range(1, num_updates + 1), redirect_stdout=True):

            log_probs = torch.zeros((self.num_steps, self.num_agents * self.num_envs)).to(self.device)
            rewards = torch.zeros((self.num_steps, self.num_agents * self.num_envs)).to(self.device)
            dones = torch.zeros((self.num_steps, self.num_agents * self.num_envs)).to(self.device)
            obs = torch.zeros((self.num_steps, self.num_agents * self.num_envs) + current_env.observation_space(current_env.agents[0]).shape).to(self.device)

            # Annealing the rate if instructed to do so.
            if anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, self.num_steps):
                global_step += 1 * self.num_envs

                obs[step] = next_obs

                action, log_prob = self.agent.get_action_and_probs(next_obs)

                action = action.cpu().numpy()
                action = reverse_flatten_list_with_agent_list(action, self.agents)

                next_obs = []
                current_rewards = []
                current_dones = []
                for current_env, current_action in zip(self.env, action):
                    current_obs, reward, terminated, truncated, info = current_env.step(current_action)
                    next_obs.append(current_obs)
                    current_rewards.append(reward)
                    current_done = {key: terminated.get(key, False) or truncated.get(key, False) for key in set(terminated) | set(truncated)}
                    current_dones.append(current_done)

                next_obs = np.array(flatten_list(next_obs))
                current_rewards = np.array(flatten_list(current_rewards))
                current_dones = np.array(flatten_list(current_dones))

                current_rewards = torch.tensor(current_rewards.flatten()).to(self.device).view(-1)
                next_obs = torch.Tensor(next_obs).to(self.device)

                log_probs[step] = log_prob
                rewards[step] = current_rewards

                if any(current_dones):
                    true_indices = np.nonzero(current_dones)[0]
                    for index in true_indices:
                        if index % self.num_agents == 0:
                            lengths_record.append(self.env[int(index / self.num_agents)].timestep)
                            rewards_record.append(self.env[int(index / self.num_agents)].episode_rewards)
                            self.env[int(index / self.num_agents)].reset()

                current_dones = torch.tensor(current_dones)
                dones[step] = current_dones

            b_rewards = rewards.reshape(-1)
            b_logprobs = log_probs.reshape(-1)
            b_dones = dones.reshape(-1)
            b_obs = obs.reshape((-1,) + self.env[0].observation_space(current_env.agents[0]).shape)

            returns = []
            R = 0
            i = 0
            for r, d in zip(torch.flip(b_rewards, dims=(0,)), torch.flip(b_dones, dims=(0,))):
                i += 1
                R = r + 0.99 * R
                returns.insert(0, R)
                if d or i % self.num_steps == 0:
                    R = 0

            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)

            policy_loss = torch.zeros(b_logprobs.shape)
            for i, (log_prob, R) in enumerate(zip(b_logprobs, returns)):
                policy_loss[i] = -log_prob * R

            optimizer.zero_grad()
            policy_loss = policy_loss.sum()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(self.agent.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                _, new_log_prob = self.agent.get_action_and_probs(b_obs)
                logratio = new_log_prob - b_logprobs
                ratio = logratio.exp()
                approx_kl = ((ratio - 1) - logratio).mean()

            average_return = sum(rewards_record[-1 * min(self.window_size, len(rewards_record)):]) / min(self.window_size, len(rewards_record))
            average_length = sum(lengths_record[-1 * min(self.window_size, len(lengths_record)):]) / min(self.window_size, len(lengths_record))

            print("SPS:", int(global_step / (time.time() - start_time)),  "Average Return:", average_return, "Episode Length:", average_length, "Loss:", policy_loss.item())

            if exp_name is not None:
                writer.add_scalar("charts/episodic_return_average", average_return, global_step)
                writer.add_scalar("charts/episodic_return_median", np.median(rewards_record[-1 * min(self.window_size, len(rewards_record)):]), global_step)
                writer.add_scalar("charts/episodic_length_average", average_length, global_step)
                writer.add_scalar("charts/episodic_length_median", np.median(lengths_record[-1 * min(self.window_size, len(lengths_record)):]), global_step)
                writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)

            del lengths_record[:-self.window_size]
            del rewards_record[:-self.window_size]