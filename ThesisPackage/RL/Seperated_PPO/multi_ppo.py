import argparse
import os
import io
import random
import time
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pettingzoo import ParallelEnv
from progressbar import progressbar
import zipfile
try:
    from agent import Agent
    from util import *
except:
    from ThesisPackage.RL.Seperated_PPO.agent import Agent
    from ThesisPackage.RL.Seperated_PPO.util import *


class PPO_Separate_Multi_Agent:
    def __init__(self, env, num_minibatches = 256, num_steps=2048, gae=True, gamma=0.99, gae_lambda=0.95, update_epochs=4, norm_adv=True, clip_coef=0.2, clip_vloss=True, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, target_kl=None, window_size=100, device="cpu"):
        self.env = env
        self.num_minibatches = num_minibatches
        self.device = device
        if isinstance(env, list):
            if isinstance(env[0], ParallelEnv):
                self.num_agents = len(env[0].agents)
                self.agents = {agent: Agent(env[0]).to(self.device) for agent in env[0].agents}
        elif isinstance(env, ParallelEnv):
            self.num_agents = len(env.agents)
            self.agents = {agent: Agent(env[0]).to(self.device) for agent in env[0].agents}
        else:
            raise ValueError("Env must be of type ParallelEnv or List[ParallelEnv]")
        self.num_steps = num_steps
        self.gae = gae
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.update_epochs = update_epochs
        self.norm_adv = norm_adv
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.window_size = window_size
        self.run_name = None

    def save(self, path: str):
        if '.pt' in path:
            path = path.replace('.pt', '')
        with zipfile.ZipFile(path + '.zip', 'w') as zipf:
            for agent in self.agents:
                torch.save(self.agents[agent].state_dict(), path + agent + ".pt")
                zipf.write(path + agent + ".pt")
                os.remove(path + agent + ".pt")

    def load(self, path: str):
        if '.zip' not in path:
            raise Exception("Provide ZIP file for multiple agents")
        zip_name = os.path.basename(path).replace('.zip', '')
        with zipfile.ZipFile(path, 'r') as zipf:
            self.agents = {}
            for file_name in zipf.namelist():
                if file_name.endswith(".pt"):
                    state_dict = zipf.open(file_name)
                    basename = os.path.basename(file_name)
                    agent_name = basename.replace(".pt", "")
                    agent_name = agent_name.replace(zip_name, "")
                    agent_model = Agent(self.env[0]).to(self.device)
                    agent_model.load_state_dict(torch.load(io.BytesIO(state_dict.read())))
                    self.agents[agent_name] = agent_model

    def calculate_advantages_returns(self, obs, rewards, dones, next_obs, next_done, values, agent):
        advantages = torch.zeros_like(rewards).to(self.device)
        returns = torch.zeros_like(rewards).to(self.device)
        
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if self.gae:
                lastgaelam = 0
                for t in reversed(range(self.num_steps)):
                    if t == self.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                for t in reversed(range(self.num_steps)):
                    if t == self.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + self.gamma * nextnonterminal * next_return
                advantages = returns - values
        
        return advantages, returns
    
    def optimize_policy_value_network(self, obs, logprobs, actions, advantages, returns, values, optimizer, agent, batch_size, minibatch_size):
        # flatten the batch
        b_obs = obs.reshape((-1,) + self.env[0].observation_space(agent).shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + self.env[0].action_space(agent).shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        random_losses = []
        for epoch in range(self.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.agents[agent].get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds].T)
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if self.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agents[agent].parameters(), self.max_grad_norm)
                optimizer.step()

            if self.target_kl is not None:
                if approx_kl > self.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        return clipfracs, random_losses, explained_var



    def train(self, total_timesteps, learning_rate=2.5e-4, anneal_lr=True, num_minibatches = 256, tensorboard_folder="results/", exp_name = None, seed = 1, torch_deterministic = True):
        """
        Trains the PPO agent.

        Args:
            total_timesteps (int): The total number of timesteps to train the agent.
            learning_rate (float, optional): The learning rate for the optimizer. Defaults to 2.5e-4.
            anneal_lr (bool, optional): Whether to anneal the learning rate during training. Defaults to True.
            num_minibatches (int, optional): The number of minibatches to use for optimization. Defaults to 256.
            exp_name (str, optional): The name of the experiment. Defaults to None.
            seed (int, optional): The random seed. Defaults to 1.
            torch_deterministic (bool, optional): Whether to set the random seed for PyTorch operations. Defaults to True.
        """

        if isinstance(self.env, list):
            num_envs = len(self.env)
        batch_size = int(num_envs * self.num_steps)
        minibatch_size = int(batch_size // num_minibatches)

        if exp_name is not None:
            run_name = f"Pong__{exp_name}__{seed}__{int(time.time())}"

            writer = SummaryWriter(f"{tensorboard_folder}/{run_name}")

        # TRY NOT TO MODIFY: seeding
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = torch_deterministic

        if isinstance(self.env, list):
            if isinstance(self.env[0], ParallelEnv):
                current_env = self.env[0]
        elif isinstance(self.env, ParallelEnv):
            current_env = self.env

        num_agents = len(current_env.agents)
        agent_names = current_env.agents

        optimizers = {}
        for agent in agent_names:
            optimizers[agent] = optim.Adam(self.agents[agent].parameters(), lr=learning_rate, eps=1e-5)

        obs = {agent: torch.zeros((self.num_steps, num_envs) + current_env.observation_space(agent).shape).to(self.device) for agent in current_env.agents}
        actions = {agent: torch.zeros((self.num_steps, num_envs) + current_env.action_space(agent).shape).to(self.device) for agent in current_env.agents}
        logprobs = {agent: torch.zeros((self.num_steps, num_envs)).to(self.device) for agent in current_env.agents}
        rewards = {agent: torch.zeros((self.num_steps, num_envs)).to(self.device) for agent in current_env.agents}
        dones = {agent: torch.zeros((self.num_steps, num_envs)).to(self.device) for agent in current_env.agents}
        values = {agent: torch.zeros((self.num_steps, num_envs)).to(self.device) for agent in current_env.agents}

        global_step = 0
        start_time = time.time()
        next_obs = [current_env.reset()[0] for current_env in self.env]
        next_obs = flatten_for_agents(next_obs, self.device)

        next_done = {agent: torch.zeros(num_envs).to(self.device) for agent in agent_names}
        num_updates = total_timesteps // batch_size

        rewards_record = []
        lengths_record = []

        lrnow = learning_rate

        for update in progressbar(range(1, num_updates + 1), redirect_stdout=True):
            # Annealing the rate if instructed to do so.
            if anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * learning_rate
                for agent in agent_names:
                    optimizers[agent].param_groups[0]["lr"] = lrnow

            for step in range(0, self.num_steps):
                global_step += 1 * num_envs
                for agent in agent_names:
                    obs[agent][step] = next_obs[agent]
                    dones[agent][step] = next_done[agent]

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action = {}
                    logprob = {}
                    for agent in agent_names:
                        cur_action, cur_logprob, _, value = self.agents[agent].get_action_and_value(next_obs[agent])
                        values[agent][step] = value.flatten()
                        logprob[agent] = cur_logprob
                        action[agent] = cur_action
                for agent in agent_names:
                    actions[agent][step] = action[agent]
                    logprobs[agent][step] = logprob[agent]

                next_obs = []
                current_rewards = []
                current_dones = []
                current_trunacted = []
                info = []
                for i, current_env in enumerate(self.env):
                    current_action = {agent: action[agent][i] for agent in agent_names}
                    new_obs, reward, terminated, truncated, info = current_env.step(current_action)
                    current_done = {key: terminated.get(key, False) or truncated.get(key, False) for key in set(terminated) | set(truncated)}
                    next_obs.append(new_obs)
                    current_rewards.append(reward)
                    current_dones.append(current_done)
                    current_trunacted.append(truncated)

                next_obs = flatten_for_agents(next_obs, self.device)
                current_rewards = flatten_for_agents(current_rewards, self.device)
                check_done = flatten_list(current_dones)
                next_done = flatten_for_agents(current_dones, self.device)

                # FIX THIS MESS!
                for agent in agent_names:
                    rewards[agent][step] = current_rewards[agent].flatten().view(-1)

                if any(check_done):
                    true_indices = np.nonzero(check_done)[0]
                    for index in true_indices:
                        if index % self.num_agents == 0:
                            lengths_record.append(self.env[int(index / self.num_agents)].timestep)
                            rewards_record.append(self.env[int(index / self.num_agents)].episode_rewards)
                            self.env[int(index / self.num_agents)].reset()

            returns = {}
            advantages = {}
            for agent in agent_names:
                current_advantages, current_returns = self.calculate_advantages_returns(obs[agent], rewards[agent], dones[agent], next_obs[agent], next_done[agent], values[agent], self.agents[agent])
                returns[agent] = current_returns
                advantages[agent] = current_advantages

            clipfracs = {}
            random_losses = {}
            explained_var = {}
            for agent in agent_names:
                current_clipfracs, current_random_losses, current_explained_var = self.optimize_policy_value_network(obs[agent], logprobs[agent], actions[agent], advantages[agent], returns[agent], values[agent], optimizers[agent], agent, batch_size, minibatch_size)
                clipfracs[agent] = current_clipfracs
                random_losses[agent] = current_random_losses
                explained_var[agent] = current_explained_var

            average_return = sum(rewards_record[-1 * min(self.window_size, len(rewards_record)):]) / min(self.window_size, len(rewards_record))
            average_length = sum(lengths_record[-1 * min(self.window_size, len(lengths_record)):]) / min(self.window_size, len(lengths_record))

            print("SPS:", int(global_step / (time.time() - start_time)),  "Average Return:", average_return, "Episode Length:", average_length)

            if exp_name is not None:
                writer.add_scalar("charts/episodic_return_average", average_return, global_step)
                writer.add_scalar("charts/episodic_return_median", np.median(rewards_record[-1 * min(self.window_size, len(rewards_record)):]), global_step)
                writer.add_scalar("charts/episodic_length_average", average_length, global_step)
                writer.add_scalar("charts/episodic_length_median", np.median(lengths_record[-1 * min(self.window_size, len(lengths_record)):]), global_step)
                writer.add_scalar("charts/learning_rate", lrnow, global_step)
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            del lengths_record[:-self.window_size]
            del rewards_record[:-self.window_size]

        if exp_name is not None:
            writer.close()