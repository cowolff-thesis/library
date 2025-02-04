import argparse
import os
import random
import time
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from gymnasium.spaces import Box
from pettingzoo import ParallelEnv
from progressbar import progressbar
try:
    from agent import Agent
    from util import *
    from vae import VAE, vae_loss_function
except:
    from ThesisPackage.RL.Worldmodel_PPO.agent import Agent
    from ThesisPackage.RL.Worldmodel_PPO.util import *
    from ThesisPackage.RL.Worldmodel_PPO.vae import VAE, vae_loss_function


class PPO_World_Model:
    def __init__(self, env, num_vae_batches=16, vae_stack=4, vae_latent_dim=32, total_vae_updates=-1, vae_beta=0.5, start_vae_epochs=10, end_vae_epochs=2, num_minibatches = 256, num_steps=2048, gae=True, gamma=0.99, gae_lambda=0.95, update_epochs=4, norm_adv=True, clip_coef=0.2, clip_vloss=True, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, target_kl=None, window_size=100, device="cpu"):
        self.env = env
        self.num_minibatches = num_minibatches
        self.device = device
        if isinstance(env, list):
            if isinstance(env[0], ParallelEnv):
                self.agent = Agent(env[0], vae_latent_dim).to(self.device)
                self.num_agents = len(env[0].agents)
                self.agents = env[0].agents
                self.vae = VAE(env[0], vae_stack, vae_latent_dim).to(self.device)
        elif isinstance(env, ParallelEnv):
            self.agent = Agent(env, vae_latent_dim).to(self.device)
            self.num_agents = len(env.agents)
            self.agents = env.agents
            self.vae = VAE(env, vae_stack, vae_latent_dim).to(self.device)
        else:
            raise ValueError("Env must be of type ParallelEnv or List[ParallelEnv]")
        self.total_vae_updates = total_vae_updates
        self.vae_stack = vae_stack
        self.vae_beta = vae_beta
        self.vae_latent_dim = vae_latent_dim
        self.start_vae_epochs = start_vae_epochs
        self.end_vae_epochs = end_vae_epochs
        self.num_vae_batches = num_vae_batches
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

    def save(self, path):
        if "." not in path:
            path = path + ".pt"
        torch.save(self.agent.state_dict(), path)

    def train(self, total_timesteps, learning_rate=2.5e-4, vae_lr=1e-3, anneal_lr=True, num_minibatches = 256, tensorboard_folder="results/", exp_name = None, seed = 1, torch_deterministic = True, vae_path=None):
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

        optimizer = optim.Adam(self.agent.parameters(), lr=learning_rate, eps=1e-5)
        vae_optimizer = optim.Adam(self.vae.parameters(), lr=vae_lr)

        if vae_path is not None:
            self.vae.load_state_dict(torch.load(vae_path))

        current_vae_epochs = self.start_vae_epochs

        if isinstance(self.env, list):
            if isinstance(self.env[0], ParallelEnv):
                current_env = self.env[0]
        elif isinstance(self.env, ParallelEnv):
            current_env = self.env

        num_agents = len(current_env.agents)

        obs_space = current_env.observation_space(current_env.agents[0])
        min_values = np.array(obs_space.low)
        max_values = np.array(obs_space.high)

        obs = torch.zeros((self.num_steps, num_agents * num_envs, self.vae_latent_dim)).to(self.device)
        actions = torch.zeros((self.num_steps, num_agents * num_envs) + current_env.action_space(current_env.agents[0]).shape).to(self.device)
        logprobs = torch.zeros((self.num_steps, num_agents * num_envs)).to(self.device)
        rewards = torch.zeros((self.num_steps, num_agents * num_envs)).to(self.device)
        dones = torch.zeros((self.num_steps, num_agents * num_envs)).to(self.device)
        values = torch.zeros((self.num_steps, num_agents * num_envs)).to(self.device)

        cur_vae_obs = torch.zeros((self.vae_stack, num_agents * num_envs) + current_env.observation_space(current_env.agents[0]).shape).to(self.device)
        vae_obs = torch.zeros((self.num_steps, num_agents * num_envs) + current_env.observation_space(current_env.agents[0]).shape).to(self.device)

        global_step = 0
        start_time = time.time()
        next_obs = [current_env.reset()[0] for current_env in self.env]
        next_obs = normalize_batch_observations(np.array(flatten_list(next_obs)), min_values, max_values)
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.zeros(num_agents * num_envs).to(self.device)
        num_updates = total_timesteps // batch_size

        rewards_record = []
        lengths_record = []

        for update in progressbar(range(1, num_updates + 1), redirect_stdout=True):
            # Annealing the rate if instructed to do so.
            if anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            for pretrain_step in range(0, self.vae_stack):
                cur_vae_obs = cur_vae_obs[:-1]
                cur_vae_obs = torch.cat((cur_vae_obs, next_obs.unsqueeze(0)), 0)
                next_obs = []
                current_rewards = []
                current_dones = []
                current_trunacted = []
                info = []
                for current_env in self.env:
                    new_obs, reward, terminated, truncated, info = current_env.step({agent: current_env.action_space(agent).sample() for agent in current_env.agents})
                    current_done = {key: terminated.get(key, False) or truncated.get(key, False) for key in set(terminated) | set(truncated)}
                    next_obs.append(new_obs)
                    current_rewards.append(reward)
                    current_dones.append(current_done)
                    current_trunacted.append(truncated)

                next_obs = normalize_batch_observations(np.array(flatten_list(next_obs)), min_values, max_values)
                current_rewards = np.array(flatten_list(current_rewards))
                current_dones = np.array(flatten_list(current_dones))

                rewards[pretrain_step] = torch.tensor(current_rewards.flatten()).to(self.device).view(-1)
                next_obs = np.array(next_obs)
                next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(current_dones).to(self.device)

                if any(current_dones):
                    true_indices = np.nonzero(current_dones)[0]
                    for index in true_indices:
                        if index % self.num_agents == 0:
                            self.env[int(index / self.num_agents)].reset()


            with torch.no_grad():
                encoded = self.vae.represent(cur_vae_obs.view(len(self.env) * self.num_agents, self.vae.input_dim))
            for step in range(0, self.num_steps):
                global_step += 1 * num_envs
                obs[step] = encoded
                vae_obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(encoded)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                action = action.cpu().numpy()
                action = reverse_flatten_list_with_agent_list(action, self.agents)

                next_obs = []
                current_rewards = []
                current_dones = []
                current_trunacted = []
                info = []
                for current_env, current_action in zip(self.env, action):
                    new_obs, reward, terminated, truncated, info = current_env.step(current_action)
                    current_done = {key: terminated.get(key, False) or truncated.get(key, False) for key in set(terminated) | set(truncated)}
                    next_obs.append(new_obs)
                    current_rewards.append(reward)
                    current_dones.append(current_done)
                    current_trunacted.append(truncated)

                next_obs = normalize_batch_observations(np.array(flatten_list(next_obs)), min_values, max_values)
                current_rewards = np.array(flatten_list(current_rewards))
                current_dones = np.array(flatten_list(current_dones))

                rewards[step] = torch.tensor(current_rewards.flatten()).to(self.device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(current_dones).to(self.device)

                with torch.no_grad():
                    cur_vae_obs = cur_vae_obs[:-1]
                    cur_vae_obs = torch.cat((cur_vae_obs, next_obs.unsqueeze(0)), 0)
                    encoded = self.vae.represent(cur_vae_obs.view(len(self.env) * self.num_agents, self.vae.input_dim))

                if any(current_dones):
                    true_indices = np.nonzero(current_dones)[0]
                    for index in true_indices:
                        if index % self.num_agents == 0:
                            lengths_record.append(self.env[int(index / self.num_agents)].timestep)
                            rewards_record.append(self.env[int(index / self.num_agents)].episode_rewards)
                            self.env[int(index / self.num_agents)].reset()

            # bootstrap value if not done
            with torch.no_grad():
                next_value = self.agent.get_value(encoded).reshape(1, -1)
                if self.gae:
                    advantages = torch.zeros_like(rewards).to(self.device)
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
                    returns = torch.zeros_like(rewards).to(self.device)
                    for t in reversed(range(self.num_steps)):
                        if t == self.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            next_return = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            next_return = returns[t + 1]
                        returns[t] = rewards[t] + self.gamma * nextnonterminal * next_return
                    advantages = returns - values

            # flatten the batch
            b_obs = obs.reshape((-1, self.vae_latent_dim))
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + self.env[0].action_space(current_env.agents[0]).shape)
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
                    
                    if isinstance(self.agent.action_space, Box):
                        _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    else:
                        _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds].T)
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
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                    optimizer.step()

                if self.target_kl is not None:
                    if approx_kl > self.target_kl:
                        break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # Update VAE
            if (self.total_vae_updates > 0 and update < self.total_vae_updates) and vae_path is None:
                b_vae_obs = vae_obs.view(-1, self.vae.input_dim)
                n, m = b_vae_obs.shape
                new_batches = n // self.vae_stack  # Integer division to find the maximum complete groups

                # Truncate the tensor to fit the new structure exactly
                b_vae_obs = b_vae_obs[:new_batches * self.vae_stack]

                vae_batch_size = int(b_vae_obs.shape[0] // self.num_vae_batches)

                for epoch in range(current_vae_epochs):
                    total_loss = 0
                    for start in range(0, b_vae_obs.shape[0], vae_batch_size):
                        end = start + vae_batch_size
                        b_vae_obs_batch = b_vae_obs[start:end]
                        recon_obs, mu, log_var = self.vae(b_vae_obs_batch)
                        vae_loss, BCE, KL = vae_loss_function(recon_obs, b_vae_obs_batch, mu, log_var, beta=self.vae_beta)
                        total_loss += vae_loss.item()
                        vae_optimizer.zero_grad()
                        vae_loss.backward()
                        vae_optimizer.step()
                    print("VAE Update Epoch:", epoch, "VAE Loss:", total_loss)
                print("Reconstructed:", recon_obs[0], "\n", "Original:", b_vae_obs_batch[0])
                print("BCE:", BCE.item(), "KL:", KL.item())

            if current_vae_epochs > self.end_vae_epochs:
                current_vae_epochs -= 1

            if len(rewards_record) > 0:
                average_return = sum(rewards_record[-1 * min(self.window_size, len(rewards_record)):]) / min(self.window_size, len(rewards_record))
                average_length = sum(lengths_record[-1 * min(self.window_size, len(lengths_record)):]) / min(self.window_size, len(lengths_record))
            else:
                average_return = 0
                average_length = 0

            print("SPS:", int(global_step / (time.time() - start_time)),  "Average Return:", average_return, "Episode Length:", average_length)

            if exp_name is not None:
                writer.add_scalar("charts/episodic_return_average", average_return, global_step)
                writer.add_scalar("charts/episodic_return_median", np.median(rewards_record[-1 * min(self.window_size, len(rewards_record)):]), global_step)
                writer.add_scalar("charts/episodic_length_average", average_length, global_step)
                writer.add_scalar("charts/episodic_length_median", np.median(lengths_record[-1 * min(self.window_size, len(lengths_record)):]), global_step)
                writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("losses/random_action_loss", np.mean(random_losses), global_step)
                writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
                writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
                writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
                writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
                writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
                writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
                writer.add_scalar("losses/explained_variance", explained_var, global_step)
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            del lengths_record[:-self.window_size]
            del rewards_record[:-self.window_size]

        if exp_name is not None:
            writer.close()