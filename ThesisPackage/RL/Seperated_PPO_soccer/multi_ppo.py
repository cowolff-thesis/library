import argparse
import os
import random
import time
import gym
import gym.spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from gymnasium.spaces import Box
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers
from progressbar import progressbar
import zipfile
import json

from ThesisPackage.RL.Decentralized_PPO.agent import Agent
from ThesisPackage.RL.Decentralized_PPO.util import *
from ThesisPackage.Wrappers.vecWrapper import PettingZooVectorizationParallelWrapper


def update_agent(agent, obs, actions, logprobs, rewards, dones, values, env, next_obs, next_done, num_steps, batch_size, minibatch_size, optimizer, gamma, gae_lambda, update_epochs, norm_adv, clip_coef, clip_vloss, ent_coef, vf_coef, max_grad_norm, target_kl, action_space, num_actions, device, gae=True):
        # bootstrap value if not done
    if isinstance(env, PettingZooVectorizationParallelWrapper):
        current_env = env.env
    elif isinstance(env, ParallelEnv) or isinstance(env[0], wrappers.BaseWrapper):
        current_env = env
    
    with torch.no_grad():
        next_value = agent.get_value(next_obs).reshape(1, -1)
        if gae:
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
        else:
            returns = torch.zeros_like(rewards).to(device)
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    next_return = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    next_return = returns[t + 1]
                returns[t] = rewards[t] + gamma * nextnonterminal * next_return
            advantages = returns - values

    # flatten the batch
    if isinstance(env, list):
        b_obs = obs.reshape((-1,) + env[0].observation_space(current_env.agents[0]).shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + env[0].action_space(current_env.agents[0]).shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
    elif isinstance(env, PettingZooVectorizationParallelWrapper):
        b_obs = obs.reshape((-1,) + env.env.observation_space(current_env.agents[0]).shape)
        b_logprobs = logprobs.reshape(-1)
        if isinstance(action_space, gym.spaces.Tuple):
            b_actions = actions.reshape((-1,) + num_actions)
        else:
            b_actions = actions.reshape((-1,) + env.env.action_space(current_env.agents[0]).shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

    # Optimizing the policy and value network
    b_inds = np.arange(batch_size)
    clipfracs = []
    for epoch in range(update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_inds = b_inds[start:end]
            
            if isinstance(agent.action_space, gym.spaces.Box):
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
            elif isinstance(agent.action_space, gym.spaces.Tuple):
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
            else:
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds].T)
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

            mb_advantages = b_advantages[mb_inds]
            if norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = newvalue.view(-1)
            if clip_vloss:
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -clip_coef,
                    clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()

        if target_kl is not None:
            if approx_kl > target_kl:
                break

    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    return agent, pg_loss, v_loss, entropy_loss, approx_kl, old_approx_kl, clipfracs, explained_var

def team_split(data, team_agents, teams):
    split_data = {team: concatenate_agent_observations({agent: torch.Tensor(content) for agent, content in data.items() if agent in team_agents[team]}) for team in teams}
    return split_data

class PPO_Multi_Agent:
    def __init__(self, env, test_env=None, num_minibatches = 256, num_steps=2048, gae=True, gamma=0.99, gae_lambda=0.95, update_epochs=4, normalize_obs=False, norm_adv=True, clip_coef=0.2, clip_vloss=True, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, target_kl=None, window_size=100, device="cpu"):
        self.env = env
        self.test_env = test_env
        self.num_minibatches = num_minibatches
        self.device = device
        if isinstance(env, list):
            if isinstance(env[0], ParallelEnv) or isinstance(env[0], wrappers.BaseWrapper):
                self.team_agents = {team: [] for team in env[0].teams}
                self.teams = env[0].teams
                self.agent_names = env[0].agents
                for agent in self.agent_names:
                    current_team = env[0].players[agent].team
                    self.team_agents[current_team].append(agent)
                self.agents = {team: Agent(env[0]).to(self.device) for team in env[0].teams}
        elif isinstance(env, PettingZooVectorizationParallelWrapper):
            current_env = env.env
            self.teams = current_env.env.env.teams
            self.team_agents = {team: [] for team in self.teams}
            self.agent_names = current_env.agents
            for agent in self.agent_names:
                current_team = current_env.env.env.players[agent].team
                self.team_agents[current_team].append(agent)
            self.agents = {team: Agent(current_env).to(self.device) for team in self.teams}
        elif isinstance(env, ParallelEnv) or isinstance(env[0], wrappers.BaseWrapper):
            self.team_agents = {team: [] for team in env.teams}
            self.teams = env.teams
            self.agent_names = env.agents
            for agent in self.agent_names:
                current_team = env.players[agent].team
                self.team_agents[current_team].append(agent)
            self.agents = {team: Agent(env).to(self.device) for team in env.teams}
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
        self.normalize_obs = normalize_obs

    def get_action(self, obs):
        """
        Get actions from the agent.

        Args:
        - obs (dict): Dictionary containing observations for each agent

        Returns:
        - action (dict): Dictionary containing actions for each agent
        """
        action = {}
        one_dim = False
        # If obs has only one dimension, expand by one
        if len(list(obs.values())[0].shape) == 1:
            one_dim = True
            obs = {player: np.expand_dims(obs[player], axis=0) for player in self.agent_names}
        obs = team_split(obs, self.team_agents, self.teams)
        obs = {team: torch.Tensor(obs[team]).to(self.device) for team in self.teams}
        with torch.no_grad():
            for team in self.teams:
                action[team], _, _, _ = self.agents[team].get_action_and_value(obs[team])
        # Create action dict for environment step from both teams
        action_dict = {}
        for team in self.teams:
            team_agent_actions = split_agent_actions(action[team], self.team_agents[team])
            action_dict.update(team_agent_actions)
        
        # Convert to numpy
        action_dict = {agent: action_dict[agent].cpu().numpy() for agent in action_dict.keys()}
        if one_dim:
            action_dict = {agent: action_dict[agent][0] for agent in action_dict.keys()}
        return action_dict

    def save_models_to_zip(self, zip_filename):
        """
        Save multiple PyTorch models into a ZIP file.

        Args:
        - models (list of tuples): List containing tuples of (model_name, model_instance)
        - zip_filename (str): Filename of the ZIP file where models will be saved
        """
        with zipfile.ZipFile(zip_filename, 'w') as zf:
            for team_name, model in self.agents.items():
                # Save each model's state_dict to a temporary file
                model_filename = f"{zip_filename}_{team_name}.pt"
                torch.save(model.state_dict(), model_filename)
                # Add the model file to the ZIP archive
                zf.write(model_filename)
                # Remove the temporary model file
                os.remove(model_filename)


    def load_models_from_zip(self, zip_filename):
        """
        Load multiple PyTorch models from a ZIP file.

        Args:
        - zip_filename (str): Filename of the ZIP file containing models
        """
        with zipfile.ZipFile(zip_filename, 'r') as zf:
            # Load models
            for info in zf.infolist():
                if info.filename.endswith('.pt'):
                    # Extract model name from filename
                    team_name = info.filename[:-3].split("_")[-1]  # Remove '.pt' extension
                    # Load model state_dict
                    with zf.open(info) as f:
                        state_dict = torch.load(f, map_location=torch.device(self.device))
                    # Create model instance and load state_dict
                    # Replace with your model creation logic
                    self.agents[int(team_name)].load_state_dict(state_dict)

    def test(self, num_episodes=50):
        next_obs, infos = self.test_env.reset()
        wins = []
        for episode in range(num_episodes):
            while True:
                next_obs = {player: np.expand_dims(next_obs[player], axis=0) for player in self.agent_names}
                next_obs = team_split(next_obs, self.team_agents, self.teams)
                next_obs = {team: torch.tensor(next_obs[team]).to(self.device) for team in self.teams}
                with torch.no_grad():
                    action_dict = {}
                    for team in self.teams:
                        action, _, _, _ = self.agents[team].get_action_and_value(next_obs[team])
                        action = action.cpu().numpy()
                        team_agent_actions = split_agent_actions(action, self.team_agents[team])
                        action_dict.update(team_agent_actions)
                action_dict = {agent: action_dict[agent][0] for agent in action_dict}
                next_obs, _, truncations, terminations, infos = self.test_env.step(action_dict)
                if any([truncations[agent] or terminations[agent] for agent in self.test_env.agents]):
                    teams = [infos[agent]["team"] for agent in self.test_env.agents if "team" in infos[agent]]
                    if len(teams) > 0:
                        wins.append(teams[0])
                    else:
                        wins.append(0)
                    next_obs, infos = self.test_env.reset()
                    break
        return wins

    def train(self, total_timesteps, learning_rate=2.5e-4, anneal_lr=True, num_minibatches = 256, tensorboard_folder="results/", exp_name = None, seed = 1, torch_deterministic = True, lr_auto_adjust = False):
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

        if anneal_lr and lr_auto_adjust:
            raise ValueError("Cannot have both anneal_lr and lr_auto_adjust set to True.")

        if isinstance(self.env, list):
            num_envs = len(self.env)
        elif isinstance(self.env, PettingZooVectorizationParallelWrapper):
            num_envs = self.env.num_envs
        else:
            num_envs = 1
        batch_size = int(num_envs * self.num_steps)
        minibatch_size = int(batch_size // num_minibatches)

        if exp_name is not None:
            run_name = f"{exp_name}__{seed}__{int(time.time())}"

            writer = SummaryWriter(f"{tensorboard_folder}/{run_name}")

        # TRY NOT TO MODIFY: seeding
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = torch_deterministic

        optimizers = {team: optim.Adam(self.agents[team].parameters(), lr=learning_rate, eps=1e-5) for team in self.teams}

        if isinstance(self.env, PettingZooVectorizationParallelWrapper):
            current_env = self.env.env
        elif isinstance(self.env, ParallelEnv) or isinstance(self.env[0], wrappers.BaseWrapper):
            current_env = self.env
        
        team_agents = self.team_agents

        action_space = current_env.action_space(current_env.agents[0])
        if isinstance(action_space, gym.spaces.Box) or isinstance(action_space, gym.spaces.Discrete) or isinstance(action_space, gym.spaces.MultiDiscrete):
            num_actions = current_env.action_space(current_env.agents[0]).shape
        elif isinstance(action_space, gym.spaces.Tuple):
            num_actions = 0
            for space in action_space:
                if isinstance(space, gym.spaces.Box):
                    num_actions += space.shape[0]
                elif isinstance(space, gym.spaces.Discrete):
                    num_actions += space.shape[0]
                elif isinstance(space, gym.spaces.MultiDiscrete):
                    num_actions += space.shape[0]
            num_actions = (num_actions,)

        obs, actions, logprobs, rewards, dones, values = {}, {}, {}, {}, {}, {}
        for team in self.teams:
            obs[team] = torch.zeros((self.num_steps, len(team_agents[team]) * num_envs) + current_env.observation_space(current_env.agents[0]).shape).to(self.device)
            actions[team] = torch.zeros((self.num_steps, len(team_agents[team]) * num_envs) + num_actions).to(self.device)
            logprobs[team] = torch.zeros((self.num_steps, len(team_agents[team]) * num_envs)).to(self.device)
            rewards[team] = torch.zeros((self.num_steps, len(team_agents[team]) * num_envs)).to(self.device)
            dones[team] = torch.zeros((self.num_steps, len(team_agents[team]) * num_envs)).to(self.device)
            values[team] = torch.zeros((self.num_steps, len(team_agents[team]) * num_envs)).to(self.device)

        global_step = 0
        start_time = time.time()

        next_obs, next_infos = current_env.reset()
        
        next_obs = team_split(next_obs, self.team_agents, self.teams)
        next_obs = {team: torch.Tensor(next_obs[team]).to(self.device) for team in self.teams}
        
        next_done = {team: torch.zeros(len(team_agents[team]) * num_envs).to(self.device) for team in self.teams}

        num_updates = total_timesteps // batch_size

        rewards_record = []
        lengths_record = []

        clipfracs = []

        team_wins = []

        approx_kl = {team: torch.tensor(0.01).to(self.device) for team in self.teams}

        for update in progressbar(range(1, num_updates + 1), redirect_stdout=True):
            # Annealing the rate if instructed to do so.
            if anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * learning_rate
                for optimizer in optimizers.values():
                    optimizer.param_groups[0]["lr"] = lrnow

            if lr_auto_adjust:
                for team in self.teams:
                    if approx_kl[team].item() > 0.01:
                        optimizers[team].param_groups[0]["lr"] /= 1.5
                    if approx_kl[team].item() < 0.003:
                        optimizers[team].param_groups[0]["lr"] *= 1.5

            for step in range(0, self.num_steps):
                global_step += 1 * num_envs
                for team in self.teams:
                    obs[team][step] = next_obs[team]
                    dones[team][step] = next_done[team]

                # ALGO LOGIC: action logic
                cur_actions = {}
                with torch.no_grad():
                    for team in self.teams:
                        action, logprob, _, value = self.agents[team].get_action_and_value(next_obs[team])
                        values[team][step] = value.flatten()
                        logprobs[team][step] = logprob
                        actions[team][step] = action
                        cur_actions[team] = action.cpu().numpy()

                # Create action dict for environment step from both teams
                action_dict = {}
                for team in self.teams:
                    team_agent_actions = split_agent_actions(cur_actions[team], team_agents[team])
                    action_dict.update(team_agent_actions)
                
                new_obs, reward, terminated, truncated, info = current_env.step(action_dict)
                
                terminated_teams = team_split(terminated, self.team_agents, self.teams)
                truncated_teams = team_split(truncated, self.team_agents, self.teams)
                next_done = {}
                for team in self.teams:
                    next_done[team] = torch.Tensor([terminated or truncated for terminated, truncated in zip(terminated_teams[team], truncated_teams[team])]).to(self.device)
                terminated = concatenate_agent_observations(terminated)
                truncated = concatenate_agent_observations(truncated)
                current_dones = np.array([terminated or truncated for terminated, truncated in zip(terminated, truncated)])
                next_obs = team_split(new_obs, self.team_agents, self.teams)
                current_rewards = team_split(reward, self.team_agents, self.teams)
                info= concatenate_agent_observations(info)

                for team in self.teams:
                    rewards[team][step] = torch.tensor(current_rewards[team].flatten()).to(self.device).view(-1)
                    next_obs[team], next_done[team] = torch.Tensor(next_obs[team]).to(self.device), torch.Tensor(next_done[team]).to(self.device)

                if any(current_dones):
                    true_indices = np.nonzero(current_dones)[0]
                    for cur_info in info[true_indices]:
                        if "team" in cur_info:
                            team_wins.append(cur_info["team"])
                        lengths_record.append(cur_info["timestep"])
                        rewards_record.append(cur_info["rewards"])

            pg_loss, v_loss, entropy_loss, approx_kl, old_approx_kl, clipfracs, explained_var = {}, {}, {}, {}, {}, {}, {}
            for team in self.teams:
                self.agents[team], pg_loss[team], v_loss[team], entropy_loss[team], approx_kl[team], old_approx_kl[team], clipfracs[team], explained_var[team] = update_agent(self.agents[team], obs[team], actions[team], logprobs[team], rewards[team], dones[team], values[team], self.env, next_obs[team], next_done[team], self.num_steps, batch_size, minibatch_size, optimizers[team], self.gamma, self.gae_lambda, self.update_epochs, self.norm_adv, self.clip_coef, self.clip_vloss, self.ent_coef, self.vf_coef, self.max_grad_norm, self.target_kl, action_space, num_actions, self.device, self.gae)

            if len(rewards_record) > 0:
                median_return = np.median(rewards_record[-1 * min(self.window_size, len(rewards_record)):])
                median_length = np.median(lengths_record[-1 * min(self.window_size, len(lengths_record)):])
            else:
                median_return = 0
                median_length = 0

            print("SPS:", int(global_step / (time.time() - start_time)),  "Median Return:", median_return, "Median Episode Length:", median_length, "Timestep", global_step)

            if self.test_env is not None:
                test_wins = self.test()
                test_one = test_wins.count(1)
                test_two = test_wins.count(-1)

            team_one = team_wins.count(1)
            team_two = team_wins.count(-1)


            if exp_name is not None:
                writer.add_scalar("charts/episodic_return_median", median_return, global_step)
                writer.add_scalar("charts/episodic_length_median", median_length, global_step)
                for team in self.teams:
                    writer.add_scalar(f"charts/{team}/learning_rate", optimizers[team].param_groups[0]["lr"], global_step)
                    writer.add_scalar(f"losses/{team}/value_loss", v_loss[team].item(), global_step)
                    writer.add_scalar(f"losses/{team}/policy_loss", pg_loss[team].item(), global_step)
                    writer.add_scalar(f"losses/{team}/entropy", entropy_loss[team].item(), global_step)
                    writer.add_scalar(f"losses/{team}/old_approx_kl", old_approx_kl[team].item(), global_step)
                    writer.add_scalar(f"losses/{team}/approx_kl", approx_kl[team].item(), global_step)
                    writer.add_scalar(f"losses/{team}/clipfrac", np.mean(clipfracs[team]), global_step)
                    writer.add_scalar(f"losses/{team}/explained_variance", explained_var[team], global_step)
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if len(team_wins) > 0:
                    writer.add_scalar("teams/team_one_wins", team_one / len(team_wins), global_step)
                    writer.add_scalar("teams/team_two_wins", team_two / len(team_wins), global_step)
                if self.test_env is not None:
                    writer.add_scalar("teams/test_team_one_wins", test_one / len(test_wins), global_step)
                    writer.add_scalar("teams/test_team_two_wins", test_two / len(test_wins), global_step)

            del lengths_record[:-self.window_size]
            del rewards_record[:-self.window_size]

            del team_wins[:-self.window_size]

        if exp_name is not None:
            writer.close()