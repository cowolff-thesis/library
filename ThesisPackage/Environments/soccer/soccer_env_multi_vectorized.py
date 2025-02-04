from gymnasium.spaces.space import Space
from pettingzoo.utils import ParallelEnv
from gym import spaces
import numpy as np
import time
import copy
import pygame
import sys

try:
    from game import Player, Ball
    from util import player_position_relative, get_direction_relative
except ImportError:
    from ThesisPackage.Environments.soccer.game import Player, Ball
    from ThesisPackage.Environments.soccer.util import player_position_relative, get_direction_relative

class SoccerGame(ParallelEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, width=7, height=5, sequence_length=1, vocab_size=1, max_timesteps=1024, noise_team=0):
        super().__init__()
        self.width = width
        self.height = height
        self.players = {
            "player_1": Player([1, self.height // 2], self.width - 1, "player_1", 1),
            "player_2": Player([self.width - 2, self.height // 2], 0, "player_2", -1),
            "player_3": Player([3, self.height // 2], self.width - 1, "player_3", 1),
            "player_4": Player([self.width - 4, self.height // 2], 0, "player_4", -1)
        }
        self.ball = Ball([self.width // 2, self.height // 2])
        self.agents = list(self.players.keys())
        self.possible_agents = self.agents[:]
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.noise_team = noise_team
        self.teams = [1, -1]
        
        self._move_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self._discrete_space = spaces.MultiDiscrete([2] + [self.vocab_size for _ in range(self.sequence_length)])
        # self._language_space = spaces.MultiDiscrete([vocab_size for _ in range(self.sequence_length)])
        self._action_space = {agent: spaces.Tuple((self._move_space, self._discrete_space)) for agent in self.agents}
        self._observation_spaces = {agent: spaces.Box(low=np.array([0] * 12 + [0, 0] + [0 for _ in range(self.sequence_length)]), high=np.array([self.width - 1, self.height - 1] * 6 + [self.width-1, 1] + [self.vocab_size for _ in range(self.sequence_length)]), dtype=np.float32) for agent in self.agents}
        self.max_timesteps = max_timesteps
        self.episode_rewards = 0
        self.utterances = {agent: [0 for _ in range(self.sequence_length)] for agent in self.agents}
        self.reset()

    def reset(self, seed=1, options=None):
        self.ball = Ball([self.width // 2, self.height // 2], initial_direction=(0, 0))
        self.players['player_1'].position = np.array([1, self.height // 2], dtype=np.float32)
        self.players['player_2'].position = np.array([self.width - 2, self.height // 2], dtype=np.float32)
        self.players['player_3'].position = np.array([3, self.height // 2], dtype=np.float32)
        self.players['player_4'].position = np.array([self.width - 4, self.height // 2], dtype=np.float32)
        self.timestep = 0
        self.episode_rewards = 0
        self.utterances = {agent: [0 for _ in range(self.sequence_length)] for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return {agent: self.observe(agent) for agent in self.agents}, infos

    def step(self, actions):
        rewards = {agent: 0 for agent in self.agents}
        terminated = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        self.timestep += 1

        # Move the players
        for agent, action in actions.items():
            self.players[agent].move(action[:2], self)

        self.ball.move(self)

        for agent, action in actions.items():
            if action[2] == 0:
                if self.ball.check_possession(self.players[agent]):
                    self.ball.set_direction(self.players[agent].get_direction())
                    self.ball.player_possesion = None
            if action[2] == 1:
                # If distance between ball and player is less than 1, player gets the ball
                if np.linalg.norm(self.ball.get_position() - self.players[agent].get_position()) < 1 and self.ball.player_possesion is None:
                    self.ball.player_possesion = self.players[agent]
                elif np.linalg.norm(self.ball.get_position() - self.players[agent].get_position()) < 1:
                    # The chance of possession switch is 33.3%
                    if np.random.rand() < 0.333:
                        self.ball.player_possesion = self.players[agent]

        for agent, action in actions.items():
            self.utterances[agent] = action[3:]

        # Reward for moving the ball in the right direction
        for agent in self.agents:
            reward = (abs(self.players[agent].goal - self.ball.old_position[0]) - abs(self.players[agent].goal - self.ball.get_position()[0])) * 0.1
            rewards[agent] += reward

        # Check if the ball has entered the goal of either player
        for agent in self.agents:
            if self.ball.get_position()[0] == self.players[agent].goal:
                rewards[agent] += 1
                # set all terminated to true
                terminated = {agent: True for agent in self.agents}
                infos[agent]['team'] = self.players[agent].team
        
        self.episode_rewards += sum(rewards.values())

        if self.max_timesteps < self.timestep:
            truncated = {agent: True for agent in self.agents}
        else:
            truncated = {agent: False for agent in self.agents}

        return {agent: self.observe(agent) for agent in self.agents}, rewards, terminated, truncated, infos

    def observe(self, agent):
        other_agents = [a for a in self.agents if a != agent]
        
        player = self.players[agent]
        agent_coord = player_position_relative(self.players[agent].get_position(), self.players[agent].goal, self.height, player.team)
        agent_direct = get_direction_relative(self.players[agent].get_direction(), player.team)
        goal_coord = self.width
        other_agent_coords = []
        for other_agent in other_agents:
            if self.players[other_agent].team != player.team:
                other_agent_coord = player_position_relative(self.players[other_agent].get_position(), self.players[other_agent].goal, self.height, player.team)
                other_agent_coords.append(other_agent_coord)
        other_agent_coords = np.concatenate(other_agent_coords)
        ball_coord = player_position_relative(self.ball.get_position(), self.players[agent].goal, self.height, player.team)
        ball_direct = get_direction_relative(self.ball.get_direction(), player.team)

        # Get utterance from other team member
        for other_agent in other_agents:
            if self.players[other_agent].team == player.team:
                if player.team == self.noise_team:
                    other_utterance = np.random.randint(0, self.vocab_size, self.sequence_length)
                else:
                    other_utterance = self.utterances[other_agent]
                break

        obs = np.concatenate((agent_coord, agent_direct, other_agent_coords, ball_coord, ball_direct, [goal_coord], [self.ball.player_possesion is not None], other_utterance), dtype=np.float32)
        
        return obs
    
    def get_player_positions(self):
        return [player.get_position() for player in self.players.values()]
    
    def observation_space(self, agent) -> Space:
        return self._observation_spaces[agent]
    
    def action_space(self, agent) -> Space:
        """
        action_space:
            Index 0: Movement
                - 0: Move forward
                - 1: Move backward
                - 2: Move left
                - 3: Move right
                - 4: Do nothing
            Index 1: Action
                - 0: Kick the ball
                - 1: Get the ball
                - 2: Do nothing
        """
        return self._action_space[agent]

    def render(self, mode='human'):
        if not hasattr(self, 'screen'):
            pygame.init()
            self.window_size = 600  # Set the size of the window
            self.cell_size = self.window_size // max(self.width, self.height)
            self.screen = pygame.display.set_mode((self.width * self.cell_size, self.height * self.cell_size))
            pygame.display.set_caption('Soccer Environment')
            self.clock = pygame.time.Clock()

        # Fill the screen with green
        self.screen.fill((0, 150, 0))
        
        # Constants for the goals
        goal_width = self.cell_size  # Width of the goal, could be wider
        goal_height = self.cell_size * 4  # Goal height, adjust as needed
        
        # Draw goals on the left and right sides
        # Left goal
        pygame.draw.rect(self.screen, (128, 128, 128), (0, 
                                                        (self.height // 2 - 2) * self.cell_size, 
                                                        goal_width, goal_height))
        # Right goal
        pygame.draw.rect(self.screen, (128, 128, 128), (self.width * self.cell_size - goal_width, 
                                                        (self.height // 2 - 2) * self.cell_size, 
                                                        goal_width, goal_height))
        
        # Draw the midfield line
        line_width = self.cell_size // 4  # Half the cell size in width
        mid_x = self.width * self.cell_size // 2 - line_width // 2  # X position centered in the field
        pygame.draw.rect(self.screen, (255, 255, 255), (mid_x, 0, line_width, self.height * self.cell_size))
        
        # Draw the ball
        ball_pos = self.ball.get_position()
        pygame.draw.circle(self.screen, (160, 160, 160), (ball_pos[0] * self.cell_size + self.cell_size // 2, 
                                                        ball_pos[1] * self.cell_size + self.cell_size // 2), 
                            self.cell_size // 2)

        # Draw the players
        for agent, player in self.players.items():
            color = (255, 0, 0) if self.players[agent].team == 1 else (0, 0, 255)
            pos = player.get_position()
            direction = player.get_direction()
            pygame.draw.rect(self.screen, color, (pos[0] * self.cell_size, 
                                                pos[1] * self.cell_size, 
                                                self.cell_size, self.cell_size))
            pygame.draw.rect(self.screen, color, ((pos[0] + direction[0]) * self.cell_size,
                                                (pos[1] + direction[1]) * self.cell_size, 
                                                self.cell_size * 0.5, self.cell_size * 0.5))

        pygame.display.flip()  # Update the full display Surface to the screen

        # Handle quitting from the window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

