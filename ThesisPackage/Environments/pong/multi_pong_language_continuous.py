import gym
from gym import spaces
import numpy as np
import random
from pettingzoo import ParallelEnv
import functools
import sys
import pygame

class PongEnv(ParallelEnv):
    """
    Custom Environment for Pong game compatible with OpenAI Gym.
    """
    metadata = {'render.modes': ['console', 'pygame'], "name": "PongEnv"}

    def __init__(self, width=20, height=10, paddle_height=3, sequence_length=1, vocab_size=2, max_episode_steps=1024):
        """
        Initializes the PongEnv class.

        Args:
            width (int): The width of the game grid. Default is 20.
            height (int): The height of the game grid. Default is 10.
            paddle_height (int): The height of the paddles. Default is 3.
            sequence_length (int): The length of the sequence of utterances. Default is 1.
            vocab_size (int): The size of the vocabulary. Default is 2.
            max_episode_steps (int): The maximum number of steps in an episode. Default is 1024.
        """
        super(PongEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 = Stay, 1 = Up, 2 = Down
        language_space = np.array([vocab_size for _ in range(sequence_length)])
        self._language_space = spaces.MultiDiscrete(language_space)
        self._move_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.int32)

        if sequence_length > 0:
            self._action_space = spaces.Tuple((self._move_space, self._language_space))
        else:
            self._action_space = self._move_space

        self.agents = ["paddle_1", "paddle_2"]
        self.possible_agents = self.agents

        self.timestep = 0
        self.episode_rewards = 0
        self.max_episode_steps = max_episode_steps

        # Game settings
        self.width = width
        self.height = height
        self.paddle_height = paddle_height

        self.vocab_size = vocab_size
        self.sequence_length = sequence_length

        self.utterances = {agent: [0 for _ in range(self.sequence_length)] for agent in self.agents}

        self.paddles = {"paddle_1": self.height // 2 + 2, "paddle_2": self.height // 2 - 2}

        self.balls = {"ball_1": {"position": [self.width // 2 - 1, self.height // 2], "direction":[random.uniform(0.5, 1), random.uniform(0.5, 1)]},
                      "ball_2": {"position": [self.width // 2 + 1, self.height // 2], "direction":[random.uniform(0.5, 1), random.uniform(0.5, 1)]}}

        # Observation space: position of paddle and ball
        low = [-1 * max(self.width, self.height) - 1 for _ in range(10)]
        high = [max(self.width, self.height) + 1 for _ in range(10)]

        # Add sequence length to observation space
        low += [0 for _ in range(self.sequence_length)]
        high += [self.vocab_size for _ in range(self.sequence_length)]

        self._observation_spaces = {agent: spaces.Box(low=np.array(low), high=np.array(high), dtype=np.float16) for agent in self.agents}

        self.__critic_space = self.__get_critic_space()

    def __get_critic_space(self):
        low = [-1 * max(self.width, self.height) for _ in range(12)]
        high = [max(self.width, self.height) for _ in range(12)]
        
        return spaces.Box(low=np.array(low), high=np.array(high), dtype=np.float16)

    def get_relative_position(self, object_pos, invert_x=False):
        """
        Calculate the relative position of an object to the center of the field.

        Args:
            object_pos (tuple): A tuple (x, y) representing the position of the object.
            invert_x (bool): If True, invert the x-axis value.

        Returns:
            tuple: A tuple (x_relative, y_relative) representing the relative
                position of the object to the center of the field. Positive values
                mean the object is to the right or above the center, negative
                values mean to the left or below. If invert_x is True, the x-axis
                values are inverted.
        """
        center_x = self.width / 2
        center_y = self.height / 2

        x_relative = object_pos[0] - center_x
        y_relative = object_pos[1] - center_y

        if invert_x:
            x_relative = -x_relative

        return (x_relative, y_relative)
    
    def __move_ball(self, ball_pos, ball_direction):
        ball_pos[0] += ball_direction[0]
        ball_pos[1] += ball_direction[1]
        if ball_pos[1] <= 0 or ball_pos[1] >= self.height - 1:
            ball_direction[1] *= -1
            ball_pos[1] += ball_direction[1]
            ball_pos[0] += ball_direction[0]
        if ball_pos[0] <= 0:
            ball_direction[0] *= -1
        rewards = {paddle: 0 for paddle in self.paddles.keys()}
        for paddle in self.paddles.keys():
            if ball_pos[0] >= self.width - 1 and self.paddles[paddle] <= ball_pos[1] < self.paddles[paddle] + self.paddle_height:
                # ball_direction[0] *= -1
                rewards[paddle] = 1
                # ball_pos[0] += ball_direction[0]
                # ball_pos[1] += ball_direction[1]
        if sum(rewards.values()) > 0:
            ball_direction[0] *= -1
            ball_pos[0] += ball_direction[0]
            ball_pos[1] += ball_direction[1]
        return ball_pos, ball_direction, rewards
    
    def check_done(self, ball_pos):
        """
        Check if the game is done based on the ball position.

        Args:
            ball_pos (tuple): The position of the ball.

        Returns:
            bool: True if the game is done, False otherwise.
        """
        if ball_pos[0] >= self.width:
            return True
        return False
    
    def __get_observation(self, paddle):

        rel_paddle_pos = self.get_relative_position((self.width - 1, self.paddles[paddle]))

        obs = np.array([rel_paddle_pos[0], rel_paddle_pos[1]], dtype=np.float16)

        for ball in self.balls.keys():
            rel_ball_pos = self.get_relative_position(self.balls[ball]["position"])
            ball_direction = self.balls[ball]["direction"]
            ball_obs = np.array([rel_ball_pos[0], rel_ball_pos[1], ball_direction[0], ball_direction[1]])
            obs = np.append(obs, ball_obs)

        other_paddle = "paddle_1" if paddle == "paddle_2" else "paddle_2"
        obs = np.append(obs, self.utterances[other_paddle])

        return obs

    def step(self, actions):
        # Update paddle position based on action
        self.timestep += 1
        for paddle in actions.keys():
            if len(actions[paddle].shape) > 0:
                action = actions[paddle][0]
            else:
                action = actions[paddle]
            action = np.clip(action, -1, 1)
            if action < 0 and self.paddles[paddle] > 0:
                self.paddles[paddle] += action
            elif action > 0 and self.paddles[paddle] < self.height - self.paddle_height:
                self.paddles[paddle] += action

        if self.sequence_length > 0:
            for paddle in self.paddles.keys():
                self.utterances[paddle] = actions[paddle][1:]

        rewards = {paddle: 0 for paddle in self.paddles.keys()}
        for ball in self.balls.keys():
            ball_pos, ball_direction, new_rewards = self.__move_ball(self.balls[ball]["position"], self.balls[ball]["direction"])
            self.balls[ball] = {"position":ball_pos, "direction":ball_direction}
            for paddle in new_rewards.keys():
                rewards[paddle] += new_rewards[paddle]

        done = False
        if any([self.check_done(self.balls[ball]["position"]) for ball in self.balls.keys()]):
            done = True
            rewards = {paddle: -1 for paddle in self.paddles.keys()}  # Negative reward for losing the ball

        self.episode_rewards += sum(rewards.values())

        terminated = {paddle: done for paddle in self.paddles.keys()}

        obs = {paddle: self.__get_observation(paddle) for paddle in self.paddles.keys()}

        if self.timestep >= self.max_episode_steps:
            truncated = {paddle: True for paddle in self.paddles.keys()}
        else:
            truncated = {paddle: False for paddle in self.paddles.keys()}

        infos = {paddle: {} for paddle in self.paddles.keys()}

        return obs, rewards, terminated, truncated, infos
    
    def state(self) -> np.ndarray:

        obs = np.array([])
        for paddle in self.agents:
            rel_paddle_pos = self.get_relative_position((self.width - 1, self.paddles[paddle]))
            obs = np.append(obs, rel_paddle_pos).flatten()
        for ball in self.balls.keys():
            rel_ball_pos = self.get_relative_position(self.balls[ball]["position"])
            ball_direction = self.balls[ball]["direction"]
            ball_obs = np.array([rel_ball_pos[0], rel_ball_pos[1], ball_direction[0], ball_direction[1]])
            obs = np.append(obs, ball_obs).flatten()
        return {agent: np.array(obs.flatten()) for agent in self.agents}

    def reset(self, seed=1, options=None):
        # Reset the game state
        self.timestep = 0
        self.episode_rewards = 0

        self.paddles = {"paddle_1": self.height // 2 + 2, "paddle_2": self.height // 2 - 2}

        self.balls = {"ball_1": {"position": [self.width // 2 - 1, self.height // 2], "direction":[random.uniform(0.5, 1), random.uniform(-1, 1)]},
                      "ball_2": {"position": [self.width // 2 + 1, self.height // 2], "direction":[random.uniform(0.5, 1), random.uniform(-1, 1)]}}

        obs = {paddle: self.__get_observation(paddle) for paddle in self.paddles}

        infos = {paddle: {} for paddle in self.paddles.keys()}
        return obs, infos
    
    def render(self, mode='console'):
        if mode == 'console':
            self.render_console()
        elif mode == 'pygame':
            self.render_pygame()
        else:
            raise NotImplementedError("The render mode is not supported.")

    def render_console(self, mode='console'):
        # Draw game board in console
        board = [[' ' for _ in range(self.width + 2)] for _ in range(self.height + 2)]
        paddle_symbolds = ['|', '/', '*', '+']
        for i, paddle in enumerate(self.paddles.keys()):
            for j in range(self.paddle_height):
                paddle_pos = int(self.paddles[paddle])
                board[paddle_pos + j][self.width - 1] = paddle_symbolds[i]

        
        for ball in self.balls.keys():
            ball_pos = self.balls[ball]["position"]
            board[int(ball_pos[1])][int(ball_pos[0])] = 'O'

        # Print game board
        print("-" * (self.width + 2))
        print('\n'.join([''.join(row) for row in board]))
        print("-" * (self.width + 2))
        print()

    def render_pygame(self, mode='pygame'):
        # Initialize PyGame if it hasn't been already
        if not hasattr(self, 'screen'):
            pygame.init()
            self.window_size = 600  # Set the size of the window
            self.cell_size = self.window_size // max(self.width, self.height)
            self.screen = pygame.display.set_mode((self.width * self.cell_size, self.height * self.cell_size))
            pygame.display.set_caption('Pong Environment')
            self.clock = pygame.time.Clock()

        # Set screen background and basic colors
        self.screen.fill((0, 0, 0))  # Black background
        paddle_color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Colors for paddles

        # Draw paddles
        paddle_width = self.cell_size
        paddle_height = self.cell_size * self.paddle_height

        for i, paddle in enumerate(self.paddles.keys()):
            paddle_pos = self.paddles[paddle]

            pygame.draw.rect(self.screen, paddle_color[i], pygame.Rect((self.width * self.cell_size - self.cell_size, paddle_pos * self.cell_size), (paddle_width, paddle_height)))

        # Draw balls
        for ball in self.balls.keys():
            ball_pos = self.balls[ball]["position"]
            pygame.draw.circle(self.screen, (255, 255, 255), (int(ball_pos[0]) * self.cell_size, int(ball_pos[1]) * self.cell_size), 10)

        # Update the display
        pygame.display.flip()

        # Handle quitting from the window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    def close(self):
        pass

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        if agent == "critic":
            return self.__critic_space
        return self._observation_spaces[agent]
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_space