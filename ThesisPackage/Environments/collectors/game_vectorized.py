import numpy as np
from gym import spaces
import copy
import random

class Player:
    def __init__(self, initial_position, name):
        self.position = np.array(initial_position, dtype=np.float32)
        initial_direction = (0, 1)
        self.direction = np.array(initial_direction, dtype=np.float32)
        self.max_angle_change = np.pi / 4
        self.name = name

    def move(self, action, environment):

        # Clip actions to -1 to 1
        action = np.clip(action, -1, 1)

        # Update the direction based on the turning action
        angle_change = action[:,1] * self.max_angle_change

        current_angle = np.arctan2(self.direction[:,1], self.direction[:,0])

        new_angle = current_angle + angle_change
        
        self.direction = np.array([np.cos(new_angle), np.sin(new_angle)])
        self.direction /= np.linalg.norm(self.direction)
        
        step_size = action[:,0]  # Assuming maximum step size of 1 unit per move

        new_position = self.position + self.direction * step_size

        if np.any(new_position[:,0] < 0) or np.any(new_position[:,0] >= environment.width):
            new_position[:,0] = np.clip(new_position[:,0], 0, environment.width - 1)
        if np.any(new_position[:,1] < 0) or np.any(new_position[:,1] >= environment.height):
            new_position[:,1] = np.clip(new_position[:,1], 0, environment.height - 1)

        # Update position if no collision
        self.position = new_position

    def get_position(self):

        return self.position

    def get_direction(self):
        return self.direction
    

class PlayerDiscrete:
    def __init__(self, num_envs, initial_position, name):
        self.position = np.array([initial_position for _ in range(num_envs)], dtype=np.float32)
        initial_direction = (0, 1)
        self.direction = np.array([initial_direction for _ in range(num_envs)], dtype=np.float32)
        self.angle_change = np.pi / 4
        self.name = name

    def move(self, action, environment):

        # Update the direction based on the turning action
        change_factors = np.zeros((len(action[:,1]),))
        change_factors = np.where(action[:,1] == 1, 1, change_factors)
        change_factors = np.where(action[:,1] == 2, -1, change_factors)

        angle_change = change_factors * self.angle_change

        current_angle = np.arctan2(self.direction[:,1], self.direction[:,0])

        new_angle = current_angle + angle_change
        
        self.direction = np.array([np.cos(new_angle), np.sin(new_angle)])
        self.direction /= np.linalg.norm(self.direction)

        step_sizes = np.zeros((len(action[:,0]),))
        step_sizes = np.where(action[:,0] == 1, 1, step_sizes)
        step_sizes = np.where(action[:,0] == 2, -1, step_sizes)

        new_position = self.position + self.direction * step_sizes

        if np.any(new_position[:,0] < 0) or np.any(new_position[:,0] >= environment.width):
            new_position[:,0] = np.clip(new_position[:,0], 0, environment.width - 1)
        if np.any(new_position[:,1] < 0) or np.any(new_position[:,1] >= environment.height):
            new_position[:,1] = np.clip(new_position[:,1], 0, environment.height - 1)

        # Update position if no collision
        self.position = new_position

    def get_position(self):

        return self.position

    def get_direction(self):
        return self.direction

class Target:
    def __init__(self, num_envs, environment, num_targets, timesteps_left=20):
        self.width, self.height = environment.width, environment.height
        self.positions = np.full((num_envs, num_targets, 2), -1, dtype=np.float32)
        self.timesteps_left = np.zeros((num_envs, num_targets), dtype=np.int32)

    def update(self):
        # Update position based on the current direction
        self.timesteps_left = self.timesteps_left - 1
        self.timesteps_left = np.where(self.timesteps_left <= 0)

        # Indizes of environments that still has an unused target slot based on the positions
        unused_targets = np.where(self.positions[:, :, 0] == -1)

    def get_position(self):
        return self.position
