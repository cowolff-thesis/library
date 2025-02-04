import numpy as np
from gym import spaces
import copy

class Player:
    def __init__(self, initial_position, goal, name, team, num_envs):
        self.position = np.array([initial_position for _ in range(num_envs)], dtype=np.float32)
        if team == -1:
            initial_direction = np.array([[-1, 0] for _ in range(num_envs)], dtype=np.float32)
        elif team == 1:
            initial_direction = np.array([[1, 0] for _ in range(num_envs)], dtype=np.float32)
        else:
            raise ValueError("team must be either -1 or 1")
        self.direction = initial_direction
        self.max_angle_change = np.pi / 4
        self.name = name
        self.goal = goal
        self.team = team

    def move(self, action, environment):
        # Store original position in case we need to revert due to a collision
        original_position = copy.deepcopy(self.position)

        # Clip actions to -1 to 1
        action = np.clip(action, -1, 1)

        # Update the direction based on the turning action
        angle_change = action[:,1] * self.max_angle_change

        current_angle = np.arctan2(self.direction[:,1], self.direction[:,0])

        new_angle = current_angle + angle_change
        
        self.direction = np.array([[np.cos(cur_angle), np.sin(cur_angle)] for cur_angle in new_angle], dtype=np.float32)
        self.direction /= np.linalg.norm(self.direction)
        
        step_size = action[:,0]  # Assuming maximum step size of 1 unit per move

        new_position = self.position + self.direction * step_size

        new_position[:,0] = np.clip(new_position[:,0], 0, environment.width - 1)
        new_position[:,1] = np.clip(new_position[:,1], 0, environment.height - 1)

        # Check for collisions
        positions = environment.get_player_positions()
        for pos in positions:
            if np.array_equal(pos, self.position):  # Ignore collision detection with self
                continue
            if np.linalg.norm(new_position - pos) < 2:  # Each player has a radius of 1, total minimum distance = 2
                self.position = original_position  # Reset to original position due to collision
                return  # Exit the function

        # Update position if no collision
        self.position = new_position

    def get_position(self):
        return self.position

    def get_direction(self):
        return self.direction

class Ball:
    def __init__(self, initial_position, initial_direction=(0, 0)):
        self.position = np.array(initial_position, dtype=np.float32)
        self.old_position = np.array(initial_position, dtype=np.float32)
        self.direction = np.array(initial_direction, dtype=np.float32)
        self.player_possesion = None

    def move(self, environment):
        # Update position based on the current direction
        self.old_position = copy.deepcopy(self.position)
        if self.player_possesion is None:
            self.position += self.direction
        else:
            self.position = self.player_possesion.get_position()
        
        # Check for collisions with the borders and reverse direction if necessary
        if self.position[0] <= 0 or self.position[0] >= environment.width - 1:
            self.direction[0] *= -1
            self.position[0] = np.clip(self.position[0], 0, environment.width - 1, dtype=np.float32)
        if self.position[1] <= 0 or self.position[1] >= environment.height - 1:
            self.direction[1] *= -1
            self.position[1] = np.clip(self.position[1], 0, environment.height - 1, dtype=np.float32)

    def set_direction(self, direction):
        self.direction = direction

    def set_possession(self, player):
        self.player_possesion = player

    def check_possession(self, player):
        return self.player_possesion == player

    def get_position(self):
        return self.position

    def get_direction(self):
        return self.direction
