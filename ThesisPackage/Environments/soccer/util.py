import random
import math
import numpy as np

def player_position_relative(position, goal_position, field_height, team=1.0):
    # player_position is a tuple (x_player, y_player)
    # goal_position is a tuple (x_goal, y_goal) which is the center of the goal
    # field_width is the width of the goal line from one end to the other

    if team not in [-1, 1]:
        raise ValueError("team must be either -1 or 1")
    
    # Calculate the midpoint of the goal line as the new origin
    x = (position[0] - goal_position) * team
    y = (position[1] - (field_height / 2)) * team
    
    return (x, y)

def get_direction_relative(direction, team=1.0):
    # direction is a tuple (dx, dy)
    # team is either -1 or 1

    if team not in [-1, 1]:
        raise ValueError("team must be either -1 or 1")

    return (direction[0] * team, direction[1] * team)

def generate_random_coordinates(x_range, y_range, min_distance):
    def distance(coord1, coord2):
        return math.sqrt((coord2[0] - coord1[0]) ** 2 + (coord2[1] - coord1[1]) ** 2)
    
    while True:
        coord1 = (random.uniform(x_range[0], x_range[1]), random.uniform(y_range[0], y_range[1]))
        coord2 = (random.uniform(x_range[0], x_range[1]), random.uniform(y_range[0], y_range[1]))
        if distance(coord1, coord2) >= min_distance:
            return np.array(coord1, dtype=np.float32), np.array(coord2, dtype=np.float32)