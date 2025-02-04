from pettingzoo.utils import wrappers
import numpy as np
from gym.spaces import Box, Tuple

class ParallelClipAction(wrappers.BaseWrapper):
    """Clip the continuous actions for all agents within the valid :class:`Box` action space bounds in a PettingZoo parallel environment, when part of a Tuple space."""

    def __init__(self, env):
        """A wrapper for clipping continuous actions for all agents within the valid bounds, specifically handling Tuple spaces containing Box types.
        
        Args:
            env: The PettingZoo environment to apply the wrapper.
        """
        super().__init__(env)

    def step(self, actions):
        """Steps through the environment with potentially clipped actions for each agent.

        Args:
            actions (dict): A dictionary where keys are agent names and values are the actions to be potentially clipped and applied.

        Returns:
            observations, rewards, dones, infos: Returns the results of stepping through the base environment.
        """
        clipped_actions = {}
        for agent, action in actions.items():
            if isinstance(self.action_space(agent), Box):
                clipped_actions[agent] = np.clip(action, self.action_space(agent).low, self.action_space(agent).high)
            elif isinstance(self.action_space(agent), Tuple):
                clipped_actions[agent] = []
            else:
                clipped_actions[agent] = action
        return self.env.step(clipped_actions)

    def action_space(self, agent):
        """Retrieves the action space of the specified agent.

        Args:
            agent (str): The name of the agent to retrieve the action space for.

        Returns:
            The action space of the agent, which can be a Box or a Tuple containing a Box.
        """
        return self.env.action_spaces[agent]