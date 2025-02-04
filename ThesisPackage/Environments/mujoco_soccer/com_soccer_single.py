from pettingzoo import ParallelEnv
import numpy as np
from gymnasium.spaces import Discrete, Box
import mujoco as mj
from ThesisPackage.RL.Decentralized_PPO.multi_ppo import PPO_Multi_Agent
from ThesisPackage.RL.Wrappers.normalizeObservation import NormalizeObservation
from ThesisPackage.RL.Wrappers.normalizeRewards import NormalizeReward
from copy import deepcopy
import glfw

class MultiAgentRL(ParallelEnv):
    metadata = {'render.modes': ['console', 'human'], "name": "PongEnv"}
    def __init__(self, num_timesteps=4096, skip_frames=4, render=False, discount_factor=0.9, max_history=10):
        self.agents = ["red_1"]
        self._action_space = Box(low=-1, high=1, shape=(8,))
        self._observation_space = Box(low=-np.inf, high=np.inf, shape=(35,))
        self.observation_spaces = {agent: self._observation_space for agent in self.agents}
        self.action_spaces = {agent: self._action_space for agent in self.agents}

        xml_path = "ThesisPackage/Environments/soccer/SingleEnv.xml"
            
        self.model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
        self.data = mj.MjData(self.model)                # MuJoCo data
        self.healthy_z_range=(0.3, 1.2)
        self.discount_factor = discount_factor
        self.max_history = max_history
        self.last_ball_distance = deepcopy(np.linalg.norm(self.data.body("ball").xpos[:2] - self.data.body("goal").xpos[:2]))
        self.timestep = 0
        self.episode_rewards = 0
        self.skip_frames = skip_frames
        self.get_obs_index()
        self.get_action_index()

        self.render_setting = render
        if render:
            glfw.init()
            self.opt = mj.MjvOption()
            self.cam = mj.MjvCamera()                    # Abstract camera
            self.window = glfw.create_window(1200, 900, "Demo", None, None)
            glfw.make_context_current(self.window)
            glfw.swap_interval(1)

            # initialize visualization data structures
            mj.mjv_defaultCamera(self.cam)
            mj.mjv_defaultOption(self.opt)

            self.scene = mj.MjvScene(self.model, maxgeom=10000)
            self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)

            glfw.set_scroll_callback(self.window, self.__scroll)
            mj.set_mjcb_control(self.__controller)

    def __controller(self, model, data):
        #put the controller here. This function is called inside the simulation.
        pass

    def __scroll(self, window, x_offset, y_offset):
        """
        Scroll the camera in the MuJoCo environment.

        Parameters:
            window: The window object.
            x_offset: The horizontal offset of the scroll.
            y_offset: The vertical offset of the scroll.
        """
        action = mj.mjtMouse.mjMOUSE_ZOOM
        mj.mjv_moveCamera(self.model, action, 0.0, -0.05 * y_offset, self.scene, self.cam)

    def __render(self):
        """Renders the environment. Only works if the environment is created with the render flag set to True """
        # get framebuffer viewport
        viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
        viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

        # Update scene and render
        mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                           mj.mjtCatBit.mjCAT_ALL.value, self.scene)
        mj.mjr_render(viewport, self.scene, self.context)

        # swap OpenGL buffers (blocking call due to v-sync)
        glfw.swap_buffers(self.window)

        # process pending GUI events, call GLFW callbacks
        glfw.poll_events()
    
    def reset(self):
        mj.mj_resetData(self.model, self.data)
        mj.mj_forward(self.model, self.data)
        self.before_pos = {agent: self.data.body(agent).xpos[:2] for agent in self.agents}
        self.positions = {agent: [] for agent in self.agents}
        self.timestep = 0
        self.episode_rewards = 0
        self.last_ball_distance = deepcopy(np.linalg.norm(self.data.body("ball").xpos[:2] - self.data.body("goal").xpos[:2]))
        self.previous_time = self.data.time
        obs = self._get_ant_obs()
        obs = self._get_ball_obs(obs)
        obs = self._get_target_goal(obs)
        return obs, {}
    
    def get_obs_index(self):
        self.obs_dict_qpos = {agent: [] for agent in self.agents}
        self.obs_dict_qvel = {agent: [] for agent in self.agents}
        qpos_index = 0
        qvel_index = 0
        for i in range(self.model.njnt):
            for agent in self.agents:
                if agent in self.data.joint(i).name:
                    qpos_indizes = [qpos_index + i for i in range(len(self.data.joint(i).qpos))]
                    qvel_indizes = [qvel_index + i for i in range(len(self.data.joint(i).qvel))]
                    self.obs_dict_qpos[agent] += qpos_indizes
                    self.obs_dict_qvel[agent] += qvel_indizes
                    qpos_index += len(self.data.joint(i).qpos)
                    qvel_index += len(self.data.joint(i).qvel)

    def get_action_index(self):
        self.action_dict = {agent: [] for agent in self.agents}
        action_index = 0
        for i in range(self.model.nu):
            for agent in self.agents:
                if agent in self.model.actuator(i).name:
                    action_indizes = [action_index + i for i in range(len(self.model.actuator(i).acc0))]
                    self.action_dict[agent] += action_indizes
                    action_index += len(self.model.actuator(i).acc0)
    
    def _get_ant_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        obs = {agent: np.concatenate((position[self.obs_dict_qpos[agent]], velocity[self.obs_dict_qvel[agent]])) for agent in self.agents}
        return obs
    
    def _get_ball_obs(self, obs):
        ball_pos = self.data.body("ball").xpos
        for agent in self.agents:
            torso_pos = self.data.body(agent).xpos
            rel_position = ball_pos - torso_pos
            obs[agent] = np.concatenate((obs[agent], rel_position))
        return obs
    
    def _get_target_goal(self, obs):
        for agent in self.agents:
            target_goal = self.data.body("goal").xpos
            torso_pos = self.data.body("red_1").xpos
            rel_position = target_goal - torso_pos
            obs[agent] = np.concatenate((obs[agent], rel_position))
        return obs
    
    def _apply_action(self, action):
        for agent in self.agents:
            self.data.ctrl[self.action_dict[agent]] = action[agent]

    def step(self, actions):
        self._apply_action(actions)
        self.timestep += 1
        for _ in range(self.skip_frames):
            mj.mj_step(self.model, self.data)
        obs = self._get_ant_obs()
        obs = self._get_ball_obs(obs)
        obs = self._get_target_goal(obs)
        rewards = {agent: self._get_ball_reward(agent) for agent in self.agents}
        rewards = {agent: rewards[agent] + self.exploration_reward(agent) for agent in self.agents}
        rewards = {agent: rewards[agent] + self.healthy_reward(agent) for agent in self.agents}
        self.episode_rewards += sum(rewards.values())
        terminations = {agent: not self.is_agent_healthy(agent) for agent in self.agents}
        terminations = {agent: terminations[agent] or self._check_ball_done() for agent in self.agents}
        if self.timestep > 4096:
            truncations = {agent: True for agent in self.agents}
        else:
            truncations = {agent: False for agent in self.agents}

        if self.render_setting and self.data.time - self.previous_time > 1.0/30.0:
            self.previous_time = self.data.time
            self.__render()
        return obs, rewards, terminations, truncations, {}
    
    def is_agent_healthy(self, agent):
        torso_pos = self.data.geom(agent + "_torso_geom").xpos
        return torso_pos[2] > self.healthy_z_range[0] and torso_pos[2] < self.healthy_z_range[1]
    
    def healthy_reward(self, agent):
        torso_pos = self.data.geom(agent + "_torso_geom").xpos
        if torso_pos[2] > self.healthy_z_range[1]:
            return -1
        elif torso_pos[2] < self.healthy_z_range[0]:
            return -1
        else:
            return 0
    
    def exploration_reward(self, agent):
        # Compute the exploration reward for the current timestep
        self.positions[agent].append(deepcopy(self.data.body(agent).xpos[:2]))

        if len(self.positions[agent]) > self.max_history:
            self.positions[agent].pop(0)

        if len(self.positions[agent]) < 2:  # Not enough data to compute exploration
            return 0
        
        t = len(self.positions[agent]) - 1
        current_position = self.positions[agent][-1]
        reward = 0

        # Sum over all previous timesteps
        for i in range(t):
            past_position = self.positions[agent][i]
            distance = np.sqrt((current_position[0] - past_position[0]) ** 2 +
                               (current_position[1] - past_position[1]) ** 2)
            decay = self.discount_factor ** (t - i)
            reward += distance * decay
        reward = reward / (t * 10)
        return reward
    
    def _get_ball_reward(self, agent):
        ball_pos = self.data.body("ball").xpos[:2]
        goal_pos = self.data.body("goal").xpos[:2]
        distance = np.linalg.norm(ball_pos - goal_pos)
        reward = self.last_ball_distance - distance
        self.last_ball_distance = deepcopy(distance)
        return reward
            
    def _check_ball_done(self):
        ball_pos = self.data.body("ball").xpos
        target_pos = self.data.body("goal").xpos
        distance = np.linalg.norm(ball_pos - target_pos)
        if distance < 0.5:
            print("GOOOOAL", distance)
        return distance < 0.5

def make_env(num_envs=16, num_timesteps=16384, render=True):
    envs = [NormalizeReward(NormalizeObservation(MultiAgentRL(num_timesteps, render=False))) for i in range(num_envs-1)]
    env = NormalizeReward(NormalizeObservation(MultiAgentRL(num_timesteps, render=True)))
    envs.append(env)
    return envs

envs = make_env(render=False, num_envs=32)
agent = PPO_Multi_Agent(envs)
agent.train(300000000, tensorboard_folder="soccer/", exp_name="single-soccer", learning_rate=0.0001)