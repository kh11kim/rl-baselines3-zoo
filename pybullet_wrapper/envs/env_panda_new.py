import numpy as np
import pybullet as p

import gym
from gym import spaces
from gym.envs.registration import register

from ..core import Bullet
from ..scene_maker import BulletSceneMaker
from ..collision_checker import BulletCollisionChecker
from ..robots import Panda
from .env_panda import PandaEnvBase

class PandaGymNewEnv(PandaEnvBase, gym.Env):
    def __init__(self, render=False, reward_type="", level=1.0):
        self.level = level
        self.reward_type = reward_type
        super().__init__(render=render)
        self.n_obs = self.robot.n_joints + 3 + 9
        self.observation_space = spaces.Dict(dict(
            joint_pos=spaces.Box(
                self.robot.joint_ll, 
                self.robot.joint_ul, 
                shape=(self.robot.n_joints,), 
                dtype=np.float32
            ),
            joint_vel=spaces.Box(
                -1., 
                1., 
                shape=(self.robot.n_joints,), 
                dtype=np.float32
            ),
            joint_goal=spaces.Box(
                self.robot.joint_ll, 
                self.robot.joint_ul, 
                shape=(self.robot.n_joints,), 
                dtype=np.float32
            ),
        ))
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.robot.n_joints,), dtype=np.float32)
        self._joint_goal = None
        self._joint_pos = None
        self._joint_vel = None
        self.eps = 0.5

    @property
    def joint_goal(self):
        return self._joint_goal.copy()
    
    @joint_goal.setter
    def joint_goal(self, arr: np.ndarray):
        self._joint_goal = arr.copy()
    
    @property
    def joint_vel(self):
        return self._joint_vel.copy()
    
    @joint_vel.setter
    def joint_vel(self, arr: np.ndarray):
        self._joint_vel = arr.copy()
    
    @property
    def joint_pos(self):
        return self._joint_pos.copy()
    
    @joint_pos.setter
    def joint_pos(self, arr: np.ndarray):
        self._joint_pos = arr.copy()

    def get_observation(self):
        return dict(
            joint_pos=self.joint_pos,
            joint_vel=self.joint_vel,
            joint_goal=self.joint_goal
        )

    def set_action(self, action: np.ndarray):
        joint_prev = self.joint_pos
        action = action.copy()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        joint_target = joint_prev.copy()
        joint_target += action * self.max_joint_change
        joint_target = np.clip(joint_target, self.robot.joint_ll, self.robot.joint_ul)
        self.robot.set_joint_angles(joint_target)
        if self.checker.is_collision():
            joint_curr = joint_prev
            self._collision_flag = True
            self.robot.set_joint_angles(joint_curr)
        elif self.is_limit():
            joint_curr = joint_target
            self._collision_flag = True
        else:
            joint_curr = joint_target
            self._collision_flag = False
        self.joint_pos = joint_curr
        self.joint_vel = action

    def is_success(self, joint_curr: np.ndarray, joint_goal: np.ndarray):
        return np.linalg.norm(joint_curr - joint_goal) < self.eps

    def reset(self):
        self.joint_goal = self.get_random_configuration(collision_free=True)
        self.robot.set_joint_angles(self.joint_goal)
        goal_ee = self.robot.get_ee_position()
        while True:
            random_start = self.get_random_configuration(collision_free=False)
            self.start = self.joint_goal + (random_start - self.joint_goal) * self.level
            self.robot.set_joint_angles(self.start)
            if not self.is_collision():
                start_ee = self.robot.get_ee_position()
                break
        
        self.joint_pos = self.start
        self.joint_vel = np.zeros(7)

        if self.is_render:
            self.scene_maker.view_position("goal", goal_ee)
            self.scene_maker.view_position("curr", start_ee)
        return self.get_observation()

    def step(self, action: np.ndarray):
        self.set_action(action)
        obs_ = self.get_observation()
        done = False
        info = dict(
            is_success=self.is_success(self.joint_pos, self.joint_goal),
            collisions=self._collision_flag,
        )
        reward = self.compute_reward(info)
        if self.is_render:
            self.scene_maker.view_position("curr", self.robot.get_ee_position())
        return obs_, reward, done, info

    def compute_reward(self, info):
        collisions = info["collisions"]
        r = - np.linalg.norm(self.joint_pos-self.joint_goal)
        
        if "step" in self.reward_type:
            r -= 1.
        
        if "col" in self.reward_type:
            r -= collisions * 1.
        
        return r

register(
    id='MyPandaReachNew-v0',
    entry_point='pybullet_wrapper_:PandaGymNewEnv',
    max_episode_steps=50,
)