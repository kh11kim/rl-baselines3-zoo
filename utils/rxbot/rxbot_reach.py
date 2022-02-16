import os
import gym
import numpy as np
from gym.envs.registration import register

from .rxbot import RxbotAbstractEnv

class RxbotReachEnv(RxbotAbstractEnv, gym.GoalEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, render=False, dim=2, reward_type="task", random_init=True, task_ll=[0,-1,0], task_ul=[1,1,1], joint_range=2*np.pi):
        super().__init__(render=render, dim=dim, task_ll=task_ll, task_ul=task_ul, joint_range=joint_range)
        self.reward_type = reward_type
        self.random_init = random_init

    def _get_observation(self):
        """ observation : joint, ee_curr, ee_goal
        """
        joints = self.robot.get_joints()
        ee_pos = self.robot.get_ee_pos()
        goal_pos = self.goal.copy()
        return dict(
            observation=joints,
            achieved_goal=ee_pos,
            desired_goal=goal_pos,
        )

    def _is_success(self, achieved_goal, desired_goal):
        return np.linalg.norm(achieved_goal - desired_goal) < self.eps

    def get_random_joint_in_task_space(self):
        for i in range(100):
            joints = self.robot.get_random_joints(set=True)
            pos = self.robot.get_ee_pos()
            if np.all(self.task_ll < pos) & np.all(pos < self.task_ul):
                return joints, pos
        raise ValueError("EE position by a random configuration seems not in task-space.")

    def reset(self):
        with self.sim.no_rendering():
            self.goal_joints, self.goal = self.get_random_joint_in_task_space()
            if self.random_init:
                self.start_joints, self.start = self.get_random_joint_in_task_space()
            else:
                self.start_joints = np.zeros(self.dim)
                self.start = self.robot.get_ee_pos()
            self.robot.set_joints(self.start_joints)
        
        if self.is_render == True:
            self.sim.view_pos("goal", self.goal)
            self.sim.view_pos("curr", self.start)
        return self._get_observation()

    def step(self, action:np.ndarray):
        self.robot.set_action(action)
        obs_ = self._get_observation()
        done = False
        info = dict(
            is_success=self._is_success(obs_["achieved_goal"], self.goal.copy()),
            joints=obs_["observation"].copy(),
            goal_joints=self.goal_joints,
        )
        reward = self.compute_reward(obs_["achieved_goal"].copy(), self.goal.copy(), info)
        if self.is_render == True:
            self.sim.view_pos("curr", obs_["achieved_goal"])
        return obs_, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        if len(achieved_goal.shape) == 2:
            r = np.zeros(len(info))
        else:
            r = 0
        
        if self.reward_type == "task":
            r -= np.linalg.norm(desired_goal - achieved_goal, axis=-1)
        else:
            raise NotImplementedError
        return r
        

register(
    id='RxbotReach-v0',
    entry_point='utils.rxbot.rxbot_reach:RxbotReachEnv',
    max_episode_steps=50,
)