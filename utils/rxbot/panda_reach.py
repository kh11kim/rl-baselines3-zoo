import os
import gym
import numpy as np
from gym.envs.registration import register

from .franka_panda import PandaAbstractEnv

class PandaReachEnv(PandaAbstractEnv, gym.GoalEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, render=False, reward_type="task", random_init=True, task_ll=[0,-1,0], task_ul=[1,1,1]):
        super().__init__(render=render, task_ll=task_ll, task_ul=task_ul)
        self.reward_type = reward_type
        self.random_init = random_init
        
        self.eps = 0.05
        
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
    
    def is_collision(self, joints):
        is_ll = np.any(joints == self.robot.joint_ll)
        is_ul = np.any(joints == self.robot.joint_ul)
        return is_ll | is_ul

    def step(self, action:np.ndarray):
        self.robot.set_action(action)
        obs_ = self._get_observation()
        done = False
        info = dict(
            is_success=self._is_success(obs_["achieved_goal"], self.goal.copy()),
            joints=obs_["observation"].copy(),
            actions=action.copy(),
            goal_joints=self.goal_joints.copy(),
            collisions=self.is_collision(obs_["observation"])
        )
        reward = self.compute_reward(obs_["achieved_goal"].copy(), self.goal.copy(), info)
        if self.is_render == True:
            self.sim.view_pos("curr", obs_["achieved_goal"])
        return obs_, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        if len(achieved_goal.shape) == 2:
            r = np.zeros(len(info))
            joints = np.array([i["joints"] for i in info])
            goal_joints = np.array([i["goal_joints"] for i in info])
            actions = np.array([i["actions"] for i in info])
            collisions = np.array([i["collisions"] for i in info])
        else:
            r = 0
            joints = info["joints"]
            goal_joints = info["goal_joints"]
            actions = info["actions"]
            collisions = info["collisions"]
        
        
        if "task" in self.reward_type:
            r -= np.linalg.norm(desired_goal - achieved_goal, axis=-1)
        if "action" in self.reward_type:
            #mask_goal = np.linalg.norm(desired_goal - achieved_goal, axis=-1) < self.eps
            r -= np.linalg.norm(actions, axis=-1) / 10
        if "joint" in self.reward_type:
            #mask1 = np.linalg.norm(desired_goal - achieved_goal, axis=-1) >= 0.5 #self.eps*2
            #mask2 = np.linalg.norm(goal_joints - joints, axis=-1) > np.pi
            r -= np.linalg.norm(joints, axis=-1) / 40
        if "col" in self.reward_type:
            r -= collisions * 1.
        return r
        

register(
    id='PandaReach-v0',
    entry_point='utils.rxbot.panda_reach:PandaReachEnv',
    max_episode_steps=50,
)