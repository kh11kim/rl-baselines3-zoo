import os
import gym
import numpy as np
from gym.envs.registration import register

from .franka_panda_dualarm import PandaDualArmAbstractEnv

class PandaDualArmReachEnv(PandaDualArmAbstractEnv, gym.GoalEnv):
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
            pos_vector = self.robot.get_ee_pos()
            task_vector_ll = np.hstack([self.task_ll, self.task_ll])
            task_vector_ul = np.hstack([self.task_ul, self.task_ul])
            if np.all(task_vector_ll < pos_vector) & np.all(pos_vector < task_vector_ul):
                return joints, pos_vector
                
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
            self.sim.view_pos("goal1", self.goal[:3])
            self.sim.view_pos("goal2", self.goal[3:])
            self.sim.view_pos("curr1", self.start[:3])
            self.sim.view_pos("curr2", self.start[3:])
        return self._get_observation()

    def step(self, action:np.ndarray):
        self.robot.set_action(action)
        obs_ = self._get_observation()
        done = False
        info = dict(
            is_success=self._is_success(obs_["achieved_goal"], self.goal.copy()),
            joints=obs_["observation"].copy(),
            actions=action.copy(),
            goal_joints=self.goal_joints.copy(),
        )
        reward = self.compute_reward(obs_["achieved_goal"].copy(), self.goal.copy(), info)
        if self.is_render == True:
            self.sim.view_pos("curr1", obs_["achieved_goal"][:3])
            self.sim.view_pos("curr2", obs_["achieved_goal"][3:])
        return obs_, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        if len(achieved_goal.shape) == 2:
            r = np.zeros(len(info))
            joints = np.array([i["joints"] for i in info])
            goal_joints = np.array([i["goal_joints"] for i in info])
            actions = np.array([i["actions"] for i in info])
        else:
            r = 0
            joints = info["joints"]
            goal_joints = info["goal_joints"]
            actions = info["actions"]
        
        
        if self.reward_type == "task":
            r -= np.linalg.norm(desired_goal - achieved_goal, axis=-1)
            #test
            mask_goal = np.linalg.norm(desired_goal - achieved_goal, axis=-1) < self.eps
            r -= np.linalg.norm(actions, axis=-1) / 5
        elif self.reward_type == "joint":
            r -= np.linalg.norm(desired_goal - achieved_goal, axis=-1)
            mask1 = np.linalg.norm(desired_goal - achieved_goal, axis=-1) >= self.eps
            mask2 = np.linalg.norm(goal_joints - joints, axis=-1) > np.pi
            r -= mask1 * mask2 * np.linalg.norm(goal_joints - joints, axis=-1) / 10
            
        else:
            raise NotImplementedError
        return r
        

register(
    id='PandaDualArmReach-v0',
    entry_point='utils.rxbot.panda_reach:PandaDualArmReachEnv',
    max_episode_steps=50,
)