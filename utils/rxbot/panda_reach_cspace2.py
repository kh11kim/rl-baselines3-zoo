import os
import gym
import numpy as np
from gym import spaces
from gym.envs.registration import register

from .franka_panda import PandaAbstractEnv

class PandaReachCspaceEnv2(PandaAbstractEnv, gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, render=False, reward_type="jointcol", random_init=True, task_ll=[-1,-1,0], task_ul=[1,1,1]):
        super().__init__(render=render, task_ll=task_ll, task_ul=task_ul)
        self.observation_space = spaces.Box(
            low=np.hstack([self.robot.joint_ll, self.robot.joint_ll]),
            high=np.hstack([self.robot.joint_ll, self.robot.joint_ll]), 
            shape=(self.robot.n_joints*2,), 
            dtype=np.float32
        )

        self.reward_type = reward_type
        self.random_init = random_init
        self._goal_joints = np.zeros(7)
        self._goal_pos = np.zeros(3)
        self.eps = 0.1

    @property
    def goal_joints(self) -> np.ndarray:
        return self._goal_joints.copy()
    
    @goal_joints.setter
    def goal_joints(self, joints):
        self._goal_joints = joints

    @property
    def goal_pos(self) -> np.ndarray:
        return self._goal_pos.copy()
    
    @goal_pos.setter
    def goal_pos(self, pos):
        self._goal_pos = pos

    def _get_observation(self):
        """ observation : joint, joint, joint_goal
        """
        joints = self.robot.get_joints()
        goal_joints = self.goal_joints
        return np.hstack([joints, goal_joints])

    def _is_success(self, joints, goal_joints):
        return np.linalg.norm(joints - goal_joints) < self.eps

    def get_random_joint_in_task_space(self):
        for i in range(100):
            joints = self.robot.get_random_joints(set=True)
            pos = self.robot.get_ee_pos()
            if np.all(self.task_ll < pos) & np.all(pos < self.task_ul):
                return joints, pos
        raise ValueError("EE position by a random configuration seems not in task-space.")
    
    def reset(self):
        with self.sim.no_rendering():
            self.goal_joints = self.robot.get_random_joints(set=True)
            self.goal_pos = self.robot.get_ee_pos()
            if self.random_init:
                self.start_joints = self.robot.get_random_joints(set=True)
                self.start_pos = self.robot.get_ee_pos()
            else:
                self.start_joints = np.zeros(self.dim)
                self.start_pos = self.robot.get_ee_pos()
            self.robot.set_joints(self.start_joints)
        
        if self.is_render == True:
            self.sim.view_pos("goal", self.goal_pos)
            self.sim.view_pos("curr", self.start_pos)
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
            is_success=self._is_success(self.robot.get_joints(), self.goal_joints),
            actions=action.copy(),
            collisions=self.is_collision(self.robot.get_joints())
        )
        reward = self.compute_reward(self.robot.get_joints(), self.goal_joints, info)
        if self.is_render == True:
            self.sim.view_pos("curr", self.robot.get_ee_pos())
        return obs_, reward, done, info

    def compute_reward(self, joints, goal_joints, info):
        return - np.linalg.norm(joints - goal_joints, axis=-1)/4
        # if len(achieved_goal.shape) == 2:
        #     is_success = np.array([1 if i["is_success"] else 0 for i in info])
        #     joints = np.array([i["joints"] for i in info])
        #     goal_joints = np.array([i["goal_joints"] for i in info])
        #     actions = np.array([i["actions"] for i in info])
        #     collisions = np.array([i["collisions"] for i in info])
        #     r = np.zeros(len(info))
        # else:
        #     is_success = info["is_success"]
        #     joints = info["joints"]
        #     goal_joints = info["goal_joints"]
        #     actions = info["actions"]
        #     collisions = info["collisions"]
        #     r = -0.
        

        # if "task" in self.reward_type:
        #     raise NotImplementedError
        #     #r -= np.linalg.norm(desired_goal - achieved_goal, axis=-1)
        # if "action" in self.reward_type:
        #     #mask_goal = np.linalg.norm(desired_goal - achieved_goal, axis=-1) < self.eps
        #     r -= np.linalg.norm(actions, axis=-1) / 10
        # if "joint" in self.reward_type:
        #     r -= np.linalg.norm(desired_goal - achieved_goal, axis=-1) / 4
        # if "col" in self.reward_type:
        #     r -= collisions * 1.
        # return r * (1 - is_success)
        

register(
    id='PandaReachCspace-v1',
    entry_point='utils.rxbot.panda_reach_cspace2:PandaReachCspaceEnv2',
    max_episode_steps=50,
)