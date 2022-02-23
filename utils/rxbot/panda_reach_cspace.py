import os
import gym
import numpy as np
from gym import spaces
from gym.envs.registration import register

from .franka_panda import PandaAbstractEnv

class PandaReachCspaceEnv(PandaAbstractEnv, gym.GoalEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, render=False, reward_type="jointcol", random_init=True, task_ll=[-1,-1,0], task_ul=[1,1,1]):
        super().__init__(render=render, task_ll=task_ll, task_ul=task_ul)
        self.observation_space = spaces.Dict(dict(
            observation=spaces.Box(-1, 1, shape=(1,), dtype=np.float32),#spaces.Box(self.robot.joint_ll, self.robot.joint_ul, shape=(self.robot.n_joints,), dtype=np.float32),
            achieved_goal=spaces.Box(self.robot.joint_ll, self.robot.joint_ul, shape=(self.robot.n_joints,), dtype=np.float32),
            desired_goal=spaces.Box(self.robot.joint_ll, self.robot.joint_ul, shape=(self.robot.n_joints,), dtype=np.float32),
        ))

        self.reward_type = reward_type
        self.random_init = random_init
        self._goal_joint = np.zeros(7)
        self._goal_pos = np.zeros(3)
        self.eps = 0.1

    @property
    def goal_joints(self):
        return self._goal_joint.copy()
    
    @goal_joints.setter
    def goal_joints(self, joints):
        self._goal_joint = joints

    @property
    def goal_pos(self):
        return self._goal_pos.copy()
    
    @goal_pos.setter
    def goal_pos(self, pos):
        self._goal_pos = pos

    def _get_observation(self):
        """ observation : joint, joint, joint_goal
        """
        joints = self.robot.get_joints()
        return dict(
            observation=0,
            achieved_goal=joints,
            desired_goal=self.goal_joints,
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
            is_success=self._is_success(obs_["achieved_goal"], obs_["desired_goal"]),
            actions=action.copy(),
            collisions=self.is_collision(obs_["achieved_goal"])
        )
        reward = self.compute_reward(obs_["achieved_goal"], obs_["desired_goal"], info)
        if self.is_render == True:
            self.sim.view_pos("curr", self.robot.get_ee_pos())
        return obs_, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        if len(achieved_goal.shape) == 2:
            is_success = np.array([1 if i["is_success"] else 0 for i in info])
            actions = np.array([i["actions"] for i in info])
            collisions = np.array([i["collisions"] for i in info])
            r = np.zeros(len(info))
        else:
            is_success = info["is_success"]
            actions = info["actions"]
            collisions = info["collisions"]
            r = -0.
        
        if "joint" in self.reward_type:
            # delta = 0.5
            # res = np.linalg.norm(desired_goal - achieved_goal, ord=1, axis=-1)
            # cond = [res < delta, res >= delta]
            # small_res = 0.5 * res**2
            # large_res = delta * res - 0.5 * delta**2
            # r -= np.select(cond, [small_res, large_res])/4
            r -= np.linalg.norm(desired_goal - achieved_goal, axis=-1) / 4

        if "sparse" in self.reward_type:
            r -= -1 * (1 - is_success)

        if "action" in self.reward_type:
            #mask_goal = np.linalg.norm(desired_goal - achieved_goal, axis=-1) < self.eps
            r -= np.linalg.norm(actions, axis=-1) / 10
        
        if "col" in self.reward_type:
            r -= collisions * 1.
        return r
        

register(
    id='PandaReachCspace-v0',
    entry_point='utils.rxbot.panda_reach_cspace:PandaReachCspaceEnv',
    max_episode_steps=50,
)