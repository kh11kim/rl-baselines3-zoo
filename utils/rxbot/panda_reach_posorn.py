import os
import gym
import numpy as np
from gym.envs.registration import register
from spatial_math_mini import SO3
from .franka_panda import PandaAbstractEnv

class PandaReachEnvPosOrn(PandaAbstractEnv, gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, render=False, reward_type="posorn", random_init=True, task_ll=[0.1,-1,0, -1,-1,-1,-1], task_ul=[1,1,1,1,1,1,1]):
        super().__init__(render=render, task_ll=task_ll, task_ul=task_ul)
        self.reward_type = reward_type
        self.random_init = random_init
        
        self.eps = 0.05
        
    def _get_observation(self):
        """ observation : joint, ee_curr, ee_goal
        """
        joints = self.robot.get_joints()
        ee_pos = self.robot.get_ee_pos()
        ee_orn = self.robot.get_ee_orn()
        ee = np.hstack([ee_pos, ee_orn])
        goal = self.goal.copy()
        return dict(
            observation=joints,
            achieved_goal=ee,
            desired_goal=goal,
        )

    def orn_distance(self, orn1, orn2):
        return 1 - np.linalg.norm(orn1@orn2)

    def _is_success(self, achieved_goal, desired_goal):
        pos1, orn1 = achieved_goal[:3], achieved_goal[3:]
        pos2, orn2 = desired_goal[:3], desired_goal[3:]
        pos_distance = np.linalg.norm(pos1 - pos2)
        orn_distance = self.orn_distance(orn1, orn2)
        return (pos_distance + orn_distance) < self.eps

    def get_random_joint_in_task_space(self):
        for i in range(100):
            joints = self.robot.get_random_joints(set=True)
            pos = self.robot.get_ee_pos()
            orn = self.robot.get_ee_orn()
            posorn = np.hstack([pos,orn])
            if np.all(self.task_ll < posorn) & np.all(posorn < self.task_ul):
                return joints, np.hstack([pos, orn])
        raise ValueError("EE position by a random configuration seems not in task-space.")

    def reset(self):
        param = 0.5
        with self.sim.no_rendering():
            for i in range(100):
                rnd_joints, _ = self.get_random_joint_in_task_space()
                if self.random_init:
                    self.start_joints, start = self.get_random_joint_in_task_space()
                else:
                    self.start_joints = self.robot.joint_mid
                    start = np.hstack([self.robot.get_ee_pos(), self.robot.get_ee_orn()])
                self.goal_joints = self.start_joints + (rnd_joints-self.start_joints) * param
                self.robot.set_joints(self.goal_joints)
                self.goal = np.hstack([self.robot.get_ee_pos(), self.robot.get_ee_orn()])
                self.robot.set_joints(self.start_joints)
                start_pose_feasible = np.all(self.task_ll < start) & np.all(start < self.task_ul)
                goal_pose_feasible = np.all(self.task_ll < self.goal) & np.all(self.goal < self.task_ul)
                if start_pose_feasible & goal_pose_feasible:
                    break
        
        if self.is_render == True:
            self.sim.view_frame("goal", self.goal[:3], self.goal[3:])
            self.sim.view_frame("curr", start[:3], start[3:])
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
            joints=obs_["observation"].copy(),
            actions=action.copy(),
            #goal_joints=self.goal_joints.copy(),
            collisions=self.is_collision(obs_["observation"])
        )
        reward = self.compute_reward(obs_["achieved_goal"].copy(), self.goal.copy(), info)
        if self.is_render == True:
            self.sim.view_frame("curr", obs_["achieved_goal"][:3], obs_["achieved_goal"][3:])
        return obs_, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        if len(achieved_goal.shape) == 2:
            r = np.zeros(len(info))
            joints = np.array([i["joints"] for i in info])
            #goal_joints = np.array([i["goal_joints"] for i in info])
            actions = np.array([i["actions"] for i in info])
            collisions = np.array([i["collisions"] for i in info])
            achieved_goal_pos = achieved_goal[:,:3]
            achieved_goal_orn = achieved_goal[:,3:]
            desired_goal_pos = desired_goal[:,:3]
            desired_goal_orn = desired_goal[:,3:]
        else:
            r = 0
            joints = info["joints"]
            #goal_joints = info["goal_joints"]
            actions = info["actions"]
            collisions = info["collisions"]
            achieved_goal_pos = achieved_goal[:3]
            achieved_goal_orn = achieved_goal[3:]
            desired_goal_pos = desired_goal[:3]
            desired_goal_orn = desired_goal[3:]
        
        if "pos" in self.reward_type:
            r -= np.linalg.norm(desired_goal_pos - achieved_goal_pos, axis=-1)
        if "orn" in self.reward_type:
            if len(achieved_goal.shape) == 2:
                dot_prod = np.array([desired_goal_orn[i] @ achieved_goal_orn[i] for i in range(len(desired_goal_orn))])
            else:
                dot_prod = desired_goal_orn @ achieved_goal_orn
            r -= 1 - np.abs(dot_prod) / 2
            
        if "action" in self.reward_type:
            #mask_goal = np.linalg.norm(desired_goal - achieved_goal, axis=-1) < self.eps
            r -= np.linalg.norm(actions, axis=-1) / 10
        if "joint" in self.reward_type:
            #mask1 = np.linalg.norm(desired_goal - achieved_goal, axis=-1) >= 0.5 #self.eps*2
            #mask2 = np.linalg.norm(goal_joints - joints, axis=-1) > np.pi
            r -= np.linalg.norm(joints - self.robot.joint_mid, axis=-1) / 40
        if "col" in self.reward_type:
            r -= collisions * 1.
        return r
        

register(
    id='PandaReachPosOrn-v0',
    entry_point='utils.rxbot.panda_reach_posorn:PandaReachEnvPosOrn',
    max_episode_steps=50,
)