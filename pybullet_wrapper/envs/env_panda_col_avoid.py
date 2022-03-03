import gym
import numpy as np
from gym import spaces
from .env_panda import PandaEnvBase
from ..collision_checker import BulletCollisionChecker

class PandaGymEnvColAvoid(PandaEnvBase, gym.Env):
    def __init__(self, render=False, reward_type="joint", level=0.5):
        self.level = level
        self.reward_type = reward_type
        task_ll = np.array([-0.5, -0.5, 0.])
        task_ul = np.array([0.5, 0.5, 1.])
        self._goal = np.zeros(7)
        self._obstacle = np.zeros(3)
        super().__init__(
            render=render,
            task_ll=task_ll,
            task_ul=task_ul,
        )        
        self.observation_space = spaces.Dict(dict(
            observation=spaces.Box(self.task_ll, self.task_ul, shape=(3,), dtype=np.float32),
            achieved_goal=spaces.Box(
                self.robot.joint_ll, 
                self.robot.joint_ul, 
                shape=(self.robot.n_joints,), 
                dtype=np.float32
            ),
            desired_goal=spaces.Box(
                self.robot.joint_ll, 
                self.robot.joint_ul, 
                shape=(self.robot.n_joints,), 
                dtype=np.float32
            ),
        ))
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.robot.n_joints,), dtype=np.float32)
        self.scene_maker.make_sphere_obstacle("obstacle", self.obstacle)
        self.checker = BulletCollisionChecker(self.bullet)
        self.eps = 0.5
        self.obs_check_list = self.checker.get_collision_check_list_by_name("panda", "obstacle")
    
    @property
    def goal(self):
        return self._goal.copy()
    
    @property
    def obstacle(self):
        return self._obstacle.copy()
    
    @goal.setter
    def goal(self, arr: np.ndarray):
        self._goal = arr

    @obstacle.setter
    def obstacle(self, arr: np.ndarray):
        self._obstacle = arr
    
    def set_action(self, action: np.ndarray):
        joint_prev = self.robot.get_joint_angles()
        action = action.copy()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        joint_target = self.robot.get_joint_angles()
        joint_target += action * self.max_joint_change
        joint_target = np.clip(joint_target, self.robot.joint_ll, self.robot.joint_ul)
        self.robot.set_joint_angles(joint_target)
        if self.is_collision():
            self.robot.set_joint_angles(joint_prev)
            self._collision_flag = True
        else:
            self._collision_flag = False
            
    def is_success(self, joint_curr: np.ndarray, joint_goal: np.ndarray):
        return np.linalg.norm(joint_curr - joint_goal) < self.eps

    def get_random_obstacle(self):
        return np.random.uniform(
            low=self.task_ll,
            high=self.task_ul,
        )

    def get_observation(self):
        return dict(
            observation=self.obstacle,
            achieved_goal=self.robot.get_joint_angles(),
            desired_goal=self.goal,
        )
    
    def min_dist_from_obstacle(self):
        distances = self.checker.compute_distances(self.obs_check_list, max_distance=1.)
        return np.min(distances)

    def reset(self):
        self.obstacle = self.get_random_obstacle()
        self.goal = self.get_random_configuration(collision_free=True)
        self.robot.set_joint_angles(self.goal)
        goal_ee = self.robot.get_ee_position()
        while True:
            random_start = self.get_random_configuration(collision_free=False)
            self.start = self.goal + (random_start - self.goal) * self.level
            self.robot.set_joint_angles(self.start)
            if not self.is_collision():
                start_ee = self.robot.get_ee_position()
                break
        
        if self.is_render:
            self.scene_maker.view_position("goal", goal_ee)
            self.scene_maker.view_position("curr", start_ee)
            self.scene_maker.make_sphere_obstacle("obstacle", self.obstacle)
        return self.get_observation()
    
    def step(self, action: np.ndarray):
        self.set_action(action)
        obs_ = self.get_observation()
        done = False
        info = dict(
            is_success=self.is_success(obs_["achieved_goal"], obs_["desired_goal"]),
            obstacles=self.obstacle,
            actions=action.copy(),
            min_dist_from_obs=self.min_dist_from_obstacle(),
            collisions=self._collision_flag,
        )
        reward = self.compute_reward(obs_["achieved_goal"], obs_["desired_goal"], info)
        if self.is_render:
            self.scene_maker.view_position("curr", self.robot.get_ee_position())
        return obs_, reward, done, info
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        if len(achieved_goal.shape) == 2:
            actions = np.array([i["actions"] for i in info])
            collisions = np.array([i["collisions"] for i in info])
            obstacles = np.array([i["min_dist_from_obs"] for i in info])
        else:
            actions = info["actions"]
            collisions = info["collisions"]
            obstacles = info["min_dist_from_obs"]
        r = - np.linalg.norm(achieved_goal - desired_goal, axis=-1)

        if "obs" in self.reward_type:
            r -= (0.2/(obstacles + 0.2))**8

        if "action" in self.reward_type:
            r -= np.linalg.norm(actions, axis=-1) / 10
        
        if "col" in self.reward_type:
            r -= collisions * 1.
        
        return r