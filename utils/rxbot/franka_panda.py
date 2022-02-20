import os
import numpy as np
from gym import spaces
from .assets import *
from .bullet import Bullet
from .bullet_robot import BulletRobot

class Panda(BulletRobot):
    def __init__(self, sim:Bullet):
        name = "panda"
        path = get_data_path() + f"/franka_description/franka_panda.urdf"
        joint_idxs = np.array([0, 1, 2, 3, 4, 5, 6])
        # joint_ll = -np.ones(dim) * joint_range / 2
        # joint_ul = np.ones(dim) * joint_range / 2
        action_space = spaces.Box(-1.0, 1.0, shape=(len(joint_idxs),), dtype=np.float32)
        super(Panda, self).__init__(
            sim, 
            name, 
            path, 
            joint_idxs,
            action_space,
            ee_idx=10 
        )
        self.max_joint_change = 0.2 

class PandaAbstractEnv:
    def __init__(self, render=False, task_ll=[-1,-1,0], task_ul=[1,1,1]):
        self.is_render = render

        self.sim = Bullet(
            render=self.is_render, 
            background_color=np.array([153, 255, 153])
        )
        self.robot = Panda(self.sim)
        self.task_ll = np.array(task_ll)
        self.task_ul = np.array(task_ul)
        self.observation_space = spaces.Dict(dict(
            observation=spaces.Box(self.robot.joint_ll, self.robot.joint_ul, shape=(self.robot.n_joints,), dtype=np.float32),
            achieved_goal=spaces.Box(self.task_ll, self.task_ul, shape=(3,), dtype=np.float32),
            desired_goal=spaces.Box(self.task_ll, self.task_ul, shape=(3,), dtype=np.float32),
        ))
        self.action_space = self.robot.action_space
        
        #visualize
        self._make_env() 
    
    def _make_env(self):
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=0.7, width=0.7, height=0.4, x_offset=0)
        self.sim.place_visualizer(
            target_position=np.zeros(3), 
            distance=1.6, 
            yaw=45, 
            pitch=-30
        )
    
    def reset(self):
        raise NotImplementedError

    def close(self):
        self.sim.close()
    
    def render(
        self,
        mode: str,
        width: int = 720,
        height: int = 480,
        target_position: np.ndarray = np.zeros(3),
        distance: float = 1.4,
        yaw: float = 45,
        pitch: float = -30,
        roll: float = 0,
    ):
        return self.sim.render(
            mode,
            width=width,
            height=height,
            target_position=target_position,
            distance=distance,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
        )

    def step(self):
        raise NotImplementedError
    
    def compute_reward(self):
        raise NotImplementedError
