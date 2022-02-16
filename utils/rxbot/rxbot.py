import os
import numpy as np
from gym import spaces
from .assets import *
from .bullet import Bullet
from .bullet_robot import BulletRobot

class Rxbot(BulletRobot):
    def __init__(self, sim:Bullet, dim=2, joint_range=2*np.pi):
        name = "r{}bot".format(dim)
        if dim == 2:
            path = get_data_path() + "/r2.urdf"
        else:
            raise NotImplementedError
        joint_idxs = np.array(range(dim))
        joint_ll = np.array([-1, -1]) * joint_range / 2
        joint_ul = np.array([1, 1]) * joint_range / 2
        action_space = spaces.Box(-1.0, 1.0, shape=(len(joint_idxs),), dtype=np.float32)
        super(Rxbot, self).__init__(
            sim, name, path, joint_idxs,
            joint_ll, joint_ul, action_space
        )
        self.max_joint_change = 0.2 

    def set_action(self, action:np.ndarray):
        action = action.copy()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        joint_target = self.get_joints() + action * self.max_joint_change
        joint_target = np.clip(joint_target, self.joint_ll, self.joint_ul) # joint limit

        self.set_joints(joint_target)
        

class RxbotAbstractEnv:
    def __init__(self, render=False, dim=2, task_ll=[-1,-1,0], task_ul=[1,1,1], joint_range=2*np.pi):
        self.is_render = render
        self.dim = dim

        # parameters
        self.task_range = 2
        self.eps = 0.05

        self.sim = Bullet(
            render=self.is_render, 
            background_color=np.array([153, 255, 153])
        )
        self.robot = Rxbot(self.sim, dim=dim, joint_range=joint_range)
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
            distance=0.9, 
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
