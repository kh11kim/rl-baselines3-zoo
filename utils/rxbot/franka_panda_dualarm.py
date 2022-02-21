import os
import numpy as np
from gym import spaces
from .assets import *
from .bullet import Bullet
from .bullet_robot import BulletRobot
from .franka_panda import Panda

class PandaDualArm:
    def __init__(self, sim:Bullet):
        #name = "panda_dualarm"
        self.sim = sim
        self.panda1 = Panda(self.sim, name="panda1", base_pos=[0,0.2,0])
        self.panda2 = Panda(self.sim, name="panda2", base_pos=[0,-0.2,0])
        self.joint_ll = np.hstack([self.panda1.joint_ll, self.panda2.joint_ll])
        self.joint_ul = np.hstack([self.panda1.joint_ul, self.panda2.joint_ul])
        self.n_joints = 14
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.n_joints,), dtype=np.float32)
        #self.panda2 = Panda(self.sim, name="panda2")
        #self.joint_idxs = np.array(range(12))
    
    def set_joints(self, joints):
        #assert len(joints) == len(self.joint_idxs)
        self.panda1.set_joints(joints[:7])
        self.panda2.set_joints(joints[7:])

    def get_joints(self):
        return np.hstack([self.panda1.get_joints(), self.panda2.get_joints()])
    
    def get_random_joints(self, set=False):
        return np.hstack([self.panda1.get_random_joints(set), self.panda2.get_random_joints(set)])
    
    def get_link_pos(self, link):
        return np.hstack([self.panda1.get_link_pos(), self.panda2.get_link_pos()])

    def get_ee_pos(self):
        return np.hstack([self.panda1.get_ee_pos(), self.panda2.get_ee_pos()])
    
    def set_action(self, action:np.ndarray):
        self.panda1.set_action(action[:7])
        self.panda2.set_action(action[7:])
    


class PandaDualArmAbstractEnv:
    def __init__(self, render=False, task_ll=[-1.5,-1.5,0], task_ul=[1.5,1.5,1]):
        self.is_render = render

        self.sim = Bullet(
            render=self.is_render, 
            background_color=np.array([153, 255, 153])
        )
        self.robot = PandaDualArm(self.sim)
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
        self.sim.create_table(length=1.0, width=1.0, height=0.4, x_offset=0)
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
