import os
import numpy as np
from gym import spaces
from .assets import *
from .bullet import Bullet
from .bullet_robot import BulletRobot

class Panda(BulletRobot):
    def __init__(self, sim:Bullet, name="panda", action_type="joint", base_pos=[0,0,0]):
        name = name
        path = get_data_path() + f"/franka_description/franka_panda.urdf"
        joint_idxs = np.array([0, 1, 2, 3, 4, 5, 6])
        self.action_type = action_type
        if self.action_type == "joint":
            action_space = spaces.Box(-1.0, 1.0, shape=(len(joint_idxs),), dtype=np.float32)
        elif self.action_type == "task":
            dim = 3
            action_space = spaces.Box(-1.0, 1.0, shape=(dim,), dtype=np.float32)
        elif self.action_type == "null":
            dim = 7
            action_space = spaces.Box(-1.0, 1.0, shape=(dim,), dtype=np.float32)
        super(Panda, self).__init__(
            sim, 
            name, 
            path, 
            joint_idxs,
            action_space,
            base_pos=base_pos,
            ee_idx=10 
        )
        self.max_joint_change = 0.2
    
    def set_action(self, action:np.ndarray):
        action = action.copy()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self.action_type == "joint":
            joint_target = self.get_joints() + action * self.max_joint_change
            joint_target = np.clip(joint_target, self.joint_ll, self.joint_ul) # joint limit
            self.set_joints(joint_target)
        elif self.action_type == "task":
            pos_err = action[:3] * 0.05  #
            #null_motion = action[3:] * 0.1
            jac = self.get_ee_jacobian()
            jac_pos = jac[:3,:]
            jac_pos_pinv = np.linalg.pinv(jac_pos)
            joint_vel = jac_pos_pinv @ pos_err
            #joint_target += (np.eye(7)-jac_pos_pinv@jac_pos) @ null_motion
            joint_target = self.get_joints() + np.clip(joint_vel, 0.1, -0.1)
            self.set_joints(joint_target)
        elif self.action_type == "null":
            pos_err = np.random.uniform(-1,1,size=3) * 0.05  #
            null_motion = action * 0.1
            jac = self.get_ee_jacobian()
            jac_pos = jac[:3,:]
            jac_pos_pinv = np.linalg.pinv(jac_pos)
            joint_task = jac_pos_pinv @ pos_err 
            joint_null = (np.eye(7)-jac_pos_pinv@jac_pos) @ null_motion
            joint_target = self.get_joints() + joint_task + joint_null
            self.set_joints(joint_target)

class PandaAbstractEnv:
    def __init__(self, render=False, task_ll=[-1,-1,0], task_ul=[1,1,1], action_type="joint"):
        self.is_render = render

        self.sim = Bullet(
            render=self.is_render, 
            background_color=np.array([153, 255, 153])
        )
        self.robot = Panda(self.sim, action_type=action_type)
        
        self.task_ll = np.array(task_ll)
        self.task_ul = np.array(task_ul)
        self.observation_space = spaces.Dict(dict(
            observation=spaces.Box(self.robot.joint_ll, self.robot.joint_ul, shape=(self.robot.n_joints,), dtype=np.float32),
            achieved_goal=spaces.Box(self.task_ll, self.task_ul, shape=(len(self.task_ll),), dtype=np.float32),
            desired_goal=spaces.Box(self.task_ll, self.task_ul, shape=(len(self.task_ll),), dtype=np.float32),
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
