import numpy as np
from .bullet import Bullet
import os
from gym import spaces

JOINT_ATRTIBUTE_NAMES = \
    ["joint_index","joint_name","joint_type",
    "q_index", "u_index", "flags", 
    "joint_damping", "joint_friction","joint_lower_limit",
    "joint_upper_limit","joint_max_force","joint_max_velocity",
    "link_name","joint_axis","parent_frame_pos","parent_frame_orn","parent_index"]

class BulletRobot:
    def __init__(
        self, 
        sim:Bullet, 
        name, 
        path, 
        joint_idxs,
        action_space:spaces.Box,
        base_pos=[0,0,0],
        ee_idx=None,
        joint_ll=None, 
        joint_ul=None,
    ):
        self.sim = sim
        self.name = name
        self.path = path
        self.joint_idxs = joint_idxs
        self.ee_idx = len(joint_idxs)
        if ee_idx is not None:
            self.ee_idx = ee_idx
        self.n_joints = len(self.joint_idxs)
        self.n_actions = self.n_joints
        self.action_space = action_space
        self.sim.loadURDF(
            body_name=self.name,
            fileName=self.path,
            basePosition=base_pos,
            useFixedBase=True,
        )
        self.joint_info = self.get_joint_info()
        self.joint_ll = np.array([self.joint_info[idx]["joint_lower_limit"] for idx in self.joint_idxs]) #from URDF
        self.joint_ul = np.array([self.joint_info[idx]["joint_upper_limit"] for idx in self.joint_idxs])
        if joint_ll is not None:
            self.joint_ll = joint_ll
        if joint_ul is not None:
            self.joint_ul = joint_ul
        self.joint_range = self.joint_ul - self.joint_ll

    def set_joints(self, joints):
        assert len(joints) == len(self.joint_idxs)
        self.sim.set_joint_angles(self.name, self.joint_idxs, joints)
    
    def get_joints(self):
        joints = []
        for idx in self.joint_idxs:
            joint = self.sim.get_joint_angle(self.name, idx)
            joints.append(joint)
        return np.array(joints)

    def get_random_joints(self, set=False):
        rnd_joints = []
        for i in range(self.n_joints):
            joint = np.random.uniform(self.joint_ll[i], self.joint_ul[i])
            rnd_joints.append(joint)
        if set:
            self.set_joints(rnd_joints)
        return np.array(rnd_joints)

    def get_joint_info(self):
        result = {}
        for i in self.joint_idxs:
            values = self.sim.get_joint_info(self.name, i)
            result[i] = {name:value for name, value in zip(JOINT_ATRTIBUTE_NAMES, values)}
        return result
    
    def get_link_pos(self, link):
        return self.sim.get_link_position(self.name, link)

    def get_ee_pos(self):
        return self.get_link_pos(self.ee_idx)
    
    def set_action(self, action:np.ndarray):
        action = action.copy()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        joint_target = self.get_joints() + action * self.max_joint_change
        joint_target = np.clip(joint_target, self.joint_ll, self.joint_ul) # joint limit

        self.set_joints(joint_target)

