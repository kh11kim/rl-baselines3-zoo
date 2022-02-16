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
        joint_ll, 
        joint_ul,
        action_space:spaces.Box
    ):
        self.sim = sim
        self.name = name
        self.path = path
        self.joint_idxs = joint_idxs
        self.ee_idx = 2
        self.n_joints = len(self.joint_idxs)
        self.n_actions = self.n_joints
        self.joint_ll = joint_ll
        self.joint_ul = joint_ul
        self.joint_range = joint_ul - joint_ll
        self.action_space = action_space
        self.sim.loadURDF(
            body_name=self.name,
            fileName=self.path,
            basePosition=[0,0,0],
            useFixedBase=True,
        )
        self.joint_info = self.get_joint_info()

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

