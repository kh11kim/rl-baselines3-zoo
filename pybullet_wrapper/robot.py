import numpy as np
from .core import Bullet
from contextlib import contextmanager

JOINT_ATRTIBUTE_NAMES = \
    ["joint_index","joint_name","joint_type",
    "q_index", "u_index", "flags", 
    "joint_damping", "joint_friction","joint_lower_limit",
    "joint_upper_limit","joint_max_force","joint_max_velocity",
    "link_name","joint_axis","parent_frame_pos","parent_frame_orn","parent_index"]

class BulletRobot:
    def __init__(
        self, 
        bullet:Bullet, 
        name, 
        path, 
        ctrl_joint_idxs=None,
        base_pos=[0,0,0],
        ee_idx=None,
    ):
        self.bullet = bullet
        self.name = name
        self.path = path
        
        self.bullet.loadURDF(
            body_name=self.name,
            fileName=self.path,
            basePosition=base_pos,
            useFixedBase=True,
        )
        self.joint_info = self.get_joint_info()
        if ctrl_joint_idxs is None:
            self.ctrl_joint_idxs = range(len(self.joint_info))
        else:
            self.ctrl_joint_idxs = ctrl_joint_idxs
        self.n_joints = len(self.ctrl_joint_idxs)
        self.ee_idx = ee_idx
        self.n_actions = self.n_joints
    
    @property
    def joint_ll(self) -> np.ndarray:
        return np.array([self.joint_info[idx]["joint_lower_limit"] for idx in self.ctrl_joint_idxs])
    
    @property
    def joint_ul(self) -> np.ndarray:
        return np.array([self.joint_info[idx]["joint_upper_limit"] for idx in self.ctrl_joint_idxs])
    
    @property
    def joint_mid(self) -> np.ndarray:
        return (self.joint_ul + self.joint_ll)/2
    
    @property
    def joint_range(self) -> np.ndarray:
        return self.joint_ul - self.joint_ll
    
    @contextmanager
    def no_set_joint(self):
        with self.bullet.no_rendering():
            joints_temp = self.get_joint_angles()
            yield
            self.set_joint_angles(joints_temp)

    def get_joint_info(self):
        result = {}
        # for i in self.ctrl_joint_idxs:
        #     values = self.bullet.get_joint_info(self.name, i)
        #     result[i] = {name:value for name, value in zip(JOINT_ATRTIBUTE_NAMES, values)}
        return self.bullet.get_joint_info(self.name, all=True)

    def set_joint_angles(self, joints):
        assert len(joints) == len(self.ctrl_joint_idxs)
        self.bullet.set_joint_angles(self.name, self.ctrl_joint_idxs, joints)
    
    def control_joint_angles(self, joint_indexes, joint_angles):
        self.bullet.control_joints(
            body=self.name,
            joints=joint_indexes,
            target_angles=joint_angles,
            forces=[100.]*len(joint_angles)
        )
    
    def get_joint_angles(self):
        joints = []
        for idx in self.ctrl_joint_idxs:
            joint = self.bullet.get_joint_angle(self.name, idx)
            joints.append(joint)
        return np.array(joints)

    def get_random_joint_angles(self, set=False):
        with self.bullet.no_rendering():
            rnd_joints = []
            for i in range(self.n_joints):
                joint = np.random.uniform(self.joint_ll[i], self.joint_ul[i])
                rnd_joints.append(joint)
            if set:
                self.set_joint_angles(rnd_joints)
        return np.array(rnd_joints)
    
    def get_link_position(self, link):
        return self.bullet.get_link_position(self.name, link)
    
    def get_link_orientation(self, link):
        return self.bullet.get_link_orientation(self.name, link)

    def get_ee_position(self):
        return self.get_link_position(self.ee_idx)
    
    def get_ee_orientation(self):
        return self.get_link_orientation(self.ee_idx)
    
    
    
    def get_ee_jacobian(self):
        joints = self.get_joint_angles()
        joints = np.hstack([joints, 0, 0]) #add finger
        jac = self.bullet.get_jacobian(self.name, self.ee_idx, joints)
        return jac[:,:-2] #remove finger
    
    def forward_kinematics(self, joint_angles):
        with self.no_set_joint():
            self.set_joint_angles(joint_angles)
            pos = self.get_ee_position()
            orn = self.get_ee_orientation()
        return pos, orn
    
    def inverse_kinematics(self, pos, orn=None):
        result = self.bullet.inverse_kinematics(
            body=self.name,
            link=self.ee_idx,
            position=pos,
            orientation=orn,
            lower_limits=self.joint_ll.tolist(),
            upper_limits=self.joint_ul.tolist(),
            joint_ranges=self.joint_range.tolist(),
            rest_poses=self.joint_mid.tolist(),
        )
        return result[self.ctrl_joint_idxs]