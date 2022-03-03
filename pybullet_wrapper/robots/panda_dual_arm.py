import numpy as np
from ..core import Bullet
from ..robots import Panda
from contextlib import contextmanager

class PandaDualArm:
    def __init__(
        self, 
        bullet:Bullet, 
        panda1_position=[0,0.2,0],
        panda2_position=[0,-0.2,0],
    ):
        #name = "panda_dualarm"
        self.bullet = bullet
        self.panda1 = Panda(self.bullet, name="panda1", base_pos=panda1_position)
        self.panda2 = Panda(self.bullet, name="panda2", base_pos=panda2_position)
        self.joint_ll = np.hstack([self.panda1.joint_ll, self.panda2.joint_ll])
        self.joint_ul = np.hstack([self.panda1.joint_ul, self.panda2.joint_ul])
        self.n_joints = 14
    
    @contextmanager
    def no_set_joint(self):
        joints_temp = self.get_joint_angles()
        yield
        self.set_joint_angles(joints_temp)

    def set_joint_angles(self, joints):
        assert len(joints) == self.n_joints
        self.panda1.set_joint_angles(joints[:7])
        self.panda2.set_joint_angles(joints[7:])

    def get_joint_angles(self):
        return np.hstack([self.panda1.get_joint_angles(), self.panda2.get_joint_angles()])
    
    def get_random_joint_angles(self, set=False):
        return np.hstack([self.panda1.get_random_joint_angles(set), self.panda2.get_random_joint_angles(set)])
    
    def get_link_position(self, link):
        return np.hstack([self.panda1.get_link_position(), self.panda2.get_link_position()])

    def get_ee_position(self):
        return np.hstack([self.panda1.get_ee_position(), self.panda2.get_ee_position()])
    
    def control_joint_angles(self, joint_angles):
        self.panda1.control_joint_angles(
            body=self.panda1.name,
            joint_indexes=list(range(7)),
            joint_angles=joint_angles[:7],
        )
        self.panda2.control_joint_angles(
            body=self.panda2.name,
            joint_indexes=list(range(7)),
            joint_angles=joint_angles[7:],
        )