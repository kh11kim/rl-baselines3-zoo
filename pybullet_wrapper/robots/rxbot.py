import numpy as np
from ..core import Bullet
from ..assets import get_data_path
from ..robot import BulletRobot

class Rxbot(BulletRobot):
    def __init__(self, bullet:Bullet, dim=2, base_pos=[0,0,0]):
        name = f"r{dim}bot"
        path = get_data_path() + f"/r{dim}.urdf"
        super(Rxbot, self).__init__(
            bullet, 
            name, 
            path, 
            base_pos=base_pos,
            ee_idx=dim 
        )
        self.max_joint_change = 0.2
        self.set_joint_angles(self.joint_mid)