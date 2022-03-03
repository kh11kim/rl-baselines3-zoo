import numpy as np
from ..core import Bullet
from ..assets import get_data_path
from ..robot import BulletRobot

class Panda(BulletRobot):
    def __init__(self, bullet:Bullet, name="panda", base_pos=[0,0,0]):
        name = name
        path = get_data_path() + f"/franka_description/franka_panda.urdf"
        ctrl_joint_idxs = np.array([0, 1, 2, 3, 4, 5, 6])
        super(Panda, self).__init__(
            bullet, 
            name, 
            path, 
            ctrl_joint_idxs,
            base_pos=base_pos,
            ee_idx=10 
        )
        self.max_joint_change = 0.2