import numpy as np
import warnings
from ..core import Bullet
from ..scene_maker import BulletSceneMaker
from ..collision_checker import BulletCollisionChecker
from ..robots import PandaDualArm

class PandaDualArmEnvBase:
    def __init__(self, render=False, arm_distance=0.4):
        self.is_render = render
        self.bullet = Bullet(render=render)
        self.scene_maker = BulletSceneMaker(self.bullet)
        self.robot = PandaDualArm(
            self.bullet,
            panda1_position=[0,arm_distance/2,0],
            panda2_position=[0,-arm_distance/2,0]
        )
        self._make_env()
        self.checker = BulletCollisionChecker(self.bullet)
        self.task_ll = [0, -0.5, 0]
        self.task_ul = [0.5, 0.5, 0.5]

    def _make_env(self):
        self.scene_maker.create_plane(z_offset=-0.4)
        self.scene_maker.create_table(length=1.0, width=1.0, height=0.4, x_offset=0)
        self.bullet.place_visualizer(
            target_position=np.zeros(3), 
            distance=1.6, 
            yaw=45, 
            pitch=-30
        )
    
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
        return self.bullet.render(
            mode,
            width=width,
            height=height,
            target_position=target_position,
            distance=distance,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
        )
    
    def get_random_configuration(self, collision_free=False):
        if not collision_free:
            return self.robot.get_random_joint_angles(set=False)
        else:
            random_joint_angles = None
            with self.robot.no_set_joint():
                for i in range(100):
                    self.robot.get_random_joint_angles(set=True)
                    if not self.checker.is_collision():
                        random_joint_angles = self.robot.get_joint_angles()
        return random_joint_angles
    
    def get_random_free_configuration_in_taskspace(self, panda1_first=True):
        if panda1_first:
            first_robot = self.robot.panda1
            second_robot = self.robot.panda2
        else:
            first_robot = self.robot.panda2
            second_robot = self.robot.panda1
        
        for robot in [first_robot, second_robot]:
            while True:
                robot.get_random_joint_angles(set=True)
                ee_position = robot.get_ee_position()
                is_collision_free = not self.checker.is_collision()
                is_in_taskspace = np.all(self.task_ll < ee_position) \
                                  & np.all(ee_position < self.task_ul)
                if is_collision_free & is_in_taskspace:
                    break
        return self.robot.get_joint_angles()

    def reset(self):
        joints_init = self.get_random_configuration(collision_free=True)
        if joints_init is not None:
            self.robot.set_joint_angles(joints_init)
            return joints_init
        else:
            warnings.warn('env.reset() can`t find feasible reset configuration')
            return None
    
    def is_collision(self, joint_angles=None):
        if joint_angles is None:
            joint_angles = self.robot.get_joint_angles()
        result = False
        with self.robot.no_set_joint():
            self.robot.set_joint_angles(joint_angles)
            if self.checker.is_collision():
                result = True
        return result
