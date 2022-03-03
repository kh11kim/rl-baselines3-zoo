import numpy as np
import gym
from gym import spaces
from gym.envs.registration import register
from ..core import Bullet
from ..scene_maker import BulletSceneMaker
from ..collision_checker import BulletCollisionChecker
from ..robots import Panda
import pybullet as p
from spatial_math_mini import *

class PandaEnvBase:
    def __init__(self, render=False):
        self.is_render = render
        self.bullet = Bullet(render=render)
        self.scene_maker = BulletSceneMaker(self.bullet)
        self.robot = Panda(
            self.bullet
        )
        self._make_env()
        self.checker = BulletCollisionChecker(self.bullet)
        
        self.task_ll = np.array([0, -0.5, 0])
        self.task_ul = np.array([0.5, 0.5, 0.5])
        self._joint_start = None
        self._joint_goal = None
        self.max_joint_change = 0.1
    
    @property
    def joint_start(self):
        return self._joint_start.copy()
    
    @property
    def joint_goal(self):
        return self._joint_goal.copy()
    
    @joint_start.setter
    def joint_start(self, joint_angles):
        self._joint_start = joint_angles

    @joint_goal.setter
    def joint_goal(self, joint_angles):
        self._joint_goal = joint_angles

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
        
        random_joint_angles = None
        with self.robot.no_set_joint():
            while True:
                self.robot.get_random_joint_angles(set=True)
                if not self.checker.is_collision():
                    random_joint_angles = self.robot.get_joint_angles()
                    break

        return random_joint_angles
    
    def get_random_free_configuration_in_taskspace(self):
        while True:
            self.robot.get_random_joint_angles(set=True)
            ee_position = self.robot.get_ee_position()
            is_collision_free = not self.checker.is_collision()
            is_in_taskspace = np.all(self.task_ll < ee_position) \
                                & np.all(ee_position < self.task_ul)
            if is_collision_free & is_in_taskspace:
                break
        return self.robot.get_joint_angles()

    def reset(self):
        self.joint_goal = self.get_random_configuration(collision_free=True)
        self.joint_start = self.get_random_configuration(collision_free=True)
        self.robot.set_joint_angles(self.joint_start)
    
    def is_collision(self, joint_angles=None):
        if joint_angles is None:
            joint_angles = self.robot.get_joint_angles()
        result = False
        with self.robot.no_set_joint():
            self.robot.set_joint_angles(joint_angles)
            if self.checker.is_collision():
                result = True
        return result

class PandaGymEnv(PandaEnvBase, gym.GoalEnv):
    def __init__(self, render=False, reward_type="pose"):
        self.reward_type = reward_type
        super().__init__(render=render)
        self.n_obs = self.robot.n_joints + 3 + 9
        self.task_ll = np.array([0, -0.5, 0])
        self.task_ul = np.array([0.5, 0.5, 0.5])
        self.orn_ll = np.array([-np.pi]*3)
        self.orn_ul = np.array([np.pi]*3)
        self.posorn = True
        if self.posorn:
                self.observation_space = spaces.Dict(dict(
                observation=spaces.Box(self.robot.joint_ll, self.robot.joint_ul, shape=(self.robot.n_joints,), dtype=np.float32),    
                achieved_goal=spaces.Box(np.hstack([self.task_ll, self.orn_ll]), np.hstack([self.task_ul, self.orn_ul]), shape=(3+3,), dtype=np.float32),
                desired_goal=spaces.Box(np.hstack([self.task_ll, self.orn_ll]), np.hstack([self.task_ul, self.orn_ul]), shape=(3+3,), dtype=np.float32),#, self.orn_ul
            ))
        else:
            self.observation_space = spaces.Dict(dict(
                observation=spaces.Box(self.robot.joint_ll, self.robot.joint_ul, shape=(self.robot.n_joints,), dtype=np.float32),    
                achieved_goal=spaces.Box(np.hstack([self.task_ll]), np.hstack([self.task_ul]), shape=(3,), dtype=np.float32),
                desired_goal=spaces.Box(np.hstack([self.task_ll]), np.hstack([self.task_ul]), shape=(3,), dtype=np.float32),#, self.orn_ul
            ))
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.robot.n_joints,), dtype=np.float32)
        self._goal = None
        self.eps = 0.1

    
    def set_debug_mode(self):
        self.bullet.physics_client.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        self.bullet.physics_client.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
        joint_angles = self.robot.get_joint_angles()
        self.param_idxs = []
        for idx in self.robot.ctrl_joint_idxs:
            name = self.robot.joint_info[idx]["joint_name"].decode("utf-8")
            param_idx = self.bullet.physics_client.addUserDebugParameter(
                name,-4,4,
                joint_angles[idx]
            )
            self.param_idxs.append(param_idx)
    
    def do_debug_mode(self):
        joint_param_values = []
        for param in self.param_idxs:
            joint_param_values.append(p.readUserDebugParameter(param))
        self.robot.set_joint_angles(joint_param_values)

    @property
    def goal(self):
        return self._goal.copy()
    
    @goal.setter
    def goal(self, arr: np.ndarray):
        self._goal = arr.copy()
    
    def get_observation(self):
        joint_angles = self.robot.get_joint_angles()
        pos = self.robot.get_ee_position()
        orn = self.robot.get_ee_orientation()
        axis, angle = p.getAxisAngleFromQuaternion(orn)
        if self.posorn:
            achieved_goal = np.hstack([pos, np.array(axis)*angle]) #, np.array(axis)*angle
        else:
            achieved_goal = np.hstack([pos]) #, np.array(axis)*angle
        desired_goal = self.goal
        return dict(
            observation=joint_angles,
            achieved_goal=achieved_goal,
            desired_goal=desired_goal,
        )
    
    def set_action(self, action: np.ndarray):
        joint_prev = self.robot.get_joint_angles()
        action = action.copy()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        joint_target = self.robot.get_joint_angles()
        joint_target += action * self.max_joint_change
        joint_target = np.clip(
            joint_target, 
            self.robot.joint_ll, 
            self.robot.joint_ul
        ) # joint limit
        self.robot.set_joint_angles(joint_target)
        if self.checker.is_collision():
            self.robot.set_joint_angles(joint_prev)
            self._collision_flag = True
        self._collision_flag = False

    def axisangle_to_qtn(self, axisangle):
        angle = np.linalg.norm(axisangle)
        axis = axisangle/angle
        s = np.sin(angle/2)
        w = np.cos(angle/2)
        x, y, z = axis * s
        return np.array([x,y,z,w])

    def distance(self, pose1: np.ndarray, pose2: np.ndarray) -> np.float64:
        pos1, orn1 = pose1[:3], self.axisangle_to_qtn(pose1[3:])
        pos2, orn2 = pose2[:3], self.axisangle_to_qtn(pose2[3:])
        lmda = orn1@orn2
        delta = np.linalg.norm(pos1 - pos2)
        delta_theta = 1 - np.abs(lmda)
        #_, diff_angle = (SO3_1.inv()@SO3_2).to_axisangle()
        #diff_mat = orn1.T@orn2
        #lmbda = (np.trace(diff_mat)-1) / 2
        #delta_theta = 1 - np.abs(lmbda)
        #delta_theta = np.arccos((np.trace(diff_mat)-1)/2)
        return delta + delta_theta/(10)

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray):
        return self.distance(achieved_goal, desired_goal) < self.eps

    def is_limit(self, joint_angles):
        is_ll = np.any(joint_angles == self.robot.joint_ll)
        is_ul = np.any(joint_angles == self.robot.joint_ul)
        return is_ll | is_ul

    def reset(self):
        self.rnd_joints1 = self.get_random_free_configuration_in_taskspace()
        self.rnd_joints2 = self.get_random_free_configuration_in_taskspace()
        self.start_joints = self.robot.joint_mid + (self.rnd_joints1 - self.robot.joint_mid) * 0.0
        self.goal_joints = self.start_joints + (self.rnd_joints2 - self.start_joints) * 0.5
        # make goal
        self.robot.set_joint_angles(self.goal_joints)
        goal_pos = self.robot.get_ee_position()
        goal_orn = self.robot.get_ee_orientation()
        goal_axis, goal_angle = p.getAxisAngleFromQuaternion(goal_orn)
        
        # make start
        self.robot.set_joint_angles(self.start_joints)
        start_pos = self.robot.get_ee_position()
        start_orn = self.robot.get_ee_orientation()
        start_axis, start_angle = p.getAxisAngleFromQuaternion(start_orn)

        if self.posorn:
            self.goal = np.hstack([goal_pos, np.array(goal_axis)*goal_angle]) #, np.array(goal_axis)*goal_angle
            self.start = np.hstack([start_pos, np.array(start_axis)*start_angle]) #, np.array(start_axis)*start_angle
        else:
            self.goal = np.hstack([goal_pos]) #, np.array(goal_axis)*goal_angle
            self.start = np.hstack([start_pos]) #, np.array(start_axis)*start_angle
        self.robot.set_joint_angles(self.start_joints)

        if self.is_render:
            self.scene_maker.view_frame("goal", goal_pos, goal_orn)
            self.scene_maker.view_frame("curr", start_pos, start_orn)
        return self.get_observation()


    def step(self, action: np.ndarray):
        self.set_action(action)
        obs_ = self.get_observation()
        done = False
        info = dict(
            is_success=self.is_success(obs_["achieved_goal"], obs_["desired_goal"]),
            actions=action.copy(),
            collisions=self._collision_flag&self.is_limit(self.robot.get_joint_angles()),
        )
        reward = self.compute_reward(obs_["achieved_goal"], obs_["desired_goal"], info)
        if self.is_render:
            self.scene_maker.view_frame("curr", self.robot.get_ee_position(), self.robot.get_ee_orientation())
        return obs_, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        if len(achieved_goal.shape) == 2:
            is_success = np.array([1 if i["is_success"] else 0 for i in info])
            actions = np.array([i["actions"] for i in info])
            collisions = np.array([i["collisions"] for i in info])
            r = np.array([
                -self.distance(a_goal, d_goal) for a_goal, d_goal in zip(achieved_goal, desired_goal)
            ])
        else:
            is_success = info["is_success"]
            actions = info["actions"]
            collisions = info["collisions"]
            r = -self.distance(achieved_goal, desired_goal) #0.
        
        #r -= self.distance(desired_goal - achieved_goal, axis=-1)
        if "action" in self.reward_type:
            r -= np.linalg.norm(actions, axis=-1) / 20
        
        if "col" in self.reward_type:
            r -= collisions * 1.
        
        return r

register(
    id='MyPandaReach-v0',
    entry_point='pybullet_wrapper:PandaGymEnv',
    max_episode_steps=100,
)