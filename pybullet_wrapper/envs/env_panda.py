import numpy as np
import gym
from gym import spaces
from gym.envs.registration import register
from ..core import Bullet
from ..scene_maker import BulletSceneMaker
from ..collision_checker import BulletCollisionChecker
from ..robots import Panda
import pybullet as p

class PandaEnvBase:
    def __init__(
        self, 
        render=False, 
        task_ll=np.array([-0.5, -0.5, 0.]),
        task_ul=np.array([0.5, 0.5, 0.7]),
    ):
        self.is_render = render
        self.task_ll = task_ll
        self.task_ul = task_ul
        self.bullet = Bullet(render=render)
        self.scene_maker = BulletSceneMaker(self.bullet)
        self.robot = Panda(
            self.bullet
        )
        self._make_env()
        self.checker = BulletCollisionChecker(self.bullet)
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
        table_length = (self.task_ul - self.task_ll)[0]
        table_width = (self.task_ul - self.task_ll)[1]
        
        self.scene_maker.create_table(
            length=table_length, 
            width=table_width, 
            height=0.4, 
            x_offset=0
        )
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
                if not self.is_collision():
                    random_joint_angles = self.robot.get_joint_angles()
                    break

        return random_joint_angles
    
    def get_random_free_configuration_in_taskspace(self, panda1_first=True):
        pass
        # if panda1_first:
        #     first_robot = self.robot.panda1
        #     second_robot = self.robot.panda2
        # else:
        #     first_robot = self.robot.panda2
        #     second_robot = self.robot.panda1
        
        # for robot in [first_robot, second_robot]:
        #     while True:
        #         robot.get_random_joint_angles(set=True)
        #         ee_position = robot.get_ee_position()
        #         is_collision_free = not self.checker.is_collision()
        #         is_in_taskspace = np.all(self.task_ll < ee_position) \
        #                           & np.all(ee_position < self.task_ul)
        #         if is_collision_free & is_in_taskspace:
        #             break
        # return self.robot.get_joint_angles()

    def reset(self):
        self.joint_goal = self.get_random_configuration(collision_free=True)
        self.joint_start = self.get_random_configuration(collision_free=True)
        self.robot.set_joint_angles(self.joint_start)
    
    def is_limit(self):
        joint_angles = self.robot.get_joint_angles()
        is_ll = np.any(joint_angles <= self.robot.joint_ll)
        is_ul = np.any(joint_angles >= self.robot.joint_ul)
        return is_ll | is_ul

    def is_collision(self, joint_angles=None):
        if joint_angles is None:
            joint_angles = self.robot.get_joint_angles()
        result = False
        #with self.robot.no_set_joint():
        self.robot.set_joint_angles(joint_angles)
        if self.checker.is_collision():
            result = True
        return result | self.is_limit()
    
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

class PandaGymEnv(PandaEnvBase, gym.Env):
    def __init__(self, render=False, reward_type="joint", level=0.5):
        self.level = level
        self.reward_type = reward_type
        super().__init__(render=render)
        self.n_obs = self.robot.n_joints + 3 + 9
        self.task_ll = np.array([-1.2, -1.2, -1.2]) #pos3 + orn9 =12
        self.task_ul = np.array([1.2, 1.2, 1.2])
        self.observation_space = spaces.Dict(dict(
            observation=spaces.Box(-1, 1, shape=(1,), dtype=np.float32),    
            achieved_goal=spaces.Box(
                self.robot.joint_ll, 
                self.robot.joint_ul, 
                shape=(self.robot.n_joints,), 
                dtype=np.float32
            ),
            desired_goal=spaces.Box(
                self.robot.joint_ll, 
                self.robot.joint_ul, 
                shape=(self.robot.n_joints,), 
                dtype=np.float32
            ),
        ))
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.robot.n_joints,), dtype=np.float32)
        self._goal = None
        self.eps = 0.1

    @property
    def goal(self):
        return self._goal.copy()
    
    @goal.setter
    def goal(self, arr: np.ndarray):
        self._goal = arr
    
    def get_observation(self):
        return dict(
            observation=0.,
            achieved_goal=self.robot.get_joint_angles(),
            desired_goal=self.goal,
        )

    

    def set_action(self, action: np.ndarray):
        joint_prev = self.robot.get_joint_angles()
        action = action.copy()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        joint_target = self.robot.get_joint_angles()
        joint_target += action * self.max_joint_change
        joint_target = np.clip(joint_target, self.robot.joint_ll, self.robot.joint_ul)
        if (self.checker.is_collision() | self.is_limit()):
            self.robot.set_joint_angles(joint_prev)
            self._collision_flag = True
        self._collision_flag = False
        self.robot.set_joint_angles(joint_target)

    # def distance(self, pose1: np.ndarray, pose2: np.ndarray) -> np.float64:
    #     pos1, orn_mat1 = pose1[:3], pose1[3:].reshape(3,3)
    #     pos2, orn_mat2 = pose2[:3], pose2[3:].reshape(3,3)
    #     diff_mat = orn_mat1.T@orn_mat2
    #     delta = np.linalg.norm(pos1 - pos2)
    #     delta_theta = np.arccos((np.trace(diff_mat)-1)/2)
    #     return delta + delta_theta/(np.pi * 2)

    def is_success(self, joint_curr: np.ndarray, joint_goal: np.ndarray):
        return np.linalg.norm(joint_curr - joint_goal) < self.eps

    def reset(self):
        random_joint1 = self.get_random_configuration(collision_free=True)
        while True:
            random_joint2 = self.get_random_configuration(collision_free=False)
            goal = random_joint1 + (random_joint2 - random_joint1) * self.level
            if not self.is_collision(goal):
                self.robot.set_joint_angles(goal)
                goal_ee = self.robot.get_ee_position()
                break
        self.start = random_joint1
        self.goal = goal
        self.robot.set_joint_angles(self.start)
        start_ee = self.robot.get_ee_position()

        if self.is_render:
            self.scene_maker.view_position("goal", goal_ee)
            self.scene_maker.view_position("curr", start_ee)
        return self.get_observation()

    def step(self, action: np.ndarray):
        self.set_action(action)
        obs_ = self.get_observation()
        done = False
        info = dict(
            is_success=self.is_success(obs_["achieved_goal"], obs_["desired_goal"]),
            actions=action.copy(),
            collisions=self._collision_flag,
        )
        reward = self.compute_reward(obs_["achieved_goal"], obs_["desired_goal"], info)
        if self.is_render:
            self.scene_maker.view_position("curr", self.robot.get_ee_position())
        return obs_, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        if len(achieved_goal.shape) == 2:
            actions = np.array([i["actions"] for i in info])
            collisions = np.array([i["collisions"] for i in info])
        else:
            actions = info["actions"]
            collisions = info["collisions"]
        r = - np.linalg.norm(achieved_goal-desired_goal, axis=-1)
        
        if "action" in self.reward_type:
            r -= np.linalg.norm(actions, axis=-1) / 10
        
        if "col" in self.reward_type:
            r -= collisions * 1.
        
        return r

register(
    id='MyPandaReach-v0',
    entry_point='dualarm.pybullet_wrapper:PandaGymEnv',
    max_episode_steps=50,
)