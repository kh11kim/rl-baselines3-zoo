from multiprocessing.sharedctypes import Value
import numpy as np
import warnings
from ..core import Bullet
from ..scene_maker import BulletSceneMaker
from ..collision_checker import BulletCollisionChecker
from ..robots import PandaDualArm
import gym
import pybullet as p
from gym import spaces
from gym.envs.registration import register

class PandaDualArmEnvBase:
    def __init__(self, render=False, arm_distance=0.4, task_ll=[0, -0.5, 0], task_ul=[0.5, 0.5, 0.5]):
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
        self.task_ll = task_ll
        self.task_ul = task_ul

    def _make_env(self):
        self.scene_maker.create_plane(z_offset=-0.4)
        self.scene_maker.create_table(length=2.0, width=1.5, height=0.4, x_offset=0.5)
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
                        break
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

    def is_limit(self):
        joint_angles = self.robot.get_joint_angles()
        is_ll = np.any(joint_angles == self.robot.joint_ll)
        is_ul = np.any(joint_angles == self.robot.joint_ul)
        return is_ll | is_ul

    def is_collision(self, joint_angles=None):
        if joint_angles is None:
            joint_angles = self.robot.get_joint_angles()
        result = False
        with self.robot.no_set_joint():
            self.robot.set_joint_angles(joint_angles)
            if self.checker.is_collision():
                result = True
        return result | self.is_limit()
    
    def set_debug_mode(self):
        self.bullet.physics_client.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        self.bullet.physics_client.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
        # joint_angles = self.worker.get_joint_angles()
        # self.param_idxs = []
        # for idx in self.worker.ctrl_joint_idxs:
        #     name = self.worker.joint_info[idx]["joint_name"].decode("utf-8")
        #     param_idx = self.bullet.physics_client.addUserDebugParameter(
        #         name,-4,4,
        #         joint_angles[idx]
        #     )
        #     self.param_idxs.append(param_idx)
    
    def do_debug_mode(self):
        joint_param_values = []
        for param in self.param_idxs:
            joint_param_values.append(p.readUserDebugParameter(param))
        self.worker.set_joint_angles(joint_param_values)

class PandaDualArmGymEnv(PandaDualArmEnvBase, gym.Env):
    """ joint
    """
    def __init__(self, render=False, reward_type="joint", level=0.1):
        self.reward_type = reward_type
        self.level = level
        super().__init__(render=render,  arm_distance=0.4)
        self.n_obs = 1
        self.task_ll = np.array([-1.2, -1.2, -1.2])
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
        self.max_joint_change = 0.1

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
        if self.checker.is_collision() & self.is_limit():
            self.robot.set_joint_angles(joint_prev)
            self._collision_flag = True
        self._collision_flag = False
        self.robot.set_joint_angles(joint_target)

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
            ee1, ee2 = start_ee[:3], start_ee[3:]
            self.scene_maker.view_position("goal1", goal_ee[:3])
            self.scene_maker.view_position("goal2", goal_ee[3:])
            self.scene_maker.view_position("curr1", start_ee[:3])
            self.scene_maker.view_position("curr2", start_ee[3:])
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
            self.scene_maker.view_position("curr1", self.robot.get_ee_position()[:3])
            self.scene_maker.view_position("curr2", self.robot.get_ee_position()[3:])
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

class PandaDualArmGymEnvSingle(PandaDualArmEnvBase, gym.Env):
    """ joint
    """
    def __init__(self, render=False, reward_type="joint", arm="left", level=1.0):
        self.arm = arm
        self.reward_type = reward_type
        self.level = level
        super().__init__(render=render,  arm_distance=0.4)
        if self.arm == "left":
            self.worker = self.robot.panda1
            self.coworker = self.robot.panda2
        elif self.arm == "right":
            self.worker = self.robot.panda2
            self.coworker = self.robot.panda1

        self.n_obs = 7
        #self.task_ll = np.array([-1.2, -1.2, -1.2])
        #self.task_ul = np.array([1.2, 1.2, 1.2])
        self.observation_space = spaces.Dict(dict(
            observation=spaces.Box(
                low=self.coworker.joint_ll,  
                high=self.coworker.joint_ul, 
                shape=(self.coworker.n_joints,), 
                dtype=np.float32
            ),    
            achieved_goal=spaces.Box(
                self.worker.joint_ll, 
                self.worker.joint_ul, 
                shape=(self.worker.n_joints,), 
                dtype=np.float32
            ),
            desired_goal=spaces.Box(
                self.worker.joint_ll, 
                self.worker.joint_ul, 
                shape=(self.worker.n_joints,), 
                dtype=np.float32
            ),
        ))
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.worker.n_joints,), dtype=np.float32)
        self._goal = None
        self.eps = 0.1
        self.max_joint_change = 0.1

    @property
    def goal(self):
        return self._goal.copy()
    
    @goal.setter
    def goal(self, arr: np.ndarray):
        self._goal = arr
    
    # def get_worker_joint_angle(self):
    #     if self.arm == "left":
    #         return self.robot.get_joint_angles()[:7]
    #     elif self.arm == "right":
    #         return self.robot.get_joint_angles()[7:]
    #     raise ValueError("wrong arm type")
    
    # def get_coworker_joint_angle(self):
    #     if self.arm == "left":
    #         return self.robot.get_joint_angles()[7:]
    #     elif self.arm == "right":
    #         return self.robot.get_joint_angles()[:7]
    #     raise ValueError("wrong arm type")
    
    # def set_worker_joint_angle(self, worker_joint_angles):
    #     curr_joint_angles = self.robot.get_joint_angles()
    #     if self.arm == "left":
    #         target_joint_angles = np.hstack([worker_joint_angles, curr_joint_angles[7:]])
    #     elif self.arm == "right":
    #         target_joint_angles = np.hstack([curr_joint_angles[:7], worker_joint_angles])
    #     else:
    #         raise ValueError("wrong arm type")
    #     self.robot.set_joint_angles(target_joint_angles)

    def get_observation(self):
        return dict(
            observation=self.coworker.get_joint_angles(),
            achieved_goal=self.worker.get_joint_angles(),
            desired_goal=self.goal,
        )

    def set_action(self, action: np.ndarray):
        joint_prev = self.worker.get_joint_angles()
        action = action.copy()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        joint_target = joint_prev.copy()
        joint_target += action * self.max_joint_change
        self.worker.set_joint_angles(joint_target)
        if self.checker.is_collision():
            self.worker.set_joint_angles(joint_prev)
            self._collision_flag = True
        else:
            self._collision_flag = False

    def is_success(self, joint_curr: np.ndarray, joint_goal: np.ndarray):
        return np.linalg.norm(joint_curr - joint_goal) < self.eps

    def reset(self):
        random_start_joints = self.get_random_configuration(collision_free=True)
        if self.arm == "left":
            random_joints_worker = random_start_joints[:7]
            random_joints_coworker = random_start_joints[7:]
        elif self.arm == "right":
            random_joints_worker = random_start_joints[7:]
            random_joints_coworker = random_start_joints[:7]
        
        while True:
            random_joints_worker2 = self.worker.get_random_joint_angles(set=True)
            #random_joint2 = self.get_random_configuration(collision_free=False)
            goal = random_joints_worker + (random_joints_worker2 - random_joints_worker) * self.level
            self.worker.set_joint_angles(goal)
            self.coworker.set_joint_angles(random_joints_coworker)
            if not self.is_collision():
                goal_ee = self.robot.get_ee_position()
                break
        self.start = random_joints_worker
        self.goal = goal
        self.robot.set_joint_angles(random_start_joints)
        start_ee = self.robot.get_ee_position()

        if self.is_render:
            self.scene_maker.view_position("goal1", goal_ee[:3])
            self.scene_maker.view_position("goal2", goal_ee[3:])
            self.scene_maker.view_position("curr1", start_ee[:3])
            self.scene_maker.view_position("curr2", start_ee[3:])
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
            self.scene_maker.view_position("curr1", self.robot.get_ee_position()[:3])
            self.scene_maker.view_position("curr2", self.robot.get_ee_position()[3:])
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
    id='MyPandaDualArmReach-v0',
    entry_point='pybullet_wrapper:PandaDualArmGymEnv',
    max_episode_steps=100,
)
register(
    id='MyPandaDualArmReach-v1',
    entry_point='pybullet_wrapper:PandaDualArmGymEnvSingle',
    max_episode_steps=100,
)