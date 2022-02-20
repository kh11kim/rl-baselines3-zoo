#import gym
# import numpy as np
# from bullet import Bullet
# from rxbot import Rxbot
from utils.rxbot.rxbot_reach import RxbotReachEnv
from stable_baselines3.common.env_checker import check_env
from utils.rxbot.franka_panda import Panda, PandaAbstractEnv
from utils.rxbot.panda_reach import PandaReachEnv
from utils.rxbot.bullet import Bullet
import numpy as np
from spatial_math_mini import SE3, SO3
#sim = Bullet(render=False)
#p = Panda(sim)
#env = PandaReachEnv(render=True)
#env = RxbotReachEnv(render=True, dim=2, reward_type="task")

qtn = SO3.Rx(45, "deg")._qtn
orn = [*qtn[1:], qtn[0]]
sim = Bullet(render=True, background_color=np.array([153, 255, 153]))

sim.view_frame("frame", [0,0,0], [0,0,0,1])
sim.view_frame("frame1", [0.2,0.2,0.2], orn)
sim.place_visualizer(
    target_position=np.zeros(3), 
    distance=1.6, 
    yaw=45, 
    pitch=-30
)
while True:
    sim.step()
env = gym.make("PandaReach-v0", render=True, reward_type="action")

obs = env.reset()
rewards = []
for i in range(1000):
    obs = env.reset()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    rewards.append(reward)
print(np.mean(rewards))
check_env(env)
input()