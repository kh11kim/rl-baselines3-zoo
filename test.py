import gym
# import numpy as np
# from bullet import Bullet
# from rxbot import Rxbot
from utils.rxbot.rxbot_reach import RxbotReachEnv
from stable_baselines3.common.env_checker import check_env
from utils.rxbot.franka_panda import Panda, PandaAbstractEnv
from utils.rxbot.panda_reach import PandaReachEnv
from utils.rxbot.bullet import Bullet

#sim = Bullet(render=False)
#p = Panda(sim)
#env = PandaReachEnv(render=True)
#env = RxbotReachEnv(render=True, dim=2, reward_type="task")

env = gym.make("PandaReach-v0", render=True, reward_type="task")

obs = env.reset()
for i in range(100):
    #obs = env.reset()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    
check_env(env)
input()