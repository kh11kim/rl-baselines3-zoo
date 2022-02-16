import gym
# import numpy as np
# from bullet import Bullet
# from rxbot import Rxbot
from rxbot_reach import RxbotReachEnv
from stable_baselines3.common.env_checker import check_env

#env = RxbotReachEnv(render=True, dim=2, reward_type="task")
env = gym.make("RxbotReach-v0", render=True, dim=2, reward_type="task")
obs = env.reset()
for i in range(100):
    env.reset()
check_env(env)
input()