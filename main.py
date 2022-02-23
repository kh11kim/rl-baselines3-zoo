import gym
from utils.rxbot.panda_reach import PandaReachEnv

import numpy as np
#from sac.sac import SAC
#from sac.replay_memory import ReplayMemory
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import itertools
from stable_baselines3.common.vec_env import VecEnv
import argparse
from stable_baselines3.common.callbacks import BaseCallback
from utils.rrt import RRT, Node
from utils.callbacks import RRTCallback

        
env = gym.make("PandaReach-v0")
callback = RRTCallback(verbose=1)
agent = SAC(
    policy="MultiInputPolicy", 
    env=env,
    learning_rate=0.001,
    learning_starts=1000,
    batch_size=512,
    gamma=0.99,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        online_sampling=True,
        goal_selection_strategy="future",
        n_sampled_goal=4,
    ),
    policy_kwargs=dict(
        net_arch=[512,512,512],
        n_critics=1,
    ),
    verbose=1, 
)
agent.learn(total_timesteps=1000000, callback=callback) # 
agent.save("temp")