import sys
import gym
import numpy as np

import gym_pygame
from gym.core import ObservationWrapper


from utils.utils import record_videos

from env.pong import Pong

#from ple.games.pong import pong


class Env(ObservationWrapper):
    "This class creates the self.environement"
    def __init__(self, max_steps, truncate, max_score, display=False):
        self.env = gym.make('Pong-PLE-v0', MAX_SCORE = max_score, cpu_speed_ratio=0.5, players_speed_ratio = 0.5, display = display)
        self.env = Pong(self.env, truncate, max_steps)
        #self.env = gym.wrappers.FrameStack(self.env, 4)
        #self.env = record_videos(self.env)
    
    def get_env(self):
        return self.env
    
    