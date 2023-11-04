import numpy as np
import matplotlib.pyplot as plt
from minigrid.core.constants import OBJECT_TO_IDX

class HumanOracle():
    """This class is a simmulated human which gives the feedback"""
    def __init__(self, env, mode):
        self.env = env
        self.mode = mode
        self.reset_arrays()

    def get_human_feedback(self):
        if self.mode == 'preference':
            return self.episode_prefered_region
        if self.mode == 'avoid':
            return -self.episode_avoid_region
        if self.mode == 'both':
            return -self.episode_avoid_region + self.episode_prefered_region
    
    def return_counts(self):
        if self.mode == 'preference':
            return self.cummulative_prefered_region, self.episode_prefered_region
        if self.mode == 'avoid':
            return self.cummulative_avoid_region, self.episode_avoid_region
        if self.mode == 'both':
            return self.cummulative_prefered_region, self.cummulative_avoid_region, self.episode_prefered_region, self.episode_avoid_region
        
    def get_feedback(self, observation, mode):
        if mode == 'avoid':
            if 22<observation[0]<28:
                self.episode_avoid_region += 1
                return -1
        elif mode == 'preference':
            if 42<observation[0]<=48 or 0<observation[0]<=8:
                self.episode_prefered_region += 1
                return 1
        elif mode == 'both':
            if 22<observation[0]<28:
                self.episode_avoid_region += 1
                return -1
            elif 42<observation[0]<=48 or 0<observation[0]<=8:
                self.episode_prefered_region += 1
                return 1
        return 0

        
    def reset_episode_count(self):
        self.episode_prefered_region = 0
        self.episode_avoid_region = 0
    
    def reset_arrays(self):
        self.cummulative_prefered_region = 0
        self.cummulative_avoid_region = 0