import numpy as np
import matplotlib.pyplot as plt
from minigrid.core.constants import OBJECT_TO_IDX

class HumanOracle():
    """This class is a simmulated human which gives the feedback"""
    def __init__(self, env, mode, hit_penalty):
        self.env = env
        self.mode = mode
        self.hit_penalty = hit_penalty
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
        
    def update_counts(self, observation):
        if 20<observation[0]<30:
            self.episode_prefered_region += 1
            self.cummulative_prefered_region += 1
        elif (0<=observation[0]<10 or 38<observation[0]<=48):
            self.episode_avoid_region += 1
            self.cummulative_avoid_region += 1
        
    def reset_episode_count(self):
        self.episode_prefered_region = 0
        self.episode_avoid_region = 0
    
    def reset_arrays(self):
        self.cummulative_prefered_region = 0
        self.cummulative_avoid_region = 0