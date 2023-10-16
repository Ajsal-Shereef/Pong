import sys
import gym
#sys.modules["gym"] = gymnasium

class Pong(gym.Wrapper):
    def __init__(self, env, max_steps):
        super().__init__(env)
        self.max_steps = max_steps
        
        self.cummulative_avoid_region = 0
        self.cummulative_prefered_region = 0
        self.reset_arrays()

    def step(self, action):
        observation, reward, self.done, info = self.env.step(action)
        if reward == -6:
            reward = -1
        elif reward == 6:
            reward = 1
        self.episode_step += 1
        truncated = self.is_episode_done()
        return observation, reward, self.done, truncated, info
    
    def reset(self):
        self.episode_step = 0
        self.episode_prefered_region = 0
        self.episode_avoid_region = 0
        reset = self.env.reset()
        return reset, {}
    
    def is_episode_complete(self):
        return self.episode_step == self.max_steps
    
    def reset_arrays(self):
        self.cummulative_avoid_region = 0
        self.cummulative_prefered_region = 0
        
    def return_counts(self):
        return self.cummulative_avoid_region, self.cummulative_prefered_region, self.episode_avoid_region, self.episode_prefered_region

    def is_episode_done(self):
        max_step_criteria = self.episode_step == self.max_steps
        return max_step_criteria