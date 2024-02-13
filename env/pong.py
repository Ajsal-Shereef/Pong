import sys
import gym
#sys.modules["gym"] = gymnasium

class Pong(gym.Wrapper):
    def __init__(self, env, truncate, max_steps, num_action):
        super().__init__(env)
        self.max_steps = max_steps
        self.num_action = num_action
        
        self.cummulative_avoid_region = 0
        self.cummulative_prefered_region = 0
        self.truncate = truncate
        self.reset_arrays()

    def step(self, action):
        observation, reward, self.done, info = self.env.step(action)
        # if action == 2:
        #    observation[1] =  0   
        if reward == -6:
            reward = -1
        elif reward == 6:
            reward = 1
        self.episode_step += 1
        if self.truncate:
            truncated = self.is_episode_done()
            if truncated:
                reward = 0
        else:
            truncated = False
        return observation, reward, self.done, truncated, info
    
    def change_truncate(self):
        self.truncate = False
        
    def change_max_step(self, num_step=1000):
        self.max_steps = num_step
    
    def reset(self):
        self.episode_step = 0
        self.episode_prefered_region = 0
        self.episode_avoid_region = 0
        reset = self.env.reset()
        return reset, {}
    
    def reset_arrays(self):
        self.cummulative_avoid_region = 0
        self.cummulative_prefered_region = 0
        
    def return_counts(self):
        return self.cummulative_avoid_region, self.cummulative_prefered_region, self.episode_avoid_region, self.episode_prefered_region

    def is_episode_done(self):
        max_step_criteria = self.episode_step == self.max_steps
        return max_step_criteria