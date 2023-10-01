import gymnasium
from gymnasium import spaces
import numpy as np

class DynamicPricingEnv(gymnasium.Env):
    def __init__(self):
        super(DynamicPricingEnv, self).__init__()
        
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        
        self.state = None

    def reset(self):
        self.state = np.random.rand(2)
        return self.state

    def step(self, action):
        next_state = np.random.rand(2)
        reward = np.random.rand(1)
        done = False
        self.state = next_state
        return next_state, reward, done, {}
