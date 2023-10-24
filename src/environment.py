import gym
import numpy as np

from .reward import reward


class Environment(gym.Env):
    def __init__(self, dataset, accept_model):
        super().__init__()

        self.dataset = dataset
        self.accept_model = accept_model
        
        self.curr_state = 0
        self.n_observation = dataset.shape[0]
        
        observation_tensor = dataset[['Tier', 'FICO', 'Term', 'Amount', 'Previous_Rate', 'Competition_rate',
                                      'Cost_Funds', 'Partner Bin', 'Car_Type_N', 'Car_Type_R','Car_Type_U']]
        
        self.observation_tensor = observation_tensor.values
        self.action_space = gym.spaces.Box(low=np.array([0]), high=np.array([100]))
        self.observation_space = gym.spaces.Box(low=-10*np.ones(11)[None, :], 
                                                high=10e+10*np.ones(11)[None, :], 
                                                shape=(1, 11))
    
    def step(self, action):
        """
          1. Update state
          2. Return reward

          Return format:
            next_s, r, done, info = self.step(a)
        """
        state = self.observation_tensor[self.curr_state]
       
        if self.curr_state >= self.n_observation - 2:
            done = True
        else:
            done = False
        
        self.curr_state +=1
        next_s = self.observation_tensor[self.curr_state]
        r = reward(action, state, self.accept_model, risk_free=0.04, loss_ratio=0.5)

        info = f'reward = {r}, at state is done = {done}'

        return next_s, r, done, info

    def reset(self):
        self.curr_state = 0
        state = self.observation_tensor[self.curr_state]
        
        return state
