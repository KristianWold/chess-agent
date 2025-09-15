import numpy as np
import torch
import math
import cmath
import random
import config


class Environment:

    def __init__(self,
                 reward_func=None,
                 max_num_actions=None,
                ):
        self.reward_func = reward_func
        self.fidelity = self.reward_func.fidelity
        self.max_num_actions = max_num_actions

        self.episode_count = 0
        self.action_count = 0
        self.action_history = []

    def get_state(self):
        """get state from current enviroment"""

        pass
        return state

    def get_reward(self):
        """get reward from current enviroment"""

        reward, _ = self.reward_func(self)
        return torch.tensor(reward, dtype=torch.float, device=config.device).view(1, 1)

    def check_done(self):
        _, done = self.reward_func(self)

        return bool(done)

    def step(self, action):
        self.action_count += 1
        self.action_history.append(action)
        
        return self.get_state(), self.get_reward(), self.check_done()

    def reset(self):
        if input_state is None:
            self.episode_count += 1
            self.U_target, _ = self.data_generator(self, seed=seed)
        else:
            self.U_target = input_state

        self.action_count = 0
        self.action_history = []
        return self.get_state(), self.check_done()
