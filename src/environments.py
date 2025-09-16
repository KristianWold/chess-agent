import numpy as np
import torch
import math
import cmath
import random
import config
import chess

from copy import deepcopy

from agents import *


class Environment:

    def __init__(self,
                 max_num_moves=None,
                ):
        self.max_num_moves = max_num_moves

        self.board = chess.Board()
        self.episode_count = 0
        self.move_count = 0

    def get_reward_and_done(self):
        if self.board.is_checkmate():
            done = True
            reward = 1 
        elif self.board.is_stalemate():
            done = True
            reward = 0  # draw
        else:
            done = False
            reward = 0

        return torch.tensor(reward, dtype=torch.float, device=config.device).view(1, 1), bool(done)

    def step(self, move):
        self.move_count += 1
        self.move_history.append(str(move))
        self.board.push(move)
        self.board = self.board.mirror()
        return deepcopy(self.board), self.get_reward_and_done()

    def reset(self):
        self.episode_count += 1
        self.move_count = 0
        self.move_history = []
        self.board = chess.Board()
        return deepcopy(self.board)
