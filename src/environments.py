import numpy as np
import torch
import math
import cmath
import random
import config

import chess

from agents import *


class Environment:

    def __init__(self,
                 max_num_moves=None,
                ):
        self.max_num_moves = max_num_moves

        self.board = chess.Board()
        self.episode_count = 0
        self.move_count = 0

    def check_done(self):
        if self.board.is_checkmate():
            done = True
            if self.board.outcome().winner:
                reward =  1  # win
            else:
                reward = -1  # loss
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
        return self.board, *self.get_reward_and_done()

    def reset(self):
        self.episode_count += 1
        self.move_count = 0
        self.move_history = []
        self.board = chess.Board()
        return self.board, self.check_done()
