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
        self.mirror = False

    def get_reward_and_done(self):
        if self.board.is_checkmate():
            done = True
            reward = 1
        elif len(self.get_legal_moves()) == 0 or self.move_count >= self.max_num_moves:
            # draw   
            done = True
            reward = 0
        else:
            done = False
            reward = 0

        return torch.tensor(reward, dtype=torch.float, device=config.device).view(1,1), bool(done)

    def step(self, move):
        self.move_count += 1
        if self.mirror:
            move = flip_move(move)
        self.board.push(move)
        self.mirror = not self.mirror
        return self.get_board(), self.get_reward_and_done()

    def reset(self):
        self.episode_count += 1
        self.mirror = False
        self.move_count = 0
        self.board = chess.Board()
        return deepcopy(self.board)
    
    def get_board(self):
        return deepcopy(self.board) if not self.mirror else self.board.mirror()

    def get_legal_moves(self):
        legal_moves = []
        for m in self.board.legal_moves:
            self.board.push(m)
            if not self.board.is_repetition(3):
                legal_moves.append(m)
            self.board.pop()

        if self.mirror:
            legal_moves = [flip_move(m) for m in legal_moves]

        return legal_moves
    

def flip_move(move: chess.Move) -> chess.Move:
    """Flips a chess.Move object to correspond to a mirrored board."""
    from_square_flipped = chess.square_mirror(move.from_square)
    to_square_flipped = chess.square_mirror(move.to_square)
    
    # Promotion types remain the same
    return chess.Move(from_square_flipped, to_square_flipped, move.promotion)
