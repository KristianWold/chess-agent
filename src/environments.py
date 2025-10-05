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
                 filter_blunder=False
                ):
        self.max_num_moves = max_num_moves
        self.filter_blunder = filter_blunder

        self.board = chess.Board()
        self.episode_count = 0
        self.mirror = False

    def get_reward_and_done(self):
        if self.board.is_checkmate():
            done = True
            reward = 1
        elif self.get_legal_moves(include_blunders=True) == [] or \
             self.board.is_insufficient_material() or \
             self.board.fullmove_number >= self.max_num_moves:
            # draw
            done = True
            reward = 0
        else:
            done = False
            reward = 0

        return torch.tensor(reward, dtype=torch.float, device=config.device).view(1,1), bool(done)

    def step(self, move):
        if move is not None:
            if self.mirror:
                move = flip_move(move)
            self.board.push(move)
        self.mirror = not self.mirror
        return self.get_board(), self.get_reward_and_done()

    def reset(self):
        self.episode_count += 1
        self.mirror = False
        self.board = chess.Board()
        return self.board

    def get_board(self):
        return self.board if not self.mirror else self.board.mirror()

    def get_legal_moves(self, include_blunders=False):
        legal_moves = list(self.board.legal_moves)

        legal_moves = self.filter_repeated_moves(legal_moves)
        if self.filter_blunder and not include_blunders:
            legal_moves = self.filter_blunder_moves(legal_moves)

        if self.mirror:
            legal_moves = [flip_move(m) for m in legal_moves]

        return legal_moves
    
    def filter_repeated_moves(self, legal_moves):
        filtered_moves = []
        for m in legal_moves:
            self.board.push(m)
            if not self.board.is_repetition(3):
                filtered_moves.append(m)
            self.board.pop()
        return filtered_moves
    
    def filter_blunder_moves(self, legal_moves):
        filtered_moves = []
        for m in legal_moves:
            self.board.push(m)
            if not self.checkmate_in_one(self.board):
                filtered_moves.append(m)
            self.board.pop()
        return filtered_moves
    
    def checkmate_in_one(self, board):
        push, pop = board.push, board.pop
        gives_check = board.gives_check
        for m in board.legal_moves:
            if not gives_check(m):
                continue
            push(m)
            if board.is_checkmate():
                pop()
                return True
            pop()
        return False
    

def flip_move(move):
    from_square_flipped = chess.square_mirror(move.from_square)
    to_square_flipped = chess.square_mirror(move.to_square)
    
    # Promotion types remain the same
    return chess.Move(from_square_flipped, to_square_flipped, move.promotion)
