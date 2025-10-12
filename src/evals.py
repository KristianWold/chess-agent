from mimetypes import init
import numpy as np
import random
import torch
import config
from collections import namedtuple, deque
from itertools import count
from torch import nn, optim
from tqdm.notebook import tqdm
from environments import *
from copy import deepcopy
from collections import deque
import random
from IPython.display import display, update_display

class EvaluateAgents:
    def __init__(self, agent1, agent2, environment, num_games, temp, transition_decay=0.95, depth=1, breadth=1):
        self.agent1 = agent1
        self.agent2 = agent2
        self.environment = environment
        self.num_games = num_games
        self.temp = temp
        self.transition_decay = transition_decay
        self.depth = depth
        self.breadth = breadth

    def play_game(self, verbose=False):
        board = self.environment.reset()
        if random.random() < 0.5:
            white = self.agent1
            black = self.agent2
        else:
            white = self.agent2
            black = self.agent1

        done = False
        temp = self.temp
        ply = 0
        while not done:

            mover = white if (ply % 2 == 0) else black

            _, action = mover.q_expand(self.environment, depth=self.depth, breadth=self.breadth, temp=temp)
            temp *= self.transition_decay

            move = mover.action_to_move(action)
            _, (reward, done) = self.environment.step(move)

            if done:
                if reward.item() != 0:
                    # Terminal reward is for the mover who just played
                    return reward.item() if mover is self.agent1 else -reward.item()
                else:
                    return 0  # draw

            ply += 1

        return 0               # draw due to max moves reached

    def evaluate(self, verbose=False):
        results = {1:0, -1:0, 0:0}  # wins for agent1, wins for agent2, draws

        for _ in tqdm(range(self.num_games)):
            result = self.play_game(verbose=verbose)
            results[result] += 1

        return results


class User:
    def __init__(self):
        self.board_logic = BoardLogic()

    def select_action(self, environment, temp=None, greedy=False):
        legal_moves = environment.get_legal_moves(include_blunders=True)
        move = None
        while move not in legal_moves:
            move = chess.Move.from_uci(input("Enter your move: "))
            if move not in legal_moves:
                print("Invalid move. Please try again.")
        action = self.board_logic.move_to_action(move)
        return action


class AgentVsUser:
    def __init__(self, agent, environment, temp, greedy=True):
        self.agent = agent
        self.environment = environment
        self.temp = temp
        self.greedy = greedy

    def play_game(self, user_start, depth1=2, breadth1=2, depth2=2, breadth2=2):
        #board = self.environment.reset()
        board =  self.environment.board
        user = User()
        if user_start:
            white = user
            black = self.agent
        else:
            white = self.agent
            black = user

        h = display(board, display_id=True)

        done = False
        temp = self.temp
        ply = 0
        while not done:

            mover = white if (ply % 2 == 0) else black
            if mover is user:
                state = self.agent.board_logic.board_to_state(board).to(config.device)

                Q1 = self.agent.online_net1(state).detach()
                Q2 = self.agent.online_net2(state).detach()
                legal_moves = self.environment.get_legal_moves()
                mask_legal = self.agent.get_mask_legal(legal_moves)

                Q1_legal = Q1[mask_legal]
                Q2_legal = Q2[mask_legal]

                Q1_legal = Q1.masked_fill(~mask_legal, -1e9)
                Q2_legal = Q2.masked_fill(~mask_legal, -1e9)
                action_star1 = torch.argmax(Q1_legal, dim=1).to(config.device)
                action_star2 = torch.argmax(Q2_legal, dim=1).to(config.device)
                score1 = Q1[0,action_star1[0]].item()
                score2 = Q2[0,action_star2[0]].item()

                print(f"Score is {(score1 + score2)/2:.3f} +- {np.abs(score1 - score2):.3f}", end='\r', flush=True)

            if mover is user:
                #action = mover.select_action(self.environment, temp=temp, greedy=self.greedy)
                _, action = self.agent.q_expand(self.environment, depth=depth1, breadth=breadth1)
            else:
                _, action = mover.q_expand(self.environment, depth=depth2, breadth=breadth2)
            temp *= 0.95

            move = mover.board_logic.action_to_move(action)
            _, (reward, done) = self.environment.step(move)

            if user_start:
                board = self.environment.board
            else:
                board = self.environment.board.mirror()
            update_display(board, display_id=h.display_id)

            if done:
                if reward.item() == 1:
                    winner = mover
                    return 1 if winner is self.agent else -1
                else:
                    return 0  # draw

            ply += 1

        return 0               # draw due to max moves reached