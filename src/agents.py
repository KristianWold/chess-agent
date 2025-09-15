import torch
import torch.nn.functional as F
from torch import nn, optim
import random
import numpy as np
import math
import cmath
import config
import chess


class NN(nn.Module):

    def __init__(self, state_dim, action_dim, num_layers, scale):
        super(NN, self).__init__()

        self.conv1 = nn.Conv2d(13, 128, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(128*8*8, 2048)
        self.fc2 = nn.Linear(2048, action_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class Agent:

    def __init__(self,
                 board_logic=None,
                 action_dim=None,
                 num_layers=None,):

        self.board_logic = board_logic
        self.action_dim = action_dim
    

        self.online_net1 = NN(self.board_logic.state_dim, action_dim, num_layers).to(config.device)
        self.online_net2 = NN(self.state_dim, action_dim, num_layers,).to(config.device)

        self.target_net1 = NN(self.state_dim, action_dim, num_layers,).to(config.device)
        self.target_net1.load_state_dict(self.online_net1.state_dict())
        self.target_net1.eval()

        self.target_net2 = NN(self.state_dim, action_dim, num_layers).to(config.device)
        self.target_net2.load_state_dict(self.online_net2.state_dict())
        self.target_net2.eval()

    def forward(self, board):
        state = self.board_logic.board_to_state(board)
        Q1 = self.online_net1(state)
        Q2 = self.online_net2(state)
        return (Q1 + Q2) / 2

    def select_action(self, board, temp=0, greedy=True):

        for m in board.legal_moves:
            board.push(m)
            if board.is_checkmate():
                board.pop()
                return m
            board.pop()

        moves = torch.tensor([self.board_logic.move_to_action(m) for m in board.legal_moves], dtype=torch.long)
        state = self.board_logic.board_to_state(board)

        with torch.no_grad():
            Q = self.forward(state)
            Q = Q[moves]

            logits = (Q - Q.mean()) / (Q.std(unbiased=False) + 1e-5)

            if greedy:
                action = moves[logits.argmax(1, keepdim=True)]
            else:
                dist = F.softmax(logits/temp, dim=1)
                action = moves[torch.multinomial(dist, num_samples=1)]

            move_next = self.board_logic.action_to_move(int(action))

            return move_next

    def update_target_net(self):
        self.target_net1.load_state_dict(self.online_net1.state_dict())
        self.target_net2.load_state_dict(self.online_net2.state_dict())


def make_move_dict():
    move_dict = {}
    num_moves = 7

    dir = [0, 1, -1]

    for dir_x in dir:
        for dir_y in dir:
            if dir_x == 0 and dir_y == 0:
                continue
            for n in range(1, num_moves+1):
                move_dict[(dir_x*n, dir_y*n)] = len(move_dict)

    # knight moves
    move_dict[(2, 1)] = len(move_dict)
    move_dict[(2, -1)] = len(move_dict)
    move_dict[(-2, 1)] = len(move_dict)
    move_dict[(-2, -1)] = len(move_dict)
    move_dict[(1, 2)] = len(move_dict)
    move_dict[(1, -2)] = len(move_dict)
    move_dict[(-1, 2)] = len(move_dict)
    move_dict[(-1, -2)] = len(move_dict)

    # promotions
    for piece in range(4):
        move_dict[(0, 1, piece)] = len(move_dict)
        move_dict[(1, 1, piece)] = len(move_dict)
        move_dict[(-1, 1, piece)] = len(move_dict)

    move_dict_inv = {v: k for k, v in move_dict.items()}

    return move_dict, move_dict_inv


class BoardLogic:

    def __init__(self):
        self.board = chess.Board()
        self.unique_pieces = sorted(set(str(self.board).replace("\n", "").replace(" ", "")))
        self.piece_to_idx = {p: i for i, p in enumerate(self.unique_pieces)}

        self.alpha_idx = {c: i for i, c in enumerate("abcdefgh")}
        self.numeric_idx = {c: i for i, c in enumerate("12345678")}
        self.promotion_idx = {c: i for i, c in enumerate("qrbn")}

        self.move_dict, self.move_dict_inv = make_move_dict()

    def board_to_state(self, board):
        state = [row.split(" ") for row in str(board).split("\n")]
        state = [[self.piece_to_idx[p] for p in row] for row in state]
        state_onehot = F.one_hot(torch.tensor(state), num_classes=len(self.unique_pieces)).float().permute(2, 0, 1)
        state_onehot = state_onehot[1:].unsqueeze(0) 
        return state_onehot

    def move_get_origin(self, move):
        move = str(move)
        x, y = self.alpha_idx[move[0]], self.numeric_idx[move[1]]
        return x + y*8

    def move_get_delta(self, move):
        move = str(move)
        x1, y1 = self.alpha_idx[move[0]], self.numeric_idx[move[1]]
        x2, y2 = self.alpha_idx[move[2]], self.numeric_idx[move[3]]
        delta = (x2-x1, y2-y1)

        if len(move) == 5:  # promotion
            piece = self.promotion_idx[move[4].lower()]
            delta = (*delta , piece)

        return self.move_dict[delta]

        
    def move_to_action(self, move):
        origin = self.move_get_origin(move)
        delta = self.move_get_delta(move)

        return origin * len(self.move_dict) + delta


    def action_to_move(self, action):
        origin, delta = action//len(self.move_dict), action%len(self.move_dict)
        x = origin % 8
        y = origin // 8

        move_delta = self.move_dict_inv[delta]
        move = "abcdefgh"[x] + str(y+1) + "abcdefgh"[x + move_delta[0]] + str(y + 1 + move_delta[1])
        if len(move_delta) == 3:  # promotion
            move += "qrbn"[move_delta[2]]

        return chess.Move.from_uci(move)
