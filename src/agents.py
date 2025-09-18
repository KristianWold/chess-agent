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

    def __init__(self):
        super(NN, self).__init__()

        self.conv1 = nn.Conv2d(12, 128, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(128*8*8, 6000)
        self.fc2 = nn.Linear(6000, 64*76)

    def forward(self, x):
        x = F.selu(self.conv1(x))
        x = F.selu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.selu(self.fc1(x))
        return self.fc2(x)

class Agent:

    def __init__(self,
                 board_logic=None,):

        self.board_logic = board_logic

        self.online_net1 = NN().to(config.device)
        self.online_net2 = NN().to(config.device)

        self.target_net1 = NN().to(config.device)
        self.target_net1.load_state_dict(self.online_net1.state_dict())
        self.target_net1.eval()

        self.target_net2 = NN().to(config.device)
        self.target_net2.load_state_dict(self.online_net2.state_dict())
        self.target_net2.eval()

    def forward(self, state_tensor):
        Q1 = self.online_net1(state_tensor)
        Q2 = self.online_net2(state_tensor)
        return (Q1 + Q2) / 2

    def select_action(self, environment, eps=0, greedy=True):
        board = environment.get_board()
        legal_moves = environment.get_legal_moves()
        for m in legal_moves:
            board.push(m)
            if board.is_checkmate():
                board.pop()
                return self.move_to_action(m)
            board.pop()

        with torch.no_grad():
            state_tensor = self.board_logic.board_to_state(board).to(config.device)
            mask_legal = self.get_mask_legal(legal_moves)
            
            Q = self.forward(state_tensor)
            
            logits = Q
            logits = logits.masked_fill(~mask_legal, -1e9)

            if greedy:
                action = logits.argmax(1, keepdim=True)
            else:
                if random.random() < eps:
                    legal_actions = mask_legal.nonzero(as_tuple=False)
                    idx = random.randint(0, len(legal_actions)-1)
                    action = legal_actions[idx][1].view(1, 1)
                else:
                    action = logits.argmax(1, keepdim=True)


            return action

    def update_target_net(self):
        self.target_net1.load_state_dict(self.online_net1.state_dict())
        self.target_net2.load_state_dict(self.online_net2.state_dict())

    def board_to_state(self, board):
        return self.board_logic.board_to_state(board).to(config.device)

    def action_to_move(self, action):
        return self.board_logic.action_to_move(int(action))

    def move_to_action(self, move):
        return torch.tensor(self.board_logic.move_to_action(move), dtype=torch.long, device=config.device)

    def get_mask_legal(self, legal_moves):
        mask_legal = torch.zeros(1, 64*76, dtype=torch.bool)

        action = torch.tensor([self.board_logic.move_to_action(m) for m in legal_moves], 
                            dtype=torch.long)
        mask_legal[0, action] = 1

        return mask_legal.to(config.device)


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


        return state_onehot.to(config.device)

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
