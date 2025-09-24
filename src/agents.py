import torch
import torch.nn.functional as F
from torch import nn, optim
import random
import numpy as np
import math
import cmath
import config
import chess

class ConvResBlock(nn.Module):

    def __init__(self, ch):
        super(ConvResBlock, self).__init__()
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias = False)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias = False)

        self.gn1 = nn.GroupNorm(8, ch)
        self.gn2 = nn.GroupNorm(8, ch)

    def forward(self, x):
        h = self.conv1(x)
        h = F.silu(self.gn1(h))
        h = self.conv2(h)
        h = self.gn2(h)
        return F.silu(x + h)


class ConvNN(nn.Module):

    def __init__(self, in_ch=14, ch=128, n_blocks=8):
        super(ConvNN, self).__init__()
        self.stem = nn.Conv2d(in_ch, ch, 3, padding=1)

        blocks = []
        for i in range(n_blocks):
            blocks.append(ConvResBlock(ch))

        self.blocks = nn.Sequential(*blocks)
        self.head_q = nn.Conv2d(ch, 76, 1)

        self.rank_plane = torch.linspace(-1, 1, 8).view(1, 8).repeat(8, 1).unsqueeze(0).unsqueeze(0)
        self.file_plane = torch.linspace(-1, 1, 8).view(8, 1).repeat(1, 8).unsqueeze(0).unsqueeze(0)


    def forward(self, x):
        x = x.float()

        r_plane = self.rank_plane.repeat(x.size(0), 1, 1, 1).to(x.device)
        f_plane = self.file_plane.repeat(x.size(0), 1, 1, 1).to(x.device)
        x = torch.cat([x, r_plane, f_plane], dim=1)

        x = F.silu(self.stem(x))
        x = self.blocks(x)
        return self.head_q(x).view(x.size(0), -1)



class Agent(nn.Module):

    def __init__(self,
                 board_logic=None,
                 in_ch=14, 
                 ch=128, 
                 n_blocks=8):
        super(Agent, self).__init__()

        self.board_logic = board_logic

        self.online_net1 = ConvNN(in_ch, ch, n_blocks).to(config.device)
        self.online_net2 = ConvNN(in_ch, ch, n_blocks).to(config.device)

        self.target_net1 = ConvNN(in_ch, ch, n_blocks).to(config.device)
        self.target_net1.load_state_dict(self.online_net1.state_dict())
        self.target_net1.eval()

        self.target_net2 = ConvNN(in_ch, ch, n_blocks).to(config.device)
        self.target_net2.load_state_dict(self.online_net2.state_dict())
        self.target_net2.eval()

    def forward(self, state_tensor, network_id=None):
        Q1 = self.online_net1(state_tensor)
        Q2 = self.online_net2(state_tensor)
        return (Q1 + Q2) / 2

    def select_action(self, environment, temp=1, greedy=True):
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
            
            Q_legal = Q.masked_fill(~mask_legal, -1e9)
            max_q = Q_legal.max(dim=1, keepdim=True).values
            logits = (Q_legal - max_q)/temp

            if greedy:
                action = logits.argmax(1, keepdim=True)
            else:
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample().view(-1,1)   

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
        mask_legal = torch.zeros(1, 76*64, dtype=torch.bool)

        action = torch.tensor([self.board_logic.move_to_action(m) for m in legal_moves], 
                            dtype=torch.long)
        mask_legal[0, action] = 1

        return mask_legal.to(config.device)
    
    def get_diff_Q(self, state, mask_legal):
        with torch.no_grad():
            Q1 = self.online_net1(state).detach()
            Q2 = self.online_net2(state).detach()

            Q1_legal = Q1[mask_legal]
            Q2_legal = Q2[mask_legal]

            diff = 2*torch.abs(Q1_legal - Q2_legal)/(torch.abs(Q1_legal) + torch.abs(Q2_legal))
            diff = diff.mean().cpu().item()

        return diff


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
    for piece in range(2,6): # knight, bishop, rook, queen
        move_dict[(0, -1, piece)] = len(move_dict)
        move_dict[(1, -1, piece)] = len(move_dict)
        move_dict[(-1, -1, piece)] = len(move_dict)

    move_dict_inv = {v: k for k, v in move_dict.items()}

    return move_dict, move_dict_inv


class BoardLogic:

    def __init__(self):
        self.board = chess.Board()
        self.unique_pieces = sorted(set(str(self.board).replace("\n", "").replace(" ", "")))
        self.piece_to_idx = {p: i for i, p in enumerate(self.unique_pieces)}

        self.move_dict, self.move_dict_inv = make_move_dict()

    def board_to_state(self, board):

        state = [row.split(" ") for row in str(board).split("\n")]
        state = [[self.piece_to_idx[p] for p in row] for row in state]
        state_onehot = F.one_hot(torch.tensor(state), num_classes=len(self.unique_pieces)).to(torch.bool).permute(2, 0, 1)
        state_onehot = state_onehot[1:].unsqueeze(0)

        return state_onehot.to(config.device)

    def move_get_origin(self, move):
        x, y = self.square_to_xy(move.from_square)
        return x + y*8

    def move_get_delta(self, move):
        x1, y1 = self.square_to_xy(move.from_square)
        x2, y2 = self.square_to_xy(move.to_square)
        delta = (x2-x1, y2-y1)

        if move.promotion:  # promotion
            delta = (*delta , move.promotion)

        return self.move_dict[delta]

        
    def move_to_action(self, move):
        origin = self.move_get_origin(move)
        delta = self.move_get_delta(move)

        return origin * len(self.move_dict) + delta


    def action_to_move(self, action):
        origin, delta = action//len(self.move_dict), action%len(self.move_dict)
        x = origin % 8
        y = origin // 8
        delta =self.move_dict_inv[delta]
        dx, dy = delta[0], delta[1]
        move = [self.xy_to_square(x, y), self.xy_to_square(x+dx, y+dy)]

        if len(delta) == 3:  # promotion
            move.append(delta[2]-1)

        return chess.Move(*move)
    
    def square_to_xy(self, square):
        x = chess.square_file(square)
        y = 7 - chess.square_rank(square)
        return x, y

    def xy_to_square(self, x, y):
        return chess.square(x, 7 - y)
