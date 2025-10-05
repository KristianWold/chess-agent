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
        return F.silu(self.gn2(h) + x)


class ConvNN(nn.Module):

    def __init__(self, in_ch=14, ch=128, n_blocks=8):
        super(ConvNN, self).__init__()
        self.stem = nn.Conv2d(in_ch, ch, 3, padding=1)

        blocks = []
        for i in range(n_blocks):
            blocks.append(ConvResBlock(ch))

        self.blocks = nn.Sequential(*blocks)
        self.head_q = nn.Conv2d(ch, 76, 1)


    def forward(self, x):
        b = x.size(0)
        x = F.silu(self.stem(x))
        x = self.blocks(x)
        return self.head_q(x).view(b, -1)


def boltzmann_policy(logits, temp=1):
    dist = torch.distributions.Categorical(logits=logits/temp)
    action = dist.sample().view(-1,1)   
    return action

def eps_greedy_policy(logits, temp=0.1, illegal_floor=-1e5):
    x = logits.squeeze(0)
    if random.random() < temp:
        idx = torch.nonzero(x > (illegal_floor / 10), as_tuple=True)[0]
        a = idx[torch.randint(idx.numel(), (1,), device=idx.device)]
    else:
        a = x.argmax()
    return a.view(1,1)

def eps_softmax_policy(logits, temp=0.1):
    logits = logits.squeeze(0).clone()
    m = torch.argmax(logits)
    if random.random() < temp:
        logits[m] = float("-inf")
        if not torch.isfinite(logits).any():
            logits[m] = 0 # forced to pick the max if all other moves are -inf
        action = torch.distributions.Categorical(logits=logits).sample()
    else:
        action = m

    return action.view(-1,1)


class Agent(nn.Module):

    def __init__(self,
                 board_logic=None,
                 in_ch=14, 
                 ch=128, 
                 n_blocks=8,
                 sample_policy=None):
        super(Agent, self).__init__()

        self.board_logic = board_logic
        self.sample_policy = sample_policy

        self.online_net1 = ConvNN(in_ch, ch, n_blocks).to(config.device)
        self.online_net2 = ConvNN(in_ch, ch, n_blocks).to(config.device)

        self.target_net1 = ConvNN(in_ch, ch, n_blocks).to(config.device)
        self.target_net1.load_state_dict(self.online_net1.state_dict())
        self.target_net1.eval()

        self.target_net2 = ConvNN(in_ch, ch, n_blocks).to(config.device)
        self.target_net2.load_state_dict(self.online_net2.state_dict())
        self.target_net2.eval()

    def forward(self, state_tensor):
        Q1 = self.online_net1(state_tensor)
        Q2 = self.online_net2(state_tensor)
        return (Q1 + Q2) / 2

    def select_action(self, environment, temp=1, greedy=True):
        board = environment.get_board()
        legal_moves = environment.get_legal_moves()

        if len(legal_moves) == 0: # no legal moves due to blunder filter
            legal_moves = environment.get_legal_moves(include_blunders=True)
            move = random.choice(legal_moves) # forced to pick a random blunder
            return self.move_to_action(move)

        for m in legal_moves: # immediate checkmate
            board.push(m)
            if board.is_checkmate():
                board.pop()
                return self.move_to_action(m)
            board.pop()

        with torch.no_grad():
            state_tensor = self.board_logic.board_to_state(board).to(config.device)
            mask_legal = self.get_mask_legal(legal_moves)
            
            Q = self.forward(state_tensor)
            
            Q_legal = Q.masked_fill(~mask_legal, float("-inf"))
            Q_max = Q_legal.max(dim=1, keepdim=True).values
            logits = (Q_legal - Q_max)

            if greedy:
                action = logits.argmax(1, keepdim=True)
            else:
                action = self.sample_policy(logits, temp=temp)

            return action

    def board_to_state(self, board):
        return self.board_logic.board_to_state(board).to(config.device)

    def action_to_move(self, action):
        return self.board_logic.action_to_move(int(action))

    def move_to_action(self, move):
        return torch.tensor(self.board_logic.move_to_action(move), dtype=torch.long, device=config.device)

    def get_mask_legal(self, legal_moves):
        mask_legal = torch.zeros(1, 76*64, dtype=torch.bool, device=config.device)

        action = torch.tensor([self.board_logic.move_to_action(m) for m in legal_moves], 
                            dtype=torch.long, device=config.device)
        mask_legal[0, action] = True

        return mask_legal
    
    def get_diff_Q(self, state, mask_legal):
        with torch.no_grad():
            Q1 = self.online_net1(state).detach()
            Q2 = self.online_net2(state).detach()

            Q1_legal = Q1[mask_legal]
            Q2_legal = Q2[mask_legal]

            diff = torch.abs(Q1_legal - Q2_legal)
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

    def __init__(self, max_num_moves=100):
        self.max_num_moves = max_num_moves
        self.board = chess.Board()
        self.unique_pieces = sorted(set(str(self.board).replace("\n", "").replace(" ", "")))
        self.piece_to_idx = {p: i for i, p in enumerate(self.unique_pieces)}

        self.move_dict, self.move_dict_inv = make_move_dict()

    def board_to_state(self, board):

        state = [row.split(" ") for row in str(board).split("\n")]
        state = [[self.piece_to_idx[p] for p in row] for row in state]
        state = F.one_hot(torch.tensor(state), num_classes=len(self.unique_pieces)).to(torch.float).permute(2, 0, 1)
        state = state[1:].unsqueeze(0)

        plane_en_passant = torch.zeros(1, 1, 8, 8, dtype=torch.float)
        ep = board.ep_square
        if ep is not None:
            x, _ = self.square_to_xy(ep)
            plane_en_passant[0, 0, :, x] = 1.0

        state = torch.cat([state, plane_en_passant], dim=1)

        rights_list = [board.has_kingside_castling_rights(chess.WHITE), 
                       board.has_queenside_castling_rights(chess.WHITE),
                       board.has_kingside_castling_rights(chess.BLACK),
                       board.has_queenside_castling_rights(chess.BLACK)]
        
        plane_rights_list = [torch.full((1, 1, 8, 8), right, dtype=torch.float) for right in rights_list]
        state = torch.cat([state] + plane_rights_list, dim=1)

        plane_rank = torch.linspace(-1, 1, 8).view(1,1,8,1).expand(1,1,8,8)
        plane_file = torch.linspace(-1, 1, 8).view(1,1,1,8).expand(1,1,8,8)  
        state = torch.cat([state, plane_rank, plane_file], dim=1)

        plane_move_count = torch.full((1, 1, 8, 8), board.fullmove_number / self.max_num_moves, dtype=torch.float)
        state = torch.cat([state, plane_move_count], dim=1)

        return state.to(config.device)

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
        action = int(action)
        origin, delta = action//len(self.move_dict), action%len(self.move_dict)
        x = origin % 8
        y = origin // 8
        delta =self.move_dict_inv[delta]
        dx, dy = delta[0], delta[1]
        move = [self.xy_to_square(x, y), self.xy_to_square(x+dx, y+dy)]

        if len(delta) == 3:  # promotion
            move.append(delta[2])

        return chess.Move(*move)
    
    def square_to_xy(self, square):
        x = chess.square_file(square)
        y = 7 - chess.square_rank(square)
        return x, y

    def xy_to_square(self, x, y):
        return chess.square(x, 7 - y)
