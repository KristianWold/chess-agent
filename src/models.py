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



class ReplayMemory():
    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.batch_size = batch_size

        self.states = torch.zeros(capacity, 12, 8, 8, dtype = torch.bool)
        self.actions = torch.zeros(capacity, 1, dtype = torch.int64)
        self.next_states = torch.zeros(capacity, 12, 8, 8, dtype = torch.bool)
        self.mask_legal = torch.zeros(capacity, 64*76, dtype = torch.bool)
        self.rewards = torch.zeros(capacity, 1, dtype = torch.float32)
        self.done = torch.zeros(capacity, 1, dtype = torch.float32)

        self.data = torch.utils.data.TensorDataset(self.states, self.actions, self.next_states, self.mask_legal, self.rewards, self.done)

        self.index = 0
        self.num_samples = 0

    def push(self, transition):
        """Save a transition"""

        state, action, next_state, mask_legal, reward, done = transition
        self.states[self.index] = state
        self.actions[self.index] = action
        self.next_states[self.index] = next_state
        self.mask_legal[self.index] = mask_legal
        self.rewards[self.index] = reward
        self.done[self.index] = done

        self.index = (self.index + 1) % self.capacity
        self.num_samples = min(self.num_samples + 1, self.capacity)

    def sample(self):
        idx = torch.randperm(self.num_samples)[:self.batch_size]
        batch = [b.to(config.device) for b in self.data[idx]]
        return batch


class Model:

    def __init__(self,
                 agent=None,
                 environment=None,
                 mem_capacity=None,
                 batch_size=None,
                 num_warmup=None,
                 policy_update=None,
                 target_update=None,
                 temp_constants=None,
                 opt_list = None,
                 scaler = None,
                 criterion = nn.SmoothL1Loss(),
                 ):

        self.agent = agent
        self.environment = environment
        self.mem_capacity = mem_capacity
        self.batch_size = batch_size
        self.num_warmup = num_warmup
        self.policy_update = policy_update
        self.target_update = target_update

        self.temp_start = temp_constants[0]
        self.temp_end = temp_constants[1]
        self.temp_min = temp_constants[2]
        self.temp_decay = temp_constants[3]

        a = environment.max_num_moves
        self.gamma = 1 - 1/a
        self.criterion = criterion

        self.counter_episode = 0
        
        self.memory_pos = ReplayMemory(mem_capacity, batch_size)
        self.memory_neg = ReplayMemory(mem_capacity, batch_size)

        self.opt_list = opt_list
        self.scaler = scaler

    #@profile
    def train(self, num_episodes, evaluate_agents=None, freq=1000):
        counter = 0
        loss = 0
        diff = 0
        for i_episode in tqdm(range(num_episodes)):
            board = self.environment.reset()
            state = self.agent.board_to_state(board)
            state_prev = None

            temp_max = self.temp_start + (self.temp_end - self.temp_start) * self.counter_episode/ self.temp_decay
            temp_max = max(self.temp_end, temp_max)

            temp = np.exp(random.uniform(np.log(self.temp_min), np.log(temp_max)))
            self.counter_episode += 1

            transition_list = []
            done = False

            while not done:
                action = self.agent.select_action(self.environment, 
                                                  temp=temp, 
                                                  greedy=False)
                temp *= 0.95 

                move = self.agent.action_to_move(action)
                board_next, (reward, done) = self.environment.step(
                    move)

                state_next = self.agent.board_to_state(board_next)
                mask_legal = self.agent.get_mask_legal(self.environment.get_legal_moves())

                done_tensor = torch.tensor(done, dtype=torch.float32, device=config.device).view(1,1)

                if state_prev is not None:
                    transition_list.append([state_prev, action_prev, state_next, mask_legal, -reward, done_tensor])

                if done:
                    transition_list.append([state, action, state_next, mask_legal, reward, done_tensor])

                board = board_next
                state_prev = state
                action_prev = action
                state = state_next

                if counter % self.policy_update == 0:
                    loss = self.optimize_agent()

                if counter % self.target_update == 0:
                    self.agent.update_target_net()

                counter += 1

            if reward.item() == 1:
                for transition in transition_list:
                    self.memory_pos.push(transition)
            else:
                for transition in transition_list:
                    self.memory_neg.push(transition)

            if i_episode % freq == 0:
                results = evaluate_agents.evaluate(verbose=False)
                transitions = self.sample_memory()
                diff = 0
                if transitions is not None:
                    _, _, next_state_batch, mask_legal_batch, _, _ = transitions
                    state_next_batch = next_state_batch
                    mask_legal_batch = mask_legal_batch
                    diff_list = []
                    for state_next, mask_legal in zip(state_next_batch, mask_legal_batch):
                        if mask_legal.sum() != 0:    
                            diff = self.agent.get_diff_Q(state_next.unsqueeze(0), mask_legal.unsqueeze(0))
                            diff_list.append(diff)
                    diff = np.mean(diff_list)

                print(results, loss, diff)

    @torch.compile
    def compute_loss(self):
        if self.memory_pos.num_samples < self.num_warmup or self.memory_neg.num_samples < self.num_warmup:
            return 0, None
        state_batch, action_batch, next_state_batch, mask_legal_batch, reward_batch, done_batch = self.sample_memory()

        if random.random() < 0.5:
            online_net = self.agent.online_net1
            target_net = self.agent.target_net2
            opt = self.opt_list[0]
        else:
            online_net = self.agent.online_net2
            target_net = self.agent.target_net1
            opt = self.opt_list[1]

        Q = online_net(state_batch)
        state_action_values = Q.gather(1, action_batch)

        with torch.no_grad():
            Q_next = online_net(next_state_batch)
            Q_next = Q_next.masked_fill(~mask_legal_batch, -float('inf'))
            action_star = Q_next.argmax(1, keepdim=True)
        
            next_state_values = target_net(next_state_batch).gather(1, action_star)
            next_state_values = next_state_values.clamp(-1.0, 1.0)

            expected_state_action_values = reward_batch + self.gamma*(1 - done_batch)*next_state_values

        loss = self.criterion(state_action_values, expected_state_action_values)
        return loss, opt
    

    def sample_memory(self):
        if self.memory_pos.num_samples < self.num_warmup or self.memory_neg.num_samples < self.num_warmup:
            return None
        
        state_batch1, action_batch1, next_state_batch1, mask_legal_batch1, reward_batch1, done_batch1 = self.memory_pos.sample()
        state_batch2, action_batch2, next_state_batch2, mask_legal_batch2, reward_batch2, done_batch2 = self.memory_neg.sample()

        state_batch = torch.cat([state_batch1, state_batch2], dim=0)
        action_batch = torch.cat([action_batch1, action_batch2], dim=0)
        next_state_batch = torch.cat([next_state_batch1, next_state_batch2], dim=0)
        mask_legal_batch = torch.cat([mask_legal_batch1, mask_legal_batch2], dim=0)
        reward_batch = torch.cat([reward_batch1, reward_batch2], dim=0)
        done_batch = torch.cat([done_batch1, done_batch2], dim=0)

        return state_batch, action_batch, next_state_batch, mask_legal_batch, reward_batch, done_batch

    
    def optimize_agent(self):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss, opt = self.compute_loss()
        if loss == 0:
            return 0

        # Optimize the model
        self.scaler.scale(loss).backward()
       
        self.scaler.unscale_(opt)
        params = [p for g in opt.param_groups for p in g['params'] if p.grad is not None]
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

        self.scaler.step(opt)
        self.scaler.update()
        opt.zero_grad(set_to_none=True)

        return loss.detach()


    def set_temp(self, temp_constants):
        self.temp_start = temp_constants[0]
        self.temp_end = temp_constants[1]
        self.temp_min = temp_constants[2]
        self.temp_decay = temp_constants[3]

        
def save_checkpoint(model, filename='checkpoint.pth'):
    checkpoint = {
        'model_state_dict': model.agent.state_dict(),
        'optimizer1_state_dict': model.opt_list[0].state_dict(),
        'optimizer2_state_dict': model.opt_list[1].state_dict(),
        'memory_pos': model.memory_pos,
        'memory_neg': model.memory_neg,
        'counter_episode': model.counter_episode,
    }
    torch.save(checkpoint, filename)


def load_checkpoint(filename, model):
    checkpoint = torch.load(filename, weights_only=False)
    model.agent.load_state_dict(checkpoint['model_state_dict'])
    
    model.opt_list[0].load_state_dict(checkpoint['optimizer1_state_dict'])
    model.opt_list[1].load_state_dict(checkpoint['optimizer2_state_dict'])

    model.memory_pos = checkpoint['memory_pos']
    model.memory_neg = checkpoint['memory_neg']

    model.counter_episode = checkpoint['counter_episode']

    return model

def group_decay_parameters(model, weight_decay=0.01, no_decay=['bias', 'GroupNorm.weight']):
    """
    Groups parameters for optimizer with weight decay and no weight decay.
    """
    param_optimizer = list(model.named_parameters())
    
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},  # Apply weight decay to these parameters
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}  # No weight decay for these parameters
    ]
    
    return optimizer_grouped_parameters


class EvaluateAgents:
    def __init__(self, agent1, agent2, environment, num_games, temp):
        self.agent1 = agent1
        self.agent2 = agent2
        self.environment = environment
        self.num_games = num_games
        self.temp = temp

    def play_game(self, verbose=False):
        board = self.environment.reset()
        if random.random() < 0.5:
            agent1 = self.agent1
            agent2 = self.agent2
            one_starts = True
        else:
            agent1 = self.agent2
            agent2 = self.agent1
            one_starts = False

        done = False
        temp = self.temp
        while not done:
            if self.environment.mirror:
                action = agent2.select_action(self.environment, temp=temp, greedy=False)
            else:
                action = agent1.select_action(self.environment, temp=temp, greedy=False)

            temp *= 0.95

            move = agent1.action_to_move(action)
            board, (reward, done) = self.environment.step(move)

            if verbose:
                print(board)
                print()

            if done:
                if reward.item() == 1:
                    if self.environment.mirror:
                        return -(1-2*one_starts)
                    else:
                        return (1-2*one_starts)
                return 0       # draw

        return 0               # draw due to max moves reached

    def evaluate(self, verbose=False):
        results = {1:0, -1:0, 0:0}  # wins for agent1, wins for agent2, draws

        for _ in tqdm(range(self.num_games)):
            result = self.play_game(verbose=verbose)
            results[result] += 1

        return results