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

class TemperatureScaler:
    def __init__(self, temp_start, temp_end, temp_min, episode_decay, transition_decay):
        self.temp_start = temp_start
        self.temp_end = temp_end
        self.temp_min = temp_min
        self.episode_decay = episode_decay
        self.transition_decay = transition_decay
        self.counter_episode = 0
        self.counter_transition = 0

    def step_episode(self):
        self.counter_episode += 1

    def step_transition(self):
        self.counter_transition += 1

    def get_temp(self):
        temp_max = self.temp_start + (self.temp_end - self.temp_start) * self.counter_episode/ self.episode_decay
        temp_max = min(self.temp_end, temp_max)

        temp = np.exp(random.uniform(np.log(self.temp_min), np.log(temp_max)))
        temp *= self.transition_decay ** self.counter_transition
        return max(self.temp_min, temp)

class Model:

    def __init__(self,
                 agent=None,
                 environment=None,
                 mem_capacity=None,
                 batch_size=None,
                 num_warmup=None,
                 policy_update=None,
                 tau = None,
                 temp_scaler=None,
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
        self.tau = tau
        self.temp_scaler = temp_scaler

        a = environment.max_num_moves
        self.gamma = 1 - 1/a
        self.criterion = criterion
        

        self.counter_episode = 0
        
        self.memory = ReplayMemory(mem_capacity, batch_size)

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
            self.counter_episode += 1

            done = False
            while not done:
                temp = self.temp_scaler.get_temp()
                action = self.agent.select_action(self.environment, 
                                                  temp=temp, 
                                                  greedy=False)

                move = self.agent.action_to_move(action)
                board_next, (reward, done) = self.environment.step(
                    move) # step also flips the board/state

                state_next = self.agent.board_to_state(board_next)
                mask_legal = self.agent.get_mask_legal(self.environment.get_legal_moves())

                done_tensor = torch.tensor(done, dtype=torch.float32, device=config.device).view(1,1)
                
                self.memory.push([state, action, state_next, mask_legal, reward, done_tensor])
                state = state_next

                if counter % self.policy_update == 0:
                    loss = self.optimize_agent()

                counter += 1
                self.temp_scaler.step_transition()

            self.temp_scaler.step_episode()

            if (i_episode + 1) % freq == 0:
                self.stats(evaluate_agents, loss)
                

    @torch.compile(mode="reduce-overhead", fullgraph=True, dynamic=False)
    def compute_loss(self, 
                     online_net, 
                     target_net, 
                     state_batch, 
                     action_batch, 
                     next_state_batch, 
                     mask_legal_batch, 
                     reward_batch, 
                     done_batch):

        Q = online_net(state_batch)
        state_action_values = Q.gather(1, action_batch)

        with torch.no_grad():
            Q_next = online_net(next_state_batch)
            Q_next = Q_next.masked_fill(~mask_legal_batch, -1e4)
            action_star = Q_next.argmax(1, keepdim=True)
        
            next_state_values = target_net(next_state_batch).gather(1, action_star)
            next_state_values = (1 - done_batch)*next_state_values

            # negative next_state_values since chess is zero-sum
            expected_state_action_values = reward_batch - self.gamma*next_state_values

        loss = self.criterion(state_action_values, expected_state_action_values)
        return loss

    
    def optimize_agent(self):

        if self.memory.num_samples < self.num_warmup:
            return None

        if random.random() < 0.5:
            online_net = self.agent.online_net1
            target_net = self.agent.target_net2
            opt = self.opt_list[0]
        else:
            online_net = self.agent.online_net2
            target_net = self.agent.target_net1
            opt = self.opt_list[1]
        
        state_batch, action_batch, next_state_batch, mask_legal_batch, reward_batch, done_batch = self.memory.sample()

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = self.compute_loss(online_net, 
                                     target_net, 
                                     state_batch, 
                                     action_batch, 
                                     next_state_batch, 
                                     mask_legal_batch, 
                                     reward_batch, 
                                     done_batch)

        # Optimize the model
        self.scaler.scale(loss).backward()
       
        self.scaler.unscale_(opt)
        params = [p for g in opt.param_groups for p in g['params'] if p.grad is not None]
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

        self.scaler.step(opt)
        self.scaler.update()
        opt.zero_grad(set_to_none=True)

        self.soft_update(target_net, online_net, self.tau)

        return loss.detach()
    
    def soft_update(self, target, online, tau: float):
        for p_t, p_o in zip(target.parameters(), online.parameters()):
            p_t.data.lerp_(p_o.data, tau) # p_t = (1 - tau) * p_t + tau * p_o

    def stats(self, evaluate_agents, loss):
        results = evaluate_agents.evaluate(verbose=False)
        transitions = self.memory.sample()
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


        
def save_checkpoint(model, filename='checkpoint.pth'):
    checkpoint = {
        'model_state_dict': model.agent.state_dict(),
        'optimizer1_state_dict': model.opt_list[0].state_dict(),
        'optimizer2_state_dict': model.opt_list[1].state_dict(),
        'memory': model.memory,
        'counter_episode': model.counter_episode,
    }
    torch.save(checkpoint, filename)


def load_checkpoint(filename, model):
    checkpoint = torch.load(filename, weights_only=False)
    model.agent.load_state_dict(checkpoint['model_state_dict'])
    
    model.opt_list[0].load_state_dict(checkpoint['optimizer1_state_dict'])
    model.opt_list[1].load_state_dict(checkpoint['optimizer2_state_dict'])

    model.memory = checkpoint['memory']

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
            white = self.agent1
            black = self.agent2
            one_starts = True
        else:
            white = self.agent2
            black = self.agent1
            one_starts = False

        done = False
        temp = self.temp
        ply = 0
        while not done:

            mover = white if (ply % 2 == 0) else black

            action = mover.select_action(self.environment, temp=temp, greedy=False)
            temp *= 0.95

            move = mover.action_to_move(action)
            _, (reward, done) = self.environment.step(move)

            if done:
                if reward.item() == 1:
                    # Terminal reward is for the mover who just played
                    winner = mover
                    return 1 if winner is self.agent1 else -1
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