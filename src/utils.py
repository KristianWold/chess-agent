import numpy as np
import random
import os
import pickle
from itertools import count
import torch



def saver(object, filename):
    pickle.dump(object, open(filename, "wb"))


def loader(filename):
    object = pickle.load(open(filename, "rb"))

    return object


def save_core(model, filename='checkpoint.pth'):
    checkpoint = {
        'model_state_dict': model.agent.state_dict(),
        'optimizer1_state_dict': model.opt_list[0].state_dict(),
        'optimizer2_state_dict': model.opt_list[1].state_dict(),
        'counter_episode': model.counter_episode,
        'temp_scaler': model.temp_scaler
    }
    torch.save(checkpoint, filename)

def save_memory(model, filename='checkpoint.pth'):
    memory = model.memory
    checkpoint = {
        "states": memory.states,
        "actions": memory.actions,
        "next_states": memory.next_states,
        "mask_legal": memory.mask_legal,
        "rewards": memory.rewards,
        "done": memory.done,
        "index": memory.index,
        "num_samples": memory.num_samples,
        "capacity": memory.capacity,
        "batch_size": memory.batch_size
    }
    torch.save(checkpoint, filename)


def load_checkpoint(core_path=None, 
                    memory_path=None, 
                    model=None):
    checkpoint = torch.load(core_path, map_location="cpu", weights_only=False)
    model.agent.load_state_dict(checkpoint['model_state_dict'])
    
    model.opt_list[0].load_state_dict(checkpoint['optimizer1_state_dict'])
    model.opt_list[1].load_state_dict(checkpoint['optimizer2_state_dict'])

    model.counter_episode = checkpoint['counter_episode']
    model.temp_scaler = checkpoint['temp_scaler']

    if not memory_path is None:
        memory = torch.load(memory_path, map_location="cpu", weights_only=False)
        model.memory.states = memory['states']
        model.memory.actions = memory['actions']
        model.memory.next_states = memory['next_states']
        model.memory.mask_legal = memory['mask_legal']
        model.memory.rewards = memory['rewards']
        model.memory.done = memory['done']
        model.memory.index = memory['index']
        model.memory.num_samples = memory['num_samples']
        model.memory.capacity = memory['capacity']
        model.memory.batch_size = memory['batch_size']

        model.memory.init()

    return model
