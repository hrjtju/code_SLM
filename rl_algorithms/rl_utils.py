from typing import Tuple
import torch
import collections
import random

def to_device(x, device):
    try:
        x.to(device)
    except AttributeError:
        return x
    
    return x.to(device)

def repack(t: Tuple):
    if isinstance(t[0], torch.Tensor):
        return torch.stack(t, dim=0)
    if isinstance(t[0], tuple):
        return [torch.stack(i, dim=0) for i in zip(*t)]

class ReplayBuffer:
    """
    Replay Buffer of tuples for DQN training
    """
    def __init__(self, capacity: int) -> None:
        self.buffer = collections.deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append(
            (state, action, reward, next_state, done)
        )
    
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        
        return repack(state), repack(action), reward, repack(next_state), done

    def size(self):
        return len(self.buffer)