"""
Ruijie He

Reference: https://hrl.boyuai.com/chapter/2/dqn%E6%94%B9%E8%BF%9B%E7%AE%97%E6%B3%95

"""


import datetime
from functools import reduce
import os
import random
import collections
from sympy import true
import torch.utils
from tqdm import tqdm
from typing import Iterable, Tuple
import warnings
import sys

from training_utils.model_utils import DQN_Log, calculate_gradient_norm

# Add the slm_model directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor, narrow
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt
import wandb
from slm_model.slm_env import SingleSLMEnvParallel1D
from training_utils.utils import IntstanceAvgMeter


def to_device(x, device):
    try:
        x.to(device)
    except AttributeError:
        return x
    
    return x.to(device)

def repack(t: Tuple):
    return torch.stack(t, dim=0)

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

class QNet(nn.Module):
    def __init__(self, 
                 max_part: int = 20, 
                 max_ori: int = 7, 
                 view_shape: Tuple[int, int] = (224, 224),
                 device: torch.device = "cpu",
                 ) -> None:
        super(QNet, self).__init__()
        
        self.max_part = max_part
        self.max_ori = max_ori
        self.view_shape = view_shape
        self.device = device
        
        self.policy = nn.Sequential(
            nn.Linear(in_features=683, out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=47),
            nn.Sigmoid()
        )
        
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        dist = self.policy(x)        
        return dist
        
class DoubleDQN:
    def __init__(self, 
                 lr: float, 
                 gamma: float, 
                 epsilon: float, 
                 target_update: int, 
                 device: torch.device,
                 max_part: int = 20, 
                 max_ori: int = 7, 
                 max_batch: int = 20,
                 view_shape: Tuple[int, int] = (224, 224),
                 
                 ) -> None:
        
        self.max_part = max_part
        self.max_ori = max_ori
        self.max_batch = max_batch
        self.view_shape = view_shape
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        
        self.update_count = 0
        self.device = device
        
        self.q_net = QNet(self.max_part, self.max_ori, self.view_shape, self.device).to(self.device)
        self.target_q_net = QNet(self.max_part, self.max_ori, self.view_shape, self.device).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        
    def take_action(self, state):
        
        # TODO: Check the two branches of these actions
        if np.random.random() < self.epsilon:
            action = (
                torch.randn(self.max_part+self.max_ori+self.max_batch).abs() + 0.01
                )
            
            if np.random.random() < self.epsilon:
                self.epsilon = max(0.05, self.epsilon * 0.99999)
        else:
            action = self.q_net(state)
        
        return action

    def split_actions(self, a: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Split action tensor into part, orientation and batch part.
        """
        # a: [batch+_size, MAX_PART_TYPE + MAX_ORIENTATION_NUM + MAX_BATCH_NUM]
        return (
            a.narrow(-1, 0, MAX_PART_TYPE),
            a.narrow(-1, MAX_PART_TYPE, MAX_ORIENTATION_NUM),
            a.narrow(-1, MAX_PART_TYPE+MAX_ORIENTATION_NUM, MAX_BATCH_NUM)
        )
    
    def batch_kronecker_product_flatten(self, tensors) -> Tensor:
        return reduce(lambda x,y:torch.einsum("ia,ib->iab", x.flatten(1), y.flatten(1)).flatten(-1), 
                      tensors)
    
    def update(self, transition_dict) -> DQN_Log:
        states = transition_dict["states"]
        actions = transition_dict["actions"]
        rewards = torch.tensor(transition_dict["rewards"]).to(self.device)
        next_states = transition_dict["next_states"]
        dones = torch.tensor(transition_dict["dones"], dtype=torch.float32).to(self.device)
        
        ## TODO: Split and apply addition to q values
        net_state_output = self.q_net(states)
        splitted_max_actions = self.split_actions(net_state_output)
        splitted_done_actions = self.split_actions(actions)
        # TODO: Split tensors into 3 parts and calculate q values individually
        max_action_sets = self.q_net(next_states)
        target_action_sets = self.target_q_net(next_states)
        splitted_max_actions_next = self.split_actions(max_action_sets)
        splitted_max_actions_target = self.split_actions(target_action_sets)
        
        # TODO: q value is defined as sum of largest q in part, orientation and batch.
        q_values = sum(map(lambda x:x[0].gather(1, x[1].max(1)[1].view(-1, 1)), zip(splitted_max_actions, splitted_done_actions)))
        max_next_q = sum(map(lambda x:x[0].gather(1, x[1].max(1)[1].view(-1, 1)), zip(splitted_max_actions_target, splitted_max_actions_next)))
        
        # ## TODO: OR split and apply kronecker product
        # tensor_max_actions = self.batch_kronecker_product_flatten(splitted_max_actions)
        # tensor_done_actions = self.batch_kronecker_product_flatten(splitted_done_actions)
        # tensor_max_actions_next = self.batch_kronecker_product_flatten(splitted_max_actions_next)
        # tensor_max_actions_target = self.batch_kronecker_product_flatten(splitted_max_actions_target)
        
        # q_values = tensor_max_actions.gather(1, tensor_done_actions.max(1)[1].view(-1, 1))
        # max_next_q = tensor_max_actions_next.gather(1, tensor_max_actions_target.max(1)[1].view(-1, 1))
        
        q_targets = rewards + self.gamma * max_next_q * (1 - dones)
        
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        
        self.optimizer.zero_grad()
        dqn_loss.backward()
        
        clip_grad_norm_(self.q_net.parameters(), 10)
        grad_norm = calculate_gradient_norm(self.q_net)
        
        self.optimizer.step()
        
        if (self.update_count+1) % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        self.update_count += 1
        
        return DQN_Log(
            loss=dqn_loss.item(),
            grad_norm=grad_norm
        )


if __name__ == "__main__":
    
    warnings.filterwarnings("ignore")
    
    wandb.init(
        project="slmflat-doubledqn",
        config={
            "actions_type": "part-orientation-batch, tensor",
            "criterion": "energy-diff",
            "lr": 5e-3,
            "num_episodes": 50000,
            "hidden_dim": 128,
            "gamma": 1.00,
            "epsilon_start": 0.5,
            "epsilon_rate": 0.999,
            "target_update": 5,
            "buffer_size": 50000,
            "minimal_size": 600,
            "batch_size": 64,
            "device": "cuda",
            "max_part_type": 20,
            "max_orientation_num": 7,
            "max_batch_num": 20,
            "time": datetime.datetime.now().strftime(r"%Y-%m-%d %H:%M:%S")
        },
    )
    
    lr = 5e-3
    num_episodes = 50000
    hidden_dim = 128
    gamma = 1.00
    epsilon = 0.5 # 0.05
    target_update = 5
    buffer_size = 50000
    minimal_size = 600
    batch_size = 64
    device = torch.device("cuda")
    
    MAX_PART_TYPE = 20
    MAX_ORIENTATION_NUM = 7
    MAX_BATCH_NUM = 20
    
    replay_buffer = ReplayBuffer(buffer_size)
    
    agent = DoubleDQN(lr, gamma, epsilon, target_update, device, 
                      max_part=MAX_PART_TYPE, max_ori=MAX_ORIENTATION_NUM, max_batch=MAX_BATCH_NUM)
    
    return_meter = IntstanceAvgMeter(window_size=200)
    energy_meter = IntstanceAvgMeter(window_size=200)
    loss_meter = IntstanceAvgMeter(window_size=2000)
    grad_norm_meter = IntstanceAvgMeter(window_size=2000)
    
    env = SingleSLMEnvParallel1D(in_path="./instances_json/", phase="Train",
                                 max_part_type=MAX_PART_TYPE, max_batch_num=MAX_BATCH_NUM, max_orientation_num=MAX_ORIENTATION_NUM)
    env_name = env.name
    
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc=f"Iteration {i}", leave=False, position=0) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state, instance = env.reset()
                done = False
                
                while not done:
                    
                    action = agent.take_action(state.to(device))
                    next_state, reward, terminate, truncate, _ = env.step(action)
                    done = terminate or truncate
                    
                    replay_buffer.add(*list(map(lambda x: to_device(x, "cpu"), [state, action, reward, next_state, done])))
                    # replay_buffer.add(state, action, reward, next_state, done)
                    
                    state = next_state
                    episode_return += reward
                    
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        # print(b_s.shape)
                        tmp_log = agent.update(
                            transition_dict=dict(
                                states = to_device(b_s, device),
                                actions = to_device(b_a, device),
                                next_states = to_device(b_ns, device),
                                rewards = to_device(b_r, device),
                                dones = to_device(b_d, device)
                            )
                        )
                        
                        loss_meter.update(instance, tmp_log.loss)
                        grad_norm_meter.update(instance, tmp_log.grad_norm)
                        
                return_meter.update(instance, episode_return)
                energy_meter.update(instance, env.solution.calculate_energy())

                episode_id = int(num_episodes / 10 * i + i_episode + 1)
                
                pbar.set_postfix({
                    "episode": f"{episode_id:4d}",
                    "return": f"{f'{return_meter.all_avg():.6e}':12s}",
                    "energy": f"{f'{energy_meter.all_avg():.6e}':12s}",
                    "loss": f"{f'{loss_meter.all_avg():.6e}':12s}",
                    "grad_norm": f"{f'{grad_norm_meter.all_avg():.6e}':12s}",
                })
                
                wandb.log({"AvgEnergy/mean": energy_meter.all_avg(),
                            "AvgEnergy/max": energy_meter.all_max_avg(),
                            "AvgEnergy/min": energy_meter.all_min_avg(),
                            "AvgReturn/mean": return_meter.all_avg(),
                            "AvgReturn/max": return_meter.all_max_avg(),
                            "AvgReturn/min": return_meter.all_min_avg(),
                           f"Instances/{instance}": env.last_criterion,
                            "epsilon": agent.epsilon, 
                            "DQN/loss": loss_meter.all_avg(),
                            "DQN/grad_norm": grad_norm_meter.all_avg()
                          }, step=episode_id)
                
                pbar.update(1)
    
    now_str = datetime.datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S")
    torch.save(agent.q_net.state_dict(), f"./model_params/{env.__class__.__name__}_{agent.__class__.__name__}_{now_str}.pt")