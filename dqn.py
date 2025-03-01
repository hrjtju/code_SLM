"""
Ruijie He

Reference: https://hrl.boyuai.com/chapter/2/dqn%E7%AE%97%E6%B3%95

"""


import datetime
import os
import random
import collections
from sympy import true
from tqdm import tqdm
from typing import Iterable, Tuple
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb

from slm_model.slm_env import SingleSLMEnv

USE_PPO = False

def to_device(ls, device):
    return [torch.tensor(x, device=device) for x in ls] if isinstance(ls, Iterable) else torch.tensor(ls, device=device)

def repack(t: Tuple):
    return list(map(lambda x: torch.stack(x, dim=0), list(zip(*t))))

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
        
        # TODO: modify the following line
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
        
        self.view_nn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=2),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=2),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )
        self.lwh_nn = nn.Sequential(
            nn.Linear(in_features=3, out_features=32),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=32, out_features=32),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=32, out_features=32),
            nn.ReLU(inplace=True)
        )
        self.part_nn = nn.Sequential(
            nn.Linear(in_features=self.max_part*(3 + 4 * self.max_ori), out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(inplace=True)
        )
        
        view_feature_len = self.view_nn(torch.rand(size=(1, *self.view_shape)))\
            .reshape(-1).shape[0]
        
        self.bottleneck = nn.Sequential(
            nn.Linear(in_features=160+view_feature_len, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(inplace=True)
        )
        
        self.policy_part = nn.Sequential(
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=self.max_part),
            nn.Sigmoid()
        )
        self.policy_orientation = nn.Sequential(
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=self.max_ori),
            nn.Sigmoid()
        )
        
    def forward(self, x: Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        view, lwh, part = list(map(lambda x:x.to(self.device), x))
        batch_size = view.shape[0]
        
        f_view = self.view_nn(view).reshape(batch_size, -1)
        f_lwh = self.lwh_nn(lwh).reshape(batch_size, -1)
        f_part = self.part_nn(part).reshape(batch_size, -1)
        
        feature = torch.concat(
            tensors=[f_view, f_lwh, f_part],
            dim=-1
        )
        p_feature = self.bottleneck(feature)
        
        part_dist = self.policy_part(p_feature)
        orientation_dist = self.policy_orientation(p_feature)
        
        return part_dist, orientation_dist
        
class DQN:
    def __init__(self, 
                 lr: float, 
                 gamma: float, 
                 epsilon: float, 
                 target_update: int, 
                 device: torch.device,
                 max_part: int = 20, 
                 max_ori: int = 7, 
                 view_shape: Tuple[int, int] = (224, 224),
                 
                 ) -> None:
        
        self.max_part = max_part
        self.max_ori = max_ori
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
                torch.randn(self.max_part).abs() + 0.01, torch.randn(self.max_ori).abs() + 0.01
                )
            action = tuple(map(lambda x:x.reshape(1, -1), action))
            self.epsilon = max(0.05, self.epsilon * 0.99999)
        else:
            action = self.q_net(state)
        
        return action

    def update(self, transition_dict):
        states = list(map(lambda x: x.clone().detach().to(self.device), 
                          transition_dict["states"]))
        actions = list(map(lambda x:x.clone().detach().to(self.device), 
                           transition_dict["actions"]))
        rewards = torch.tensor(transition_dict["rewards"]).to(self.device)
        next_states = list(map(lambda x: x.clone().detach().to(self.device), 
                          transition_dict["next_states"]))
        dones = torch.tensor(transition_dict["dones"], dtype=torch.float32).to(self.device)
        
        q_values = torch.concat(self.q_net(states), -1)\
            .gather(1, torch.concat(list(map(lambda x: torch.argmax(x, -1), actions)), -1))\
                .sum(-1)
        max_next_q = sum(map(lambda x: x.max(1)[0], self.target_q_net(next_states))).reshape(-1)
        q_targets = rewards + self.gamma * max_next_q * (1 - dones)
        
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()
        
        if (self.update_count+1) % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        self.update_count += 1


if __name__ == "__main__":
    now_str = str(datetime.datetime.now()).split('.')[0].replace(':', '_').replace(' ', '_')
    
    # os.mkdir(f'./tf-logs/{now_str}')
    
    warnings.filterwarnings("ignore")
    
    wandb.init(
        project="slm2d-dqn",
        config={
            "action_type": "part-orientation, tuple",
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
            "time": str(datetime.datetime.now())
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
    
    replay_buffer = ReplayBuffer(buffer_size)
    
    agent = DQN(lr, gamma, epsilon, target_update, device)
    
    return_list = []
    energy_list = []
    instances_dict = {}
    
    env = SingleSLMEnv(in_path="./instances_json/", device=device)
    env_name = env.name
    
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc=f"Iteration {i}", leave=False, position=0) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state, instance = env.reset()
                done = False
                
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, terminate, truncate, _ = env.step(action)
                    done = terminate or truncate
                    
                    replay_buffer.add(*list(map(lambda x: to_device(x, "cpu"), [state, action, reward, next_state, done])))
                    # replay_buffer.add(state, action, reward, next_state, done)
                    
                    state = next_state
                    episode_return += reward
                    
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        agent.update(
                            transition_dict=dict(
                                states = to_device(b_s, device),
                                actions = to_device(b_a, device),
                                next_states = to_device(b_ns, device),
                                rewards = to_device(b_r, device),
                                dones = to_device(b_d, device)
                            )
                        )
                        
                return_list.append(episode_return)
                energy_list.append(env.last_criterion)

                episode_id = int(num_episodes / 10 * i + i_episode + 1)
                moving_avg_return = np.mean(return_list[-100:] if len(return_list) > 100 else np.mean(return_list))
                moving_ene_return = np.mean(energy_list[-200:] if len(energy_list) > 200 else np.mean(energy_list))
                
                pbar.set_postfix({
                    "episode": f"{episode_id:4d}",
                    "return": f"{f'{moving_avg_return:.6e}':12s}"
                })

                instances_dict[instance] = 1 if instance not in instances_dict else instances_dict[instance]+1
                
                wandb.log({"Avg Episode Return": moving_avg_return, 
                           "Avg Energy Return": moving_ene_return,
                           f"{instance}": env.last_criterion}, step=episode_id)
                
                pbar.update(1)
    
    now_str = datetime.datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S")
    torch.save(agent.q_net.state_dict(), f"./model_params/{env.__class__.__name__}_{agent.__class__.__name__}_{now_str}.pt")