from importlib.metadata import distributions
from re import M
from ssl import ALERT_DESCRIPTION_DECOMPRESSION_FAILURE
from typing import Callable, Tuple, List
import gymnasium as gym
import pydantic
from pandas import Categorical
import torch
from torch import Tensor
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from slm_model.slm_env import SingleSLMEnvParallel1D

class PPO_Log(pydantic.BaseModel):
    avg_actor_loss: float
    avg_critic_loss: float
    avg_actor_grad_norm: float
    avg_critic_grad_norm: float


def calculate_gradient_norm(model: torch.nn.Module) -> float:
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def compute_advantage(gamma: float, lmbda: float, td_delta: Tensor) -> Tensor:
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)

class Transition:
    def __init__(self, device) -> None:
        self.states: List[Tensor] = []
        self.actions: List[List[int]] = []
        self.next_states: List[Tensor] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.masks: List[bool] = []
        self.device = device
    
    def append_history(self, state, action, next_state, reward, done, mask=None) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.dones.append(done)
        if mask is not None:
            self.masks.append(mask)
    
    def readout(self) -> Tuple[Tensor, List[Tensor], Tensor, Tensor, Tensor]\
                        |Tuple[Tensor, List[Tensor], Tensor, Tensor, Tensor, Tensor]:
        states_ = torch.stack(self.states, dim=0).to(self.device)
        actions_ = torch.stack([torch.tensor([i]).view(-1).to(self.device) for i in self.actions], dim=0).to(self.device)
        rewards_ = torch.tensor([self.rewards], dtype=torch.float).to(self.device)
        next_states_ = torch.stack(self.next_states, dim=0).to(self.device)
        dones_ = torch.tensor([self.dones], dtype=torch.float).to(self.device)
        
        if len(self.masks) == len(self.dones):
            masks_ = torch.stack(self.masks, dim=0).bool().to(self.device)
            return states_, actions_, rewards_, next_states_, dones_, masks_
        
        return states_, actions_, rewards_, next_states_, dones_

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, max_part_type, max_ori_num, max_batch_num, env: SingleSLMEnvParallel1D, device):
        super(PolicyNet, self).__init__()
        self.env = env 
        self.device = device
        
        self.policy_stem = nn.Sequential(
            nn.Linear(in_features=683, out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(inplace=True)
        )
        
        self.policy_batch = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, max_batch_num),
            nn.Softmax(dim=-1)
        )
        self.policy_part = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, max_part_type),
            nn.Softmax(dim=-1)
        )
        self.policy_orientation = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, max_ori_num),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):        
        feature_stem = self.policy_stem(x)
        batch_dist = self.policy_batch(feature_stem)
        part_dist = self.policy_part(feature_stem)
        ori_dist = self.policy_orientation(feature_stem)
        
        return part_dist, ori_dist, batch_dist
        
class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.value = nn.Sequential(
            nn.Linear(in_features=683, out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=1)
        )
        
    def forward(self, x):
        return self.value(x)

class PPO:
    """PPO CLIP Version"""
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device,
                 max_part_type, max_ori_num, max_batch_num, env: SingleSLMEnvParallel1D):
        self.env = env 
        self.device = device
        self.max_ori_num = max_ori_num
        
        self.actor: Callable[[torch.Tensor], Tuple[torch.Tensor]] \
            = PolicyNet(state_dim, hidden_dim, action_dim, max_part_type, max_ori_num, max_batch_num, env, device).to(device)
        self.critic: Callable[[torch.Tensor], torch.Tensor] \
            = ValueNet(state_dim, hidden_dim).to(device)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), critic_lr)
        
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device
    
    def normalize(self, p):
        return p / torch.sum(p, dim=-1, keepdim=True)
    
    def take_action(self, state: torch.Tensor, curr_mask: torch.Tensor):
        # TODO: Update According to mask
        
        state = state.reshape(1, -1)
        part_dist, ori_dist, batch_dist = self.actor(state)
        
        # print(part_dist)
        part_dist = self.normalize(part_dist * torch.sum(curr_mask, dim=(-1, -2)).bool())
        # print(part_dist)
        part = torch.distributions.Categorical(part_dist).sample()
        
        ori_dist = self.normalize(ori_dist * torch.sum(curr_mask[part], dim=-1).bool())
        ori = torch.distributions.Categorical(ori_dist).sample()
        
        batch_dist = self.normalize(batch_dist * curr_mask[part, ori].bool())
        batch = torch.distributions.Categorical(batch_dist).sample()
        
        return [part.item(), ori.item(), batch.item()]
    
    def get_filtered_dist_with_action(self, 
                                      actions: Tuple[int, int, int], 
                                      dists: Tuple[Tensor, Tensor, Tensor], 
                                      mask: Tensor
                                      ):
        # print(actions, actions[0].shape, len(actions))
        # part, ori shape = [n_step, 1]
        part, ori = actions[:, 0], actions[:, 1]
        
        # [n_step, sub_action_dim]
        pd, od, bd = dists
        
        # print(pd.shape, od.shape, bd.shape)
        
        # [n_step, n_parts]
        part_mask = torch.sum(mask, dim=(-1, -2)).bool()
        pd = self.normalize(pd * part_mask)
        
        # TODO: Check: Ori mask
        ori_mask = torch.sum(mask[torch.range(0, mask.shape[0]-1).long(), part.long()], dim=-1).bool()
        # print(mask.shape, ori_mask.shape, od.shape)
        od = self.normalize(ori_mask * od)
        
        # TODO: Check: Batch mask
        batch_mask = mask[torch.range(0, mask.shape[0]-1).long(), part.long(), ori.long()]
        bd = self.normalize(batch_mask * bd)
        
        # print(pd.shape, od.shape, bd.shape)
        
        return [pd, od, bd]
        
    
    def update(self, transition: Transition) -> PPO_Log:
        # states = torch.stack(transition.states, dim=0).to(self.device)
        # actions = [torch.tensor([i]).view(-1, 1).to(self.device) for i in transition.actions]
        # rewards = torch.tensor([transition.rewards], dtype=torch.float).to(self.device)
        # next_states = torch.stack(transition.next_states, dim=0).to(self.device)
        # dones = torch.tensor([transition.dones], dtype=torch.float).to(self.device)
        
        # masks: [n_step, n_part, n_ori, n_batch]
        states, actions, rewards, next_states, dones, masks = transition.readout()
        
        rewards = (rewards + 8.0) / 8.0
        td_target: Tensor = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = rewards - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        
        # dists = [part_dist, ori_dist, batch_dist]
        dists = self.actor(states)
        
        
        slice_a = lambda x: (x[:, 0], x[:, 1], x[:, 2]) 
        
        # TODO: Fix log probs
        # TODO: 按照 part, orientation, batch 的顺序依次过滤 policy distribution
        # TODO: 有没有什么更好的做法？（最差的做法就是一个循环）
        # TODO: Check
        old_log_probs_ls =  list(map(
            lambda x: torch.log(x[0].gather(1, x[1][..., None])).detach(),
            zip(self.get_filtered_dist_with_action(actions, dists, masks), slice_a(actions))
        ))
        
        avg_actor_loss_ls: List[float] = []
        avg_critic_loss_ls: List[float] = []
        avg_actor_grad_norm_ls: List[float] = []
        avg_critic_grad_norm_ls: List[float] = []
        
        for _ in range(self.epochs):
            
            dists = self.actor(states)
            
            # TODO: Fix log probs
            # TODO: 同理如上
            # TODO: Check
            log_probs_ls = list(map(
                lambda x: torch.log(x[0].gather(1, x[1][..., None])),
                zip(self.get_filtered_dist_with_action(actions, dists, masks), slice_a(actions))
            ))
            
            ratio_ls = [torch.exp(lp - olp) for (lp, olp) in zip(log_probs_ls, old_log_probs_ls)]
            surr1 = sum(map(lambda x:x*advantage, ratio_ls))
            surr2 = sum(map(lambda x:torch.clamp(x, 1 - self.eps, 1 + self.eps) * advantage,
                            ratio_ls))
            
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
            
            avg_actor_loss_ls.append(actor_loss.item())
            avg_critic_loss_ls.append(actor_loss.item())
            avg_actor_grad_norm_ls.append(calculate_gradient_norm(self.actor))
            avg_critic_grad_norm_ls.append(calculate_gradient_norm(self.critic))
            
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            
            actor_loss.backward()
            critic_loss.backward()
            
            self.actor_optimizer.step()
            self.critic_optimizer.step()
        
        avg = lambda x: sum(x) / len(x)
        return PPO_Log(avg_actor_loss=avg(avg_actor_loss_ls), 
                       avg_critic_loss=avg(avg_critic_loss_ls),
                       avg_actor_grad_norm=avg(avg_actor_grad_norm_ls),
                       avg_critic_grad_norm=avg(avg_critic_grad_norm_ls)
                       )
