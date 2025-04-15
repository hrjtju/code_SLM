"""
Ruijie He

Reference: https://hrl.boyuai.com/chapter/2/dqn%E6%94%B9%E8%BF%9B%E7%AE%97%E6%B3%95

"""


import datetime
from functools import reduce
import os
import random
import collections
import heapq
from sympy import true
import torch.utils
from tqdm import tqdm
from typing import List, Tuple
import warnings
import sys

from training_utils.model_utils import DQN_Log, calculate_gradient_norm, init_weights_normal

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

from dataclasses import dataclass, field
from typing import Any

@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Tuple[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor], int]=field(compare=False)


class PrioritizedReplayBuffer:
    """
    Replay Buffer of tuples for DQN training
    
    1. 按照td-error的优先级进入，只保存较大td-error的条目
    2. 根据时效性，周期删除 10% 较老的条目
    """
    def __init__(self, capacity: int) -> None:
        self.buffer: List[PrioritizedItem] = []
        self.num = 0
        self.capacity = capacity
        # self.seen = set()
        self.in_num = 0
    
    # def hash_(self, tup: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]) -> int:
    #     return hash(tuple(torch.concat(tup, dim=0).reshape(-1).detach().numpy()))
    
    #! update to priority queue, using -|td_error| as priority
    def add(self, state: Tensor, action: Tensor, reward: Tensor, next_state: Tensor, done: Tensor, abs_td_error: float):
        priority = abs_td_error
        tup = (state, action, reward, next_state, done)
        
        self.num += 1
        self.in_num += 1
        
        # if self.hash_(tup) in self.seen:
        #     for i in self.buffer:
        #         if i.item == tup:
        #             print(f"\nHash hit: {i.priority:.2f} --> {priority:.2f}")
        #             i.priority = priority
        #             i.item[1] = self.in_num
        #             heapq.heapify(self.buffer)
        #             break
        # else:
        self.num += 1
        
        # 先丢已经在里面的，这样就不会出现前期td-err大的一直占在里面，后面的进不来的情况
        if self.num > self.capacity:
            self.num = self.capacity
            pop = heapq.heappop(self.buffer)
            # self.seen.remove(self.hash_(pop.item[0]))
            
            if random.random() < 0.1:
                print("\nRemoving old history ...")
                self.buffer = sorted(self.buffer, key=lambda x:x.item[1])[self.capacity//10:]
                self.num = len(self.buffer)
                heapq.heapify(self.buffer)
        
        heapq.heappush(self.buffer, PrioritizedItem(priority=priority, item=(tup, self.in_num)))
        # self.seen.add(self.hash_(tup))
    
    def sample(self, batch_size):
        transitions = [i.item[0] for i in random.sample((self.buffer), batch_size)]
        state, action, reward, next_state, done = zip(*transitions)
        
        return repack(state), repack(action), reward, repack(next_state), done
    
    def size(self):
        return len(self.buffer)
    
    def get_priority_dist(self) -> Tuple[float, float]:
        arr = np.array([i.priority for i in self.buffer])
        return arr.mean(), arr.std()
        

class ClassicalReplayBuffer:
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

    
    def sample(self, batch_size):
        transitions = [i.item for i in random.sample((self.buffer), batch_size)]
        state, action, reward, next_state, done = zip(*transitions)
        
        return repack(state), repack(action), reward, repack(next_state), done

    def size(self):
        return self.num

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
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=256, out_features=128),
            nn.LeakyReLU(inplace=True),
            nn.LayerNorm(128),
            nn.Linear(in_features=128, out_features=47),
            # nn.Softmax(dim=-1) #! 按道理来说应该分组 softmax，或者直接换成三个头
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
        self.q_net.apply(init_weights_normal)
        
        self.target_q_net = QNet(self.max_part, self.max_ori, self.view_shape, self.device).to(self.device)
        self.q_net.apply(init_weights_normal)
        
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        
    def take_action(self, state: torch.Tensor, curr_mask: torch.Tensor, test: bool = False):
        
        # TODO: Check the two branches of these actions
        if not test and np.random.random() < self.epsilon:
            action = (
                torch.randn(self.max_part+self.max_ori+self.max_batch).abs() + 0.01
                ).to(self.device)
            
            if np.random.random() < self.epsilon:
                self.epsilon = max(0.05, self.epsilon * 0.99999)
        else:
            action = self.q_net(state)
        
        part_d, ori_d, batch_d = self.split_actions(action)
        
        part_d[~torch.sum(curr_mask, dim=(-1, -2)).bool()] = -torch.inf
        ori_d[~torch.sum(curr_mask[part:=part_d.argmax(-1)], dim=-1).bool()] = -torch.inf
        batch_d[~curr_mask[part, ori_d.argmax(-1)].bool()] = -torch.inf
        
        return torch.concat([part_d, ori_d, batch_d])

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
    
    def get_td_target(self, state, action, reward, next_state, done, test=False):
        net_state_output = self.q_net(state)
        splitted_max_actions = self.split_actions(net_state_output)
        splitted_done_actions = self.split_actions(action)
        
        max_action_sets = self.q_net(next_state)
        target_action_sets = self.target_q_net(next_state)
        splitted_max_actions_next = self.split_actions(max_action_sets)
        splitted_max_actions_target = self.split_actions(target_action_sets)
        
        q_values = sum(map(lambda x:x[0].gather(1, x[1].max(1)[1].view(-1, 1)), zip(splitted_max_actions, splitted_done_actions)))
        max_next_q = sum(map(lambda x:x[0].gather(1, x[1].max(1)[1].view(-1, 1)), zip(splitted_max_actions_target, splitted_max_actions_next)))
        
        q_targets = reward + self.gamma * max_next_q * (1 - done)
        
        if test:
            return q_values.detach(), q_targets.detach()
        else:
            return q_values, q_targets.detach()
    
    def update(self, transition_dict) -> DQN_Log:
        states = transition_dict["states"]
        actions = transition_dict["actions"]
        rewards = torch.tensor(transition_dict["rewards"]).to(self.device)
        next_states = transition_dict["next_states"]
        dones = torch.tensor(transition_dict["dones"], dtype=torch.float32).to(self.device)
        
        q_values, q_targets = self.get_td_target(states, actions, rewards, next_states, dones)
        
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        
        self.optimizer.zero_grad()
        
        dqn_loss.backward()
        
        clip_grad_norm_(self.q_net.parameters(), 100)
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
            "name": (trial_name:="slmflat-doubledqn-penalty2-bs64-lr01"),
            "actions_type": "part-orientation-batch, tensor",
            "criterion": "energy-diff",
            "lr": 0.01,
            "num_episodes": 50000,
            "hidden_dim": 128,
            "gamma": 1.00,
            "epsilon_start": 0.5,
            "epsilon_rate": 0.999,
            "target_update": 5,
            "buffer_size": 50000,
            "minimal_size": 600,
            "batch_size": 256,
            "device": "cuda",
            "max_part_type": 20,
            "max_orientation_num": 7,
            "max_batch_num": 20,
            "cuda_grad_norm": 100,
            "time": datetime.datetime.now().strftime(r"%Y-%m-%d %H:%M:%S")
        },
    )
    
    lr = 0.01
    num_episodes = 100000
    hidden_dim = 128
    gamma = 1.00
    epsilon = 0.5 # 0.05
    target_update = 5
    buffer_size = 50000
    minimal_size = 600
    batch_size = 256
    device = torch.device("cuda")
    
    MAX_PART_TYPE = 20
    MAX_ORIENTATION_NUM = 7
    MAX_BATCH_NUM = 20
    
    replay_buffer = PrioritizedReplayBuffer(buffer_size)
    
    agent = DoubleDQN(lr, gamma, epsilon, target_update, device, 
                      max_part=MAX_PART_TYPE, max_ori=MAX_ORIENTATION_NUM, max_batch=MAX_BATCH_NUM)
    
    return_meter = IntstanceAvgMeter(window_size=200)
    energy_meter = IntstanceAvgMeter(window_size=200)
    loss_meter = IntstanceAvgMeter(window_size=2000)
    grad_norm_meter = IntstanceAvgMeter(window_size=2000)
    
    test_meter = IntstanceAvgMeter(window_size=10)
    
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
                    
                    action = agent.take_action(state.to(device), env.mask_tensor.to(device), test=False)
                    next_state, reward, terminate, truncate, _ = env.step(action)
                    done = terminate or truncate
                    
                    # TODO: Calculate TD_error
                    q_value, q_target = agent.get_td_target(state.to(device).reshape(1, -1), 
                                                            action.to(device).reshape(1, -1), 
                                                            torch.tensor([[reward]], device=device), 
                                                            next_state.to(device).reshape(1, -1), 
                                                            torch.tensor([[done]], device=device).float(),
                                                            test=True
                                                            )
                        
                    td_err = torch.sum(torch.abs(q_value - q_target)).item()
                    
                    if any((not isinstance(i, torch.Tensor)) for i in [state, action, next_state]):
                        ...
                        
                    replay_buffer.add(*list(map(lambda x: to_device(x, "cpu").reshape(-1), [state, action, torch.Tensor([reward]), next_state, torch.Tensor([done])])), td_err)
                    
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
                
                td_err_mean, td_err_std = replay_buffer.get_priority_dist()
                
                wandb.log({"AvgEnergy/mean": energy_meter.all_avg(),
                            "AvgEnergy/max": energy_meter.all_max_avg(),
                            "AvgEnergy/min": energy_meter.all_min_avg(),
                            "AvgReturn/mean": return_meter.all_avg(),
                            "AvgReturn/max": return_meter.all_max_avg(),
                            "AvgReturn/min": return_meter.all_min_avg(),
                            "epsilon": agent.epsilon, 
                            "DQN/loss": loss_meter.all_avg(),
                            "DQN/grad_norm": grad_norm_meter.all_avg(),
                            "DQN/fail_allocate_num": env.fail_allocate_num, # TODO: Check failed allocation num, and its relation to loss
                            "ReplayBuffer/TD_Error_Mean": td_err_mean,
                            "ReplayBuffer/TD_Error_Std": td_err_std,
                          }, step=episode_id)
                
                if i_episode % 500 == 0:
                    
                    # test model with greedy action-selection
                    for instance_f in os.listdir("./instances_json/"):
                        test_env = SingleSLMEnvParallel1D(in_path=f"./instances_json/{instance_f}", phase="Test",
                                    max_part_type=MAX_PART_TYPE, max_batch_num=MAX_BATCH_NUM, max_orientation_num=MAX_ORIENTATION_NUM)
                        
                        state_, instance_ = test_env.curr_state, test_env.in_path
                        done = False
                        
                        with torch.no_grad():
                            while not done:
                                action = agent.take_action(state_.to(device), test_env.mask_tensor.to(device), test=True)
                                next_state, _, terminate, truncate, _ = test_env.step(action)
                                done = terminate or truncate
                                
                                state = next_state
                            
                            test_meter.update(os.path.basename(instance_), test_env.solution.calculate_energy())
                    
                    wandb.log(test_meter.dict_avg("Instances/"))
                            
                            
                pbar.update(1)
    
        now_str = datetime.datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S")
        torch.save(agent.q_net.state_dict(), f"./model_params/{trial_name}_{i}_{now_str}.pt")