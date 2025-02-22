from importlib.metadata import distributions
from re import M
from typing import Callable, Tuple
import gymnasium as gym
from pandas import Categorical
import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from slm_model.slm_env import SingleSLMEnvParallel1D

def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)

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
            nn.Linear(128, max_batch_num),
            nn.Softmax(dim=-1)
        )
        self.policy_part = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, max_part_type),
            nn.Softmax(dim=-1)
        )
        self.policy_orientation = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, max_ori_num),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):        
        feature_stem = self.policy_stem(x)
        batch_dist = self.policy_batch(feature_stem)
        part_dist = self.policy_part(feature_stem)
        ori_dist = self.policy_orientation(feature_stem)
        
        # filter out infeasible parts and batches
        part_mask = 1 - self.env.get_unavailable_parts_mask().float()
        batch_mask = 1 - (self.env.solution.get_current_view(show=True).reshape(-1) >= 1).float()
        print(part_dist.shape)
        
        # TODO: Fix this
        masked_part_dist = torch.nn.functional.softmax(part_dist, dim=-1) * part_mask.reshape(1, -1).to(self.device)
        masked_part_dist = masked_part_dist / masked_part_dist.sum(-1, keepdim=True)
        # masked_batch_dist = torch.nn.functional.softmax(batch_dist, dim=-1) * batch_mask.reshape(1, -1).to(self.device)
        # masked_batch_dist = masked_batch_dist / masked_batch_dist.sum(-1, keepdim=True)
        masked_batch_dist = batch_dist
        
        return masked_part_dist, ori_dist, masked_batch_dist
        
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
    
    def log_prob(self, state_actions: torch.Tensor, actions_history):
        return torch.log(action.gather(1))
    
    def take_action(self, state: torch.Tensor):
        state = state.reshape(1, -1)
        part_dist, ori_dist, batch_dist = self.actor(state)
        action_dists = list(map(lambda x:torch.distributions.Categorical(x), [part_dist, batch_dist]))
        part, batch = list(map(lambda x:x.sample(), action_dists))
       
        # mask out infeasible orientations
        # The line `ori_num = len(self.env.slm_metadata.parts[int(part.item())].build_params)` is
        # calculating the number of build parameters associated with a specific part selected by the
        # agent during the action selection process. Let's break down the purpose of this line:
        # The code snippet `ori_num = len(self.env.slm_metadata.parts[int(part.item())].build_params)`
        # is calculating the number of build parameters for a specific part in the environment. Let's
        # break it down:
        ori_num = len(self.env.slm_metadata.parts[int(part.item())].build_params)
        ori_mask = 1 - torch.tensor(
            [(0 if i < ori_num else 1) for i in range(self.max_ori_num)]
        )
        masked_ori_dist = torch.nn.functional.softmax(ori_dist, dim=-1) * ori_mask.reshape(1, -1).to(self.device)
        masked_ori_dist = masked_ori_dist / masked_ori_dist.sum()
        
        ori = torch.distributions.Categorical(masked_ori_dist).sample()
        
        return [part.item(), ori.item(), batch.item()]
    
    def update(self, transition_dict):
        states = torch.stack(transition_dict["states"], dim=0).to(self.device)
        actions = [torch.tensor([i]).view(-1, 1).to(self.device) for i in transition_dict["actions"]]
        rewards = torch.tensor([transition_dict["rewards"]], dtype=torch.float).to(self.device)
        next_states = torch.stack(transition_dict["next_states"], dim=0).to(self.device)
        dones = torch.tensor([transition_dict["dones"]], dtype=torch.float).to(self.device)
        
        rewards = (rewards + 8.0) / 8.0
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = rewards - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        
        dists = self.actor(states)
        old_log_probs_ls =  list(map(
            lambda x: torch.log(x[0].gather(1, x[1])).detach(),
            zip(dists, actions)
        ))
        
        for _ in range(self.epochs):
            
            # TODO: Split the batch performation in to individual steps
            # TODO: As the mask may change
            
            dists = self.actor(states)
            log_probs_ls = list(map(
                lambda x: torch.log(x[0].gather(1, x[1])).detach(),
                zip(dists, actions)
            ))
            
            ratio_ls = [torch.exp(lp - olp) for (lp, olp) in zip(log_probs_ls, old_log_probs_ls)]
            surr1 = sum(map(lambda x:x*advantage, ratio_ls))
            surr2 = sum(map(lambda x:torch.clamp(x, 1 - self.eps, 1 + self.eps) * advantage,
                            ratio_ls))
            
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
            
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            
            actor_loss.backward()
            critic_loss.backward()
            
            self.actor_optimizer.step()
            self.critic_optimizer.step()


if __name__ == "__main__":
    
    actor_lr = 1e-4
    critic_lr = 1e-4
    num_episodes = 5000
    hidden_dim = 128
    gamma = 0.9
    lmbda = 0.9
    epochs = 100
    eps = 0.2
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    env_name ="Pendulum-v1"
    env: gym.Env = gym.make(env_name)
    
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device, env)
    
    return_list = []
    for i in range(epochs):
        with tqdm(total=int(num_episodes/epochs), desc=f"Iteration {i}") as pbar:
            for i_episode in range(int(num_episodes / epochs)):
                episode_return = 0
                
                transition_dict = {
                    "states": [],
                    "actions": [],
                    "next_states": [],
                    "rewards": [],
                    "dones": []
                }
                
                state, _ = env.reset()
                done = False 
                
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, terminated, truncated, *_ = env.step(action)
                    done = terminated or truncated
                    
                    transition_dict["states"].append(state)
                    transition_dict["actions"].append(action)
                    transition_dict["next_states"].append(next_state)
                    transition_dict["rewards"].append(reward)
                    transition_dict["dones"].append(done)
                    
                    state = next_state
                    episode_return += reward
                    
                return_list.append(episode_return)
                agent.update(transition_dict)
                
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        "Episode": f"{(num_episodes/epochs * i + i_episode):>4d}",
                        "Return": f"{np.mean(return_list[-10:]):.3f}"
                    })
                pbar.update(1)
                