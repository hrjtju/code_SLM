import warnings
import datetime
import os
import torch
from tqdm import tqdm
import numpy as np

from rl_algorithms.rl_utils import ReplayBuffer, to_device
from doubledqn_flat import DoubleDQN
from rl_algorithms.ppo_test_flat import PPO, Transition
from slm_model.slm_env import SingleSLMEnv, SingleSLMEnvParallel1D

import wandb

warnings.filterwarnings("ignore")

wandb.init(
    project="slmflat-PPO",
    config={
        "actions_type": "part-orientation-batch, tensor",
        "criterion": "energy-diff",
        "lr": 5e-3,
        "num_episodes": 10000,
        "hidden_dim": 128,
        "gamma": 1.00,
        "device": "cuda",
        "max_part_type": 20,
        "max_orientation_num": 7,
        "max_batch_num": 20,
        "time": datetime.datetime.now().strftime(r"%Y-%m-%d %H:%M:%S")
    },
)
    

lr = 0.01
num_episodes = 10000
hidden_dim = 128
gamma = 1.00
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
epochs=20

MAX_PART_TYPE = 20
MAX_ORIENTATION_NUM = 7
MAX_BATCH_NUM = 20

env = SingleSLMEnvParallel1D(in_path="./instances_json/", phase="Train",
                             max_part_type=MAX_PART_TYPE, max_batch_num=MAX_BATCH_NUM, max_orientation_num=MAX_ORIENTATION_NUM, 
                             seed=25, 
                             ppo=True)
env_name = env.name

agent = PPO(state_dim=923,
            hidden_dim=512,
            action_dim=None,
            actor_lr=1e-4,
            critic_lr=1e-4,
            lmbda=0.9,
            epochs=100,
            eps=0.2,
            gamma=0.9,
            device=torch.device("cuda"),
            max_part_type=MAX_PART_TYPE,
            max_batch_num=MAX_BATCH_NUM,
            max_ori_num=MAX_ORIENTATION_NUM,
            env=env)

now_str = str(datetime.datetime.now()).split('.')[0].replace(':', '_').replace(' ', '_')
# os.mkdir(f'./tf-logs/{agent.__class__.__name__}_{now_str}')
# writer = SummaryWriter(f'./tf-logs/{agent.__class__.__name__}_{now_str}')

return_list = []
energy_list = []
instances_dict = {}


for i in range(epochs):
    with tqdm(total=int(num_episodes//epochs), desc=f"Iteration {i}", leave=False, position=0) as pbar:
        for i_episode in range(int(num_episodes//epochs)):
            episode_return = 0
            
            transition = Transition(device=device)
            
            state, instance = env.reset()
            done = False
            
            while not done:
                curr_mask = env.mask_tensor
                action = agent.take_action(state.to(device), curr_mask.to(device))
                next_state, reward, terminate, truncate, _ = env.step(action)
                done = terminate or truncate
                
                transition.append_history(state, action, next_state, reward, done, curr_mask)
                
                state = next_state
                episode_return += reward
                
            return_list.append(episode_return)
            energy_list.append(env.last_criterion)
            
            episode_log = agent.update(transition)

            episode_id = int(num_episodes / 10 * i + i_episode + 1)
            moving_avg_return = np.mean(return_list[-200:] if len(return_list) > 200 else np.mean(return_list))
            moving_ene_return = np.mean(energy_list[-200:] if len(energy_list) > 200 else np.mean(energy_list))
            
            pbar.set_postfix({
                "episode": f"{episode_id:4d}",
                "return": f"{f'{moving_avg_return:.6e}':12s}",
                "energy": f"{f'{moving_ene_return:.6e}':12s}",
                "avg_actor_loss": f"{f'{episode_log.avg_actor_loss:.6e}':12s}",
                "avg_critic_loss": f"{f'{episode_log.avg_critic_loss:.6e}':12s}",
                "avg_actor_grad_norm": f"{f'{episode_log.avg_actor_grad_norm:.6e}':12s}",
                "avg_critic_grad_norm": f"{f'{episode_log.avg_critic_grad_norm:.6e}':12s}",
            })

            instances_dict[instance] = 1 if instance not in instances_dict else instances_dict[instance]+1
            
            wandb.log({"Avg Episode Return": moving_avg_return, 
                        "Avg Energy Return": moving_ene_return,
                        f"{instance}": env.last_criterion,
                        "avg_actor_loss": episode_log.avg_actor_loss,
                        "avg_critic_loss": episode_log.avg_critic_loss,
                        "avg_actor_grad_norm": episode_log.avg_actor_grad_norm,
                        "avg_critic_grad_norm": episode_log.avg_critic_grad_norm,
            })
            
            pbar.update(1)

now_str = datetime.datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S")
torch.save(agent.q_net.state_dict(), f"./model_params/{env.__class__.__name__}_{agent.__class__.__name__}_{now_str}.pt")