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

# from tensorboardX import SummaryWriter

warnings.filterwarnings("ignore")

lr = 5e-5
num_episodes = 10000
hidden_dim = 128
gamma = 1.00
epsilon = 0.05 # 0.05
target_update = 5
buffer_size = 20000
minimal_size = 600
batch_size = 16
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
epochs=10

MAX_PART_TYPE = 20
MAX_ORIENTATION_NUM = 7
MAX_BATCH_NUM = 20

replay_buffer = ReplayBuffer(buffer_size)

env = SingleSLMEnvParallel1D(in_path="./instances_json/", phase="Train",
                             max_part_type=MAX_PART_TYPE, max_batch_num=MAX_BATCH_NUM, max_orientation_num=MAX_ORIENTATION_NUM, 
                             seed=25, 
                             ppo=True)
env_name = env.name

# agent = DoubleDQN(lr, gamma, epsilon, target_update, device, 
#                   max_part=MAX_PART_TYPE, max_ori=MAX_ORIENTATION_NUM, max_batch=MAX_BATCH_NUM)
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
            
            transition = Transition()
            
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
            agent.update(transition)

            episode_id = int(num_episodes / 10 * i + i_episode + 1)
            moving_avg_return = np.mean(return_list[-200:] if len(return_list) > 200 else np.mean(return_list))
            moving_ene_return = np.mean(energy_list[-200:] if len(energy_list) > 200 else np.mean(energy_list))
            
            pbar.set_postfix({
                "episode": f"{episode_id:4d}",
                "return": f"{f'{moving_avg_return:.6e}':12s}",
                "energy": f"{f'{moving_ene_return:.6e}':12s}"
            })

            instances_dict[instance] = 1 if instance not in instances_dict else instances_dict[instance]+1
            
            # writer.add_scalar("Avg 
            # Episode Return", moving_avg_return, episode_id)
            # writer.add_scalar("Avg Energy Return", moving_ene_return, episode_id)
            # writer.add_scalar(f"{instance}", scalar_value=episode_return, global_step=instances_dict.get(instance))
            
            pbar.update(1)

torch.save(agent.q_net.state_dict(), f"./model_params/{agent.__class__.__name__}_{now_str}.pt")