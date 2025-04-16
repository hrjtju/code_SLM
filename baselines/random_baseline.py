import os
import warnings
import sys

# append the parent directory to sys.path
sys.path.extend([os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')),
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))])

import torch
from tqdm import tqdm

from doubledqn_flat import DoubleDQN, PrioritizedReplayBuffer, to_device
from slm_model.slm_env import SingleSLMEnvParallel1D
from training_utils.utils import IntstanceAvgMeter


warnings.filterwarnings("ignore")


lr = 0.01
num_episodes = 100000
hidden_dim = 128
gamma = 1.00
epsilon = 2 # RANDOM ACT
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

test_meter = IntstanceAvgMeter(window_size=10)

env = SingleSLMEnvParallel1D(in_path="./instances_json/", phase="Train",
                                max_part_type=MAX_PART_TYPE, max_batch_num=MAX_BATCH_NUM, max_orientation_num=MAX_ORIENTATION_NUM)
env_name = env.name

for _ in range(10):
    print("Testing... Iteration: ", _, flush=True)
    # test model with greedy action-selection
    for instance_f in os.listdir("./instances_json/"):
        test_env = SingleSLMEnvParallel1D(in_path=f"./instances_json/{instance_f}", phase="Test",
                    max_part_type=MAX_PART_TYPE, max_batch_num=MAX_BATCH_NUM, max_orientation_num=MAX_ORIENTATION_NUM)
        
        state_, instance_ = test_env.curr_state, test_env.in_path
        done = False
        
        with torch.no_grad():
            while not done:
                action = agent.take_action(state_.to(device), test_env.mask_tensor.to(device))
                next_state, _, terminate, truncate, _ = test_env.step(action)
                done = terminate or truncate
                
                state = next_state
            
            test_meter.update(os.path.basename(instance_), test_env.solution.calculate_energy())

with open("./test_results/baseline_random.txt", "w") as f:
    for k, v in test_meter.dict_avg().items():
        f.write(f"{k}: {v}\n") 