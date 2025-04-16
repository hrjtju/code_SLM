import os
from typing import Literal
import warnings
import sys

# append the parent directory to sys.path
sys.path.extend([os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')),
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))])

from regex import T
import torch
from tqdm import tqdm

from doubledqn_flat import DoubleDQN, PrioritizedReplayBuffer
from slm_model.slm_env import SingleSLMEnvParallel1D
from training_utils.utils import IntstanceAvgMeter


warnings.filterwarnings("ignore")

# parse command line arguments
import argparse

class Arguments:
    load_path: str
    mode: Literal["greedy", "epsilon"]
    epsilon: float
    trial_num: int
    test_num: int
    early_stop: bool
    
    def __str__(args):
        return f"{os.path.basename(args.load_path)}_{args.mode}_{str(args.epsilon).replace('.', '-')}_{args.trial_num}_{args.test_num}_{'E' if args.early_stop else 'NE'}"

parser = argparse.ArgumentParser(description="Test DQN Flat")

parser.add_argument("--load_path", type=str, help="Path of the pretrained params")
parser.add_argument("--mode", type=str, choices=["greedy", "epsilon"], default="greedy", help="Action selection mode")
parser.add_argument("--epsilon", type=float, default=0.1, help="Epsilon value for epsilon-greedy action selection")
parser.add_argument("--trial_num", type=int, default=3, help="Trial number for testing w/ epsilon-greedy action selection")
parser.add_argument("--test_num", type=int, default=10, help="Test number for getting mean value")
parser.add_argument("--early_stop", type=bool, default=False, help="Choose to early stop or not under epsilon-greedy action selection")

args: Arguments = parser.parse_args()

print(f"Arguments: {args}")

lr = 0.01
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

agent.q_net.load_state_dict(torch.load(args.load_path))
agent.q_net.to(device)
torch.compile(agent.q_net)
agent.q_net.eval()

test_meter = IntstanceAvgMeter(window_size=10)

env = SingleSLMEnvParallel1D(in_path="./instances_json/", phase="Train",
                                max_part_type=MAX_PART_TYPE, max_batch_num=MAX_BATCH_NUM, max_orientation_num=MAX_ORIENTATION_NUM)
env_name = env.name

# start timing
import time
start_time = time.time()

for _ in range(args.test_num):
    print("Testing... Iteration: ", _, flush=True)
    # test model with greedy action-selection
    for instance_f in tqdm(os.listdir("./instances_json/")):
        
        results = []
        
        for _ in (range(args.trial_num) if args.mode=="epsilon" else [0]):
        
            test_env = SingleSLMEnvParallel1D(in_path=f"./instances_json/{instance_f}", phase="Test",
                        max_part_type=MAX_PART_TYPE, max_batch_num=MAX_BATCH_NUM, max_orientation_num=MAX_ORIENTATION_NUM)
            
            state_, instance_ = test_env.curr_state, test_env.in_path
            done = False
            early_stop_flag = False
            
            with torch.no_grad():
                while not done:
                    action = agent.take_action(state_.to(device), test_env.mask_tensor.to(device), test=args.mode=="greedy")
                    next_state, _, terminate, truncate, _ = test_env.step(action)
                    done = terminate or truncate
                    
                    state = next_state
                    
                    # early stop if the energy is already HIGHER than the best solution
                    if len(results) > 0 and args.early_stop and results[-1] < test_env.solution.calculate_energy():
                        early_stop_flag = True
                        break
            
            if early_stop_flag:
                continue
            
            results.append(test_env.solution.calculate_energy())
            
        test_meter.update(os.path.basename(instance_), min(results))

# stop timing
end_time = time.time()
elapsed_time = end_time - start_time

with open(result_str:=f"./test_results/dqnflat_{Arguments.__str__(args)}.txt", "w") as f:
    for k, v in test_meter.dict_avg().items():
        f.write(f"{k}: {v}\n") 
        
print(f"Results saved to {result_str}")
print(f"Elapsed time: {elapsed_time:.4f} seconds")
print(f"Average time per instance: {elapsed_time/args.test_num:.4f} seconds")