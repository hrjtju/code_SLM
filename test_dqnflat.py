import os
from typing import Literal
import warnings
import sys

# append the parent directory to sys.path
sys.path.extend([os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')),
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))])

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
    mode: Literal["greedy", "epsilon", "sample"]
    sample: bool
    epsilon: float
    trial_num: int
    test_num: int
    early_stop: bool
    mask: bool
    lr: float
    penalty: float
    num_episodes: int
    hidden_dim: int
    gamma: float
    target_update: int
    buffer_size: int
    minimal_size: int
    batch_size: int
    max_part_type: int
    max_orientation_num: int
    max_batch_num: int
    clip_grad_norm: float
    train_dir: str
    eval_dir: str
    trial_name: str
    
    def __str__(args):
        return f"{os.path.basename(args.load_path)}_{args.mode}_{str(args.epsilon).replace('.', '-')}_{args.trial_num}_{args.test_num}_{'E' if args.early_stop else 'NE'}"

parser = argparse.ArgumentParser(description="Test DQN Flat")

parser.add_argument("--load_path", type=str, help="Path of the pretrained params")
parser.add_argument("--mode", type=str, choices=["greedy", "epsilon", "sample"], default="greedy", help="Action selection mode")
parser.add_argument("--sample", action='store_true', help="Whether to sample from the action space according to the output distribution")

parser.add_argument("--epsilon", type=float, default=0.1, help="Epsilon value for epsilon-greedy action selection")
parser.add_argument("--trial_num", type=int, default=3, help="Trial number for testing w/ epsilon-greedy action selection")
parser.add_argument("--test_num", type=int, default=10, help="Test number for getting mean value")
parser.add_argument("--early_stop", type=bool, default=False, help="Choose to early stop or not under epsilon-greedy action selection")

parser.add_argument("--mask", type=bool, default=False, help="Use mask for action selection")
parser.add_argument("--epsilon_final", type=float, default=0.05, help="Final epsilon (unused at test time)")
parser.add_argument("--epsilon_decay_steps", type=int, default=50000, help="Unused at test time")
parser.add_argument("--obs_norm", action="store_true", help="Normalize observations (auto-disabled for legacy ckpts)")
parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
parser.add_argument("--penalty", type=float, default=0.1, help="Penalty for invalid actions")
parser.add_argument("--num_episodes", type=int, default=100000, help="Number of episodes")
parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension")
parser.add_argument("--gamma", type=float, default=1.00, help="Discount factor")
parser.add_argument("--target_update", type=int, default=5, help="Target network update frequency")
parser.add_argument("--buffer_size", type=int, default=50000, help="Replay buffer size")
parser.add_argument("--minimal_size", type=int, default=600, help="Minimal size for sampling")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
parser.add_argument("--max_part_type", type=int, default=20, help="Max part type")
parser.add_argument("--max_orientation_num", type=int, default=7, help="Max orientation number")
parser.add_argument("--max_batch_num", type=int, default=20, help="Max batch number")
parser.add_argument("--clip_grad_norm", type=float, default=100.0, help="Gradient clipping norm")
parser.add_argument("--train_dir", type=str, default="./instances_json/", help="Directory for training instances")
parser.add_argument("--eval_dir", type=str, default="./instances_json/", help="Directory for testing instances")

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_PART_TYPE = 20
MAX_ORIENTATION_NUM = 7
MAX_BATCH_NUM = 20

replay_buffer = PrioritizedReplayBuffer(buffer_size)

agent = DoubleDQN(device, args)

# T1/T16: tolerant loader -- handles both the new checkpoint dict and a bare state_dict
agent.load_checkpoint(args.load_path)
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

for _ in range(args.test_num if (args.mode != "greedy") else 1):
    print("Testing... Iteration: ", _, flush=True)
    # test model with greedy action-selection
    for instance_f in tqdm(os.listdir("./instances_json/")):
        
        results = []
        
        for _ in range(args.trial_num if (args.mode != "greedy") else 1):
        
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
                    
                    state_ = next_state  # T6: was `state = next_state` (agent never saw progress)
                    
                    # early stop if the energy is already HIGHER than the best solution
                    if len(results) > 0 and args.early_stop and results[-1] < test_env.solution.calculate_energy():
                        early_stop_flag = True
                        break
            
            if early_stop_flag:
                continue
            
            
            test_env.solution.show(out_dir=f"./solution/dqnflat_{Arguments.__str__(args)}/{instance_f.split('.')[0]}")
            
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