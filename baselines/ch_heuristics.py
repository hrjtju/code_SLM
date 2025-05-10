from functools import reduce
import os
from sre_constants import SUCCESS
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

import argparse
parser = argparse.ArgumentParser(description="Test DQN Flat")

parser.add_argument("--load_path", type=str, help="Path of the pretrained params")
parser.add_argument("--mode", type=str, choices=["MH", "MS", "MA"], default="MH", help="Action selection mode")

args = parser.parse_args()

match args.mode:
    case "MH":
        key_ = "H"
    case "MS":
        key_ = "S"
    case "MA":
        key_ = "A"
    case _:
        raise ValueError("Invalid heuristic choice.", args.mode)


test_meter = IntstanceAvgMeter(window_size=10)


# test model with greedy action-selection
for instance_f in os.listdir(args.load_path):
    test_env = SingleSLMEnvParallel1D(in_path=os.path.join(args.load_path, instance_f), phase="Test",
                max_part_type=MAX_PART_TYPE, max_batch_num=MAX_BATCH_NUM, max_orientation_num=MAX_ORIENTATION_NUM)
    
    parts_ls = test_env.slm_metadata.parts
    
    # choose the orientation of each kind of parts to minimize... 
    # height OR suppret volume OR projection area
    best_choice = {int(p.part_type): p.get_min_ori(key_) for p in parts_ls}
    all_parts_ls_sorted = sorted(reduce(lambda x, y: x + y, [[(p, int(p.part_type), best_choice[int(p.part_type)]) for _ in range(int(p.num_part))] for p in parts_ls]), 
                            key=lambda x: x[0].build_params[x[2]]["H"], reverse=True)
    
    current_batch = 0
    part_to_assign = 0
    
    state_, instance_ = test_env.curr_state, test_env.in_path
    done = False
    
    with torch.no_grad():
        while part_to_assign < len(all_parts_ls_sorted):
            
            part_type = all_parts_ls_sorted[part_to_assign][1] - 1
            part_ori = all_parts_ls_sorted[part_to_assign][2]
            
            _, success, _ = test_env.solution.add_part(part=test_env.slm_metadata.parts[part_type],
                                                            orientation=part_ori,
                                                            idx=current_batch)
            
            if not success:
                current_batch += 1
                test_env.solution.add_part(part=test_env.slm_metadata.parts[part_type],
                                                            orientation=part_ori,
                                                            idx=current_batch)

            part_to_assign += 1
            
        test_env.solution.show(out_dir=f"./solution/baseline_{args.mode}/{instance_f.split('.')[0]}")
        test_meter.update(os.path.basename(instance_), test_env.solution.calculate_energy())

with open(f"./test_results/baseline_{args.mode}.txt", "w+") as f:
    for k, v in test_meter.dict_avg().items():
        f.write(f"{k}: {v}\n") 