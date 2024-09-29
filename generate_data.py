import os
import numpy as np
import random
import json
from copy import deepcopy
from tqdm import tqdm

with open(r".\instances_json\ece_100-7.json", 'r') as f:
        json_template_d = json.load(f)

target_dir = "./instances_generated_json"
if not os.path.exists(target_dir):
    os.mkdir(target_dir)

generate_num = 1000

min_type_num, max_type_num = 15, 20
min_parts_num, max_parts_num = 80, 100
min_parts_ori, max_parts_ori = 3, 7

for _ in tqdm(range(generate_num)):
    parts_num = random.randint(min_parts_num, max_parts_num)
    parts_ori = random.randint(min_parts_ori, max_parts_ori)
    types_num = random.randint(min_type_num, max_type_num)
    
    count = 0
    fname = f"g_ece_{parts_num}_{types_num}_{parts_ori}_{count}"
    while os.path.exists(f'{target_dir}/{fname}.json'):
        count += 1
        fname = f"g_ece_{parts_num}_{types_num}_{parts_ori}_{count}"
    
    num_dist_list = [0 for _ in range(types_num)]
    for _ in range(parts_num):
        num_dist_list[random.randint(0, types_num-1)] += 1
    
    new_part_info = []
    for part_id, (item, new_num) in enumerate(zip(random.choices(json_template_d["part_info"], k=types_num), 
                             num_dist_list)):
        new_part_info.append(deepcopy(item))
        
        new_part_info[-1]["num_part"] = new_num
        new_part_info[-1]["part_type"] = part_id
        new_part_info[-1]["build_params"] = random.choices(new_part_info[-1]["build_params"], 
                                                           k=parts_ori)
        for i, item in enumerate(new_part_info[-1]["build_params"]):
            item["O"] = i
    
    
    new_json_dict = deepcopy(json_template_d)
    new_json_dict["part_info"] = new_part_info
    new_json_dict["file_name"] = fname
    new_json_dict["instance_type"]["num_parts"] = parts_num
    new_json_dict["instance_type"]["type_parts"] = types_num
    new_json_dict["instance_type"]["num_orientations"] = parts_ori
    
    with open(f'{target_dir}/{fname}.json', 'w') as fp:
        json.dump(new_json_dict, fp)
        