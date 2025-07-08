import os
import tqdm
import torch
import random
import matplotlib.pyplot as plt


from slm_classes import MetaData, load_json_to_class
from batch_solution import Batch, Solution
from slm_env import SingleSLMEnv

torch.set_printoptions(precision=1, sci_mode=False)

# file = random.choice(list(filter(lambda x: "-1." not in x, os.listdir("./instances_json"))))
# file_dir = os.path.join("./instances_json", file)

# print(f"Testing file name: \t{file_dir}")

# metadata = load_json_to_class(file_dir)

# part_num = len(metadata.parts)
# ori_num = len(min(metadata.parts, key=lambda x: len(x.build_params)).build_params)

# print(f"{part_num = }, {ori_num = }")

# -------------------slm_classes.py-------------------
# print("\n-------------------slm_classes.py-------------------\n")
# Testing ALL params
# metadata.show()

# Init state
# print(metadata.init_state())

# Mask Matrix
# print(metadata.mask_matrix())


# -------------------batch_solution.py-------------------
# print("\n-------------------batch_solution.py-------------------\n")
# solution_test = Solution(view_shape=(224, 224))
# solution_test.add_batch(metadata.machine, metadata.process)

# # Add type 0

# print(f"\n Add type 0 orientation 0 \n")
# print(f"{solution_test.get_batch().empty() = }")
# print(f"{metadata.parts[0].get_part_info(0) = }")
# print(f"{solution_test.get_batch().slice_number}")

# # The get_part_info() method already takes in orientation parameter, 
# # add_part() can be simplified to taking only one param
# # by letting get_part_info() to output a dict that contains key "O".
# _, success = solution_test.add_part(metadata.parts[0], 0)

# print(f"{success = }")
# print(f"{solution_test.get_batch().empty() = }")
# print(f"{solution_test.get_batch().get_total_part_volume() = }")
# print(f"{solution_test.get_batch().get_total_surface_area() = }")
# print(f"{solution_test.get_batch().get_total_support_volume() = }")
# print(f"{solution_test.get_batch().get_rest_area() = }")

# view = solution_test.get_current_view(stretch=False, show=True)
# plt.imshow(view)

# # Add type 0 orientation 1

# print(f"\n Add type 0 orientation 1 \n")
# print(f"{solution_test.get_batch().empty() = }")
# print(f"{metadata.parts[0].build_params = }")
# print(f"{metadata.parts[0].get_part_info(1) = }")
# print(f"{solution_test.get_batch().slice_number}")

# solution_test.add_part(metadata.parts[0], 1)

# print(f"{solution_test.get_batch().empty() = }")
# print(f"{solution_test.get_batch().get_total_part_volume() = }")
# print(f"{solution_test.get_batch().get_total_surface_area() = }")
# print(f"{solution_test.get_batch().get_total_support_volume() = }")
# print(f"{solution_test.get_batch().get_rest_area() = }")

# view = solution_test.get_current_view(stretch=False, show=True)
# plt.imshow(view)

# print(f"{solution_test.get_batch().slice_number}")


# -------------------batch_solution.py-------------------
# # TODO: Check accuracy of functions 
# print("\n-------------------batch_solution.py-------------------\n")
# solution_test = Solution(view_shape=(224, 224))

# # Add type 0

# for _ in range(10):
#     solution_test.add_batch(metadata.machine, metadata.process)
#     for _ in range(20):
#         type_id, ori_id = random.randint(0, part_num-1), random.randint(0, ori_num-1)
#         _, success = solution_test.add_part(metadata.parts[type_id], ori_id)

# solution_test.show(out_dir="./solution_test")

# ------------------- slm_env.py -------------------

# TODO: Random Benchmark
# TODO: Rank by height Benchmark

env = SingleSLMEnv(in_path="./instances_json", device="cpu")
done = False
episode = 0

# Fetching Outputs
for _ in tqdm.tqdm(range(1000)):
    if done == True:
        env.solution.show(out_dir=f"./solution_test/test_episode_{episode}_{env.load_path.split('.')[0]}")
        episode += 1
        env.reset()
    
    *_, terminated, truncated, _ = env.step(
        action=[torch.rand((20)), torch.rand((7))]
        )
    done = terminated or truncated
    

# For debug mode only
# env = SingleSLMEnv(in_path="./instances_json", device="cpu")
# done = False

# # For debug mode only
# while not done:
    
#     *_, terminated, truncated, _ = env.step(
#         action=[torch.rand((20)), torch.rand((7))]
#         )
#     done = terminated or truncated
    
