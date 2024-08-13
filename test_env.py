import os
import tqdm
import torch
import random
import matplotlib.pyplot as plt


from slm_classes import MetaData, load_json_to_class
from batch_solution import Batch, Solution
from slm_env import SLMEnv

torch.set_printoptions(precision=1, sci_mode=False)

file = random.choice(os.listdir("./instances_json"))
file_dir = os.path.join("./instances_json", file)

print(f"Testing file name: \t{file_dir}")

metadata = load_json_to_class(r"./instances_json/ec_30-1.json")

# -------------------slm_classes.py-------------------
print("\n-------------------slm_classes.py-------------------\n")
# Testing ALL params
metadata.show()

# Init state
print(metadata.init_state())

# Mask Matrix
print(metadata.mask_matrix())


# -------------------batch_solution.py-------------------
print("\n-------------------batch_solution.py-------------------\n")
solution_test = Solution(view_shape=(224, 224))

# Add type 0

print(f"\n Add type 0 orientation 0 \n")
print(f"{solution_test.get_batch().empty() = }")
print(f"{metadata.parts[0].get_part_info() = }")
print(f"{solution_test.get_batch().slice_number}")

solution_test.add_part(metadata.parts[0].get_part_info(), 0)

print(f"{solution_test.get_batch().empty() = }")
print(f"{solution_test.get_batch().get_total_part_volume() = }")
print(f"{solution_test.get_batch().get_total_surface_area() = }")
print(f"{solution_test.get_batch().get_total_support_volume() = }")
print(f"{solution_test.get_batch().get_rest_area() = }")

view = solution_test.get_current_view(stretch=False, show=True)
plt.imshow(view)

# Add type 0 orientation 1

print(f"\n Add type 0 orientation 1 \n")
print(f"{solution_test.get_batch().empty() = }")
print(f"{metadata.parts[0].get_part_info() = }")
print(f"{solution_test.get_batch().slice_number}")

solution_test.add_part(metadata.parts[0].get_part_info(), 1)

print(f"{solution_test.get_batch().empty() = }")
print(f"{solution_test.get_batch().get_total_part_volume() = }")
print(f"{solution_test.get_batch().get_total_surface_area() = }")
print(f"{solution_test.get_batch().get_total_support_volume() = }")
print(f"{solution_test.get_batch().get_rest_area() = }")

view = solution_test.get_current_view(stretch=False, show=True)
plt.imshow(view)

print(f"{solution_test.get_batch().slice_number}")
