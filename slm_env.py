from collections.abc import ValuesView
from functools import reduce
from operator import add
import os
import random
from typing import Any, Dict, Tuple, List, Literal
from numpy import negative
import torch
from torch import Tensor as Tensor
import joyrl
from rectpack import newPacker, PackingMode
from math import ceil, floor

from time_energy_model import calculate_batch_energy, calculate_batch_time
from slm_classes import Machine, Part, Process, load_json_to_class
from bin_packing import allocate_bin_packing_2d

class Batch:
    """
    A Batch contains:
        - A TRUE INSTANCE of 2d-bin (for bin-packing algorithm)
        - A discretized VIEW of the 2d-bin (for neural network)
        - A container containing information of parts allocated to this batch,
            - each item of which contains the original dict of the category(kind) 
              the part belongs to with `build_param` equaling the value 
              corresponding the orientation chosen
    """
    def __init__(self, 
                 machine: Machine,
                 process: Process,
                 view_shape: tuple
                 ) -> None:
        self.L = machine.build_l
        self.W = machine.build_w
        self.H = machine.build_h
        self.machine = machine
        self.process = process
        
        self.view_shape = view_shape
        
        # Trial, Using RectPack Algorithm
        self.bin_true = newPacker(mode=PackingMode.Online,
                                       rotation=False
                                       ) # depends on the packing algorithm
        self.bin_true.add_bin(width=self.L, height=self.W)
        self.bin_view: Tensor = None # should finally be a fixed size tensor
        
        # {
        #     "volume": self.volume,
        #     "surface_ares": self.surface_area,
        #     "L": build_param["L"]+gap,
        #     "W": build_param["W"]+gap,
        #     "H": build_param["H"],
        #     "S": build_param["S"],
        # } 
        self.parts_info: List[dict] = []
    
    def get_current_view(self) -> Tensor:
        """
        Returns the current view of the batch as one of the neural network inputs
        """
        grid = torch.zeros(size=tuple(map(ceil, 
                                          (self.L, self.W))))
        for (c, x, y, w, h, _) in self.bin_true.rect_list():
            grid[floor(x):ceil(x+w), floor(y):ceil(y+h)] = 1
            grid[floor(x+1):ceil(x+w-1), floor(y+1):floor(y+h-1)] = -1
        
        grid = torch.nn.functional.interpolate(grid, size=self.view_shape, mode="bilinear")
        
        return grid
    
    def get_rest_area(self) -> float:
        """
        Returns rest area of a batch
        """
        total_area = self.L * self.W
        occupied_area = sum(map(lambda x:x["L"]*x["W"], self.parts_info))
        return total_area - occupied_area
    
    def get_total_surface_area(self) -> float:
        ...
    
    def get_total_part_volume(self) -> float:
        ...
    
    def get_total_support_volume(self) -> float:
        ...
    
    def add_part(self, part: Part, orientation: int) -> Tuple[Tensor, bool]:
        """
        Add part to this batch.
        
        Returns
            - `True`  , if part is successfully added
            - `False` , otherwise
        """
        
        # Checks if the projection area of the part is smaller than the 
        # area available in this batch.
        if part.get_proj_area(orientation) < self.get_rest_area():
            return False
        
        # Try to add the part into the batch using the bin-packing algorithm
        # If it cannot be packed, return False.
        self.bin_true, success = allocate_bin_packing_2d(self.bin_true, part.get_part_info(orientation))
        
        if success:
            # update list parts_info 
            self.parts_info.append(part.get_part_info(orientation))
            # Update the current view
            self.bin_view = self.get_current_view()
            
            return self.bin_view, True
        else:
            # don't update anything
            return self.bin_view, False
      
    def empty(self):
        return len(self.parts_info) < 1


class Solution:
    """
    A Solution to a instance contains multiple batches.
    """
    def __init__(self, view_shape: Tuple[int, int]) -> None:
        self.view_shape = view_shape
        self.batches: List[Batch] = []
    
    def add_batch(self, machine: Machine, process: Process) -> None:
        """
        Add an empty batch to the solution
        """
        self.batches.append(Batch(machine, process, self.view_shape))
    
    def get_batch(self) -> Batch:
        """
        Get the current batch
        """
        assert len(self.batches) > 0, "There is no batches in this solution!"
        return self.batches[-1]
    
    def get_current_view(self) -> Tensor:
        return self.get_batch().get_current_view()
    
    def add_part(self, part: Part, orientation: int) -> Tuple[Tensor, bool]:
        """
        Try to add the part to the current batch
        """
        return self.get_batch().add_part(part, orientation)
    
    # calculate time cost
    def calculate_time(self) -> float:
        """
        Calculate the time needed for the solution UNTIL NOW
        """
        
        # Equivalent to:
        # sum_time = 0
        # for b in self.batches:
        #     sum_time += calculate_batch_time(b)
        # return sum 
        
        return reduce(add, map(calculate_batch_time, self.batches))
    
    # calculate energy cost
    def calculate_energy(self) -> float:
        """
        Calculate the power needed for the solution UNTIL NOW
        """
        return reduce(add, map(calculate_batch_energy, self.batches))


# only supports assigning a part to a batch
# and the 2D bin packing algorithm puts the part into a target position
class SLMEnv:
    """
    Environment of SLM Machine Optimization
    """
    
    # load instance-meta data
    def __init__(self, 
                 in_path: str = None, # Dir path in training and .json file path in testing stage
                 phase: Literal["Train", "Test"] = "Train", 
                 view_shape: Tuple[int, int] = (224, 224), 
                 seed: float = 0,
                 **kwargs
                 ):
        random.seed(seed)
        
        self.phase = phase
        self.in_path = in_path
        self.view_shape = view_shape
        
        # if random == True, ignore the in_path
        if self.phase == "Train":
            # randomly pick a json file in the training dir
            # and pack the training data into a class
            load_path = random.choice(os.listdir(self.in_path))
            self.metadata = load_json_to_class(os.path.join(self.in_path, load_path))
        elif self.phase == "Test":
            # load the specified in_path
            self.metadata = load_json_to_class(self.in_path)
        else:
            raise NotImplementedError
        
        # -----------------------------------------------------------
        # transform the loaded data into states
        
        # set dict of available parts
        # length of this dict is the same size of output allocation vector.
        self.part_unavailable_mask = None
        
        self.last_state = None
        
        
        self.L = self.metadata.machine.build_l
        self.W = self.metadata.machine.build_w
        self.H = self.metadata.machine.build_h
        
        self.solution = Solution(view_shape=self.view_shape)
        self.solution.add_batch(self.metadata.machine,
                                self.metadata.process)
        
        # TODO: Create object indicating the parts packed in every batch, 
        # TODO: i.e. (number&types of parts, positions and orientations)
        # observation of the current batch
        # TODO: Redefine State
        self.curr_state = (
            self.solution.get_current_view(),        # Current discretized view of the batch, Variable
            torch.tensor([self.L, self.W, self.H]),  # Real size of the batch, Constant
            self.metadata.init_state()               # Situation of all parts, Variable
        )
    
    def get_unavailable_mask(self):
        return (self.state[-1][:, 0] > 0).reshape(-1)
    
    # verify physical constraints
    def check_geo_constraints(self) -> float:
        """
        Used only if DRL model is allowed to determine 
        the position and orientation of a part in a batch.
        
        Checks if the allocation meets the geometry constraints.
        """
        ...
    
    def reset(self, seed = 0):
        if self.phase == "Train":
            self.__init__(in_path=self.in_path, 
                          phase=self.phase, 
                          view_shape=self.view_shape, 
                          seed=seed
                          )
        else:
            exit(0)
        
        info = "Reset Env"
        
        return self.curr_state, info
    
    def step(
        self, 
        action: Tensor
        ) -> Tuple[Tensor, float, bool, bool, str]:
        
        # TODO: Add printing orientation. 
        
        # action consists of two vectors
        # i.e. two distributions on part_types and batches accordingly
        
        terminated = False
        truncated = False
        info = None
        reward = 0
        
        
        allocated = False
        
        # TODO: Remember to add a softmax layer to the policy network
        parts_rank = torch.argsort((action.reshape(-1) * self.get_unavailable_mask()), 
                                   descending=True)
        # [0, 2, 0, 3, 0, 5]
        # [5, 3, 1, 0, 2, 4]
        
        
        negative_reward = 1
        
        # Select the highest id
        part_id = parts_rank[0]
        
        view, allocated = self.solution.add_part(self.metadata.parts[part_id].get_part_info())
        
        reward = 0
        
        # If allocating failure, add a new batch and reallocate
        if allocated:
            self.last_state = self.curr_state
            
            temp_part_state = self.last_state[-1]
            temp_part_state[part_id, 0] -= 1
            
            self.curr_state = (
                view,
                self.last_state[1],
                temp_part_state
            )    
            
            # If all parts are allocated, terminate this episode.
            if all(v==0 for v in self.available_parts.values()):
                terminated = True
        else:
            # Add a new batch (same size)
            self.solution.add_batch(self.metadata.machine, self.metadata.process)
            view, allocated = self.solution.add_part(self.metadata.parts[part_id].get_part_info())
            
            if allocated:
                self.last_state = self.curr_state
                
                temp_part_state = self.last_state[-1]
                temp_part_state[part_id, 0] -= 1
                
                self.curr_state = (
                    view,
                    self.last_state[1],
                    temp_part_state
                )    
                
                # If all parts are allocated, terminate this episode.
                if all(v==0 for v in self.available_parts.values()):
                    terminated = True
            else:
                truncated = True
            
        return self.curr_state, reward, terminated, truncated, info

