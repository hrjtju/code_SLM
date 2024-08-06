from typing import Any, Dict, Tuple, List, Literal
import torch
from torch import Tensor as Tensor
import joyrl
from rectpack import newPacker, PackingMode
from math import ceil, floor

from slm_classes import Part, load_json_to_class
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
                 L: float, 
                 W: float, 
                 H: float, 
                 view_shape: tuple
                 ) -> None:
        self.L = L
        self.W = W
        self.H = H
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
    
    def add_part(self, part: Part, orientation: int) -> bool:
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
            # Update the current view
            self.bin_view = self.get_current_view()
            # update list parts_info 
            self.parts_info.append(part.get_part_info(orientation))
            return True
        else:
            # don't update anything
            return False

class Solution:
    """
    A Solution to a instance contains multiple batches.
    """
    def __init__(self, grid_length) -> None:
        self.grid_length = grid_length
        self.batches: List[Batch] = []
    
    def add_batch(self, L, W, H) -> None:
        """
        Add an empty batch to the solution
        """
        self.batches.append(Batch(L, W, H, self.grid_length))
    
    def get_batch(self) -> Batch:
        """
        Get the current batch
        """
        assert len(self.batches) > 0, "There is no batches in this solution!"
        return self.batches[-1]
    
    def add_part(self, part: Part, orientation: int) -> None:
        """
        Try to add the part to the current batch
        """
        success = self.get_batch().add_part(part, orientation)
        
        # TODO: Check the seq logic
        if not success:
            self.add_batch()
            s = self.get_batch().add_part(part, orientation)
            
            if not s:
                return False
        else:
            return True

# only supports assigning a part to a batch
# and the 2D bin packing algorithm puts the part into a target position
class SLMEnv:
    """
    Environment of SLM Machine Optimization
    """
    
    # load instance-meta data
    def __init__(self, 
                 random: bool = True, 
                 in_path: str = None, 
                 phase: Literal["Train", "Test"] = "Train", 
                 grid_length: float =1, 
                 **kwargs
                 ):
        self.phase = phase
        
        # if random == True, ignore the in_path
        if random == True:
            # randomly pick a json file in the training dir
            # and pack the training data into a class
            load_path = ...
            metadata = load_json_to_class(load_path)
        else:
            # load the specified in_path
            assert (in_path is not None), "in_path should not be None if random is set to False"
            metadata = load_json_to_class(in_path)
        
        # -----------------------------------------------------------
        # transform the loaded data into states
        
        # set dict of available parts
        # length of this dict is the same size of output allocation vector.
        self.available_parts = {part["part_type"]:part["num_part"] for part in metadata.parts}
        self.part_unavailable_mask = torch.tensor([(1 if value > 0 else 0) \
            for _,value in sorted(self.available_parts.values(), key=lambda x:x[0])]).reshape(1, -1)
        
        self.last_state = None
        
        # TODO: Create object indicating the parts packed in every batch, 
        # TODO: i.e. (number&types of parts, positions and orientations)
        
        # observation of the current batch
        # TODO: Redefine State
    
    # calculate power cost
    def calculate_power(self) -> float:
        """
        Calculate the power needed for the solution UNTIL NOW
        """
        ...
    
    # calculate time cost
    def calculate_time(self) -> float:
        """
        Calculate the time needed for the solution UNTIL NOW
        """
        ...
    
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
            ...
        else:
            raise NotImplementedError
        
        self.last_state = None
        self.curr_state = ...
        info = None
        
        return self.curr_state, info
    
    def step(
        self, 
        action: Tuple[Tensor, Tensor]
        ) -> Tuple[Tensor, float, bool, bool, str]:
        
        # TODO: Add printing orientation. 
        
        # action consists of two vectors
        # i.e. two distributions on part_types and batches accordingly
        part_distribution, batch_distribution = action
        
        terminated = False
        truncated = False
        info = None
        reward = 0
        
        # Try to allocate the part to the batch with the highest output score.
        # if the batch_no allocated cannot hold the part, switch to batch_no 
        # that has the 2-nd highest output score and apply a negative reward,
        # and so-on so forth.
        # If the part cannot be assigned to any of the batches, select another 
        # part with a 2-nd highest score and repeat the aforementioned steps.
        # if the remaining all parts cannot be assigned to any batches, apply a 
        # large negative reward and terminate this episode.
        
        allocation_status = False
        
        parts_rank, batch_rank = list(map(lambda x:torch.argsort(
                                                                 x, 
                                                                 descending=True
                                                                 ).reshape(-1), 
                                          action))
        part_id, batch_id = parts_rank[0], batch_rank[0]
        
        # Empty maximal space criterion is no longer needed if bin-packing algorithm
        # is to be used.
        
        self.last_state = self.curr_state
        
        self.curr_state = allocate_bin_packing_2d(self.curr_state, part_id, batch_id)
        
        # If all parts are allocated, terminate this episode.
        if all(v==0 for v in self.available_parts.values()):
            terminated = True
        
        return self.curr_state, reward, terminated, truncated, info

