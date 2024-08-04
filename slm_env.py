from typing import Tuple
import torch
from torch import Tensor as Tensor
import joyrl

from slm_classes import load_json_to_class
from bin_packing import allocate_bin_packing_2d

# only supports assigning a part to a batch
# and the 2D bin packing algorithm puts the part into a target position
class SLMEnv:
    """
    Environment of SLM Machine Optimization
    """
    
    # load instance-meta data
    def __init__(self, random=True, in_path=None, phase="Train", grid_length=1, num_batches=10, **kwargs):
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
        self.curr_state = torch.zeros(size=(
            num_batches, 
            metadata.machine.build_l // grid_length,
            metadata.machine.build_w // grid_length,
            metadata.machine.build_h // grid_length,
        ))
    
    # calculate power cost
    def calculate_power(self) -> float:
        ...
    
    # calculate time cost
    def calculate_time(self) -> float:
        ...
    
    # verify physical constraints
    def check_constraints(self) -> float:
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

