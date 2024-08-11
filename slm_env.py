import os
import random
from typing import Any, Dict, Tuple, List, Literal
import torch
from torch import Tensor as Tensor
import joyrl

from slm_classes import load_json_to_class
from batch_solution import Batch, Solution

#! Add Unit Test


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
                 max_part_type: int = 20,
                 max_orientation_num: int = 7,
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
        
        self.metadata.max_part_type = max_part_type
        self.metadata.max_orientation_num = max_orientation_num
        
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
        
        # observation of the current batch
        self.curr_state = (
            self.solution.get_current_view(),        # Current discretized view of the batch, Variable
            torch.tensor([self.L, self.W, self.H]),  # Real size of the batch, Constant
            self.metadata.init_state()               # Situation of all parts, Variable
        )
        self.last_criterion = 0
    
    def get_unavailable_mask(self):
        """
        get unavailable mask of the part types. The indices of the output vector is 1 iff
            - The remaining number of parts is greater than 0
            - There is at least 1 legal orientation
        """
        return ((self.state[-1][:, 0] > 0) * (torch.sum(self.metadata.mask_matrix(), dim=-1) > 0)).reshape(-1)
    
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
    
    def update_state(self, view: Tensor, part_id: int) -> None:
        self.last_state = self.curr_state
            
        temp_part_state = self.last_state[-1]
        temp_part_state[part_id, 0] -= 1
        
        self.curr_state = (
            view,
            self.last_state[1],
            temp_part_state
        )
    
    def done(self) -> None:
        """
        Check if all the parts are allocated
        """
        # If any of the kind of part have unallocated instances, return False
        return not any(self.curr_state[-1][:, 0])
    
    def step(
        self, 
        action: Tuple[Tensor, Tensor]
        ) -> Tuple[Tensor, float, bool, bool, str]:
        
        terminated = False
        truncated = False
        info = None
        reward = 0
        
        out_matrix = action.reshape(self.metadata.max_part_type, self.metadata.max_orientation_num)
        # Perform softmax to ensure all the instances are strictly greater than 0
        feasible_matrix = torch.softmax(out_matrix, dim=None) * self.metadata.mask_matrix()
        
        allocated = False
        
        # Select part_id and orientation_id
        part_id = torch.argmax(torch.sum(feasible_matrix, dim=-1))
        orientation_rank = torch.argsort(feasible_matrix[part_id].reshape(-1))
        rank_ptr = 0
        orientation_id = orientation_rank[rank_ptr]
        
        # Try allocating the part according to the orientation selected.
        view, allocated = self.solution.add_part(self.metadata.parts[part_id].get_part_info(), orientation=orientation_id)
                
        # If allocation fails, try other orientations
        # If the part still cannot be allocated, then add a new bin and reallocate
        # Terminate the env if all parts are allocated
        # Truncate the env if part cannot be allocated by adding a new bin.
        if allocated:
            self.update_state(view=view, part_id=part_id)   
            
            # If all parts are allocated, terminate this episode.
            # 
            if self.done():
                terminated = True
        else:
            # Try other printing orientations, If all orientations does not fit, truncate the env.
            while not allocated:
                rank_ptr += 1
                orientation_id = orientation_rank[rank_ptr]
                
                # If the orientation is not feasible, then break the loop
                if feasible_matrix[part_id, orientation_id] == 0:
                    break
                    
                view, allocated = self.solution.add_part(self.metadata.parts[part_id].get_part_info(), orientation=orientation_id)
            
            if allocated:
                self.update_state(view=view, part_id=part_id)
                
                # If all parts are allocated, terminate this episode.
                if self.done():
                    terminated = True
            else:
                # reset orientation
                rank_ptr = 0
                # Add a new batch (same size) and retry allocating
                self.solution.add_batch(self.metadata.machine, self.metadata.process)
                view, allocated = self.solution.add_part(self.metadata.parts[part_id].get_part_info(),
                                                         orientation=orientation_rank[rank_ptr])
                
                if allocated:
                    self.update_state(view=view, part_id=part_id)   
                    
                    # If all parts are allocated, terminate this episode.
                    if self.done():
                        terminated = True
                        
                else:
                    truncated = True
        
        # Reward assignment: Average power, Total energy cost, Total time consumption
        # Weighted Sum ? >>> Difference as reward <<<
        criterion = self.solution.calculate_energy()
        reward = self.last_criterion - criterion
        self.last_criterion = criterion
        
        return self.curr_state, reward, terminated, truncated, info


if __name__ == "__main__":
    
    # TODO: Add testing instances
    ...