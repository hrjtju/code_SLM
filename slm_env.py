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
        
    
    # TODO: Change this method to be align with the 2d-mask method
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
        action: Tuple[Tensor, Tensor]
        ) -> Tuple[Tensor, float, bool, bool, str]:
        
        terminated = False
        truncated = False
        info = None
        reward = 0
        
        out_matrix = action.reshape(self.metadata.max_part_type, self.metadata.max_orientation_num)
        feasible_matrix = out_matrix * self.metadata.mask_matrix()
        
        allocated = False
        
        # TODO: Remember to add a softmax layer to the policy network
        parts_rank = torch.argmax(feasible_matrix)
        part_id, orientation_id = divmod(parts_rank, self.metadata.max_part_type)
        
        # [0, 2, 0, 3, 0, 5]
        # [5, 3, 1, 0, 2, 4]
        
        negative_reward = 1
        
        view, allocated = self.solution.add_part(self.metadata.parts[part_id].get_part_info(), orientation=orientation_id)
        
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
                # TODO: Try other printing orientations, If all orientations does not fit, truncate the env.
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