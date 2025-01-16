from math import trunc
import os
import random
from typing import Any, Dict, Tuple, List, Literal
import torch
from torch import Tensor as Tensor
import joyrl
from gymnasium import Env, spaces

from slm_classes import MetaData, load_json_to_class
from batch_solution import Solution, SolutionParallel1D

#! Add Unit Test


# only supports assigning a part to a batch
# and the 2D bin packing algorithm puts the part into a target position
class SingleSLMEnv(Env):
    """
    Environment of SLM Machine Optimization
    """
    
    # load instance-meta data
    def __init__(self, 
                 in_path: str, # Dir path in training and .json file path in testing stage
                 device: torch.device,
                 phase: Literal["Train", "Test"] = "Train", 
                 view_shape: Tuple[int, int] = (224, 224), 
                 max_part_type: int = 20,
                 max_orientation_num: int = 7,
                 seed: float = 0,
                 **kwargs
                 ):
        """Initialise a SLM Env.

        Args:
            `in_path` (`str`): path for instances, dir path for training phase while path of .json file for
                testing phase. 
                Defaults to `None`.
            `view_shape` (`Tuple[int, int]`): shape of the reshaped view for NN processing. Defaults to `(224, 224)`.
            `max_part_type` (`int`): maximum of number of types accepted. Defaults to `20`.
            `max_orientation_num` (`int`): maximum of number of orientation numbers accepted. Defaults to `7`.
            `seed` (`float`): random seed to be passed to function random.seed(). Defaults to `0`.

        Raises:
            NotImplementedError: If self.phase is not among ["Train", "Test"].
        """
        
        random.seed(seed)
        
        self.device = device
        
        self.name = "SingleSLMEnv"
        self.phase = phase
        self.in_path = in_path
        self.view_shape = view_shape
        self.max_part_type = max_part_type
        self.max_orientation_num = max_orientation_num
        
        # randomly pick a json file in the training dir
        # and pack the training data into a class
        if self.phase == "Train":
            self.load_path = random.choice(os.listdir(self.in_path))
            self.slm_metadata = load_json_to_class(os.path.join(self.in_path, self.load_path))
            
        # if the phase is Test, choose the file indicated by the path.
        elif self.phase == "Test":
            # load the specified in_path
            self.slm_metadata = load_json_to_class(self.in_path)
            
        # Raise Error if self.phase is not among the two strings above.
        else:
            raise NotImplementedError
        
        # pass the max params to self.slm_metadata for generating state matrix and mask matrix
        self.slm_metadata.max_part_type = max_part_type
        self.slm_metadata.max_orientation_num = max_orientation_num
        
        # -----------------------------------------------------------
        # transform the loaded data into states
        
        # set dict of available parts
        # length of this dict is the same size of output allocation vector.
        self.part_unavailable_mask = None
        self.last_state = None
        
        # Get the shape of the working area
        self.L = self.slm_metadata.machine.build_l
        self.W = self.slm_metadata.machine.build_w
        self.H = self.slm_metadata.machine.build_h
        
        # Create an instance of class Solution and initialize it by calling .add_batch() method.
        self.solution = Solution(view_shape=self.view_shape, instance_name=self.load_path)
        self.solution.add_batch(self.slm_metadata.machine,
                                self.slm_metadata.process)
        self.view_process_shape = (1, view_shape[0], view_shape[1])
        
        # observation of the current batch
        self.curr_state = (
            self.solution.get_current_view().reshape(*self.view_process_shape), # Current discretized view of the batch, Variable
            torch.tensor([self.L, self.W, self.H]),  # Real size of the batch, Constant
            self.slm_metadata.init_state().reshape(-1) # Situation of all parts, Variable
        )
        
        # Initial reference for comparing criterion numbers.
        self.last_criterion = 0
        
        # Gymnasium Env Spaces Setup
        self.observation_space = spaces.Tuple(spaces=[
            spaces.Box(low=0, high=float("inf"), shape=(1, view_shape[0], view_shape[1])),
            spaces.Box(low=0, high=float("inf"), shape=(3,)),
            spaces.Box(low=0, high=float("inf"), shape=(max_part_type*max_orientation_num, ))
        ])
        self.action_space = spaces.Tuple(spaces=[
            spaces.Box(low=0, high=float("inf"), shape=(max_part_type, )), 
            spaces.Box(low=0, high=float("inf"), shape=(max_orientation_num, ))
        ])
    
    def get_unavailable_mask(self):
        """
        get unavailable mask of the part types. The indices of the output vector is 1 iff
            - The remaining number of parts is greater than 0
            - There is at least 1 legal orientation
        """
        return ((self.state[-1][:, 0] > 0) * (torch.sum(self.slm_metadata.mask_matrix(), dim=-1) > 0)).reshape(-1)
    
    # verify physical constraints
    def check_geo_constraints(self) -> float:
        """
        NOT IMPLEMENTED YET
        
        Used only if DRL model is allowed to determine 
        the position and orientation of a part in a batch.
        
        Checks if the allocation meets the geometry constraints.
        """
        ...
    
    def reset(self):
        """
        Reset the env according to self.phase
        
        Returns: self.curr_state, info
        """
        if self.phase == "Train":
            self.__init__(in_path=self.in_path, 
                          device=self.device,
                          phase=self.phase, 
                          view_shape=self.view_shape, 
                          seed=random.random()
                          )
        elif self.phase == "Test":
            exit(0)
        else:
            raise NotImplementedError
        
        return self.curr_state, self.load_path
    
    def update_state(self, view: Tensor, part_id: int) -> None:
        """
        Update self.current state according to part_id and view after a successful packing
        """
        self.last_state = self.curr_state
        
        # get the state matrix of parts at this timestamp
        temp_part_state = self.last_state[-1]
        assert temp_part_state.reshape(self.max_part_type, -1)[part_id, 0] > 0, \
            "Error, trying to allocate type of part which is already 0 parts."
        
        # Decrement the number of type part_id by 1 (with assertion that it must greater than 0)
        temp_part_state.view(self.max_part_type, -1)[part_id, 0] -= 1
        
        # Create new self.curr_state.
        self.curr_state = (
            view.reshape(*self.view_process_shape),
            self.last_state[1],
            temp_part_state.reshape(-1)
        )
    
    def done(self) -> None:
        """
        Check if all the parts are allocated
        """
        # If any of the kind of part have unallocated instances, return False
        return not any(self.curr_state[-1].reshape(self.max_part_type, -1)[:, 0])
    
    def step(
        self, 
        action: Tensor
        ) -> Tuple[Tensor, float, bool, bool, str]:
        """
        Update the Env according to the action.
        Action: long tensor shaped 
            [ max_part_type * max_orientation_num ]
        Returns: 
            self.curr_state, reward, terminated, truncated, info
        """
        
        terminated = False
        truncated = False
        info = None
        reward = 0
        
        part_d, ori_d = list(map(lambda x:torch.softmax(x, -1), action))
        
        # Flag var, Whether the part is allocated successfully
        allocated = False
        
        # Select part_id and orientation_id
        # TODO: 
        part_id = torch.argmax(part_d.reshape(-1).to(self.device) * \
            (self.curr_state[-1].reshape(self.max_part_type, -1)[:, 0] > 0).to(self.device), dim=-1)
        
        # Get the rank of different orientations of the part selected according to the output score
        masked_ranks = ori_d.reshape(-1).to(self.device)\
            * self.slm_metadata.mask_matrix()[part_id].reshape(-1).to(self.device)
        orientation_rank = torch.argsort((masked_ranks), descending=True)
        orientation_rank = list(filter(lambda x:self.slm_metadata.mask_matrix()[part_id].reshape(-1)[x],
                                       orientation_rank))
        
        # Initialize pointer, and get the orientation id according to the pointer
        rank_ptr = 0
        orientation_id = orientation_rank[rank_ptr]
        
        # Try allocating the part according to the orientation selected.
        view, allocated = self.solution.add_part(self.slm_metadata.parts[part_id], orientation=orientation_id)
                
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
                
                # If the orientation is not feasible, then break the loop
                if (not self.slm_metadata.mask_matrix()[part_id, orientation_id].item()) \
                        or rank_ptr >= len(orientation_rank):
                    break
                
                orientation_id = orientation_rank[rank_ptr]
                    
                view, allocated = self.solution.add_part(self.slm_metadata.parts[part_id], orientation=orientation_id)
            
            if allocated:
                self.update_state(view=view, part_id=part_id)
                
                # If all parts are allocated, terminate this episode.
                if self.done():
                    terminated = True
            else:
                # reset orientation
                rank_ptr = 0
                # Add a new batch (same size) and retry allocating
                self.solution.add_batch(self.slm_metadata.machine, self.slm_metadata.process)
                view, allocated = self.solution.add_part(self.slm_metadata.parts[part_id],
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

class SingleSLMEnvParallel1D(SingleSLMEnv):
    """
    State: Concatenation of three flattened tensors
    
    Action: Concatenation of three tensors
        [part_distribution, orientation_distribution, batch_distribution]
    """
    def __init__(self,
                 in_path: str,
                 phase: Literal["Train", "Test"] = "train",
                 max_part_type: int = 20,
                 max_orientation_num: int = 7,
                 max_batch_num: int = 20,
                 seed: float = 0,
                 **kwargs
                 ):
        
        random.seed(seed)
        
        self.name = "SingleSLMEnvParallel1D"
        self.phase = phase
        self.in_path = in_path
        self.max_part_type = max_part_type
        self.max_orientation_num = max_orientation_num
        self.max_batch_num = max_batch_num
        
        # randomly pick a json file in the training dir
        # and pack the training data into a class
        if self.phase == "Train":
            self.load_path = random.choice(os.listdir(self.in_path))
            self.slm_metadata = load_json_to_class(os.path.join(self.in_path, self.load_path))
            
        # if the phase is Test, choose the file indicated by the path.
        elif self.phase == "Test":
            # load the specified in_path
            self.slm_metadata = load_json_to_class(self.in_path)
            
        # Raise Error if self.phase is not among the two strings above.
        else:
            raise NotImplementedError
        
        # select slm instance and load it
        self.slm_metadata = self.get_metadata()
        
        # pass the max params to self.slm_metadata for generating state matrix and mask matrix
        self.slm_metadata.max_part_type = max_part_type
        self.slm_metadata.max_orientation_num = max_orientation_num
        
        self.lwh = self.slm_metadata.machine.get_lwh()
        
        self.solution = SolutionParallel1D(instance_name=self.load_path)
        self.solution.init_batches(
            machine=self.slm_metadata.machine,
            process=self.slm_metadata.process
        )
        
        # total length of space:
        # num_batch * num_param_batch + 3 + num_init_states
        self.curr_state = torch.stack(
            tensors=[self.solution.get_current_view().reshape(-1), 
                     torch.tensor(self.lwh),
                     self.slm_metadata.init_state().reshape(-1)],
            dim=0
        )
        
        self.last_criterion = 0
        
        self.observation_space = spaces.Box(low=0, high=float("inf"), 
                                            shape=self.curr_state.shape)
        self.action_space = spaces.Tuple(spaces=[
            spaces.Box(low=0, high=float("inf"), shape=(max_batch_num, )), 
            spaces.Box(low=0, high=float("inf"), shape=(max_part_type, )), 
            spaces.Box(low=0, high=float("inf"), shape=(max_orientation_num, ))
        ])
    
    def update_state(self, view: Tensor, part_id: int) -> None:
        self.last_state = self.curr_state
        
        # get the state matrix of parts at this timestamp
        temp_part_state = self.last_state[-1]
        assert temp_part_state.reshape(self.max_part_type, -1)[part_id, 0] > 0, \
            "Error, trying to allocate type of part which is already 0 parts."
        
        # Decrement the number of type part_id by 1 (with assertion that it must greater than 0)
        temp_part_state.view(self.max_part_type, -1)[part_id, 0] -= 1
        
        self.curr_state = (
            view.reshape(-1),
            self.last_state[1],
            temp_part_state.reshape(-1)
        )
    
    def done(self) -> None:
        """
        Check if all the parts are allocated
        """
        # If any of the kind of part have unallocated instances, return False
        return not any(self.curr_state[-1].reshape(self.max_part_type, -1)[:, 0])
    
    def slice_action(self, action: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns: `[part_dist, orientation_dist, batch_dist]`
        """
        return action[:self.max_part_type],\
            action[self.max_part_type: self.max_part_type+self.max_orientation_num],\
            action[self.max_part_type+self.max_orientation_num:]

    def step(
        self, 
        action: Tensor
        ) -> Tuple[Tensor, float, bool, bool, str]:
        """
        Action: Concatenation of three tensors
            [part_distribution, orientation_distribution, batch_distribution]
        """
        terminated, truncated = False, False
        info, reward = None, 0        
        
        distributions = self.slice_action(action=action)
        part_dist, ori_dist, batch_dist = distributions
        
        penalty = 0
        
        for part_id in torch.argsort(part_dist, descending=True):
            if success:
                break
            if self.curr_state[-1][part_id, 0] < 1:
                continue
            for ori_id in torch.argsort(ori_dist, descending=True):
                if success:
                    break
                for batch_id in torch.argsort(batch_dist, descending=True):
                    new_view, success, penalty_tmp = self.solution.add_part(part=part_id,
                                                            orientation=ori_id,
                                                            idx=batch_id)
                    penalty += penalty_tmp
                    if success:
                        self.update_state(view=new_view, part_id=part_id)   
                        
                        # If all parts are allocated, terminate this episode.
                        # 
                        if self.done():
                            terminated = True
                        break
        
        criterion = self.solution.calculate_energy()
        reward = self.last_criterion - criterion - penalty
        self.last_criterion = criterion

        return self.curr_state, reward, terminated, truncated, info

if __name__ == "__main__":
    print(joyrl.__version__) # print version
    yaml_path = "./yaml_configurations/SingleSLMEnv-v0-DQN.yaml"
    slm_single_env = SingleSLMEnvParallel1D(in_path="./instances_json/", phase="Train")
    joyrl.run(yaml_path=yaml_path, env=slm_single_env)

