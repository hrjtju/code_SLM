from math import trunc
import os
import random
from typing import Any, Dict, Tuple, List, Literal
import torch
from torch import Tensor as Tensor
from gymnasium import Env, spaces

from slm_model.slm_classes import MetaData, load_json_to_class
from slm_model.batch_solution import Solution, SolutionParallel1D

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
                 seed: float | None = None,
                 ppo: bool = False,
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
        
        # T11: only seed the *global* RNG when a seed is explicitly requested.
        # Re-seeding on every reset() (with a value drawn from the same stream)
        # also re-seeds the RNG used by the replay buffer sampling, which makes
        # the sampling pattern structurally repeat across episodes.
        if seed is not None:
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
        return ((self.curr_state[-1][:, 0] > 0) * (torch.sum(self.slm_metadata.mask_matrix(), dim=-1) > 0)).reshape(-1)
    
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
                          seed=None
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
        action: Tuple[Tensor, Tensor]   
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
        view, allocated, penalty = self.solution.add_part(self.slm_metadata.parts[part_id], orientation=orientation_id)
        
        all_penalty = penalty
        
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
                    
                view, allocated, penalty = self.solution.add_part(self.slm_metadata.parts[part_id], orientation=orientation_id)
                all_penalty += penalty
                
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
                view, allocated, penalty = self.solution.add_part(self.slm_metadata.parts[part_id],
                                                         orientation=orientation_rank[rank_ptr])
                all_penalty += penalty + 0.5
                
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
                 seed: float | None = None,
                 ppo: bool = False,
                 penalty: float = 0,
                 **kwargs
                 ):
        
        # T11: see SingleSLMEnv.__init__ -- do not touch the global RNG unless asked.
        if seed is not None:
            random.seed(seed)
        
        self.ppo = ppo
        # print(f"{self.ppo=}")
        self.name = "SingleSLMEnvParallel1D"
        self.phase = phase
        self.in_path = in_path
        self.max_part_type = max_part_type
        self.max_orientation_num = max_orientation_num
        self.max_batch_num = max_batch_num
        
        self.penalty = penalty
        
        self.fail_allocate_num: int = 0
        
        # T2: the (part, orientation, batch) triple that was *actually* executed by
        # the last step(). The action vector handed to step() only defines a
        # *priority order*; the environment falls back to lower-ranked candidates
        # when the argmax triple is infeasible, so the learner must be told which
        # action really happened.
        self.executed_action: Tuple[int, int, int] | None = None
        
        # randomly pick a json file in the training dir
        # and pack the training data into a class
        if self.phase == "Train":
            self.load_path = random.choice([*filter(lambda x:".json" in x, os.listdir(self.in_path))])
            self.slm_metadata = load_json_to_class(os.path.join(self.in_path, self.load_path))
            
        # if the phase is Test, choose the file indicated by the path.
        elif self.phase == "Test":
            # load the specified in_path
            self.load_path = self.in_path
            self.slm_metadata = load_json_to_class(self.in_path)
            
        # Raise Error if self.phase is not among the two strings above.
        else:
            raise NotImplementedError
        
        # pass the max params to self.slm_metadata for generating state matrix and mask matrix
        self.slm_metadata.max_part_type = max_part_type
        self.slm_metadata.max_orientation_num = max_orientation_num
        
        self.lwh = self.slm_metadata.machine.get_lwh()
        
        self.solution = SolutionParallel1D(instance_name=self.load_path, batch_num=max_batch_num)
        self.solution.init_batches(
            machine=self.slm_metadata.machine,
            process=self.slm_metadata.process
        )
        
        # total length of space:
        # num_batch * num_param_batch + 3 + num_init_states
        self.curr_state = torch.concatenate(
            tensors=[self.solution.get_current_view().reshape(-1), 
                     torch.tensor(self.lwh),
                     self.slm_metadata.init_state().reshape(-1)],
            dim=0
        )
        
        self.last_criterion = 0
        
        self.observation_space = spaces.Box(low=0, high=float("inf"), 
                                            shape=self.curr_state.shape)
        self.action_space = spaces.Box(low=0, high=float("inf"), shape=(max_batch_num+max_part_type+max_orientation_num, ))
        
        self.mask_tensor: Tensor = None
        self.parts_info_mtx: Tensor = None
        self.project_spaces: Tensor = None
        
        self.init_mask()
        
        # assert torch.equal(self.curr_state, self.transform_state(self.transform_state(self.curr_state)))
        # assert all(torch.equal(a, b) for (a, b) in zip(self.transform_state(self.curr_state), 
        #                    self.transform_state(self.transform_state(self.transform_state(self.curr_state)))))
    
    def init_mask(self):
        """
        First initialize the mask tensor
        
        set all valid (part, orientation) pairs to True, this corresponds to mask_tensor[:parts_num, :orientations_num, :]
        """
        
        
        # [n, o, b]
        self.mask_tensor = torch.ones(self.max_part_type, self.max_orientation_num, self.max_batch_num, dtype=torch.bool)
        
        # [n, o]
        feasible_orientations = self.slm_metadata.mask_matrix()

        # T20: fail fast with a readable message. `project_spaces` (used by
        # `update_mask`) is sized by the *instance*, so `max_orientation_num` smaller
        # than the instance's orientation count used to blow up much later with a
        # cryptic broadcasting error.
        assert feasible_orientations.shape[1] <= self.max_orientation_num, (
            f"max_orientation_num={self.max_orientation_num} is smaller than the "
            f"instance's orientation count {feasible_orientations.shape[1]}"
        )
        assert feasible_orientations.shape[0] <= self.max_part_type, (
            f"max_part_type={self.max_part_type} is smaller than the instance's "
            f"part-type count {feasible_orientations.shape[0]}"
        )

        self.mask_tensor = self.mask_tensor * feasible_orientations[..., None]
    
    def update_mask(self):
        # TODO: Check correctness
        
        # Get the remain space of the batches
        # get_current_view: [n_batches, 3], where 3 refers to **rest_area, occupies_ratio, max_part_height**
        # [n_batches,]
        remain_spaces = self.solution.get_current_view(show=False)[:, 0].reshape(-1)
        
        # get the project space of all feasible (part, orientation) pairs.
        # [x, x, x, l1, w1, *, *, l2, w2, *, *]
        # l = [:, 4::4]
        if self.parts_info_mtx is None or self.project_spaces is None:
            self.parts_info_mtx = self.slm_metadata.init_state()
            self.project_spaces = self.parts_info_mtx[:, 3::4] * self.parts_info_mtx[:, 4::4]
            # T20: readable failure instead of a broadcasting RuntimeError below
            assert self.project_spaces.shape[1] <= self.max_orientation_num, (
                f"max_orientation_num={self.max_orientation_num} is smaller than the "
                f"instance's orientation count {self.project_spaces.shape[1]} "
                f"({self.in_path})"
            )
            # print(self.parts_info_mtx, self.parts_info_mtx[:, 3::4], self.parts_info_mtx[:, 4::4])
        
        # [num_parts, num_orientations]
        # print(remain_spaces, self.project_spaces)
        space_feasible_flag = self.project_spaces[..., None] < remain_spaces[None, None, ...]
        self.mask_tensor = (space_feasible_flag * self.mask_tensor).bool()
        
        # Get remaining parts type
        feasible_parts_type = (~self.get_unavailable_parts_mask()).reshape(-1)
        self.mask_tensor = (feasible_parts_type[..., None, None] * self.mask_tensor).bool()
    
    def get_unavailable_parts_mask(self):
        """
        get unavailable mask of the part types. The indices of the output vector is 1 iff
            - The remaining number of parts is greater than 0
            - There is at least 1 legal orientation
        """
        return self.transform_state(self.curr_state)[-1][:, 0] < 1
    
    def transform_state(self, state: Tensor|Tuple[Tensor, Tensor, Tensor]):
        
        if isinstance(state, Tuple):
            return torch.concatenate(
                tensors=[x.reshape(-1) for x in state],
                dim=0
            )   
        else:
            view_shape = self.solution.get_current_view().shape 
            view_flattened_len = self.solution.get_current_view().reshape(-1).shape[0]
            parts_state_shape = self.slm_metadata.init_state().shape
            return (
                state[:view_flattened_len].reshape(*view_shape),
                state[view_flattened_len:view_flattened_len+3],
                state[view_flattened_len+3:].reshape(*parts_state_shape)
            )

    def reset(self):
        """
        Reset the env according to self.phase
        
        Returns: self.curr_state, info
        """
        
        if self.phase == "Train":
            self.__init__(in_path=self.in_path, 
                          phase=self.phase, 
                          max_part_type=self.max_part_type,
                          max_orientation_num=self.max_orientation_num,
                          max_batch_num=self.max_batch_num,
                          seed=None,
                          ppo=self.ppo,
                          penalty=self.penalty
                          )
        elif self.phase == "Test":
            return 
        else:
            raise NotImplementedError
        
        self.check_state_dim()  # T20
        return self.curr_state, self.load_path

    def update_state(self, view: Tensor, part_id: int) -> None:
        self.last_state = self.curr_state
        
        # get the state matrix of parts at this timestamp
        temp_part_state = self.transform_state(self.last_state)[-1]
        assert temp_part_state[part_id, 0] > 0, \
            "Error, trying to allocate type of part which is already 0 parts."
        
        # Decrement the number of type part_id by 1 (with assertion that it must greater than 0)
        temp_part_state[part_id, 0] -= 1
        
        self.curr_state = torch.concatenate(tensors=[
            view.reshape(-1),
            self.transform_state(self.last_state)[1],
            temp_part_state.reshape(-1)
        ], dim=0)
    
    def done(self) -> None:
        """
        Check if all the parts are allocated
        """
        # If any of the kind of part have unallocated instances, return False
        return not any(self.transform_state(self.curr_state)[-1].reshape(self.max_part_type, -1)[:, 0])
    
    def slice_action(self, action: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns: `[part_dist, orientation_dist, batch_dist]`
        """
        action = action.reshape(-1)
        
        return action[:self.max_part_type],\
            action[self.max_part_type: self.max_part_type+self.max_orientation_num],\
            action[self.max_part_type+self.max_orientation_num:]

    @property
    def action_dim(self) -> int:
        return self.max_part_type + self.max_orientation_num + self.max_batch_num

    @property
    def state_dim(self) -> int:
        """T20: expected flat-state length for the configured max_* dimensions."""
        return (self.max_batch_num * 3 + 3
                + self.max_part_type * (3 + 4 * self.max_orientation_num))

    def check_state_dim(self) -> None:
        """
        T20: the flat state only has a fixed layout when every `max_*` is >= the
        instance's own dimensions. Otherwise the observation length silently varies
        per instance and the replay buffer fails later with an unrelated
        "stack expects each tensor to be equal size" error.
        """
        actual = int(self.curr_state.reshape(-1).shape[0])
        assert actual == self.state_dim, (
            f"state length {actual} != expected {self.state_dim} for "
            f"max_part_type={self.max_part_type}, max_orientation_num="
            f"{self.max_orientation_num}, max_batch_num={self.max_batch_num}; "
            f"the max_* settings are too small for instance {self.in_path}"
        )

    def onehot_action(self, triple: Tuple[int, int, int]) -> Tensor:
        """
        T2: build the action vector whose per-head argmax equals `triple`.
        
        This is what has to be stored in the replay buffer so that the Q-learning
        update gathers the value of the action the environment really executed.
        """
        part_id, ori_id, batch_id = (int(x) for x in triple)
        action = torch.zeros(self.action_dim)
        action[part_id] = 1.0
        action[self.max_part_type + ori_id] = 1.0
        action[self.max_part_type + self.max_orientation_num + batch_id] = 1.0
        return action

    def head_masks(self) -> Tensor:
        """
        T18: per-head feasibility marginals of `self.mask_tensor`, laid out like an
        action vector: `[part_valid (P) | ori_valid (O) | batch_valid (B)]`.
        
        The factored Q function maximises each head independently, so these marginals
        are exactly what is needed to stop the bootstrapped target from maximising
        over actions that cannot be executed.
        """
        m = self.mask_tensor.bool()
        return torch.concatenate([
            m.any(dim=-1).any(dim=-1),   # [P]
            m.any(dim=0).any(dim=-1),    # [O]
            m.any(dim=0).any(dim=0),     # [B]
        ], dim=0)

    def step(
        self, 
        actions: Tensor
        ) -> Tuple[Tensor, float, bool, bool, str]:
        """
        Action: Concatenation of three tensors
            [part_distribution, orientation_distribution, batch_distribution]
        """
        terminated, truncated = False, False
        info, reward = {}, 0        
        
        if isinstance(actions, torch.Tensor):
            part, ori, batch = self.slice_action(actions.detach().cpu())
        else:
            part, ori, batch = actions.detach().cpu()
        
        penalty = 0
        success = False
        self.fail_allocate_num = 0
        self.executed_action = None
        # print(f"{self.ppo=}")
        
        if not self.ppo:
            # TODO: 考虑使用 combinations 改写？
            for part_id in torch.argsort(part, descending=True):
                if success:
                    break
                if self.transform_state(self.curr_state)[-1][part_id, 0] < 1:
                    continue
                for ori_id in torch.argsort(ori, descending=True):
                    if success:
                        break
                    for batch_id in torch.argsort(batch, descending=True):
                        new_view, success, penalty_tmp = self.solution.add_part(part=self.slm_metadata.parts[part_id],
                                                                orientation=ori_id,
                                                                idx=batch_id)
                        penalty += penalty_tmp
                        
                        if success:
                            # T2: remember which triple was *really* executed
                            self.executed_action = (int(part_id), int(ori_id), int(batch_id))
                            self.update_state(view=new_view, part_id=part_id)   
                            
                            # If all parts are allocated, terminate this episode.
                            # 
                            if self.done():
                                terminated = True
                            break
                        else:
                            self.fail_allocate_num += 1
            
            # T12: nothing could be placed anywhere -> episode is *truncated*
            # (a dead end), not terminated.
            if not success:
                truncated = True
        else:
            new_view, success, penalty_tmp = self.solution.add_part(part=self.slm_metadata.parts[part], 
                                                                    orientation=ori, 
                                                                    idx=batch,
                                                                    hard=False)
        
            if success:
                self.executed_action = (int(part), int(ori), int(batch))
                self.update_state(new_view, part)
                if self.done():
                    terminated = True
            else:
                truncated = True
                    
            penalty = penalty_tmp
        
        criterion = self.solution.calculate_energy()
        reward = self.last_criterion - criterion - penalty * self.penalty
        self.last_criterion = criterion

        self.update_mask()
        
        info = {
            "executed_action": self.executed_action,
            "success": success,
            "fail_allocate_num": self.fail_allocate_num,
            "penalty": penalty,
        }
        
        return self.curr_state, reward, terminated, truncated, info


