from functools import reduce
from operator import add
from typing import Any, Dict, Tuple, List
from numpy import negative
import torch
from torch import Tensor as Tensor
from rectpack import newPacker, PackingMode
from math import ceil, floor

from time_energy_model import calculate_batch_energy, calculate_batch_time
from slm_classes import Machine, Part, Process
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
                 view_shape: tuple,
                 gap: float,
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
        
        #! Added Margin between parts and platform edges
        self.bin_true.add_bin(width=self.L - self.process.min_distance_part_platform, 
                              height=self.W - self.process.min_distance_part_platform
                              )
        
        self.bin_view: Tensor = None # should finally be a fixed size tensor
        
        # {
        #     "volume": self.volume,
        #     "surface_area": self.surface_area,
        #     "L": build_param["L"] + gap,
        #     "W": build_param["W"] + gap,
        #     "H": build_param["H"],
        #     "S": build_param["S"],
        # } 
        self.parts_info: List[dict] = []
    
    @property
    def slice_number(self) -> float:
        return ceil(max(self.parts_info, key=lambda x:x["H"]) / self.process.layer_thickness)
    
    
    def get_current_view(self, stretch: bool = True) -> Tensor:
        """
        Returns the current view of the batch as one of the neural network inputs
        Height information in also included.
        """
        grid = torch.zeros(size=tuple(map(ceil, 
                                          (self.L, self.W))))
        for (_, x, y, w, h, rid) in self.bin_true.rect_list():
            grid[floor(x):ceil(x+w), floor(y):ceil(y+h)] = rid["height"]
            # grid[floor(x+1):ceil(x+w-1), floor(y+1):floor(y+h-1)] = -1
        
        if stretch == True:
            return torch.nn.functional.interpolate(grid, size=self.view_shape, mode="bilinear")
        else:
            return grid
            
    def get_rest_area(self) -> float:
        """
        Returns rest area of a batch
        """
        total_area = self.L * self.W
        occupied_area = sum(map(lambda x:x["L"]*x["W"], self.parts_info))
        return total_area - occupied_area
    
    def get_total_surface_area(self) -> float:
        return sum(map(lambda x:x["surface_area"], self.parts_info))
    
    def get_total_part_volume(self) -> float:
        return sum(map(lambda x:x["volume"], self.parts_info))
    
    def get_total_support_volume(self) -> float:
        return sum(map(lambda x:x["S"], self.parts_info))
    
    def add_part(self, part: Part, orientation: int) -> Tuple[Tensor, bool]:
        """
        Add part to this batch.
        
        Returns
            - `True`  , if part is successfully added
            - `False` , otherwise
        """
        
        # Checks if the projection area of the part is smaller than the 
        # area available in this batch.
        if part.get_proj_area(orientation, self.process.min_distance_parts) < self.get_rest_area():
            return False
        
        # Try to add the part into the batch using the bin-packing algorithm
        # If it cannot be packed, return False.
        self.bin_true, success = allocate_bin_packing_2d(self.bin_true, 
                                                         part.get_part_info(orientation, 
                                                                            self.process.min_distance_parts))
        
        if success:
            # update list parts_info 
            self.parts_info.append(part.get_part_info(orientation, 
                                                      self.process.min_distance_parts))
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
        return sum(map(calculate_batch_time, self.batches))
    
    # calculate energy cost
    def calculate_energy(self) -> float:
        """
        Calculate the power needed for the solution UNTIL NOW
        """
        return sum(map(calculate_batch_energy, self.batches))
    
    # TODO: Complete this method
    def show(self) -> None:
        ...
