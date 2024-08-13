import os
from typing import Tuple, List
import torch
from torch import Tensor as Tensor
from rectpack import newPacker, PackingMode
from math import ceil, floor
from _typeshed import SupportsWrite
import matplotlib.pyplot as plt

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
        
        # Fetch LWH params from instance of class Machine
        #! Real space for bin packing: (L - 2*Margin) * (W - 2*Margin)
        self.L = machine.build_l - 2 * self.process.min_distance_part_platform
        self.W = machine.build_w - 2 * self.process.min_distance_part_platform
        self.H = machine.build_h
        
        # define self.machine and self.process for easier access of params
        self.machine = machine
        self.process = process
        
        # view_shape is for reshaping view of the batch 
        # for neural network processing
        self.view_shape = view_shape
        
        # Trial, Using RectPack Algorithm
        self.bin_true = newPacker(mode=PackingMode.Online,
                                       rotation=False
                                       ) # depends on the packing algorithm
        
        # Added Margin between parts and platform edges
        self.bin_true.add_bin(width=self.L, 
                              height=self.W
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
        # self.parts_into contains list of dicts containing parts info with orientations already specified.
        self.parts_info: List[dict] = []
    
    @property
    def slice_number(self) -> float:
        """
        Get the slice number of build

        slice_number = ceil( maximum_of_build_height / layer_thickness )  
        """
        return ceil(max(self.parts_info, key=lambda x:x["H"]) / self.process.layer_thickness)
    
    def get_current_view(self, stretch: bool = True, show: bool = False) -> Tensor:
        """
        Returns the current view of the batch as one of the neural network inputs
        Height information in also included.
        """
        grid = torch.zeros(size=tuple(map(ceil, 
                                          (self.L, self.W))))
        for (_, x, y, w, h, rid) in self.bin_true.rect_list():
            
            # Add hights corresponding to the area occupied by each part, for reference of NN.
            grid[floor(x):ceil(x+w), floor(y):ceil(y+h)] = rid["height"]
            
            if show:
                # Add lacing to each rectangle for visual reference.
                grid[floor(x+1):ceil(x+w-1), floor(y+1):floor(y+h-1)] = -1
        
        # Stretch the view into standard size to fit in to NN.
        if stretch == True:
            return torch.nn.functional.interpolate(grid, size=self.view_shape, mode="bilinear")
        
        # retain the original shape of the view for human-eye reference.
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
        """
        Returns total <u>surface area</u> of all parts in this batch
        """
        return sum(map(lambda x:x["surface_area"], self.parts_info))
    
    def get_total_part_volume(self) -> float:
        """
        Returns total <u>volume</u> of all parts in this batch
        """
        return sum(map(lambda x:x["volume"], self.parts_info))
    
    def get_total_support_volume(self) -> float:
        """
        Returns total <u>volume of support</u> of all parts in this batch
        """
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
      
    def empty(self) -> bool:
        """
        Checks whether the batch is empty
        """
        return len(self.parts_info) < 1

    def show_parts(self, fp: SupportsWrite[str]) -> None:
        """
        Print self.parts_into into a file.
        """
        for idx, part in self.parts_info:
            print(f"{idx = }, {part}", file=fp)
    
    def show_view(self, dir: str) -> None:
        """
        Print view of the batch in in terms of a matrix as an image
        """
        view = self.get_current_view(stretch=False, show=True)
        plt.imshow(view)
        plt.colorbar()
        plt.grid()
        plt.savefig(dir)
        

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
        """
        Get the view of the current batch
        """
        return self.get_batch().get_current_view()
    
    def add_part(self, part: Part, orientation: int) -> Tuple[Tensor, bool]:
        """
        Try to add the part to the current batch
        """
        return self.get_batch().add_part(part, orientation)
    
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
    
    def calculate_energy(self) -> float:
        """
        Calculate the power needed for the solution UNTIL NOW
        """
        return sum(map(calculate_batch_energy, self.batches))
    
    def show(self, out_dir: str = f"./solution/") -> None:
        """
        Display all the part_ls and view of all batches of the solution.
        """
        with open(os.path.join(out_dir, "contains.txt"), 'w') as f:
            for bid, b in enumerate(self.batches):
                print(f"{'=' * 30}\n\t\tBatch No. {bid}{'=' * 30}", file=f)
                b.show_parts(f)
                b.show_view(f"{out_dir}/batch_{bid:02d}.img")
