import os
from typing import Tuple, List, Dict, Union
import torch
from torch import Tensor as Tensor
from rectpack import newPacker, PackingMode
from math import ceil, floor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
                 view_shape: tuple
                 ) -> None:
        
        # Fetch LWH params from instance of class Machine
        #! Real space for bin packing: (L - 2*Margin) * (W - 2*Margin)
        self.L = machine.build_l - 2 * process.min_distance_part_platform
        self.W = machine.build_w - 2 * process.min_distance_part_platform
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
        return ceil(max(self.parts_info, key=lambda x:x["H"])["H"] / self.process.layer_thickness) \
            if not self.empty() else 0
    
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
                grid[floor(x):ceil(x+w), floor(y):ceil(y+h)] = -1
                grid[floor(x+1):ceil(x+w-1), floor(y+1):floor(y+h-1)] = rid["height"]
        
        # Stretch the view into standard size to fit in to NN.
        if stretch == True:
            return torch.nn.functional.interpolate(grid.reshape(1, 1, *grid.shape), 
                                                   size=self.view_shape, 
                                                   mode="bilinear"
                                                   )
        
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
        if part.get_proj_area(orientation) > self.get_rest_area():
            return None, False
        
        # Try to add the part into the batch using the bin-packing algorithm
        # If it cannot be packed, return False.
        self.bin_true, success = allocate_bin_packing_2d(self.bin_true, 
                                                         part.get_part_info(orientation))
        
        if success:
            # update list parts_info 
            self.parts_info.append(part.get_part_info(orientation))
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

    def show_parts(self, fp) -> None:
        """
        Print self.parts_into into a file.
        """
        for idx, part in enumerate(self.parts_info):
            print(f"{idx = }, {part}", file=fp)

        print(f"rest_area: {self.get_rest_area()}", file=fp)
        print(f"slice_number: {self.slice_number}", file=fp)
        print(f"get_total_surface_area: {self.get_total_surface_area()}", file=fp)
        print(f"get_total_part_volume: {self.get_total_part_volume()}", file=fp)
        print(f"get_total_support_volume: {self.get_total_support_volume()}", file=fp)
        print(f"Time: {calculate_batch_time(self)['total_time']}", file=fp)
        print(f"Energy: {calculate_batch_energy(self)['EPC']}", file=fp)
    
    def show_view(self, dir: str) -> None:
        """
        Print view of the batch in in terms of a matrix as an image
        """
        view = self.get_current_view(stretch=False, show=True)
        plt.figure(dpi=100, figsize=(6, 5))
        ax = plt.imshow(view, cmap="Blues")
        plt.colorbar()
        plt.grid(alpha=0.1)
        plt.savefig(dir)
        
        del ax
         

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
    
    def get_current_view(self, stretch: bool = True, show: bool = False) -> Tensor:
        """
        Get the view of the current batch
        """
        return self.get_batch().get_current_view(stretch=stretch, show=show)
    
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
        return 0 if self.empty() else sum(map(lambda x:calculate_batch_time(x)["total_time"], self.batches))
    
    def calculate_energy(self) -> float:
        """
        Calculate the power needed for the solution UNTIL NOW
        """
        return 0 if self.empty() else sum(map(lambda x:calculate_batch_energy(x)["EPC"], self.batches))
    
    def show(self, out_dir: str = f"./solution/") -> None:
        """
        Display all the part_ls and view of all batches of the solution.
        """
        with open(os.path.join(out_dir, "contains.txt"), 'w') as f:
            print(f"Solution Time: {self.calculate_time()}", file=f)
            print(f"Solution Energy: {self.calculate_energy()}", file=f)
            
            for bid, b in enumerate(self.batches):
                print(f"{'=' * 30}\n\t\tBatch No. {bid}\n{'=' * 30}", file=f)
                b.show_parts(f)
                b.show_view(f"{out_dir}/batch_{bid:02d}.jpg")

    def empty(self) -> bool:
        return (len(self.batches) < 1) or all(map(lambda x:x.empty(), self.batches))
    

CONCENTRATION_OXYGEN_INITIAL = 21
CONCENTRATION_OXYGEN_END = 0.1

def calculate_batch_time(
    b: Batch
    ) -> Dict[str, float]:
    
    if b.empty():
        return 0
    
    process = b.process
    
    #! No longer needed, since it is included in the heat_time
    # t0 = Machine.C * np.log(CONCENTRATION_OXYGEN_INITIAL) - Machine.D
    # t1 = Machine.C * np.log(CONCENTRATION_OXYGEN_END) - Machine.D
    # delta_t_oxygen = t0 - t1
    
    # ====================================================================
    # Calculating time durations
    delta_t_heater = process.heat_time
    delta_t_cooling = process.cool_time
    
    heater_time = delta_t_heater
    
    #! Check if the term "layer_thickness" refer to "layer_thickness"
    scanning_border_time = b.get_total_surface_area() / (process.num_laser \
        * process.laser_speed_border * process.layer_thickness)
        
    fill_contour_time = b.get_total_surface_area() / (process.num_laser \
        * process.laser_speed_contour * process.layer_thickness)
        
    volume_hatching_time = b.get_total_part_volume() / (process.num_laser \
        * process.laser_speed_volume * process.layer_thickness * process.hatch_distance_volume)
        
    support_building_time = b.get_total_support_volume() / (process.num_laser \
        * process.laser_speed_support * process.layer_thickness * process.hatch_distance_support)
    
    #! MISSING
    # TODO: Report the issue and find ways to fill them up, and ALL .json files
    # TODO: should be modified (May completed quickly using Regex Expressions).
    process.recoater_time_single = 1 #! Temporarily solve
    recoater_time_all = process.recoater_time_single * b.slice_number
    
    cooling_time = delta_t_cooling
    
    building_time = scanning_border_time + fill_contour_time \
        + volume_hatching_time + support_building_time
    
    # Sum the times up
    total_time = heater_time + building_time + recoater_time_all + cooling_time
    
    return {
        "total_time": total_time,
        "heater_time": heater_time,
        "scanning_border_time": scanning_border_time,
        "fill_contour_time": fill_contour_time,
        "volume_hatching_time": volume_hatching_time,
        "support_building_time": support_building_time,
        "recoater_time_all": recoater_time_all,
        "cooling_time": cooling_time
    }

# calculate time cost
def calculate_batch_energy(
    b: Batch
    ) -> Dict[str, Union[float, np.ndarray, pd.DataFrame]]:
    
    if b.empty():
        return 0
    
    process = b.process
    machine = b.machine
    
    # ==========================================================================
    # Calculating real power of some subsystems
    real_power_scanning_border = process.num_laser \
        * (Machine.A + machine.power_subsystems["laser_scanning_border"] * Machine.B)
    real_power_scanning_fill_contour = process.num_laser \
        * (Machine.A + machine.power_subsystems['laser_filling_contour'] * Machine.B)
    real_power_scanning_volume_hatching = process.num_laser \
        * (Machine.A + machine.power_subsystems['laser_volume_hatching'] * Machine.B)
    real_power_scanning_support_volume = process.num_laser \
        * (Machine.A + machine.power_subsystems['laser_supports_building'] * Machine.B)
    
    # ==========================================================================
    # Building power array
    P = np.array([
        [machine.power_subsystems["basic_subsystem"]], 
        [machine.power_subsystems["platform_heater"]], 
        [machine.power_subsystems["water_cooling_unit"]], 
        [machine.power_subsystems["water_circulation_unit"]], 
        [real_power_scanning_border],
        [real_power_scanning_fill_contour],
        [real_power_scanning_volume_hatching],
        [real_power_scanning_support_volume],           
        [machine.power_subsystems["recoater_motor"]], 
        [machine.power_subsystems["electric_valves"]], 
        [machine.power_subsystems["gas_circulation_pump_motor"]], 
    ])
    
    # ==========================================================================
    # Building time array
    batch_time_dict = calculate_batch_time(b)
    T = np.array([
        [batch_time_dict["heater_time"]], 
        [batch_time_dict["scanning_border_time"]], 
        [batch_time_dict["fill_contour_time"]], 
        [batch_time_dict["volume_hatching_time"]], 
        [batch_time_dict["support_building_time"]], 
        [batch_time_dict["recoater_time_all"]], 
        [batch_time_dict["cooling_time"]]
        ])
    
    # ==========================================================================
    # Calculating total energy and anergy matrix
    
    K = machine.power_coefficient.values
    # Equivalent to EPC = np.dot(np.dot(P, K), T.T)
    EPC = (P.T @ K @ T).reshape(-1).item() # An number
    
    # PKT[i][j] = P[i] K[i][j] T[j]
    # Hadamard product with broadcasting
    # Equivalent to the following
    # for i in range(11):
    #     K_row=[]
    #     for j in range(7):
    #         K_row.append(int(P[i] * K[i][j] * T[j]))
    #     PKT.append(K_row)
    PKT = (P * K * T.T).astype(np.int64)
    
    energy_matrix = pd.DataFrame(
        data=PKT, 
        index=machine.power_coefficient.index,
        columns=machine.power_coefficient.columns
        )
      
    subsystem_energy_arr = PKT.sum(axis=1)
    subprocess_energy_arr = PKT.sum(axis=0)
    
    return {
        "subsystem_energy_arr": subsystem_energy_arr,
        "subprocess_energy_arr": subprocess_energy_arr,
        "energy_matrix": energy_matrix,
        "EPC": EPC,
        "PKT": PKT
    }
