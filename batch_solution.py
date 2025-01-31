import os
from typing import Tuple, List, Dict, Union
import torch
from torch import Tensor as Tensor, tensor
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
          each item of which contains the original dict of the category(kind) 
          the part belongs to with `build_param` equaling the value corresponding 
          the orientation chosen
    """
    def __init__(self, 
                 machine: Machine, 
                 process: Process, 
                 view_shape: tuple, 
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
                                       rotation=True
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
    
    def get_occupied_ratio(self) -> float:
        """
        Returns the proportion of area occupied by parts out of all area of the machine that can be used
        0 <= output_value <= 1 
        """
        return 0 if self.empty() else 1 - (self.get_rest_area() / (self.L * self.W))
        
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
        for idx, (part, (_, x, y, w, *_)) in enumerate(zip(self.parts_info, self.bin_true.rect_list())):
            print(f"{idx = :03d}, {x = :.2f}, {y = :.2f}, Rotated = {str(w == part['L']):5s},"
                  f" {dict(map(lambda x:(x[0], x[1].item()) if isinstance(x[1], Tensor) else x, part.items()))}", file=fp)

        print(f"occupied_ratio: {self.get_occupied_ratio() * 100:.2f}%", file=fp)
        print(f"rest_area: {self.get_rest_area()}", file=fp)
        print(f"slice_number: {self.slice_number}", file=fp)
        print(f"total_surface_area: {self.get_total_surface_area()}", file=fp)
        print(f"total_part_volume: {self.get_total_part_volume()}", file=fp)
        print(f"total_support_volume: {self.get_total_support_volume()}", file=fp)
        print(f"Time: {calculate_batch_time(self)['total_time']}", file=fp)
        print(f"Energy: {calculate_batch_energy(self)['EPC']}", file=fp)
    
    def show_view(self, dir: str) -> None:
        """
        Print view of the batch in in terms of a matrix as an image
        """
        view = self.get_current_view(stretch=False, show=True)
        plt.figure(dpi=200, figsize=(6, 5))
        ax = plt.imshow(view, cmap="Blues")
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.grid(alpha=0.1)
        plt.savefig(dir)
        
        try:
            plt.close()
        except:
            pass
        
        del ax
         

class Solution:
    """
    A Solution to a instance contains multiple batches.
    """
    def __init__(self, 
                 view_shape: Tuple[int, int],
                 instance_name: str = "None",
                 ) -> None:
        self.instance_name = instance_name
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
        return 0 if self.empty() else sum(map(lambda x:calculate_batch_energy(x)["EPC"], self.batches)) / 1e6
    
    def show(self, out_dir: str = f"./solution/") -> None:
        """
        Display all the part_ls and view of all batches of the solution.
        """
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        
        with open(os.path.join(out_dir, "contains.txt"), 'w') as f:
            print(f"\n\t {self.instance_name} \n", file=f)
            
            print(f"Solution Time: {self.calculate_time()}", file=f)
            print(f"Solution Energy: {self.calculate_energy()}", file=f)
            
            for bid, b in enumerate(self.batches):
                print(f"{'=' * 30}\n\t\tBatch No. {bid}\n{'=' * 30}", file=f)
                b.show_parts(f)
                b.show_view(f"{out_dir}/batch_{bid:02d}.jpg")

    def empty(self) -> bool:
        return (len(self.batches) < 1) or all(map(lambda x:x.empty(), self.batches))


class BatchParallel1D(Batch):
    def __init__(self,
                 machine: Machine,
                 process: Process,
                 ) -> None:
        # Fetch LWH params from instance of class Machine
        #! Real space for bin packing: (L - 2*Margin) * (W - 2*Margin)
        self.L = machine.build_l - 2 * process.min_distance_part_platform
        self.W = machine.build_w - 2 * process.min_distance_part_platform
        self.H = machine.build_h
        
        # define self.machine and self.process for easier access of params
        self.machine = machine
        self.process = process
        
        self.parts_info: List[dict] = []
    
    @property
    def slice_number(self) -> float:
        return super().slice_number
    
    def get_current_view(self, show: bool=False) -> Tensor:
        return torch.tensor([self.get_occupied_ratio()]) if show \
                else torch.tensor([self.get_rest_area(), 
                             self.get_occupied_ratio(), 
                             self.get_largest_height()])
    
    def get_rest_area(self) -> float:
        return super().get_rest_area()
    
    def get_largest_height(self) -> float:
        if self.empty():
            return 0
        return max(self.parts_info, key=lambda x:x["H"])["H"]
    
    def get_occupied_ratio(self) -> float:
        return super().get_occupied_ratio()
    
    def get_total_surface_area(self) -> float:
        return super().get_total_surface_area()
    
    def get_total_part_volume(self) -> float:
        return super().get_total_part_volume()
        
    def get_total_support_volume(self) -> float:
        return super().get_total_support_volume()
    
    def add_part(self, part: Part, orientation: int) -> Tuple[Tensor, bool]:
        # Checks if the projection area of the part is smaller than the 
        # area available in this batch.
        
        try: 
            proj_area = part.get_proj_area(orientation)
        except IndexError:
            return None, False
        
        if proj_area > self.get_rest_area():
            return None, False
        else:
            self.parts_info.append(part.get_part_info(orientation))
            # Update the current view
            self.bin_view = self.get_current_view()
            
            return self.bin_view, True

    def empty(self) -> bool:
        return len(self.parts_info) < 1
    
    def show_parts(self, fp):
        return super().show_parts(fp)
    
    def show_view(self, dir: str) -> None:
        return
    

# TODO: Complete 1D parallel version of solution
class SolutionParallel1D(Solution):
    def __init__(self, 
                 instance_name: str = None,
                 batch_num: int = 100,
                 ) -> None:
        self.instance_name = instance_name
        self.batch_num = batch_num
        self.batches: List[BatchParallel1D] = []
    
    def init_batches(self, machine: Machine, process: Process) -> None:
        """
        Add an empty batch to the solution
        """
        self.batches = [
            BatchParallel1D(machine, process) for _ in range(self.batch_num)
        ]
    
    def get_batch(self, idx: None|int) -> BatchParallel1D:
        
        assert len(self.batches) > 0, "There is no batches in this solution!"
        if idx is None:
            return self.batches[-1]
        else:
            return self.batches[idx]
    
    def get_current_view(self, show: bool = False) -> Tensor:
    
        current_views = [
            b.get_current_view(show=show) \
                for b in self.batches
        ]
        
        # shape: [n, 3]
        return torch.stack(tensors=current_views, dim=0)
    
    def add_part(self, part: Part, orientation: int, idx: None|int) -> Tuple[Tensor, bool, float]:
        
        _, success = self.get_batch(idx).add_part(part, orientation)

        return (
            self.get_current_view(show=False), 
            success,
            0 if success else 0.01
        )
    
    def calculate_time(self) -> float:
        return 0 if self.empty() else sum(map(lambda x:calculate_batch_time(x)["total_time"], self.batches))
    
    def calculate_energy(self) -> float:
        return 0 if self.empty() else sum(map(lambda x:calculate_batch_energy(x)["EPC"], self.batches)) / 1e9
    
    def show(self, out_dir: str = f"./solution/") -> None:
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        
        with open(os.path.join(out_dir, "contains.txt"), 'w') as f:
            print(f"\n\t {self.instance_name} \n", file=f)
            
            print(f"Solution Time: {self.calculate_time()}", file=f)
            print(f"Solution Energy: {self.calculate_energy()}", file=f)
            
        batch_status = list(self.get_current_view(show=True)[:, -1])
        plt.bar(range(len(batch_status)), batch_status)
        plt.savefig(f"{out_dir}/batch_all.jpg")
        plt.close()

CONCENTRATION_OXYGEN_INITIAL = 21
CONCENTRATION_OXYGEN_END = 0.1

def calculate_batch_time(
    b: Batch|BatchParallel1D
    ) -> Dict[str, float]:
    
    if b.empty():
        return {"total_time": 0}
    
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
        return {"EPC": 0}
    
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
        [machine.power_subsystems["water_circulation_unit"]], 
        [machine.power_subsystems["water_cooling_unit"]], 
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
