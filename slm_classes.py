from typing import List, Self
import numpy as np
import pandas as pd
import json
from pprint import pprint
from operator import add
from functools import reduce

import torch
from torch import Tensor

class ItemFromJson:
    def __init__(self) -> None:
        ...
    
    def get_from_dict(self, d:dict) -> None:
        for k,v in d.items():
            setattr(self, k, v)
    
    def show(self) -> None:
        # pprint(self.__dict__.items())
        
        pprint(list(map(lambda x: (x[0], x[1].__dict__) if isinstance(x[1], ItemFromJson)\
                            else ((x[0], [k.__dict__ for k in x[1]]) if isinstance(x[1], List)\
                                else x), 
                        self.__dict__.items())))

# instance class. for getting instance info from json file.
class Instance(ItemFromJson):
    def __init__(self) -> None:
        self.num_parts = None
        self.num_orientations = None
        self.type_parts = None

# collects machine params
class Machine(ItemFromJson):
    def __init__(self) -> None:
        self.build_l, self.build_w, self.build_h = None, None, None
        self.power_subsystems = dict()
        self.power_coefficient = None
    
    def get_from_dict(self, d:dict) -> None:
        self.build_l, self.build_w, self.build_h = d.get("build_lwh")
        
        for k, v in d.get("power_subsystems").items():
            self.power_subsystems[k] = v
        
        columns = d.get("power_coefficient").get("column_names")
        
        self.power_coefficient = pd.DataFrame.from_dict(
            d.get("power_coefficient").get("rows"),
            columns=columns,
            orient="index"
        )
    
    def get_power_coeff_ij(row, col) -> float:
        ...

# collects process params
class Process(ItemFromJson):
    def __init__(self):
        self.min_distance_parts = None 
        self.min_distance_part_platform = None
        self.num_laser = None
        self.hatch_distance_volume = None
        self.hatch_distance_support = None 
        self.laser_speed_border = None
        self.laser_speed_contour = None 
        self.laser_speed_volume = None
        self.laser_speed_support = None 
        self.layer_thickness = None
        self.heat_time = None
        self.cool_time = None

# collects params for each part
class Part(ItemFromJson):
    def __init__(self, d: dict) -> None:
        self.part_type = None
        self.num_part = None
        self.volume = None
        self.surface_area = None
        self.build_params: List[dict] = None
        # self.orientation = None
        
        self.get_from_dict(d)
    
    # def set_orientation(self, orientation: int) -> Self:
    #     self.orientation = orientation
    #     return self
    
    # def get_proj_area(self, orientation: int, gap: float) -> float:
    #     """
    #     Returns the projection area given orientation.
    #     """
    #     # TODO: Check if this statement is correct
    #     return (self.build_params[orientation]["L"]+gap) * (self.build_params[orientation]["W"]+gap)
    
    def get_part_info(self, orientation: int, gap: float) -> dict:
        """
        Returns a specific dict when orientation is specified for batch state.
        """
        build_param = self.build_params[orientation]
        return {
            "volume": self.volume,
            "surface_ares": self.surface_area,
            "L": build_param["L"]+gap,
            "W": build_param["W"]+gap,
            "H": build_param["H"],
            "S": build_param["S"],
        }      

# collects all data in one json file.
class MetaData(ItemFromJson):
    def __init__(self, max_part_type: int = 20, max_orientation_num: int = 7) -> None:
        self.max_part_type = max_part_type
        self.max_orientation_num = max_orientation_num
        
        self.instance = None
        self.machine = None
        self.process = None
        self.parts: List[Part] = None
        
        self.mask_mtx = None
    
    def load(self, 
             instance: Instance, 
             machine: Machine, 
             process: Process, 
             parts: List[Part],
             ) -> None:
        self.instance = instance
        self.machine = machine
        self.process = process
        self.parts = parts
        
        self.mask_mtx = self.mask_matrix()
        
    def get_from_dict(self, d: dict) -> None:
        raise NotImplementedError
    
    # get a vector indicating the part information
    def part_vec(self, part_id: int) -> Tensor:
        max_vec_len = 3 + 4*self.max_orientation_num
        
        if part_id >= len(self.parts):
            return torch.zeros(int(max_vec_len))
        
        selected_part = self.parts[part_id]
        
        info_list = [selected_part.num_part, selected_part.volume, selected_part.surface_area]\
                        + list(reduce(add, ([ori['L'], ori['W'], ori['H'], ori['S']] for ori in selected_part.build_params)))
        
        info_list.extend([0 for _ in range(max_vec_len - len(info_list))])
        
        return torch.tensor(info_list)
                        
    def mask_vec(self, part_id: int) -> Tensor:
        out_vec = torch.zeros(self.max_orientation_num)
        
        if part_id >= len(self.parts):
            return out_vec
        
        orient_num = len(self.parts[part_id].build_params)
        out_vec[:orient_num] = 1
        return out_vec
    
    # get mask_dict of this instance
    def init_state(self) -> Tensor:
        return torch.stack([self.part_vec(i) for i in range(self.max_part_type)], dim=0)
    
    def mask_matrix(self) -> Tensor:
        return torch.stack([self.mask_vec(i) for i in range(self.max_part_type)], dim=0)

def load_json_to_class(path: str) -> MetaData:
    with open(path, 'r') as f:
        json_dict = json.load(f)
    
    instance_json = Instance()
    machine_json = Machine()
    process_json = Process()
    
    instance_json.get_from_dict(json_dict["instance_type"])
    machine_json.get_from_dict(json_dict["machine_params"])
    process_json.get_from_dict(json_dict["process_params"])
    parts = [Part(d) for d in json_dict["part_info"]]
    
    metadata = MetaData()
    metadata.load(instance=instance_json, 
                  machine=machine_json,
                  process=process_json,
                  parts=parts)
    
    return metadata

if __name__ == "__main__":
    
    torch.set_printoptions(precision=1, sci_mode=False)
    
    metadata = load_json_to_class(r"./instances_json/ec_30-1.json")
    
    print(metadata.init_state())
    print(metadata.mask_mtx)
    