from typing import List
import numpy as np
import pandas as pd
import json
from pprint import pprint

class ItemFromJson:
    def __init__(self) -> None:
        ...
    
    def get_from_dict(self, d:dict) -> None:
        for k,v in d.items():
            setattr(self, k, v)
    
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
    def __init__(self) -> None:
        self.part_type = None
        self.num_part = None
        self.volume = None
        self.surface_area = None
        self.build_params: List[dict] = None

# collects all data in one json file.
class MetaData:
    def __init__(self) -> None:
        self.instance = Instance()
        self.machine = Machine()
        self.process = Process()
        self.parts: List[Part] = []
    
    def load(self, d:dict) -> None:
        ...



if __name__ == "__main__":
    
    with open("./test.json", 'r') as f:
        json_dict = json.load(f)
    
    pprint(json_dict)
    
    instance_json = Instance()
    machine_json = Machine()
    process_json = Process()
    
    instance_json.get_from_dict(json_dict["instance_type"])
    machine_json.get_from_dict(json_dict["machine_params"])
    process_json.get_from_dict(json_dict["process_params"])
    