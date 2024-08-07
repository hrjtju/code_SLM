# calculate time cost
from slm_env import Batch
from slm_classes import Machine, Process
import numpy as np

CONCENTRATION_OXYGEN_INITIAL = 21
CONCENTRATION_OXYGEN_END = 0.1

def calculate_batch_time(b: Batch) -> float:
    process = b.process
    
    # TODO: Check if the commented lines are needed
    # t0 = Machine.C * np.log(CONCENTRATION_OXYGEN_INITIAL) - Machine.D
    # t1 = Machine.C * np.log(CONCENTRATION_OXYGEN_END) - Machine.D
    # delta_t_oxygen = t0 - t1
    
    delta_t_heater = process.heat_time
    delta_t_cooling = process.cool_time
    
    heater_time = delta_t_heater
    
    scanning_border_time = b.get_total_surface_area() / (process.num_laser \
        * process.laser_speed_border * process.laser_thickness)
        
    fill_contour_time = b.get_total_surface_area() / (process.num_laser \
        * process.laser_speed_contour * process.laser_thickness)
        
    volume_hatching_time = b.get_total_part_volume() / (process.num_laser \
        * process.laser_speed_volume * process.laser_thickness * process.hatch_distance_volume)
        
    support_building_time = b.get_total_support_volume() / (process.num_laser \
        * process.laser_speed_support * process.laser_thickness * process.hatch_distance_support)
    
    # TODO: Find out where `recoater_time_single` and `slice_number` are
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
    

def calculate_batch_energy(b: Batch) -> float:
    ...
