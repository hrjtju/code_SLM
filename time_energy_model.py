# calculate time cost
from typing import Dict, Union
import pandas as pd
from slm_env import Batch
from slm_classes import Machine, Process
import numpy as np

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
    
    scanning_border_time = b.get_total_surface_area() / (process.num_laser \
        * process.laser_speed_border * process.laser_thickness)
        
    fill_contour_time = b.get_total_surface_area() / (process.num_laser \
        * process.laser_speed_contour * process.laser_thickness)
        
    volume_hatching_time = b.get_total_part_volume() / (process.num_laser \
        * process.laser_speed_volume * process.laser_thickness * process.hatch_distance_volume)
        
    support_building_time = b.get_total_support_volume() / (process.num_laser \
        * process.laser_speed_support * process.laser_thickness * process.hatch_distance_support)
    
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
    EPC = P.T @ K @ T # An number
    
    # PKT[i][j] = P[i] K[i][j] T[j]
    # Hadamard product with broadcasting
    # Equivalent to the following
    # for i in range(11):
    #     K_row=[]
    #     for j in range(7):
    #         K_row.append(int(P[i] * K[i][j] * T[j]))
    #     PKT.append(K_row)
    PKT = (P.T * K * T).astype(np.int64)
    
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
