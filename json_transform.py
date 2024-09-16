import json
import os
from tqdm import tqdm

def convert(s):
    try:
        return float(s)
    except ValueError:
        return None

source_dir = input("Please key in the source dir:\n")
target_dir = input("Please key in the target dir:\n")

for file_name in tqdm(os.listdir(source_dir)):

    print(file_name)
    
    fname = file_name.split('.')[0]
    
    with open(os.path.join(source_dir, file_name)) as f:
        lines = f.readlines()
    
    # ----------------------------------------------------------
    # parse lines into list of (lists of integers)
    parsed_list = [
        list(map(convert, line.strip().split(' '))) for line in lines
        ]
    
    # cut parsed list into slides indicating different
    # fields in the .json file
    instance_ls = parsed_list[0]
    machine_ls = parsed_list[2:15]
    process_ls = parsed_list[16]
    parts_ls = parsed_list[18:]
    
    # repacking the parts_ls by separating data belonging
    # to different parts
    temp_ls = []
    part_ls = []
    for line in parts_ls:
        if line[0] is not None:
            part_ls.append(line)
        else:
            temp_ls.append(part_ls)
            part_ls = []
    parts_ls_repacked = temp_ls
    
    # ----------------------------------------------------------
    # pack all information into sub-dictionaries
    instance_type = {
        "num_parts": instance_ls[0],
        "num_orientations": instance_ls[1],
        "type_parts": instance_ls[2]
    }

    column_names = [
                "preheating",
                "scanning_border",
                "filling_contour",
                "volume_hatching",
                "supports_soliding",
                "recoating",
                "cooling"
                ]

    machine_params = {
        "build_lwh": machine_ls[0],
        "power_subsystems": {
                "basic_subsystem": machine_ls[1][0],
                "platform_heater": machine_ls[1][1],
                "water_circulation_unit": machine_ls[1][2],
                "water_cooling_unit": machine_ls[1][3],
                "laser_scanning_border": machine_ls[1][4],
                "laser_filling_contour": machine_ls[1][5],
                "laser_volume_hatching": machine_ls[1][6],
                "laser_supports_building": machine_ls[1][7],
                "recoater_motor": machine_ls[1][8],
                "electric_valves": machine_ls[1][9],
                "gas_circulation_pump_motor": machine_ls[1][10]
        },
        "power_coefficient": {
            "column_names": column_names, 
            "rows": {
                "basic_subsystem": machine_ls[2], 
                "platform_heater": machine_ls[3], 
                "water_circulation_unit": machine_ls[4], 
                "water_cooling_unit": machine_ls[5], 
                "laser_scanning_border": machine_ls[6], 
                "laser_filling_contour": machine_ls[7], 
                "laser_volume_hatching": machine_ls[8], 
                "laser_supports_building": machine_ls[9], 
                "recoater_motor": machine_ls[10], 
                "electric_valves": machine_ls[11], 
                "gas_circulation_pump_motor": machine_ls[12],
            } 
        } 
    }

    process_params = {
        "min_distance_parts": process_ls[0], 
        "min_distance_part_platform": process_ls[1],
        "num_laser": process_ls[2],
        "hatch_distance_volume": process_ls[3],
        "hatch_distance_support": process_ls[4], 
        "laser_speed_border": process_ls[5],
        "laser_speed_contour": process_ls[6], 
        "laser_speed_volume": process_ls[7],
        "laser_speed_support": process_ls[8], 
        "layer_thickness": process_ls[9],
        "recoater_time_single": 11,
        "heat_time": process_ls[10],
        "cool_time": process_ls[11]
    }
    
    part_info = [
        {
            "part_type": title[0],
            "num_part": title[1],
            "volume": title[2],
            "surface_area": title[3],
            "build_params": [
                {
                    "O": idx, 
                    "L": item[0], 
                    "W": item[1], 
                    "H": item[2], 
                    "S": item[3]
                    }
                for idx,item in enumerate(data)
            ]
        }
        for title, *data in parts_ls_repacked
    ]
    
    # ----------------------------------------------------------
    # compose all we get into a json dictionary and store it in target dir
    json_dict = {
        "file_name": file_name,
        "instance_type": instance_type,
        "machine_params": machine_params,
        "process_params": process_params,
        "part_info": part_info
    }

    with open(f'{target_dir}/{fname}.json', 'w') as fp:
        json.dump(json_dict, fp)
