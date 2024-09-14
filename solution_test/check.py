import os, re
from tqdm import tqdm # type: ignore

for test_name in tqdm(filter(lambda x:"episode" in x, os.listdir("./solution_test/"))):
    
    with open(os.path.join("./solution_test/", test_name, "contains.txt")) as f:
        result_doc = f.read()
    
    # Time accumulation
    time_re = re.compile(pattern=r"(Solution )?Time: ([.0-9]+)\n")
    time_results = list(map(lambda x:float(x[-1]), time_re.findall(result_doc)))
    assert abs(time_results[0] - sum(time_results[1:])) <= 1e-6, \
        f"Time assertion fails at {test_name}"
    
    # Energy Accumulation
    energy_re = re.compile(pattern=r"(Solution )?Energy: ([.0-9]+)\n")
    energy_results = list(map(lambda x:float(x[-1]), energy_re.findall(result_doc)))
    assert abs(energy_results[0] - sum(energy_results[1:])) <= 1e-6, \
        f"Energy assertion fails at {test_name}"

    batch_info = list(map(lambda x:x.strip(), 
                          re.split(r"=+\n\s+Batch No. \d+\s*\n\s*=+", result_doc)))[1:]
    sum_map_findall = lambda k: list(map(lambda x:sum(map(float, re.findall(k[0], x))), 
                            k[1]))
    
    # Support Accumulation
    assert sum(map(lambda x:abs(x[0]-x[1]), 
                   zip(sum_map_findall((r"'S': ([\d.]+)", batch_info)), 
                       sum_map_findall((r"total_support_volume: ([\d.]+)", batch_info))
                       ))) <= 1e-5, f"Support accumulation fails at {test_name}"
    
    # Surface Area Accumulation
    assert sum(map(lambda x:abs(x[0]-x[1]), 
                zip(sum_map_findall((r"'surface_area': ([\d.]+)", batch_info)), 
                    sum_map_findall((r"total_surface_area: ([\d.]+)", batch_info))
                    ))) <= 1e-5, f"Support accumulation fails at {test_name}"
    
    # Volume Accumulation
    assert sum(map(lambda x:abs(x[0]-x[1]), 
                zip(sum_map_findall((r"'volume': ([\d.]+)", batch_info)), 
                    sum_map_findall((r"total_part_volume: ([\d.]+)", batch_info))
                    ))) <= 1e-5, f"Support accumulation fails at {test_name}"
    

print("All tests passed")