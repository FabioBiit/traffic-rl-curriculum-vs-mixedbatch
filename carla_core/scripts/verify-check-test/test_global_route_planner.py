import sys
sys.path.insert(0, r"c:\Users\kyros\OneDrive\Desktop\TESI_PROJECT_traffic-rl-curriculum-vs-mixedbatch\traffic-rl-curriculum-vs-mixedbatch")

import carla
import numpy as np
from carla_core.envs.route_planner import CARLARoutePlanner

client = carla.Client("127.0.0.1", 2000)
client.set_timeout(20.0)
world = client.get_world()
carla_map = world.get_map()
spawn_points = carla_map.get_spawn_points()

planner = CARLARoutePlanner(carla_map, sampling_resolution=2.0)
origin = spawn_points[0].location

veh_route = planner.plan_vehicle_route(origin, 200.0, spawn_points, np.random.default_rng(42))
ped_route = planner.plan_pedestrian_route_by_distance(origin, 40.0, carla_map)

print("vehicle route ok:", veh_route is not None, "wps:", 0 if veh_route is None else len(veh_route))
print("ped route ok:", ped_route is not None, "wps:", 0 if ped_route is None else len(ped_route))
