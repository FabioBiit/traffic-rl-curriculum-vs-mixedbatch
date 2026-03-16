"""
CarlaEnv v0.1 — Wrapper Gymnasium per CARLA (Single-Agent)
==========================================================
Compatibile con gymnasium.Env e RLlib.

Observation space (vector, dim=24):
    [0:3]   ego velocity (vx, vy, vz) m/s
    [3:6]   ego acceleration (ax, ay, az) m/s²
    [6]     ego speed (scalar, km/h normalized 0-1)
    [7:9]   ego forward vector (fx, fy) — heading direction
    [9]     distance to next waypoint (m, normalized)
    [10]    angle to next waypoint (rad, normalized -1..1)
    [11]    road lane offset (m, normalized -1..1)
    [12]    fraction of route completed (0..1)
    [13]    is_at_traffic_light (0 or 1)
    [14]    traffic_light_state (0=green,0.5=yellow,1=red)
    [15:24] nearest 3 vehicles: (rel_x, rel_y, rel_speed) × 3

Action space (continuous, dim=2):
    [0] throttle/brake: -1 (full brake) .. +1 (full throttle)
    [1] steer: -1 (full left) .. +1 (full right)

Requires:
    - CARLA server running
    - pip install carla==0.9.16 gymnasium pyyaml numpy
"""

import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import yaml

try:
    import carla
except ImportError:
    raise ImportError("carla package not found. Install with: pip install carla==0.9.16")


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

DEFAULT_ENV_CONFIG = {
    "simulator": {
        "host": "127.0.0.1",
        "port": 2000,
        "timeout_seconds": 20.0,
        "sync_mode": True,
        "fixed_delta_seconds": 0.05,
    },
    "world": {
        "map": "Town03",
        "weather_preset": "ClearNoon",
        "no_rendering": True,
    },
    "traffic": {
        "enabled": True,
        "n_vehicles": 40,
        "n_pedestrians": 20,
        "seed": 42,
    },
    "ego": {
        "spawn_mode": "random",
        "spawn_index": 0,
        "blueprint_filter": "vehicle.tesla.model3",
        "role_name": "hero",
    },
    "wrapper": {
        "obs_type": "vector",
        "action_type": "continuous",
        "frame_stack": 1,
        "normalize_obs": True,
    },
    "episode": {
        "max_steps": 1000,
        "terminate_on_collision": True,
        "terminate_on_offroad": True,
        "terminate_on_route_completion": True,
    },
}


def _merge_nested_dict(base: dict, override: dict):
    """Recursively merge override into base."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _merge_nested_dict(base[k], v)
        else:
            base[k] = v


def load_env_config(config_path: Optional[str] = None) -> dict:
    """Load env.yaml and merge with defaults for missing keys."""
    if config_path is None:
        # Look relative to this file's location
        config_path = Path(__file__).resolve().parent.parent / "configs" / "env.yaml"

    config = deepcopy(DEFAULT_ENV_CONFIG)

    if Path(config_path).exists():
        with open(config_path, "r") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Invalid CARLA env config format in {config_path}: expected mapping.")
        _merge_nested_dict(config, loaded)

    return config


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

# Obs dimensions
N_NEARBY_VEHICLES = 3
OBS_DIM = 15 + N_NEARBY_VEHICLES * 3  # 24


class CarlaEnv(gym.Env):
    """Gymnasium-compatible single-agent CARLA environment."""

    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(self, config: Optional[dict] = None, config_path: Optional[str] = None):
        super().__init__()

        # Merge config: explicit dict > yaml file > defaults
        file_cfg = load_env_config(config_path)
        if config:
            # Allow RLlib to pass env_config dict
            self._merge_dict(file_cfg, config)
        self.cfg = file_cfg

        sim = self.cfg["simulator"]
        self.ep_cfg = self.cfg["episode"]

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32,
        )

        # CARLA connection (lazy — connects on first reset)
        self._client: Optional[carla.Client] = None
        self._world: Optional[carla.World] = None
        self._map: Optional[carla.Map] = None
        self._tm: Optional[carla.TrafficManager] = None
        self._ego: Optional[carla.Vehicle] = None
        self._collision_sensor: Optional[carla.Actor] = None
        self._npc_vehicles: list = []
        self._npc_walkers: list = []
        self._walker_controllers: list = []
        self._spawn_points: list = []
        self._route_waypoints: list = []
        self._current_wp_idx: int = 0

        # Episode state
        self._step_count: int = 0
        self._collision_flag: bool = False
        self._collision_intensity: float = 0.0
        self._original_settings: Optional[carla.WorldSettings] = None
        self._original_tm_sync_mode: Optional[bool] = None
        self._connected: bool = False

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        if not self._connected:
            self._connect()

        self._cleanup_actors()
        self._setup_ego()
        self._setup_collision_sensor()
        self._setup_route()

        if self.cfg["traffic"]["enabled"]:
            self._spawn_traffic()

        # Warm up — let physics settle
        for _ in range(10):
            self._world.tick()

        self._step_count = 0
        self._collision_flag = False
        self._collision_intensity = 0.0

        obs = self._get_obs()
        info = {"route_length": len(self._route_waypoints)}
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._apply_action(action)
        self._world.tick()
        self._step_count += 1

        obs = self._get_obs()
        reward = self._compute_reward()
        terminated, truncated = self._check_done()

        info = {
            "step": self._step_count,
            "speed_kmh": self._get_speed_kmh(),
            "collision": self._collision_flag,
            "route_completion": self._route_completion(),
        }

        return obs, reward, terminated, truncated, info

    def close(self):
        self._cleanup_actors()
        if self._tm is not None:
            try:
                self._tm.set_synchronous_mode(
                    self._original_tm_sync_mode
                    if self._original_tm_sync_mode is not None
                    else False
                )
            except Exception:
                pass
        if self._world is not None and self._original_settings is not None:
            self._world.apply_settings(self._original_settings)
        self._connected = False

    # ------------------------------------------------------------------
    # Connection & setup
    # ------------------------------------------------------------------

    def _connect(self):
        sim = self.cfg["simulator"]
        self._client = carla.Client(sim["host"], sim["port"])
        self._client.set_timeout(sim["timeout_seconds"])

        # Load map
        target_map = self.cfg["world"]["map"]
        current_map = self._client.get_world().get_map().name.split("/")[-1]
        if current_map != target_map:
            self._world = self._client.load_world(target_map)
        else:
            self._world = self._client.get_world()

        self._map = self._world.get_map()
        self._spawn_points = self._map.get_spawn_points()

        # Sync mode
        self._original_settings = self._world.get_settings()
        self._original_tm_sync_mode = bool(self._original_settings.synchronous_mode)
        settings = self._world.get_settings()
        settings.synchronous_mode = sim["sync_mode"]
        settings.fixed_delta_seconds = sim["fixed_delta_seconds"]
        if self.cfg["world"].get("no_rendering", False):
            settings.no_rendering_mode = True
        self._world.apply_settings(settings)

        # Traffic manager
        self._tm = self._client.get_trafficmanager(8000)
        self._tm.set_synchronous_mode(bool(sim.get("sync_mode", True)))
        self._tm.set_global_distance_to_leading_vehicle(2.5)
        self._tm.set_random_device_seed(self.cfg["traffic"].get("seed", 42))

        # Weather
        weather_preset = self.cfg["world"].get("weather_preset", "ClearNoon")
        if hasattr(carla.WeatherParameters, weather_preset):
            self._world.set_weather(getattr(carla.WeatherParameters, weather_preset))

        self._connected = True

    def _setup_ego(self):
        ego_cfg = self.cfg["ego"]
        bp_lib = self._world.get_blueprint_library()
        bp = bp_lib.find(ego_cfg["blueprint_filter"])
        bp.set_attribute("role_name", ego_cfg["role_name"])

        if ego_cfg["spawn_mode"] == "fixed_index":
            transform = self._spawn_points[ego_cfg["spawn_index"]]
        else:
            transform = self.np_random.choice(self._spawn_points)

        self._ego = self._world.spawn_actor(bp, transform)
        self._world.tick()  # Register actor

    def _setup_collision_sensor(self):
        bp = self._world.get_blueprint_library().find("sensor.other.collision")
        transform = carla.Transform(carla.Location(z=1.0))
        self._collision_sensor = self._world.spawn_actor(bp, transform, attach_to=self._ego)
        self._collision_sensor.listen(self._on_collision)

    def _on_collision(self, event):
        self._collision_flag = True
        impulse = event.normal_impulse
        self._collision_intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)

    def _setup_route(self, route_length: int = 50):
        """Generate a route of waypoints from ego's current position."""
        ego_loc = self._ego.get_location()
        current_wp = self._map.get_waypoint(ego_loc, project_to_road=True)

        self._route_waypoints = [current_wp]
        for _ in range(route_length):
            nexts = self._route_waypoints[-1].next(5.0)  # 5m ahead
            if not nexts:
                break
            self._route_waypoints.append(nexts[0])  # Follow first lane
        self._current_wp_idx = 0

    def _spawn_traffic(self):
        """Spawn NPC vehicles and pedestrians with autopilot."""
        bp_lib = self._world.get_blueprint_library()
        t_cfg = self.cfg["traffic"]

        # Vehicles
        vehicle_bps = [bp for bp in bp_lib.filter("vehicle.*")
                       if int(bp.get_attribute("number_of_wheels").as_int()) >= 4]
        available_points = [sp for sp in self._spawn_points
                            if sp.location.distance(self._ego.get_location()) > 20.0]
        self.np_random.shuffle(available_points)

        for i in range(min(t_cfg["n_vehicles"], len(available_points))):
            bp = self.np_random.choice(vehicle_bps)
            if bp.has_attribute("color"):
                bp.set_attribute("color",
                    self.np_random.choice(bp.get_attribute("color").recommended_values))
            npc = self._world.try_spawn_actor(bp, available_points[i])
            if npc:
                npc.set_autopilot(True, self._tm.get_port())
                self._npc_vehicles.append(npc)

        # Pedestrians
        walker_bps = bp_lib.filter("walker.pedestrian.*")
        walker_ctrl_bp = bp_lib.find("controller.ai.walker")

        for _ in range(t_cfg["n_pedestrians"]):
            loc = self._world.get_random_location_from_navigation()
            if loc is None:
                continue
            bp = self.np_random.choice(list(walker_bps))
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")
            walker = self._world.try_spawn_actor(bp, carla.Transform(loc))
            if walker:
                self._npc_walkers.append(walker)

        self._world.tick()

        for walker in self._npc_walkers:
            ctrl = self._world.try_spawn_actor(walker_ctrl_bp, carla.Transform(), walker)
            if ctrl:
                self._walker_controllers.append(ctrl)

        self._world.tick()

        for ctrl in self._walker_controllers:
            target = self._world.get_random_location_from_navigation()
            if target:
                ctrl.start()
                ctrl.go_to_location(target)
                ctrl.set_max_speed(1.4)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _apply_action(self, action: np.ndarray):
        throttle_brake = float(np.clip(action[0], -1.0, 1.0))
        steer = float(np.clip(action[1], -1.0, 1.0))

        control = carla.VehicleControl()
        if throttle_brake >= 0:
            control.throttle = throttle_brake
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = -throttle_brake
        control.steer = steer
        control.hand_brake = False
        control.reverse = False

        self._ego.apply_control(control)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(OBS_DIM, dtype=np.float32)

        transform = self._ego.get_transform()
        velocity = self._ego.get_velocity()
        accel = self._ego.get_acceleration()
        fwd = transform.get_forward_vector()

        # Ego velocity (normalized by ~30 m/s ≈ 108 km/h)
        obs[0] = velocity.x / 30.0
        obs[1] = velocity.y / 30.0
        obs[2] = velocity.z / 30.0

        # Ego acceleration (normalized by ~10 m/s²)
        obs[3] = accel.x / 10.0
        obs[4] = accel.y / 10.0
        obs[5] = accel.z / 10.0

        # Speed scalar (km/h normalized)
        speed_kmh = self._get_speed_kmh()
        obs[6] = min(speed_kmh / 120.0, 1.0)

        # Forward vector (heading)
        obs[7] = fwd.x
        obs[8] = fwd.y

        # Waypoint tracking
        self._advance_waypoint()
        if self._current_wp_idx < len(self._route_waypoints):
            target_wp = self._route_waypoints[self._current_wp_idx]
            target_loc = target_wp.transform.location
            ego_loc = transform.location

            # Distance to next waypoint (normalized by 50m)
            dist = ego_loc.distance(target_loc)
            obs[9] = min(dist / 50.0, 1.0)

            # Angle to next waypoint (normalized to -1..1)
            dx = target_loc.x - ego_loc.x
            dy = target_loc.y - ego_loc.y
            target_angle = math.atan2(dy, dx)
            ego_yaw = math.radians(transform.rotation.yaw)
            angle_diff = target_angle - ego_yaw
            # Normalize to [-pi, pi]
            angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
            obs[10] = angle_diff / math.pi

            # Lane offset (normalized by 4m — typical lane width)
            wp_at_ego = self._map.get_waypoint(ego_loc, project_to_road=True)
            if wp_at_ego:
                lane_offset = ego_loc.distance(wp_at_ego.transform.location)
                # Determine sign via cross product
                wp_fwd = wp_at_ego.transform.get_forward_vector()
                cross = (ego_loc.x - wp_at_ego.transform.location.x) * wp_fwd.y - \
                        (ego_loc.y - wp_at_ego.transform.location.y) * wp_fwd.x
                obs[11] = np.clip(math.copysign(lane_offset, cross) / 4.0, -1.0, 1.0)

        # Route completion
        obs[12] = self._route_completion()

        # Traffic light
        if self._ego.is_at_traffic_light():
            obs[13] = 1.0
            tl = self._ego.get_traffic_light()
            if tl:
                state = tl.get_state()
                if state == carla.TrafficLightState.Red:
                    obs[14] = 1.0
                elif state == carla.TrafficLightState.Yellow:
                    obs[14] = 0.5
                else:
                    obs[14] = 0.0

        # Nearby vehicles
        self._fill_nearby_vehicles(obs, start_idx=15)

        return np.clip(obs, -1.0, 1.0)

    def _fill_nearby_vehicles(self, obs: np.ndarray, start_idx: int):
        """Fill obs with relative position/speed of N nearest vehicles."""
        ego_loc = self._ego.get_location()
        ego_vel = self._ego.get_velocity()
        ego_speed = math.sqrt(ego_vel.x**2 + ego_vel.y**2)

        # Get all vehicles except ego
        vehicles = self._world.get_actors().filter("vehicle.*")
        dists = []
        for v in vehicles:
            if v.id == self._ego.id:
                continue
            d = ego_loc.distance(v.get_location())
            if d < 50.0:  # 50m detection radius
                dists.append((d, v))

        dists.sort(key=lambda x: x[0])

        ego_fwd = self._ego.get_transform().get_forward_vector()
        ego_right_x = -ego_fwd.y
        ego_right_y = ego_fwd.x

        for i, (dist, v) in enumerate(dists[:N_NEARBY_VEHICLES]):
            v_loc = v.get_location()
            dx = v_loc.x - ego_loc.x
            dy = v_loc.y - ego_loc.y

            # Relative position in ego frame (forward/right)
            rel_fwd = dx * ego_fwd.x + dy * ego_fwd.y
            rel_right = dx * ego_right_x + dy * ego_right_y

            # Relative speed
            v_vel = v.get_velocity()
            v_speed = math.sqrt(v_vel.x**2 + v_vel.y**2)
            rel_speed = v_speed - ego_speed

            idx = start_idx + i * 3
            obs[idx] = np.clip(rel_fwd / 50.0, -1.0, 1.0)
            obs[idx + 1] = np.clip(rel_right / 50.0, -1.0, 1.0)
            obs[idx + 2] = np.clip(rel_speed / 30.0, -1.0, 1.0)

    def _advance_waypoint(self):
        """Advance route waypoint index if ego is close enough."""
        if self._current_wp_idx >= len(self._route_waypoints):
            return
        ego_loc = self._ego.get_location()
        wp_loc = self._route_waypoints[self._current_wp_idx].transform.location
        if ego_loc.distance(wp_loc) < 5.0:
            self._current_wp_idx += 1

    def _route_completion(self) -> float:
        if len(self._route_waypoints) == 0:
            return 0.0
        return min(self._current_wp_idx / len(self._route_waypoints), 1.0)

    def _get_speed_kmh(self) -> float:
        v = self._ego.get_velocity()
        return 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self) -> float:
        """
        Reward v1 — dense, shaped:
            + speed_along_route    (incentiva movimento verso il goal)
            + waypoint_progress    (bonus per raggiungimento waypoint)
            - collision_penalty
            - lane_deviation
            - excessive_steer      (comfort/smoothness)
        """
        reward = 0.0

        # 1. Speed along route direction
        speed_kmh = self._get_speed_kmh()
        if self._current_wp_idx < len(self._route_waypoints):
            target_loc = self._route_waypoints[self._current_wp_idx].transform.location
            ego_loc = self._ego.get_location()
            ego_vel = self._ego.get_velocity()

            # Direction to waypoint
            dx = target_loc.x - ego_loc.x
            dy = target_loc.y - ego_loc.y
            dist = math.sqrt(dx**2 + dy**2)
            if dist > 0.01:
                dir_x, dir_y = dx / dist, dy / dist
                # Project velocity onto direction
                speed_along = ego_vel.x * dir_x + ego_vel.y * dir_y
                reward += np.clip(speed_along / 10.0, -0.5, 1.0)  # max ~1.0 at 36 km/h

        # 2. Waypoint progress bonus
        # Handled via _advance_waypoint: if wp_idx increased this step
        # We approximate by checking if route_completion changed
        reward += self._route_completion() * 0.1  # Small continuous bonus

        # 3. Collision penalty
        if self._collision_flag:
            reward -= 10.0

        # 4. Lane deviation penalty
        ego_loc = self._ego.get_location()
        wp = self._map.get_waypoint(ego_loc, project_to_road=True)
        if wp:
            lane_dev = ego_loc.distance(wp.transform.location)
            reward -= np.clip(lane_dev / 4.0, 0.0, 1.0) * 0.3

        # 5. Steer smoothness penalty
        ctrl = self._ego.get_control()
        reward -= abs(ctrl.steer) * 0.05

        return float(reward)

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def _check_done(self) -> tuple[bool, bool]:
        terminated = False
        truncated = False

        # Collision
        if self.ep_cfg["terminate_on_collision"] and self._collision_flag:
            terminated = True

        # Route completed
        if self.ep_cfg["terminate_on_route_completion"]:
            if self._current_wp_idx >= len(self._route_waypoints):
                terminated = True

        # Off-road
        if self.ep_cfg["terminate_on_offroad"]:
            ego_loc = self._ego.get_location()
            wp = self._map.get_waypoint(ego_loc, project_to_road=True)
            if wp is None or ego_loc.distance(wp.transform.location) > 5.0:
                terminated = True

        # Max steps
        if self._step_count >= self.ep_cfg["max_steps"]:
            truncated = True

        return terminated, truncated

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _cleanup_actors(self):
        """Destroy all spawned actors safely."""
        for ctrl in self._walker_controllers:
            try:
                ctrl.stop()
                ctrl.destroy()
            except Exception:
                pass
        self._walker_controllers.clear()

        for w in self._npc_walkers:
            try:
                w.destroy()
            except Exception:
                pass
        self._npc_walkers.clear()

        for v in self._npc_vehicles:
            try:
                v.destroy()
            except Exception:
                pass
        self._npc_vehicles.clear()

        if self._collision_sensor and self._collision_sensor.is_alive:
            self._collision_sensor.destroy()
            self._collision_sensor = None

        if self._ego and self._ego.is_alive:
            self._ego.destroy()
            self._ego = None

        self._route_waypoints.clear()
        self._current_wp_idx = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_dict(base: dict, override: dict):
        """Recursively merge override into base."""
        for k, v in override.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                CarlaEnv._merge_dict(base[k], v)
            else:
                base[k] = v
