"""
CarlaEnv v0.3 — Wrapper Gymnasium per CARLA (Single-Agent)
==========================================================
Changelog v0.3:
  - Reward v2: speed limit penalty, waypoint reach bonus +5,
    throttle jerk penalty, increased lane/steer penalties
  - Nuovi campi episodio: _prev_wp_idx, _prev_throttle

Observation space (vector, dim=24):
    [0:3]   ego velocity (vx, vy, vz) m/s
    [3:6]   ego acceleration (ax, ay, az) m/s²
    [6]     ego speed (scalar, km/h normalized 0-1)
    [7:9]   ego forward vector (fx, fy)
    [9]     distance to next waypoint (normalized)
    [10]    angle to next waypoint (normalized -1..1)
    [11]    road lane offset (normalized -1..1)
    [12]    fraction of route completed (0..1)
    [13]    is_at_traffic_light (0 or 1)
    [14]    traffic_light_state (0=green,0.5=yellow,1=red)
    [15:24] nearest 3 vehicles: (rel_x, rel_y, rel_speed) × 3

Action space (continuous, dim=2):
    [0] throttle/brake: -1..+1
    [1] steer: -1..+1
"""

import logging
import math
import signal
import sys
from copy import deepcopy
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import yaml

try:
    import carla
except ImportError:
    raise ImportError("pip install carla==0.9.16")

logger = logging.getLogger(__name__)

# Suppress CARLA C++ destroy warnings on Windows
if sys.platform == "win32":
    signal.signal(signal.SIGABRT, signal.SIG_IGN)

# ---------------------------------------------------------------------------
# Config
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
        "n_vehicles": 10, 
        "n_pedestrians": 10,
        "seed": 42, 
        "persist_traffic": True,
    },
    "ego": {
        "spawn_mode": "random", 
        "spawn_index": 0,
        "blueprint_filter": 
        "vehicle.ford.mustang",
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


def _merge(base: dict, override: dict):
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _merge(base[k], v)
        else:
            base[k] = v


def load_env_config(config_path: Optional[str] = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "configs" / "env.yaml"
    config = deepcopy(DEFAULT_ENV_CONFIG)
    p = Path(config_path)
    if p.exists():
        with open(p) as f:
            loaded = yaml.safe_load(f) or {}
        _merge(config, loaded)
    return config


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

N_NEARBY_VEHICLES = 3
OBS_DIM = 15 + N_NEARBY_VEHICLES * 3  # 24


class CarlaEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(self, config: Optional[dict] = None, config_path: Optional[str] = None):
        super().__init__()
        file_cfg = load_env_config(config_path)
        if config:
            _merge(file_cfg, config)
        self.cfg = file_cfg
        self.ep_cfg = self.cfg["episode"]

        # Fix RLlib horizon warning
        self.spec = type("Spec", (), {
            "id": "CarlaEnv-v0",
            "max_episode_steps": self.ep_cfg["max_steps"],
        })()

        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]),
            dtype=np.float32,
        )

        # CARLA state (lazy connect)
        self._client = None
        self._world = None
        self._map = None
        self._tm = None
        self._ego = None
        self._collision_sensor = None
        self._npc_vehicles: list = []
        self._npc_walkers: list = []
        self._walker_controllers: list = []
        self._spawn_points: list = []
        self._route_waypoints: list = []
        self._current_wp_idx: int = 0
        self._step_count: int = 0
        self._collision_flag: bool = False
        self._collision_intensity: float = 0.0
        self._prev_wp_idx: int = 0
        self._prev_throttle: float = 0.0
        self._original_settings = None
        self._connected: bool = False
        self._traffic_spawned: bool = False

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if not self._connected:
            self._connect()

        self._cleanup_ego()
        self._setup_ego()
        self._setup_collision_sensor()
        self._setup_route()

        # Spawn traffic once if persist_traffic=True
        if self.cfg["traffic"]["enabled"]:
            if not self._traffic_spawned or not self.cfg["traffic"].get("persist_traffic", True):
                self._cleanup_traffic()
                self._spawn_traffic()
                self._traffic_spawned = True

        for _ in range(10):
            self._world.tick()

        self._step_count = 0
        self._collision_flag = False
        self._collision_intensity = 0.0
        self._prev_wp_idx = 0
        self._prev_throttle = 0.0
        return self._get_obs(), {"route_length": len(self._route_waypoints)}

    def step(self, action):
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
        self._cleanup_ego()
        self._cleanup_traffic()
        if self._tm:
            try:
                self._tm.set_synchronous_mode(False)
            except Exception:
                pass
        if self._world and self._original_settings:
            try:
                self._world.apply_settings(self._original_settings)
            except Exception:
                pass
        self._connected = False

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _connect(self):
        sim = self.cfg["simulator"]
        self._client = carla.Client(sim["host"], sim["port"])
        self._client.set_timeout(sim["timeout_seconds"])

        target_map = self.cfg["world"]["map"]
        current_map = self._client.get_world().get_map().name.split("/")[-1]
        if current_map != target_map:
            self._world = self._client.load_world(target_map)
        else:
            self._world = self._client.get_world()

        self._map = self._world.get_map()
        self._spawn_points = self._map.get_spawn_points()

        self._original_settings = self._world.get_settings()
        settings = self._world.get_settings()
        settings.synchronous_mode = sim["sync_mode"]
        settings.fixed_delta_seconds = sim["fixed_delta_seconds"]
        if self.cfg["world"].get("no_rendering", False):
            settings.no_rendering_mode = True
        self._world.apply_settings(settings)

        self._tm = self._client.get_trafficmanager(8000)
        self._tm.set_synchronous_mode(True)
        self._tm.set_global_distance_to_leading_vehicle(2.5)
        self._tm.set_random_device_seed(self.cfg["traffic"].get("seed", 42))

        weather = self.cfg["world"].get("weather_preset", "ClearNoon")
        if hasattr(carla.WeatherParameters, weather):
            self._world.set_weather(getattr(carla.WeatherParameters, weather))

        self._connected = True

    # ------------------------------------------------------------------
    # Ego setup
    # ------------------------------------------------------------------

    def _setup_ego(self):
        ego_cfg = self.cfg["ego"]
        bp = self._world.get_blueprint_library().find(ego_cfg["blueprint_filter"])
        bp.set_attribute("role_name", ego_cfg["role_name"])

        if ego_cfg["spawn_mode"] == "fixed_index":
            candidates = [self._spawn_points[ego_cfg["spawn_index"]]]
        else:
            candidates = list(self._spawn_points)
            self.np_random.shuffle(candidates)

        # try_spawn_actor returns None on collision instead of raising
        for transform in candidates:
            self._ego = self._world.try_spawn_actor(bp, transform)
            if self._ego is not None:
                self._world.tick()
                return

        raise RuntimeError(
            f"Could not spawn ego on any of {len(candidates)} spawn points. "
            "Reduce NPC count or use a larger map."
        )

    def _setup_collision_sensor(self):
        bp = self._world.get_blueprint_library().find("sensor.other.collision")
        self._collision_sensor = self._world.spawn_actor(
            bp, carla.Transform(carla.Location(z=1.0)), attach_to=self._ego
        )
        self._collision_sensor.listen(self._on_collision)

    def _on_collision(self, event):
        self._collision_flag = True
        imp = event.normal_impulse
        self._collision_intensity = math.sqrt(imp.x**2 + imp.y**2 + imp.z**2)

    def _setup_route(self, route_length: int = 50):
        ego_loc = self._ego.get_location()
        current_wp = self._map.get_waypoint(ego_loc, project_to_road=True)
        self._route_waypoints = [current_wp]
        for _ in range(route_length):
            nexts = self._route_waypoints[-1].next(5.0)
            if not nexts:
                break
            self._route_waypoints.append(nexts[0])
        self._current_wp_idx = 0

    # ------------------------------------------------------------------
    # Traffic
    # ------------------------------------------------------------------

    def _spawn_traffic(self):
        bp_lib = self._world.get_blueprint_library()
        t_cfg = self.cfg["traffic"]

        # Vehicles
        vehicle_bps = [bp for bp in bp_lib.filter("vehicle.*")
                       if int(bp.get_attribute("number_of_wheels").as_int()) >= 4]
        pts = [sp for sp in self._spawn_points
               if sp.location.distance(self._ego.get_location()) > 20.0]
        self.np_random.shuffle(pts)

        for i in range(min(t_cfg["n_vehicles"], len(pts))):
            bp = self.np_random.choice(vehicle_bps)
            if bp.has_attribute("color"):
                bp.set_attribute("color",
                    self.np_random.choice(bp.get_attribute("color").recommended_values))
            npc = self._world.try_spawn_actor(bp, pts[i])
            if npc:
                npc.set_autopilot(True, self._tm.get_port())
                self._npc_vehicles.append(npc)

        # Pedestrians
        walker_bps = list(bp_lib.filter("walker.pedestrian.*"))
        ctrl_bp = bp_lib.find("controller.ai.walker")

        for _ in range(t_cfg["n_pedestrians"]):
            loc = self._world.get_random_location_from_navigation()
            if loc is None:
                continue
            bp = self.np_random.choice(walker_bps)
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")
            w = self._world.try_spawn_actor(bp, carla.Transform(loc))
            if w:
                self._npc_walkers.append(w)

        self._world.tick()

        for w in self._npc_walkers:
            ctrl = self._world.try_spawn_actor(ctrl_bp, carla.Transform(), w)
            if ctrl:
                self._walker_controllers.append(ctrl)

        self._world.tick()

        for ctrl in self._walker_controllers:
            target = self._world.get_random_location_from_navigation()
            if target:
                ctrl.start()
                ctrl.go_to_location(target)
                ctrl.set_max_speed(1.4)

        logger.info(f"Traffic spawned: {len(self._npc_vehicles)}V + {len(self._npc_walkers)}P")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _apply_action(self, action):
        tb = float(np.clip(action[0], -1.0, 1.0))
        st = float(np.clip(action[1], -1.0, 1.0))
        ctrl = carla.VehicleControl()
        ctrl.throttle = max(tb, 0.0)
        ctrl.brake = max(-tb, 0.0)
        ctrl.steer = st
        ctrl.hand_brake = False
        ctrl.reverse = False
        self._ego.apply_control(ctrl)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_obs(self):
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        transform = self._ego.get_transform()
        vel = self._ego.get_velocity()
        acc = self._ego.get_acceleration()
        fwd = transform.get_forward_vector()

        obs[0:3] = [vel.x / 30.0, vel.y / 30.0, vel.z / 30.0]
        obs[3:6] = [acc.x / 10.0, acc.y / 10.0, acc.z / 10.0]
        obs[6] = min(self._get_speed_kmh() / 120.0, 1.0)
        obs[7:9] = [fwd.x, fwd.y]

        self._advance_waypoint()
        if self._current_wp_idx < len(self._route_waypoints):
            t_loc = self._route_waypoints[self._current_wp_idx].transform.location
            e_loc = transform.location

            dist = e_loc.distance(t_loc)
            obs[9] = min(dist / 50.0, 1.0)

            dx, dy = t_loc.x - e_loc.x, t_loc.y - e_loc.y
            angle = math.atan2(dy, dx) - math.radians(transform.rotation.yaw)
            angle = (angle + math.pi) % (2 * math.pi) - math.pi
            obs[10] = angle / math.pi

            wp_ego = self._map.get_waypoint(e_loc, project_to_road=True)
            if wp_ego:
                ld = e_loc.distance(wp_ego.transform.location)
                wf = wp_ego.transform.get_forward_vector()
                cross = (e_loc.x - wp_ego.transform.location.x) * wf.y - \
                        (e_loc.y - wp_ego.transform.location.y) * wf.x
                obs[11] = np.clip(math.copysign(ld, cross) / 4.0, -1.0, 1.0)

        obs[12] = self._route_completion()

        if self._ego.is_at_traffic_light():
            obs[13] = 1.0
            tl = self._ego.get_traffic_light()
            if tl:
                s = tl.get_state()
                obs[14] = 1.0 if s == carla.TrafficLightState.Red else \
                           0.5 if s == carla.TrafficLightState.Yellow else 0.0

        self._fill_nearby_vehicles(obs, 15)
        return np.clip(obs, -1.0, 1.0)

    def _fill_nearby_vehicles(self, obs, start_idx):
        e_loc = self._ego.get_location()
        e_vel = self._ego.get_velocity()
        e_spd = math.sqrt(e_vel.x**2 + e_vel.y**2)
        fwd = self._ego.get_transform().get_forward_vector()
        rx, ry = -fwd.y, fwd.x

        dists = []
        for v in self._world.get_actors().filter("vehicle.*"):
            if v.id == self._ego.id:
                continue
            d = e_loc.distance(v.get_location())
            if d < 50.0:
                dists.append((d, v))
        dists.sort(key=lambda x: x[0])

        for i, (_, v) in enumerate(dists[:N_NEARBY_VEHICLES]):
            vl = v.get_location()
            dx, dy = vl.x - e_loc.x, vl.y - e_loc.y
            idx = start_idx + i * 3
            obs[idx] = np.clip((dx * fwd.x + dy * fwd.y) / 50.0, -1, 1)
            obs[idx+1] = np.clip((dx * rx + dy * ry) / 50.0, -1, 1)
            vv = v.get_velocity()
            obs[idx+2] = np.clip((math.sqrt(vv.x**2+vv.y**2) - e_spd) / 30.0, -1, 1)

    def _advance_waypoint(self):
        if self._current_wp_idx >= len(self._route_waypoints):
            return
        e = self._ego.get_location()
        w = self._route_waypoints[self._current_wp_idx].transform.location
        if e.distance(w) < 5.0:
            self._current_wp_idx += 1

    def _route_completion(self):
        if not self._route_waypoints:
            return 0.0
        return min(self._current_wp_idx / len(self._route_waypoints), 1.0)

    def _get_speed_kmh(self):
        v = self._ego.get_velocity()
        return 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self):
        """
        Reward v2 — dense, shaped:
            + speed_along_route  (capped, incentiva movimento moderato)
            + waypoint_bonus     (+5.0 per ogni WP raggiunto questo step)
            - speed_limit        (penalità quadratica sopra 40 km/h)
            - collision          (-10.0)
            - lane_deviation     (aumentata a 0.5)
            - steer_penalty      (aumentata a 0.1)
            - throttle_jerk      (variazione brusca throttle)
        """
        reward = 0.0
        speed_kmh = self._get_speed_kmh()

        # 1. Speed along route (capped a 0.8 per non incentivare velocità estrema)
        if self._current_wp_idx < len(self._route_waypoints):
            t_loc = self._route_waypoints[self._current_wp_idx].transform.location
            e_loc = self._ego.get_location()
            ev = self._ego.get_velocity()
            dx, dy = t_loc.x - e_loc.x, t_loc.y - e_loc.y
            d = math.sqrt(dx**2 + dy**2)
            if d > 0.01:
                reward += np.clip((ev.x * dx/d + ev.y * dy/d) / 10.0, -0.5, 0.8)

        # 2. Waypoint reach bonus (+5 per ogni WP raggiunto questo step)
        wp_delta = self._current_wp_idx - self._prev_wp_idx
        if wp_delta > 0:
            reward += wp_delta * 5.0
        self._prev_wp_idx = self._current_wp_idx

        # 3. Speed limit penalty (quadratica sopra 40 km/h)
        if speed_kmh > 40.0:
            reward -= ((speed_kmh - 40.0) ** 2) / 1600.0  # -1.0 a 80 km/h

        # 4. Collision penalty
        if self._collision_flag:
            reward -= 10.0

        # 5. Lane deviation penalty (aumentata)
        e_loc = self._ego.get_location()
        wp = self._map.get_waypoint(e_loc, project_to_road=True)
        if wp:
            reward -= np.clip(e_loc.distance(wp.transform.location) / 4.0, 0, 1) * 0.5

        # 6. Steer smoothness penalty (aumentata)
        ctrl = self._ego.get_control()
        reward -= abs(ctrl.steer) * 0.1

        # 7. Throttle jerk penalty (variazione brusca)
        current_throttle = ctrl.throttle - ctrl.brake
        reward -= abs(current_throttle - self._prev_throttle) * 0.1
        self._prev_throttle = current_throttle

        return float(reward)

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def _check_done(self):
        terminated = False
        truncated = False

        if self.ep_cfg["terminate_on_collision"] and self._collision_flag:
            terminated = True
        if self.ep_cfg["terminate_on_route_completion"] and \
           self._current_wp_idx >= len(self._route_waypoints):
            terminated = True
        if self.ep_cfg["terminate_on_offroad"]:
            e = self._ego.get_location()
            wp = self._map.get_waypoint(e, project_to_road=True)
            if wp is None or e.distance(wp.transform.location) > 5.0:
                terminated = True
        if self._step_count >= self.ep_cfg["max_steps"]:
            truncated = True
        return terminated, truncated

    # ------------------------------------------------------------------
    # Cleanup (silent)
    # ------------------------------------------------------------------

    def _safe_destroy(self, actor):
        """Destroy actor only if alive, suppress errors."""
        try:
            if actor is not None and actor.is_alive:
                actor.destroy()
        except RuntimeError:
            pass  # Actor already destroyed by server

    def _cleanup_ego(self):
        if self._collision_sensor:
            self._safe_destroy(self._collision_sensor)
            self._collision_sensor = None
        if self._ego:
            self._safe_destroy(self._ego)
            self._ego = None
        self._route_waypoints.clear()
        self._current_wp_idx = 0

    def _cleanup_traffic(self):
        for ctrl in self._walker_controllers:
            try:
                if ctrl.is_alive:
                    ctrl.stop()
            except Exception:
                pass
            self._safe_destroy(ctrl)
        self._walker_controllers.clear()

        for w in self._npc_walkers:
            self._safe_destroy(w)
        self._npc_walkers.clear()

        for v in self._npc_vehicles:
            self._safe_destroy(v)
        self._npc_vehicles.clear()

        self._traffic_spawned = False

    @staticmethod
    def _merge_dict(base, override):
        _merge(base, override)