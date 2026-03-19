"""
CarlaMultiAgentEnv v0.1 — PettingZoo ParallelEnv per CARLA
===========================================================
Multi-agent: N veicoli RL + M pedoni RL + NPC autopilot.
Compatibile con RLlib multi-agent via ParallelPettingZooEnv.

Agent IDs:
    "vehicle_0", "vehicle_1", ...
    "pedestrian_0", "pedestrian_1", ...

Policy mapping:
    vehicle_* -> vehicle_policy
    pedestrian_* -> pedestrian_policy

Vehicle obs (24D): same as CarlaEnv v0.3
Pedestrian obs (18D):
    [0:3]   position (x, y, z) normalized
    [3:6]   velocity (vx, vy, vz) normalized
    [6]     speed scalar normalized
    [7:9]   forward vector (fx, fy)
    [9]     distance to goal (normalized)
    [10]    angle to goal (normalized -1..1)
    [11]    on_sidewalk (0 or 1)
    [12:18] nearest 2 vehicles: (rel_x, rel_y, rel_speed) x 2

Vehicle action (2D continuous): [throttle_brake, steer]
Pedestrian action (2D continuous): [speed_frac, direction_delta]
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
from pettingzoo import ParallelEnv

try:
    import carla
except ImportError:
    raise ImportError("pip install carla==0.9.16")

logger = logging.getLogger(__name__)

if sys.platform == "win32":
    signal.signal(signal.SIGABRT, signal.SIG_IGN)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VEHICLE_OBS_DIM = 24
PEDESTRIAN_OBS_DIM = 18
N_NEARBY_VEHICLES_FOR_VEHICLE = 3
N_NEARBY_VEHICLES_FOR_PEDESTRIAN = 2
PEDESTRIAN_MAX_SPEED = 5.0  # m/s (~18 km/h, running)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_MA_CONFIG = {
    "simulator": {
        "host": "127.0.0.1", "port": 2000,
        "timeout_seconds": 20.0, "sync_mode": True,
        "fixed_delta_seconds": 0.05,
    },
    "world": {
        "map": "Town03", "weather_preset": "ClearNoon",
        "no_rendering": True,
    },
    "agents": {
        "n_vehicles_rl": 1,
        "n_pedestrians_rl": 1,
        "vehicle_blueprint": "vehicle.tesla.model3",
        "pedestrian_blueprint": "walker.pedestrian.*",
    },
    "traffic": {
        "enabled": True, "n_vehicles_npc": 10, "n_pedestrians_npc": 10,
        "seed": 42, "persist_traffic": True,
    },
    "episode": {
        "max_steps": 1000,
        "terminate_on_collision": True,
        "route_length_vehicle": 50,
        "route_length_pedestrian": 20,
    },
}


def _merge(base, override):
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _merge(base[k], v)
        else:
            base[k] = v


def load_ma_config(config_path=None):
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "configs" / "multi_agent.yaml"
    config = deepcopy(DEFAULT_MA_CONFIG)
    p = Path(config_path)
    if p.exists():
        with open(p) as f:
            loaded = yaml.safe_load(f) or {}
        _merge(config, loaded)
    return config


# ---------------------------------------------------------------------------
# Agent data container
# ---------------------------------------------------------------------------

class AgentData:
    """Stores per-agent state."""
    __slots__ = [
        "agent_id", "agent_type", "actor", "collision_sensor",
        "collision_flag", "collision_intensity",
        "route_waypoints", "current_wp_idx", "prev_wp_idx",
        "goal_location", "prev_throttle",
    ]

    def __init__(self, agent_id: str, agent_type: str):
        self.agent_id = agent_id
        self.agent_type = agent_type  # "vehicle" or "pedestrian"
        self.actor = None
        self.collision_sensor = None
        self.collision_flag = False
        self.collision_intensity = 0.0
        self.route_waypoints = []
        self.current_wp_idx = 0
        self.prev_wp_idx = 0
        self.goal_location = None
        self.prev_throttle = 0.0


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class CarlaMultiAgentEnv(ParallelEnv):
    """PettingZoo ParallelEnv for CARLA multi-agent."""

    metadata = {"render_modes": ["human"], "name": "carla_multi_agent_v0"}

    def __init__(self, config=None, config_path=None):
        super().__init__()
        file_cfg = load_ma_config(config_path)
        if config:
            _merge(file_cfg, config)
        self.cfg = file_cfg

        ag = self.cfg["agents"]
        self._n_veh = ag["n_vehicles_rl"]
        self._n_ped = ag["n_pedestrians_rl"]

        # Build agent list
        self.possible_agents = []
        for i in range(self._n_veh):
            self.possible_agents.append(f"vehicle_{i}")
        for i in range(self._n_ped):
            self.possible_agents.append(f"pedestrian_{i}")
        self.agents = list(self.possible_agents)

        # Spaces per agent type
        self._vehicle_obs_space = gym.spaces.Box(-1, 1, (VEHICLE_OBS_DIM,), np.float32)
        self._vehicle_act_space = gym.spaces.Box(
            np.array([-1, -1], dtype=np.float32),
            np.array([1, 1], dtype=np.float32),
        )
        self._pedestrian_obs_space = gym.spaces.Box(-1, 1, (PEDESTRIAN_OBS_DIM,), np.float32)
        self._pedestrian_act_space = gym.spaces.Box(
            np.array([0, -1], dtype=np.float32),
            np.array([1, 1], dtype=np.float32),
        )

        # CARLA state
        self._client = None
        self._world = None
        self._map = None
        self._tm = None
        self._original_settings = None
        self._connected = False
        self._traffic_spawned = False
        self._spawn_points = []
        self._npc_vehicles = []
        self._npc_walkers = []
        self._npc_controllers = []

        # Per-agent data
        self._agent_data: dict[str, AgentData] = {}
        self._step_count = 0

    # ------------------------------------------------------------------
    # PettingZoo API
    # ------------------------------------------------------------------

    def observation_space(self, agent):
        return self._vehicle_obs_space if agent.startswith("vehicle") \
            else self._pedestrian_obs_space

    def action_space(self, agent):
        return self._vehicle_act_space if agent.startswith("vehicle") \
            else self._pedestrian_act_space

    @property
    def observation_spaces(self):
        return {a: self.observation_space(a) for a in self.possible_agents}

    @property
    def action_spaces(self):
        return {a: self.action_space(a) for a in self.possible_agents}

    def reset(self, seed=None, options=None):
        if not self._connected:
            self._connect()

        self._cleanup_agents()
        self._setup_agents()

        if self.cfg["traffic"]["enabled"]:
            if not self._traffic_spawned or not self.cfg["traffic"].get("persist_traffic", True):
                self._cleanup_traffic()
                self._spawn_traffic()
                self._traffic_spawned = True

        for _ in range(10):
            self._world.tick()

        self._step_count = 0
        self.agents = list(self.possible_agents)

        # Reset per-agent flags
        for ad in self._agent_data.values():
            ad.collision_flag = False
            ad.collision_intensity = 0.0
            ad.prev_wp_idx = 0
            ad.prev_throttle = 0.0

        observations = {a: self._get_obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return observations, infos

    def step(self, actions):
        # Apply actions for all alive agents
        for agent_id, action in actions.items():
            if agent_id in self._agent_data:
                self._apply_action(agent_id, action)

        self._world.tick()
        self._step_count += 1

        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        for agent_id in list(self.agents):
            ad = self._agent_data[agent_id]

            # Advance waypoints
            if ad.agent_type == "vehicle":
                self._advance_vehicle_waypoint(ad)
            else:
                self._advance_pedestrian_goal(ad)

            observations[agent_id] = self._get_obs(agent_id)
            rewards[agent_id] = self._compute_reward(agent_id)
            term, trunc = self._check_done(agent_id)
            terminations[agent_id] = term
            truncations[agent_id] = trunc
            infos[agent_id] = {
                "step": self._step_count,
                "collision": ad.collision_flag,
            }

        # Remove terminated/truncated agents
        self.agents = [a for a in self.agents
                       if not terminations.get(a, False) and not truncations.get(a, False)]

        return observations, rewards, terminations, truncations, infos

    def close(self):
        self._cleanup_agents()
        self._cleanup_traffic()
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
    # Agent setup
    # ------------------------------------------------------------------

    def _setup_agents(self):
        bp_lib = self._world.get_blueprint_library()
        rng = np.random.default_rng(self.cfg["traffic"].get("seed", 42) + self._step_count)

        # Shuffle spawn points for vehicles
        veh_points = list(self._spawn_points)
        rng.shuffle(veh_points)

        # Spawn RL vehicles
        veh_bp = bp_lib.find(self.cfg["agents"]["vehicle_blueprint"])
        for i in range(self._n_veh):
            agent_id = f"vehicle_{i}"
            ad = AgentData(agent_id, "vehicle")

            veh_bp.set_attribute("role_name", f"rl_vehicle_{i}")
            actor = None
            for pt in veh_points:
                actor = self._world.try_spawn_actor(veh_bp, pt)
                if actor:
                    veh_points.remove(pt)
                    break
            if actor is None:
                raise RuntimeError(f"Cannot spawn {agent_id}")

            ad.actor = actor
            self._agent_data[agent_id] = ad

        self._world.tick()

        # Setup collision sensors and routes for vehicles
        for agent_id, ad in self._agent_data.items():
            if ad.agent_type == "vehicle":
                self._setup_collision_sensor(ad)
                self._setup_vehicle_route(ad)

        # Spawn RL pedestrians
        ped_bps = list(bp_lib.filter(self.cfg["agents"]["pedestrian_blueprint"]))
        for i in range(self._n_ped):
            agent_id = f"pedestrian_{i}"
            ad = AgentData(agent_id, "pedestrian")

            bp = rng.choice(ped_bps)
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")

            actor = None
            for _ in range(20):  # retry
                loc = self._world.get_random_location_from_navigation()
                if loc:
                    actor = self._world.try_spawn_actor(bp, carla.Transform(loc))
                    if actor:
                        break
            if actor is None:
                raise RuntimeError(f"Cannot spawn {agent_id}")

            ad.actor = actor
            self._agent_data[agent_id] = ad

        self._world.tick()

        # Setup collision sensors and goals for pedestrians
        for agent_id, ad in self._agent_data.items():
            if ad.agent_type == "pedestrian":
                self._setup_collision_sensor(ad)
                self._setup_pedestrian_goal(ad)

    def _setup_collision_sensor(self, ad: AgentData):
        bp = self._world.get_blueprint_library().find("sensor.other.collision")
        sensor = self._world.spawn_actor(bp, carla.Transform(carla.Location(z=0.5)),
                                         attach_to=ad.actor)

        def on_collision(event, _ad=ad):
            _ad.collision_flag = True
            imp = event.normal_impulse
            _ad.collision_intensity = math.sqrt(imp.x**2 + imp.y**2 + imp.z**2)

        sensor.listen(on_collision)
        ad.collision_sensor = sensor

    def _setup_vehicle_route(self, ad: AgentData):
        loc = ad.actor.get_location()
        wp = self._map.get_waypoint(loc, project_to_road=True)
        route_len = self.cfg["episode"].get("route_length_vehicle", 50)
        ad.route_waypoints = [wp]
        for _ in range(route_len):
            nexts = ad.route_waypoints[-1].next(5.0)
            if not nexts:
                break
            ad.route_waypoints.append(nexts[0])
        ad.current_wp_idx = 0

    def _setup_pedestrian_goal(self, ad: AgentData):
        """Set a random navigation goal for the pedestrian."""
        for _ in range(10):
            goal = self._world.get_random_location_from_navigation()
            if goal:
                ad.goal_location = goal
                return
        # Fallback: 50m ahead
        loc = ad.actor.get_location()
        ad.goal_location = carla.Location(x=loc.x + 50, y=loc.y, z=loc.z)

    # ------------------------------------------------------------------
    # Traffic (NPC)
    # ------------------------------------------------------------------

    def _spawn_traffic(self):
        bp_lib = self._world.get_blueprint_library()
        t = self.cfg["traffic"]
        rng = np.random.default_rng(t.get("seed", 42))

        # NPC Vehicles
        veh_bps = [b for b in bp_lib.filter("vehicle.*")
                    if int(b.get_attribute("number_of_wheels").as_int()) >= 4]
        rl_locs = [ad.actor.get_location() for ad in self._agent_data.values()
                   if ad.agent_type == "vehicle"]
        pts = [sp for sp in self._spawn_points
               if all(sp.location.distance(rl) > 20.0 for rl in rl_locs)]
        rng.shuffle(pts)

        for i in range(min(t["n_vehicles_npc"], len(pts))):
            bp = rng.choice(veh_bps)
            if bp.has_attribute("color"):
                bp.set_attribute("color", rng.choice(bp.get_attribute("color").recommended_values))
            npc = self._world.try_spawn_actor(bp, pts[i])
            if npc:
                npc.set_autopilot(True, self._tm.get_port())
                self._npc_vehicles.append(npc)

        # NPC Pedestrians
        walker_bps = list(bp_lib.filter("walker.pedestrian.*"))
        ctrl_bp = bp_lib.find("controller.ai.walker")

        for _ in range(t["n_pedestrians_npc"]):
            loc = self._world.get_random_location_from_navigation()
            if not loc:
                continue
            bp = rng.choice(walker_bps)
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")
            w = self._world.try_spawn_actor(bp, carla.Transform(loc))
            if w:
                self._npc_walkers.append(w)

        self._world.tick()

        for w in self._npc_walkers:
            ctrl = self._world.try_spawn_actor(ctrl_bp, carla.Transform(), w)
            if ctrl:
                self._npc_controllers.append(ctrl)

        self._world.tick()

        for ctrl in self._npc_controllers:
            target = self._world.get_random_location_from_navigation()
            if target:
                ctrl.start()
                ctrl.go_to_location(target)
                ctrl.set_max_speed(1.4)

        logger.info(f"NPC: {len(self._npc_vehicles)}V + {len(self._npc_walkers)}P")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _apply_action(self, agent_id: str, action: np.ndarray):
        ad = self._agent_data[agent_id]
        if not ad.actor or not ad.actor.is_alive:
            return

        if ad.agent_type == "vehicle":
            tb = float(np.clip(action[0], -1, 1))
            st = float(np.clip(action[1], -1, 1))
            ctrl = carla.VehicleControl()
            ctrl.throttle = max(tb, 0.0)
            ctrl.brake = max(-tb, 0.0)
            ctrl.steer = st
            ctrl.hand_brake = False
            ctrl.reverse = False
            ad.actor.apply_control(ctrl)

        elif ad.agent_type == "pedestrian":
            speed_frac = float(np.clip(action[0], 0, 1))
            dir_delta = float(np.clip(action[1], -1, 1))

            speed = speed_frac * PEDESTRIAN_MAX_SPEED
            rotation = ad.actor.get_transform().rotation
            yaw = math.radians(rotation.yaw) + dir_delta * math.pi * 0.25  # max ±45°/step
            direction = carla.Vector3D(
                x=math.cos(yaw), y=math.sin(yaw), z=0.0
            )
            ad.actor.apply_control(carla.WalkerControl(
                speed=speed, direction=direction, jump=False
            ))

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_obs(self, agent_id: str) -> np.ndarray:
        ad = self._agent_data[agent_id]
        if ad.agent_type == "vehicle":
            return self._get_vehicle_obs(ad)
        else:
            return self._get_pedestrian_obs(ad)

    def _get_vehicle_obs(self, ad: AgentData) -> np.ndarray:
        """Same 24D obs as CarlaEnv v0.3."""
        obs = np.zeros(VEHICLE_OBS_DIM, dtype=np.float32)
        t = ad.actor.get_transform()
        vel = ad.actor.get_velocity()
        acc = ad.actor.get_acceleration()
        fwd = t.get_forward_vector()

        obs[0:3] = [vel.x / 30, vel.y / 30, vel.z / 30]
        obs[3:6] = [acc.x / 10, acc.y / 10, acc.z / 10]
        speed = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        obs[6] = min(speed / 120, 1)
        obs[7:9] = [fwd.x, fwd.y]

        if ad.current_wp_idx < len(ad.route_waypoints):
            tl = ad.route_waypoints[ad.current_wp_idx].transform.location
            el = t.location
            dist = el.distance(tl)
            obs[9] = min(dist / 50, 1)
            dx, dy = tl.x - el.x, tl.y - el.y
            angle = math.atan2(dy, dx) - math.radians(t.rotation.yaw)
            angle = (angle + math.pi) % (2 * math.pi) - math.pi
            obs[10] = angle / math.pi
            wp_ego = self._map.get_waypoint(el, project_to_road=True)
            if wp_ego:
                ld = el.distance(wp_ego.transform.location)
                wf = wp_ego.transform.get_forward_vector()
                cross = (el.x - wp_ego.transform.location.x) * wf.y - \
                        (el.y - wp_ego.transform.location.y) * wf.x
                obs[11] = np.clip(math.copysign(ld, cross) / 4, -1, 1)

        obs[12] = self._vehicle_route_completion(ad)

        if ad.actor.is_at_traffic_light():
            obs[13] = 1.0
            tl = ad.actor.get_traffic_light()
            if tl:
                s = tl.get_state()
                obs[14] = 1.0 if s == carla.TrafficLightState.Red else \
                           0.5 if s == carla.TrafficLightState.Yellow else 0.0

        # Nearby vehicles
        self._fill_nearby(obs, 15, ad, N_NEARBY_VEHICLES_FOR_VEHICLE, "vehicle.*")
        return np.clip(obs, -1, 1)

    def _get_pedestrian_obs(self, ad: AgentData) -> np.ndarray:
        """18D obs for pedestrian."""
        obs = np.zeros(PEDESTRIAN_OBS_DIM, dtype=np.float32)
        t = ad.actor.get_transform()
        vel = ad.actor.get_velocity()
        loc = t.location

        # Position (normalized by map extent ~500m)
        obs[0:3] = [loc.x / 500, loc.y / 500, loc.z / 50]

        # Velocity
        obs[3:6] = [vel.x / 5, vel.y / 5, vel.z / 5]
        speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        obs[6] = min(speed / PEDESTRIAN_MAX_SPEED, 1)

        # Forward vector
        fwd = t.get_forward_vector()
        obs[7:9] = [fwd.x, fwd.y]

        # Goal
        if ad.goal_location:
            dx = ad.goal_location.x - loc.x
            dy = ad.goal_location.y - loc.y
            dist = math.sqrt(dx**2 + dy**2)
            obs[9] = min(dist / 100, 1)
            angle = math.atan2(dy, dx) - math.radians(t.rotation.yaw)
            angle = (angle + math.pi) % (2 * math.pi) - math.pi
            obs[10] = angle / math.pi

        # On sidewalk (approximate: check if waypoint has lane_type sidewalk)
        wp = self._map.get_waypoint(loc, project_to_road=False)
        if wp and str(wp.lane_type) == "Sidewalk":
            obs[11] = 1.0

        # Nearby vehicles (danger detection)
        self._fill_nearby(obs, 12, ad, N_NEARBY_VEHICLES_FOR_PEDESTRIAN, "vehicle.*")
        return np.clip(obs, -1, 1)

    def _fill_nearby(self, obs, start_idx, ad, n, filter_str):
        """Fill obs with relative position/speed of nearest actors."""
        e_loc = ad.actor.get_location()
        e_vel = ad.actor.get_velocity()
        e_spd = math.sqrt(e_vel.x**2 + e_vel.y**2)
        fwd = ad.actor.get_transform().get_forward_vector()
        rx, ry = -fwd.y, fwd.x

        dists = []
        for v in self._world.get_actors().filter(filter_str):
            if v.id == ad.actor.id:
                continue
            d = e_loc.distance(v.get_location())
            if d < 50:
                dists.append((d, v))
        dists.sort(key=lambda x: x[0])

        for i, (_, v) in enumerate(dists[:n]):
            vl = v.get_location()
            dx, dy = vl.x - e_loc.x, vl.y - e_loc.y
            idx = start_idx + i * 3
            obs[idx] = np.clip((dx * fwd.x + dy * fwd.y) / 50, -1, 1)
            obs[idx + 1] = np.clip((dx * rx + dy * ry) / 50, -1, 1)
            vv = v.get_velocity()
            obs[idx + 2] = np.clip((math.sqrt(vv.x**2 + vv.y**2) - e_spd) / 30, -1, 1)

    # ------------------------------------------------------------------
    # Waypoint / goal tracking
    # ------------------------------------------------------------------

    def _advance_vehicle_waypoint(self, ad: AgentData):
        if ad.current_wp_idx >= len(ad.route_waypoints):
            return
        loc = ad.actor.get_location()
        wp_loc = ad.route_waypoints[ad.current_wp_idx].transform.location
        if loc.distance(wp_loc) < 5.0:
            ad.current_wp_idx += 1

    def _advance_pedestrian_goal(self, ad: AgentData):
        """Check if pedestrian reached goal, set new one if so."""
        if not ad.goal_location:
            return
        loc = ad.actor.get_location()
        if loc.distance(ad.goal_location) < 3.0:
            self._setup_pedestrian_goal(ad)  # New random goal

    def _vehicle_route_completion(self, ad: AgentData) -> float:
        if not ad.route_waypoints:
            return 0.0
        return min(ad.current_wp_idx / len(ad.route_waypoints), 1.0)

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def _compute_reward(self, agent_id: str) -> float:
        ad = self._agent_data[agent_id]
        if ad.agent_type == "vehicle":
            return self._vehicle_reward(ad)
        else:
            return self._pedestrian_reward(ad)

    def _vehicle_reward(self, ad: AgentData) -> float:
        """Reward v2 — same as CarlaEnv v0.3."""
        reward = 0.0
        vel = ad.actor.get_velocity()
        speed_kmh = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

        # Speed along route
        if ad.current_wp_idx < len(ad.route_waypoints):
            tl = ad.route_waypoints[ad.current_wp_idx].transform.location
            el = ad.actor.get_location()
            dx, dy = tl.x - el.x, tl.y - el.y
            d = math.sqrt(dx**2 + dy**2)
            if d > 0.01:
                reward += np.clip((vel.x * dx/d + vel.y * dy/d) / 10, -0.5, 0.8)

        # Waypoint bonus
        wp_delta = ad.current_wp_idx - ad.prev_wp_idx
        if wp_delta > 0:
            reward += wp_delta * 5.0
        ad.prev_wp_idx = ad.current_wp_idx

        # Speed limit
        if speed_kmh > 40:
            reward -= ((speed_kmh - 40) ** 2) / 1600

        # Collision
        if ad.collision_flag:
            reward -= 10.0

        # Lane deviation
        el = ad.actor.get_location()
        wp = self._map.get_waypoint(el, project_to_road=True)
        if wp:
            reward -= np.clip(el.distance(wp.transform.location) / 4, 0, 1) * 0.5

        # Steer
        ctrl = ad.actor.get_control()
        reward -= abs(ctrl.steer) * 0.1

        # Throttle jerk
        cur_t = ctrl.throttle - ctrl.brake
        reward -= abs(cur_t - ad.prev_throttle) * 0.1
        ad.prev_throttle = cur_t

        return float(reward)

    def _pedestrian_reward(self, ad: AgentData) -> float:
        """
        Pedestrian reward:
            + progress toward goal
            + goal reach bonus
            - collision penalty
            - leaving sidewalk penalty
            - speed penalty (too fast)
        """
        reward = 0.0
        loc = ad.actor.get_location()
        vel = ad.actor.get_velocity()
        speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

        # Progress toward goal
        if ad.goal_location:
            dx = ad.goal_location.x - loc.x
            dy = ad.goal_location.y - loc.y
            dist = math.sqrt(dx**2 + dy**2)
            if dist > 0.01:
                dir_x, dir_y = dx / dist, dy / dist
                speed_toward = vel.x * dir_x + vel.y * dir_y
                reward += np.clip(speed_toward / 2.0, -0.5, 1.0)

            # Goal reach bonus
            if dist < 3.0:
                reward += 10.0

        # Collision
        if ad.collision_flag:
            reward -= 10.0

        # Sidewalk bonus / road penalty
        wp = self._map.get_waypoint(loc, project_to_road=False)
        if wp:
            if str(wp.lane_type) == "Sidewalk":
                reward += 0.1  # small bonus for staying on sidewalk
            elif str(wp.lane_type) == "Driving":
                reward -= 0.5  # penalty for being on road

        # Speed penalty (running too fast)
        if speed > 2.0:  # faster than brisk walk
            reward -= (speed - 2.0) * 0.2

        return float(reward)

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def _check_done(self, agent_id: str) -> tuple[bool, bool]:
        ad = self._agent_data[agent_id]
        terminated = False
        truncated = False

        # Collision
        if self.cfg["episode"]["terminate_on_collision"] and ad.collision_flag:
            terminated = True

        # Vehicle: route complete
        if ad.agent_type == "vehicle":
            if ad.current_wp_idx >= len(ad.route_waypoints):
                terminated = True
            # Off-road
            el = ad.actor.get_location()
            wp = self._map.get_waypoint(el, project_to_road=True)
            if wp is None or el.distance(wp.transform.location) > 5.0:
                terminated = True

        # Max steps
        if self._step_count >= self.cfg["episode"]["max_steps"]:
            truncated = True

        return terminated, truncated

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _safe_destroy(self, actor):
        try:
            if actor and actor.is_alive:
                actor.destroy()
        except RuntimeError:
            pass

    def _cleanup_agents(self):
        for ad in self._agent_data.values():
            if ad.collision_sensor:
                self._safe_destroy(ad.collision_sensor)
            self._safe_destroy(ad.actor)
        self._agent_data.clear()

    def _cleanup_traffic(self):
        for ctrl in self._npc_controllers:
            try:
                if ctrl.is_alive:
                    ctrl.stop()
            except Exception:
                pass
            self._safe_destroy(ctrl)
        self._npc_controllers.clear()
        for w in self._npc_walkers:
            self._safe_destroy(w)
        self._npc_walkers.clear()
        for v in self._npc_vehicles:
            self._safe_destroy(v)
        self._npc_vehicles.clear()
        self._traffic_spawned = False
