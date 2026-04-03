"""
CarlaMultiAgentEnv v0.2 — PettingZoo ParallelEnv per CARLA
===========================================================
Multi-agent: N veicoli RL + M pedoni RL + NPC autopilot.
Compatibile con RLlib multi-agent via ParallelPettingZooEnv.

Agent IDs:
    "vehicle_0", "vehicle_1", ...
    "pedestrian_0", "pedestrian_1", ...

Policy mapping:
    vehicle_* -> vehicle_policy
    pedestrian_* -> pedestrian_policy

Vehicle obs (25D): same as CarlaEnv v0.3
Pedestrian obs (19D):
    [0:3]   position (x, y, z) normalized
    [3:6]   velocity (vx, vy, vz) normalized
    [6]     speed scalar normalized
    [7:9]   forward vector (fx, fy)
    [9]     distance to current waypoint (normalized)
    [10]    angle to current waypoint (normalized -1..1)
    [11]    on_sidewalk (0 or 1)
    [12:18] nearest 2 vehicles: (rel_x, rel_y, rel_speed) x 2
    [18]    route_completion (0..1)

Vehicle action (2D continuous): [throttle_brake, steer]
Pedestrian action (2D continuous): [speed_frac, direction_delta]

Changelog v0.2:
  - Pedestrians now use waypoint routes (like vehicles) instead of random goals
  - _setup_pedestrian_route() replaces _setup_pedestrian_goal()
  - _advance_pedestrian_waypoint() replaces _advance_pedestrian_goal()
  - _pedestrian_reward() v5: waypoint-based, aligned with vehicle reward
  - _route_completion() unified for both agent types
  - _NavPoint helper class for pedestrian waypoint interface compatibility
  - Reward v5 for both vehicles and pedestrians (5 terms each)
  - Pedestrian route completion now terminates episode
"""

import logging
import math
import signal
import sys
from copy import deepcopy
from pathlib import Path
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

VEHICLE_OBS_DIM = 25
PEDESTRIAN_OBS_DIM = 19
N_NEARBY_VEHICLES_FOR_VEHICLE = 3
N_NEARBY_VEHICLES_FOR_PEDESTRIAN = 2
PEDESTRIAN_MAX_SPEED = 5.0  # m/s (~18 km/h, running)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _NavPoint:
    """Lightweight wrapper to give a carla.Location the same .transform.location
    interface as carla.Waypoint, so pedestrian routes reuse vehicle WP logic."""
    __slots__ = ["transform"]

    def __init__(self, location):
        self.transform = type("T", (), {"location": location})()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_MA_CONFIG = {
    "simulator": {
        "host": "127.0.0.1", "port": 2000,
        "traffic_manager_port": 8000,
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
        "route_length_vehicle": 10, # Waypoints for vehicle route
        "route_length_pedestrian": 10, # Sidewalk waypoints for pedestrian route
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
        "agent_id",
        "agent_type",
        "actor",
        "collision_sensor",
        "collision_flag",
        "collision_step",
        "route_waypoints",
        "current_wp_idx",
        "prev_wp_idx",
        "goal_location",
        "prev_dist_to_wp",
        "prev_steer",
        "position_history",
        "last_wp_advance_step",
        "loop_penalty_active",
        "route_optimal_length",     # somma distanze WP-to-WP (calcolata al reset)
        "actual_distance_traveled", # accumulata step-by-step
        "prev_location"             # per calcolo distanza incrementale
    ]

    def __init__(self, agent_id: str, agent_type: str):
        self.agent_id = agent_id
        self.agent_type = agent_type  # "vehicle" or "pedestrian"
        self.actor = None
        self.collision_sensor = None
        self.collision_flag = False
        self.collision_step = 0
        self.route_waypoints = []
        self.current_wp_idx = 0
        self.prev_wp_idx = 0
        self.goal_location = None
        self.prev_dist_to_wp = 0.0
        self.prev_steer = 0.0
        self.position_history = []
        self.last_wp_advance_step = 0
        self.loop_penalty_active = False
        self.route_optimal_length = 0.0
        self.actual_distance_traveled = 0.0
        self.prev_location = None


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
        runtime_cfg = self.cfg.get("runtime", {}) or {}
        self._close_mode = str(runtime_cfg.get("close_mode", "legacy")).strip().lower()
        if self._close_mode not in {"legacy", "robust"}:
            self._close_mode = "legacy"
        self._closing = False
        self._closed = False
        self._traffic_spawned = False
        self._spawn_points = []
        self._npc_vehicles = []
        self._npc_walkers = []
        self._npc_controllers = []

        # Per-agent data
        self._agent_data: dict[str, AgentData] = {}
        self._step_count = 0
        self._reset_count = 0
        self._terminated_agent_infos = {}

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

        # Apply seed BEFORE any spawn
        self._reset_count += 1
        respawn_traffic = False
        if seed is not None:
            self.cfg["traffic"]["seed"] = seed
            self._reset_count = 0  # deterministic replay from explicit seed
            if self._tm:
                self._tm.set_random_device_seed(seed)
            # Force traffic respawn for full scene replay
            if self._traffic_spawned:
                self._cleanup_traffic()
            respawn_traffic = True

        self._cleanup_agents()
        self._setup_agents()

        if self.cfg["traffic"]["enabled"]:
            if respawn_traffic:
                self._spawn_traffic()
                self._traffic_spawned = True
            elif not self._traffic_spawned or not self.cfg["traffic"].get("persist_traffic", True):
                self._cleanup_traffic()
                self._spawn_traffic()
                self._traffic_spawned = True

        for _ in range(10):
            self._world.tick(10.0)  # 10 second timeout

        self._step_count = 0
        self._terminated_agent_infos = {}
        self.agents = list(self.possible_agents)

        # Reset per-agent flags
        for ad in self._agent_data.values():
            ad.collision_flag = False
            ad.collision_step = 0
            ad.prev_wp_idx = 0
            ad.prev_dist_to_wp = 0.0
            ad.prev_steer = 0.0
            ad.position_history = []
            ad.last_wp_advance_step = 0
            ad.loop_penalty_active = False
            ad.route_optimal_length = 0.0
            ad.actual_distance_traveled = 0.0
            ad.prev_location = None

        observations = {a: self._get_obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return observations, infos

    def step(self, actions):
        # Apply actions for all alive agents
        for agent_id, action in actions.items():
            if agent_id in self._agent_data:
                self._apply_action(agent_id, action)

        self._world.tick(10.0)  # 10 second timeout
        self._step_count += 1

         # --- Accumulate actual distance traveled ---
        for agent_id, ad in self._agent_data.items():
            if not ad.actor or not ad.actor.is_alive:
                continue
            loc = ad.actor.get_location()
            if ad.prev_location is not None:
                ad.actual_distance_traveled += ad.prev_location.distance(loc)
            ad.prev_location = loc

        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        # --- Update position history and loop detection ---
        for agent_id, ad in self._agent_data.items():
            if not ad.actor or not ad.actor.is_alive:
                continue

            # Position history sampling (every 50 steps) + loop detection
            if self._step_count % 50 == 0:
                loc = ad.actor.get_location()
                ad.position_history.append((loc.x, loc.y))
                if len(ad.position_history) > 20:
                    ad.position_history.pop(0)

                # Loop detection: bounding box of last 10 samples (500 steps)
                if len(ad.position_history) >= 10:
                    recent = ad.position_history[-10:]
                    xs = [p[0] for p in recent]
                    ys = [p[1] for p in recent]
                    bbox = max(max(xs) - min(xs), max(ys) - min(ys))
                    ad.loop_penalty_active = bbox < 5.0
                else:
                    ad.loop_penalty_active = False

        for agent_id in list(self.agents):
            ad = self._agent_data[agent_id]

            # Advance waypoints
            if ad.agent_type == "vehicle":
                self._advance_vehicle_waypoint(ad)
                # Fix finding 1: update last_wp_advance_step HERE, after advance
                if ad.current_wp_idx > ad.prev_wp_idx:
                    ad.last_wp_advance_step = self._step_count
            else:
                self._advance_pedestrian_waypoint(ad)

            reward = self._compute_reward(agent_id)
            rewards[agent_id] = self._validate_reward(reward, agent_id)
            self._refresh_route_if_needed(ad)

            # Collision cooldown: reset flag after 10 steps (For Future: es Hard Map)
            if ad.collision_flag:
                if ad.collision_step == 0:
                    ad.collision_step = self._step_count  # mark when collision happened
                elif self._step_count - ad.collision_step >= 10:
                    ad.collision_flag = False
                    ad.collision_step = 0

            term, trunc = self._check_done(agent_id)
            
            #Fix Run 9 -> terminate_on_collision: True
            terminations[agent_id] = term
            truncations[agent_id] = trunc

            # Path efficiency: optimal / actual (1.0 = perfect, <1.0 = inefficient)
            if ad.actual_distance_traveled > 0.1:
                path_eff = min(ad.route_optimal_length / ad.actual_distance_traveled, 1.0)
            else:
                path_eff = 0.0

            # --- Determine termination reason ---
            if term or trunc:
                if ad.collision_flag:
                    termination_reason = "collision"
                elif ad.agent_type == "vehicle" and ad.current_wp_idx >= len(ad.route_waypoints):
                    termination_reason = "route_complete"
                elif ad.agent_type == "pedestrian" and ad.current_wp_idx >= len(ad.route_waypoints):
                    termination_reason = "route_complete"
                elif ad.agent_type == "vehicle" and self.cfg["episode"].get("terminate_on_offroad", True):
                    el = ad.actor.get_location()
                    wp_check = self._map.get_waypoint(el, project_to_road=True)
                    if wp_check is None or el.distance(wp_check.transform.location) > 5.0:
                        termination_reason = "offroad"
                    elif trunc:
                        # Stuck: timeout + (low route_completion OR loop_penalty)
                        rc = self._route_completion(ad)
                        if rc < 0.3 or ad.loop_penalty_active:
                            termination_reason = "stuck"
                        else:
                            termination_reason = "timeout"
                    else:
                        termination_reason = "timeout"
                elif trunc:
                    rc = self._route_completion(ad)
                    if rc < 0.3 or ad.loop_penalty_active:
                        termination_reason = "stuck"
                    else:
                        termination_reason = "timeout"
                else:
                    termination_reason = "timeout"
            else:
                termination_reason = "alive"

            # Build info dict for this agent
            agent_info = {
                "step": self._step_count,
                "collision": ad.collision_flag,
                "route_completion": self._route_completion(ad),
                "path_efficiency": path_eff,
                "termination_reason": termination_reason,
            }

            if not term and not trunc:
                # Agent alive: emit both obs and info (RLlib contract)
                observations[agent_id] = self._get_obs(agent_id)
                infos[agent_id] = agent_info
            else:
                # Agent terminated: store in side-channel for callback
                self._terminated_agent_infos[agent_id] = agent_info

        # Remove terminated/truncated agents
        self.agents = [a for a in self.agents
                       if not terminations.get(a, False) and not truncations.get(a, False)]

        return observations, rewards, terminations, truncations, infos

    def close(self):
        if self._close_mode == "robust":
            if self._closed or self._closing:
                return

            self._closing = True
            tm = self._tm
            world = self._world
            original_settings = self._original_settings

            self._connected = False

            try:
                self._cleanup_agents_robust()
                self._cleanup_traffic_robust()
                if tm:
                    try:
                        tm.set_synchronous_mode(False)
                    except Exception:
                        pass
                if world and original_settings:
                    try:
                        world.apply_settings(original_settings)
                    except Exception:
                        pass
            finally:
                self._tm = None
                self._world = None
                self._map = None
                self._client = None
                self._original_settings = None
                self._spawn_points = []
                self._terminated_agent_infos = {}
                self.agents = []
                self._closed = True
                self._closing = False
            return

        self._cleanup_agents()
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

    def set_close_mode(self, mode: str):
        mode = str(mode).strip().lower()
        if mode not in {"legacy", "robust"}:
            raise ValueError(f"Unsupported close mode: {mode}")
        self._close_mode = mode
        self._closed = False
        self._closing = False

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _connect(self):
        sim = self.cfg["simulator"]
        self._client = carla.Client(sim["host"], sim["port"])
        self._client.set_timeout(sim["timeout_seconds"])
        tm_port = int(sim.get("traffic_manager_port", 8000))

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
        assert sim.get("sync_mode", True), \
            "CarlaMultiAgentEnv requires sync_mode=True for deterministic RL training"
        settings.fixed_delta_seconds = sim["fixed_delta_seconds"]
        if self.cfg["world"].get("no_rendering", False):
            settings.no_rendering_mode = True
        self._world.apply_settings(settings)

        self._tm = self._client.get_trafficmanager(tm_port)
        self._tm.set_synchronous_mode(True)
        self._tm.set_global_distance_to_leading_vehicle(2.5)
        self._tm.set_random_device_seed(self.cfg["traffic"].get("seed", 42))

        weather = self.cfg["world"].get("weather_preset", "ClearNoon")
        if hasattr(carla.WeatherParameters, weather):
            self._world.set_weather(getattr(carla.WeatherParameters, weather))

        self._connected = True
        self._closed = False
        self._closing = False

    # ------------------------------------------------------------------
    # Agent setup
    # ------------------------------------------------------------------

    def _setup_agents(self):
        bp_lib = self._world.get_blueprint_library()
        rng = np.random.default_rng(self.cfg["traffic"].get("seed", 42) + self._reset_count)

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

        self._world.tick(10.0)  # 10 second timeout

        # Setup collision sensors and routes for vehicles
        for agent_id, ad in self._agent_data.items():
            if ad.agent_type == "vehicle":
                self._setup_collision_sensor(ad)
                self._setup_vehicle_route(ad, skip_current_wp=True)

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

        self._world.tick(10.0)  # 10 second timeout

        # Setup collision sensors and routes for pedestrians
        for agent_id, ad in self._agent_data.items():
            if ad.agent_type == "pedestrian":
                self._setup_collision_sensor(ad)
                self._setup_pedestrian_route(ad)

    def _setup_collision_sensor(self, ad: AgentData):
        bp = self._world.get_blueprint_library().find("sensor.other.collision")
        sensor = self._world.spawn_actor(bp, carla.Transform(carla.Location(z=0.5)),
                                         attach_to=ad.actor)

        def on_collision(event, _ad=ad):
            _ad.collision_flag = True
            _ad.collision_step = 0  # signal: new collision, step() will set actual value

        sensor.listen(on_collision)
        ad.collision_sensor = sensor

    def _get_sidewalk_waypoint(self, loc: carla.Location):
        """Resolve the closest sidewalk waypoint for pedestrian routing."""
        wp = self._map.get_waypoint(
            loc,
            project_to_road=False,
            lane_type=carla.LaneType.Sidewalk,
        )
        if wp is not None and wp.lane_type == carla.LaneType.Sidewalk:
            return wp

        wp = self._map.get_waypoint(
            loc,
            project_to_road=True,
            lane_type=carla.LaneType.Sidewalk,
        )
        if wp is not None and wp.lane_type == carla.LaneType.Sidewalk:
            return wp

        return None

    def _setup_pedestrian_route_fallback(self, ad: AgentData, route_len: int):
        """Fallback to navmesh sampling if no sidewalk waypoint is resolvable."""
        ad.route_waypoints = []
        ad.goal_location = None
        current_loc = ad.actor.get_location()

        for _ in range(route_len):
            goal = None
            for _ in range(20):
                candidate = self._world.get_random_location_from_navigation()
                if candidate and current_loc.distance(candidate) > 5.0:
                    goal = candidate
                    break
            if goal is None:
                break
            ad.route_waypoints.append(_NavPoint(goal))
            current_loc = goal

        if ad.route_waypoints:
            ad.goal_location = ad.route_waypoints[0].transform.location
        else:
            loc = ad.actor.get_location()
            ad.goal_location = carla.Location(x=loc.x + 30.0, y=loc.y, z=loc.z)
            ad.route_waypoints = [_NavPoint(ad.goal_location)]

        ad.route_optimal_length = self._compute_route_optimal_length(ad)
        ad.prev_location = ad.actor.get_location()

    def _setup_vehicle_route(self, ad: AgentData, skip_current_wp: bool = False):
        loc = ad.actor.get_location()
        wp = self._map.get_waypoint(loc, project_to_road=True)
        route_len = self.cfg["episode"].get("route_length_vehicle", 10) # 50 -> 10 Reward v5 - Waypoints for vehicle route
        ad.route_waypoints = []
        current_wp = wp

        if not skip_current_wp:
            ad.route_waypoints.append(current_wp)

        for _ in range(route_len):
            nexts = current_wp.next(2.0)  # 2.0m spacing vehicle (ratio 1:1)
            if not nexts:
                break
            current_wp = nexts[0]
            ad.route_waypoints.append(current_wp)

        if not ad.route_waypoints and not skip_current_wp:
            ad.route_waypoints = [wp]

        ad.current_wp_idx = 0
        ad.route_optimal_length = self._compute_route_optimal_length(ad)
        ad.prev_location = ad.actor.get_location()

    def _setup_pedestrian_route(self, ad: AgentData):
        """Build a pedestrian route by chaining sidewalk waypoints like vehicles."""
        route_len = self.cfg["episode"].get("route_length_pedestrian", 10)
        ad.route_waypoints = []
        ad.current_wp_idx = 0
        ad.goal_location = None

        start_wp = self._get_sidewalk_waypoint(ad.actor.get_location())
        if start_wp is None:
            self._setup_pedestrian_route_fallback(ad, route_len)
            return

        current_wp = start_wp

        for _ in range(route_len):
            nexts = [
                wp for wp in current_wp.next(2.5)
                if wp.lane_type == carla.LaneType.Sidewalk
            ]
            if not nexts:
                break

            current_wp = nexts[0]
            ad.route_waypoints.append(current_wp)

        if not ad.route_waypoints:
            self._setup_pedestrian_route_fallback(ad, route_len)
            return

        ad.goal_location = ad.route_waypoints[0].transform.location
        ad.route_optimal_length = self._compute_route_optimal_length(ad)
        ad.prev_location = ad.actor.get_location()

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

        self._world.tick(10.0)  # 10 second timeout

        for w in self._npc_walkers:
            ctrl = self._world.try_spawn_actor(ctrl_bp, carla.Transform(), w)
            if ctrl:
                self._npc_controllers.append(ctrl)

        self._world.tick(10.0)  # 10 second timeout

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

    def _validate_action(self, action: np.ndarray, agent_id: str) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,):
            raise ValueError(f"[{agent_id}] invalid action shape: {action.shape}")
        if not np.isfinite(action).all():
            raise ValueError(f"[{agent_id}] non-finite action: {action}")
        return action

    def _sanitize_obs(self, obs: np.ndarray, agent_id: str) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        obs = np.clip(obs, -1.0, 1.0)
        if not np.isfinite(obs).all():
            raise ValueError(f"[{agent_id}] non-finite observation after sanitization")
        return obs.astype(np.float32, copy=False)

    def _validate_reward(self, reward: float, agent_id: str) -> float:
        reward = float(reward)
        if not np.isfinite(reward):
            raise ValueError(f"[{agent_id}] non-finite reward: {reward}")
        return reward

    def _apply_action(self, agent_id: str, action: np.ndarray):
        ad = self._agent_data[agent_id]
        if not ad.actor or not ad.actor.is_alive:
            return

        action = self._validate_action(action, agent_id)

        if ad.agent_type == "vehicle":
            tb = float(np.clip(action[0], -1, 1))
            st = float(np.clip(action[1], -1, 1))

            ctrl = carla.VehicleControl()
            ctrl.throttle = max(tb, 0.0)
            ctrl.brake = max(-tb, 0.0)
            ctrl.steer = st
            ctrl.reverse = False
            ad.actor.apply_control(ctrl)

        elif ad.agent_type == "pedestrian":
            speed_frac = float(np.clip(action[0], 0, 1))
            dir_delta = float(np.clip(action[1], -1, 1))

            speed = speed_frac * PEDESTRIAN_MAX_SPEED
            rotation = ad.actor.get_transform().rotation
            yaw = math.radians(rotation.yaw) + dir_delta * math.pi * 0.25  # max +/-45 deg/step
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

        obs[12] = self._route_completion(ad)

        if ad.actor.is_at_traffic_light():
            obs[13] = 1.0
            tl = ad.actor.get_traffic_light()
            if tl:
                s = tl.get_state()
                obs[14] = 1.0 if s == carla.TrafficLightState.Red else \
                           0.5 if s == carla.TrafficLightState.Yellow else 0.0

        # Nearby vehicles
        self._fill_nearby(obs, 15, ad, N_NEARBY_VEHICLES_FOR_VEHICLE, "vehicle.*")

        # Previous steering value (Block 4.3)
        obs[24] = np.clip(ad.prev_steer, -1.0, 1.0)

        return self._sanitize_obs(obs, ad.agent_id)

    def _get_pedestrian_obs(self, ad: AgentData) -> np.ndarray:
        """18D obs for pedestrian — now uses waypoint route like vehicles."""
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

        # Current waypoint (replaces old goal_location logic)
        if ad.current_wp_idx < len(ad.route_waypoints):
            wp_loc = ad.route_waypoints[ad.current_wp_idx].transform.location
            dx = wp_loc.x - loc.x
            dy = wp_loc.y - loc.y
            dist = math.sqrt(dx**2 + dy**2)
            obs[9] = min(dist / 100, 1)
            angle = math.atan2(dy, dx) - math.radians(t.rotation.yaw)
            angle = (angle + math.pi) % (2 * math.pi) - math.pi
            obs[10] = angle / math.pi
        elif ad.goal_location:
            # Fallback to goal_location if route exhausted
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

        # Route completion fraction (Block 4.3)
        obs[18] = self._route_completion(ad)

        return self._sanitize_obs(obs, ad.agent_id)

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
    # Waypoint / route tracking
    # ------------------------------------------------------------------

    def _advance_vehicle_waypoint(self, ad: AgentData):
        """Advance vehicle along waypoint route"""
        if ad.current_wp_idx >= len(ad.route_waypoints):
            return
        loc = ad.actor.get_location()
        wp_loc = ad.route_waypoints[ad.current_wp_idx].transform.location
        if loc.distance(wp_loc) < 2.0: # 2.0m threshold vehicle (ratio 1:1)
            ad.current_wp_idx += 1
            ad.prev_dist_to_wp = 0.0

    def _advance_pedestrian_waypoint(self, ad: AgentData):
        """Advance pedestrian along waypoint route"""
        if ad.current_wp_idx >= len(ad.route_waypoints):
            return
        loc = ad.actor.get_location()
        wp_loc = ad.route_waypoints[ad.current_wp_idx].transform.location
        if loc.distance(wp_loc) < 2.0: # 2.0m threshold for pedestrians
            ad.current_wp_idx += 1
            ad.prev_dist_to_wp = 0.0
            # Update goal_location for obs fallback
            if ad.current_wp_idx < len(ad.route_waypoints):
                ad.goal_location = ad.route_waypoints[ad.current_wp_idx].transform.location

    def _route_completion(self, ad: AgentData) -> float:
        """Unified route completion for both vehicles and pedestrians."""
        if not ad.route_waypoints:
            return 0.0
        return min(ad.current_wp_idx / len(ad.route_waypoints), 1.0)

    def _compute_route_optimal_length(self, ad: AgentData) -> float:
        """Sum of consecutive WP-to-WP distances = optimal path length."""
        if len(ad.route_waypoints) < 2:
            return 0.0
        total = 0.0
        for i in range(len(ad.route_waypoints) - 1):
            loc_a = ad.route_waypoints[i].transform.location
            loc_b = ad.route_waypoints[i + 1].transform.location
            total += loc_a.distance(loc_b)
        return total

    def _refresh_route_if_needed(self, ad: AgentData):
        ep_cfg = self.cfg["episode"]
        if ad.current_wp_idx < len(ad.route_waypoints):
            return
        if ep_cfg.get("terminate_on_route_completion", True):
            return

        ad.route_waypoints = []
        ad.current_wp_idx = 0
        ad.prev_wp_idx = 0
        ad.prev_dist_to_wp = 0.0
        ad.goal_location = None

        if ad.agent_type == "vehicle":
            self._setup_vehicle_route(ad, skip_current_wp=True)
        else:
            self._setup_pedestrian_route(ad)

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
        """Vehicle Reward v5"""
        reward = 0.0
        vel = ad.actor.get_velocity()
        speed_kmh = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        el = ad.actor.get_location()

        # ---- 1. Waypoint reach bonus (DOMINANT — sole positive signal) ----
        wp_delta = ad.current_wp_idx - ad.prev_wp_idx
        if wp_delta > 0:
            reward += wp_delta * 50.0  # very high, must be THE reason to move
        ad.prev_wp_idx = ad.current_wp_idx

        # ---- 2. Distance to next WP (dense guidance — gets closer = good) ----
        if ad.current_wp_idx < len(ad.route_waypoints):
            wp_loc = ad.route_waypoints[ad.current_wp_idx].transform.location
            curr_dist = el.distance(wp_loc)
            if ad.prev_dist_to_wp > 0:
                # Positive when getting closer, negative when getting farther
                reward += (ad.prev_dist_to_wp - curr_dist) * 2.0
            ad.prev_dist_to_wp = curr_dist

        # ---- 3. Collision penalty (large, immediate) ----
        if ad.collision_flag and ad.collision_step == 0:
            reward -= 50.0

            # Future Finetuing: scale by speed at collision (normalized)
            # reward -= (20.0 + speed_kmh * 0.5)  # range: -20 (fermo) a -45 (50 km/h)

        # ---- 4. Off-lane penalty (stay on road) ----
        wp = self._map.get_waypoint(el, project_to_road=True)
        if wp:
            lane_dist = el.distance(wp.transform.location)
            if lane_dist > 2.0:
                reward -= lane_dist * 0.5

        # ---- 5. Speed target (moderate speed = good, too fast/slow = bad) ----
        # Target: 15-30 km/h. Below 5 = idle penalty. Above 50 = speed penalty
        if speed_kmh < 2.5:
            reward -= 0.15  # idle penalty
        elif 15.0 <= speed_kmh <= 50.0:
            reward += 0.3  # optimal speed bonus
        elif speed_kmh > 50.0:
            reward -= (speed_kmh - 50.0) * 0.10 # speed penalty (scaled)

        # ---- 6. Steering smoothness ----
        ctrl = ad.actor.get_control()
        steer_delta = abs(ctrl.steer - ad.prev_steer)
        if steer_delta < 0.1:
            reward += 0.1   # smooth driving bonus
        elif steer_delta > 0.5:
            reward -= 0.3   # jerk penalty
        ad.prev_steer = ctrl.steer

        # ---- 7. Anti-loop penalty ----
        if ad.loop_penalty_active:
            reward -= 1.0

        return float(reward)

    def _pedestrian_reward(self, ad: AgentData) -> float:
        """Pedestrian Reward v5"""
        reward = 0.0
        loc = ad.actor.get_location()
        vel = ad.actor.get_velocity()
        speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

        # ---- 1. Waypoint reach bonus (DOMINANT) ----
        wp_delta = ad.current_wp_idx - ad.prev_wp_idx
        if wp_delta > 0:
            reward += wp_delta * 50.0
        ad.prev_wp_idx = ad.current_wp_idx

        # ---- 2. Distance to next WP (dense guidance) ----
        if ad.current_wp_idx < len(ad.route_waypoints):
            wp_loc = ad.route_waypoints[ad.current_wp_idx].transform.location
            curr_dist = loc.distance(wp_loc)
            if ad.prev_dist_to_wp > 0:
                reward += (ad.prev_dist_to_wp - curr_dist) * 2.0
            ad.prev_dist_to_wp = curr_dist

        # ---- 3. Collision penalty ----
        if ad.collision_flag and ad.collision_step == 0:
            reward -= 50.0

        # ---- 4. Sidewalk bonus / road penalty ----
        wp = self._map.get_waypoint(loc, project_to_road=False)
        if wp:
            if str(wp.lane_type) == "Sidewalk":
                reward += 0.2
            elif str(wp.lane_type) == "Driving":
                reward -= 0.3

        # ---- 5. Speed target (walking pace) ----
        if speed < 0.15:
            reward -= 0.15  # idle
        elif 0.8 <= speed <= 1.8:
            reward += 0.3  # comfortable walking pace
        elif speed > 3.0:
            reward -= (speed - 3.0) * 0.7  # running too fast

        # ---- 6. Anti-loop penalty ----
        if ad.loop_penalty_active:
            reward -= 1.0

        return float(reward)

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def _check_done(self, agent_id: str) -> tuple[bool, bool]:
        ad = self._agent_data[agent_id]
        ep_cfg = self.cfg["episode"]
        terminated = False
        truncated = False

        # Collision
        if ep_cfg.get("terminate_on_collision", True) and ad.collision_flag:
            terminated = True

        # Vehicle: route complete or off-road
        if ad.agent_type == "vehicle":
            if ep_cfg.get("terminate_on_route_completion", True):
                if ad.current_wp_idx >= len(ad.route_waypoints):
                    terminated = True

            if ep_cfg.get("terminate_on_offroad", True):
                el = ad.actor.get_location()
                wp = self._map.get_waypoint(el, project_to_road=True)
                if wp is None or el.distance(wp.transform.location) > 5.0:
                    terminated = True

        # Pedestrian: route complete
        if ad.agent_type == "pedestrian":
            if ep_cfg.get("terminate_on_route_completion", True):
                if ad.current_wp_idx >= len(ad.route_waypoints):
                    terminated = True

        # Max steps
        if self._step_count >= ep_cfg["max_steps"]:
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

    def _safe_destroy_robust(self, actor):
        try:
            if actor and actor.is_alive:
                actor.destroy()
            return True
        except Exception:
            return False
        return True

    def _collect_batch_actor_refs(self, actors):
        actor_refs = {}
        for actor in actors:
            if actor is None:
                continue
            try:
                actor_id = int(actor.id)
            except Exception:
                continue
            if actor_id <= 0 or actor_id in actor_refs:
                continue
            actor_refs[actor_id] = actor
        return actor_refs

    def _alive_actor_ids(self, actor_ids):
        actor_ids = [int(actor_id) for actor_id in actor_ids if int(actor_id) > 0]
        if not actor_ids or self._world is None:
            return set()
        try:
            return {int(actor.id) for actor in self._world.get_actors(actor_ids)}
        except Exception:
            alive = set()
            for actor_id in actor_ids:
                try:
                    actor = self._world.get_actor(actor_id)
                except Exception:
                    actor = None
                if actor is not None:
                    alive.add(int(actor_id))
            return alive

    def _batch_destroy_actors(self, actors, *, label):
        actor_refs = self._collect_batch_actor_refs(actors)
        if not actor_refs:
            return set()

        if self._client is None or self._world is None:
            for actor in actor_refs.values():
                self._safe_destroy_robust(actor)
            return self._alive_actor_ids(actor_refs.keys())

        commands = [carla.command.DestroyActor(actor_id) for actor_id in actor_refs]
        try:
            self._client.apply_batch_sync(
                commands,
                bool(self.cfg["simulator"].get("sync_mode", True)),
            )
        except Exception:
            logger.exception(
                "CARLA batch destroy failed for %s; falling back to per-actor destroy",
                label,
            )
            for actor in actor_refs.values():
                self._safe_destroy_robust(actor)

        return set(actor_refs.keys())

    def _cleanup_agents(self):
        # Stop sensors BEFORE destroying to prevent callback-during-destroy deadlock
        for ad in self._agent_data.values():
            if ad.collision_sensor:
                try:
                    ad.collision_sensor.stop()
                except Exception:
                    pass
        for ad in self._agent_data.values():
            if ad.collision_sensor:
                self._safe_destroy(ad.collision_sensor)
            self._safe_destroy(ad.actor)
        self._agent_data.clear()

    def _cleanup_agents_robust(self):
        agent_data = list(self._agent_data.values())
        self._terminated_agent_infos = {}
        failed_agents = {}

        for ad in agent_data:
            if ad.collision_sensor:
                try:
                    ad.collision_sensor.stop()
                except Exception:
                    pass

        sensor_actor_ids = self._batch_destroy_actors(
            [ad.collision_sensor for ad in agent_data],
            label="collision sensors",
        )
        agent_actor_ids = self._batch_destroy_actors(
            [ad.actor for ad in agent_data],
            label="rl agents",
        )
        remaining_actor_ids = self._alive_actor_ids(
            list(sensor_actor_ids) + list(agent_actor_ids)
        )

        for ad in agent_data:
            sensor_id = getattr(ad.collision_sensor, "id", None)
            actor_id = getattr(ad.actor, "id", None)
            sensor_alive = (
                sensor_id is not None and int(sensor_id) in remaining_actor_ids
            )
            actor_alive = (
                actor_id is not None and int(actor_id) in remaining_actor_ids
            )

            if not sensor_alive:
                ad.collision_sensor = None
            if not actor_alive:
                ad.actor = None

            if sensor_alive or actor_alive:
                failed_agents[ad.agent_id] = ad

        self._agent_data = failed_agents

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

    def _cleanup_traffic_robust(self):
        controllers = list(self._npc_controllers)
        walkers = list(self._npc_walkers)
        vehicles = list(self._npc_vehicles)

        failed_controllers = []
        failed_walkers = []
        failed_vehicles = []

        for ctrl in controllers:
            try:
                if ctrl.is_alive:
                    ctrl.stop()
            except Exception:
                pass

        controller_actor_ids = self._batch_destroy_actors(
            controllers,
            label="traffic controllers",
        )
        walker_actor_ids = self._batch_destroy_actors(
            walkers,
            label="traffic walkers",
        )
        vehicle_actor_ids = self._batch_destroy_actors(
            vehicles,
            label="traffic vehicles",
        )
        remaining_actor_ids = self._alive_actor_ids(
            list(controller_actor_ids) + list(walker_actor_ids) + list(vehicle_actor_ids)
        )

        for ctrl in controllers:
            actor_id = getattr(ctrl, "id", None)
            if actor_id is not None and int(actor_id) in remaining_actor_ids:
                failed_controllers.append(ctrl)
        for walker in walkers:
            actor_id = getattr(walker, "id", None)
            if actor_id is not None and int(actor_id) in remaining_actor_ids:
                failed_walkers.append(walker)
        for vehicle in vehicles:
            actor_id = getattr(vehicle, "id", None)
            if actor_id is not None and int(actor_id) in remaining_actor_ids:
                failed_vehicles.append(vehicle)

        self._npc_controllers = failed_controllers
        self._npc_walkers = failed_walkers
        self._npc_vehicles = failed_vehicles
        self._traffic_spawned = False
