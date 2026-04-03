import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import random

INNER_ORBIT = 125.0
OUTER_ORBIT = 225.0

class OrbitEnv(gym.Env):
    """
    Orbit Mania RL Environment.
    Uses the original polar-coordinate mechanics (angle + smooth radius lerp),
    wrapped in a Gymnasium interface for PPO training.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.MAX_FUEL = 100.0
        self.ANGULAR_SPEED = 0.8   # degrees per frame — matches original
        self.RADIUS_LERP = 0.44    # px per frame — matches original transition feel

        # Difficulty scaling constants
        self.OBSTACLE_INTERVAL_START = 180  # frames between spawns at t=0
        self.OBSTACLE_INTERVAL_MIN   = 45   # floor: never faster than this
        self.OBSTACLE_SPEED_START    = (1.0, 2.0)   # (min, max) speed at t=0
        self.OBSTACLE_SPEED_MAX      = (2.5, 5.0)   # speed ceiling

        # Action space: [orbit_switch (0=no, 1=yes), shield (0=off, 1=on)]
        self.action_space = spaces.MultiDiscrete([2, 2])

        # Observation: [sin(angle), cos(angle), radius_norm, fuel_norm, shield_active,
        #               nearest_ob_rel_x, nearest_ob_rel_y, nearest_ob_vx, nearest_ob_vy,
        #               nearest_ep_rel_x, nearest_ep_rel_y,
        #               obstacle_count_norm, dist_to_orbit_edge_norm, is_transitioning]  — size 14
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        )

        self.reset()

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.angle = 270.0          # degrees; start at top of inner orbit
        self.radius = INNER_ORBIT
        self.target_radius = INNER_ORBIT
        self.fuel = self.MAX_FUEL
        self.shield_active = False

        self.obstacles = []
        self.energy_packets = []
        self.step_count = 0
        self.last_obstacle_step = 0
        self.last_packet_step = 0

        self._update_player_pos()
        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def _update_player_pos(self):
        """Compute Cartesian position from polar coords (planet at origin)."""
        rad = math.radians(self.angle)
        self.player_pos = np.array(
            [self.radius * math.cos(rad),
             self.radius * math.sin(rad)],
            dtype=np.float32
        )

    # ------------------------------------------------------------------
    def switch_orbit(self):
        """Request an orbit switch. Only acts if ship is on a stable orbit."""
        if abs(self.radius - self.target_radius) < 1.0:   # fully settled
            if self.target_radius <= INNER_ORBIT:
                self.target_radius = OUTER_ORBIT
            else:
                self.target_radius = INNER_ORBIT

    # ------------------------------------------------------------------
    def _get_obs(self):
        obs = np.zeros(14, dtype=np.float32)
        obs[0] = math.sin(math.radians(self.angle))
        obs[1] = math.cos(math.radians(self.angle))
        obs[2] = (self.radius - INNER_ORBIT) / (OUTER_ORBIT - INNER_ORBIT)
        obs[3] = self.fuel / self.MAX_FUEL
        obs[4] = 1.0 if self.shield_active else 0.0

        nearest_ob_dist = np.inf
        if self.obstacles:
            dists = [np.linalg.norm(ob['pos'] - self.player_pos) for ob in self.obstacles]
            idx = int(np.argmin(dists))
            nearest_ob_dist = dists[idx]
            obs[5:7] = self.obstacles[idx]['pos'] - self.player_pos
            obs[7:9] = self.obstacles[idx]['vel']

        if self.energy_packets:
            dists = [np.linalg.norm(ep['pos'] - self.player_pos) for ep in self.energy_packets]
            idx = int(np.argmin(dists))
            obs[9:11] = self.energy_packets[idx]['pos'] - self.player_pos

        # --- Richer context signals ---
        # [11] How crowded is the screen? (normalised, cap at 10)
        obs[11] = min(len(self.obstacles), 10) / 10.0

        # [12] Proximity to the nearest orbit ring (0 = on an orbit, 1 = halfway between)
        dist_inner = abs(self.radius - INNER_ORBIT)
        dist_outer = abs(self.radius - OUTER_ORBIT)
        orbit_gap = (OUTER_ORBIT - INNER_ORBIT) / 2.0
        obs[12] = min(dist_inner, dist_outer) / orbit_gap

        # [13] Is the ship currently mid-transition between orbits?
        obs[13] = 0.0 if abs(self.radius - self.target_radius) < 1.0 else 1.0

        # Store for reward shaping
        self._nearest_ob_dist = nearest_ob_dist

        return obs

    # ------------------------------------------------------------------
    def step(self, action):
        self.step_count += 1
        switch_action = int(action[0])
        shield_action = int(action[1])

        self.shield_active = (shield_action == 1)

        # Orbit switch request
        if switch_action == 1:
            self.switch_orbit()

        # Smooth radius lerp — the original animation
        diff = self.target_radius - self.radius
        if abs(diff) > self.RADIUS_LERP:
            self.radius += math.copysign(self.RADIUS_LERP, diff)
        else:
            self.radius = self.target_radius

        # Advance orbital angle
        self.angle += self.ANGULAR_SPEED
        self._update_player_pos()

        # Passive fuel drain
        if self.shield_active:
            self.fuel -= 0.2
        else:
            self.fuel -= 0.05
        self.fuel = max(0.0, self.fuel)

        # Difficulty: ramp up every 600 steps (~10 s at 60 fps)
        # interval shrinks linearly; speed range grows linearly
        difficulty = min(1.0, self.step_count / 3600)  # 0 → 1 over 60 s
        spawn_interval = int(
            self.OBSTACLE_INTERVAL_START
            - difficulty * (self.OBSTACLE_INTERVAL_START - self.OBSTACLE_INTERVAL_MIN)
        )
        speed_min = self.OBSTACLE_SPEED_START[0] + difficulty * (self.OBSTACLE_SPEED_MAX[0] - self.OBSTACLE_SPEED_START[0])
        speed_max = self.OBSTACLE_SPEED_START[1] + difficulty * (self.OBSTACLE_SPEED_MAX[1] - self.OBSTACLE_SPEED_START[1])

        if self.step_count - self.last_obstacle_step >= spawn_interval:
            self.last_obstacle_step = self.step_count
            x = random.randint(400, 800)
            y = random.randint(-300, 300)
            speed = random.uniform(speed_min, speed_max)
            self.obstacles.append({
                'pos': np.array([x, y], dtype=np.float32),
                'vel': np.array([-speed, 0.0], dtype=np.float32),
                'radius': 15.0,
            })

        # Spawn energy packets (every ~5 s = 300 frames, max 5 on screen)
        if self.step_count - self.last_packet_step >= 300 and len(self.energy_packets) < 5:
            self.last_packet_step = self.step_count
            r = random.choice([INNER_ORBIT, OUTER_ORBIT])
            a = math.radians(random.randint(0, 35) * 10)
            self.energy_packets.append({
                'pos': np.array([r * math.cos(a), r * math.sin(a)], dtype=np.float32),
                'radius': 10.0,
            })

        # Move obstacles
        for ob in self.obstacles:
            ob['pos'] += ob['vel']

        # Cull off-screen obstacles
        self.obstacles = [ob for ob in self.obstacles if ob['pos'][0] > -1000]

        # --- Collision detection (using index-based removal to avoid numpy equality bug) ---
        terminated = False
        reward = 0.1   # alive reward

        # Shield-spam penalty: discourage holding shield when no threat is near
        if self.shield_active:
            nearest_dist = getattr(self, '_nearest_ob_dist', np.inf)
            if nearest_dist > 150.0:
                reward -= 0.1

        player_r = 25.0 if self.shield_active else 15.0

        keep_obs = []
        for ob in self.obstacles:
            dist = float(np.linalg.norm(self.player_pos - ob['pos']))
            if dist < player_r + ob['radius']:
                if self.shield_active:
                    reward += 5.0   # destroyed by shield
                else:
                    terminated = True
                    reward = -100.0
            else:
                keep_obs.append(ob)
        self.obstacles = keep_obs

        keep_ep = []
        for ep in self.energy_packets:
            dist = float(np.linalg.norm(self.player_pos - ep['pos']))
            if dist < player_r + ep['radius']:
                self.fuel = min(self.MAX_FUEL, self.fuel + 20.0)
                reward += 20.0
            else:
                keep_ep.append(ep)
        self.energy_packets = keep_ep

        if self.fuel <= 0:
            terminated = True
            reward -= 20.0

        return self._get_obs(), reward, terminated, False, {}
