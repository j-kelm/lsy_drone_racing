from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import gymnasium
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.sim.sim import Sim
from lsy_drone_racing.utils import check_gate_pass

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class DroneRacingEnv(gymnasium.Env):
    CONTROLLER = "mellinger"  # specifies controller type

    def __init__(self, config: dict, render_mode="human"):
        """Initialize the DroneRacingEnv.

        Args:
            config: Configuration dictionary for the environment.
        """
        super().__init__()
        self.config = config
        self.sim = Sim(
            track=config.env.track,
            sim_freq=config.sim.sim_freq,
            ctrl_freq=config.sim.ctrl_freq,
            disturbances=getattr(config.sim, "disturbances", {}),
            randomization=getattr(config.env, "randomization", {}),
            gui=config.sim.gui,
            n_drones=1,
            physics=config.sim.physics,
        )
        self.sim.seed(config.env.seed)
        self.action_space = spaces.Box(low=-1, high=1, shape=(13,))
        self.observation_space = spaces.Dict(
            {
                "pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
                "rpy": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
                "vel": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
                "ang_vel": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
            }
        )
        self.target_gate = 0
        self.symbolic = self.sim.symbolic() if config.env.symbolic else None
        self._steps = 0
        self._last_drone_pos = np.zeros(3)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if self.config.env.reseed:
            self.sim.seed(self.config.env.seed)
        if seed is not None:
            self.sim.seed(seed)
        self.sim.reset()
        self.target_gate = 0
        self._steps = 0
        self.sim.drone.reset(self.sim.drone.pos, self.sim.drone.rpy, self.sim.drone.vel)
        self._last_drone_pos[:] = self.sim.drone.pos
        if self.sim.n_drones > 1:
            raise NotImplementedError("Firmware wrapper does not support multiple drones.")
        info = self.info
        info["sim.sim_freq"] = self.config.sim.sim_freq
        info["sim.ctrl_freq"] = self.config.sim.ctrl_freq
        info["sim.drone.mass"] = self.sim.drone.params.mass
        info["env.freq"] = self.config.env.freq
        return self.obs, info

    def step(self, action: NDArray[np.floating]):
        """Step the firmware_wrapper class and its environment.

        This function should be called once at the rate of ctrl_freq. Step processes and high level
        commands, and runs the firmware loop and simulator according to the frequencies set.

        Args:
            action: Full-state command [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate]
                to follow.
        """
        action = action.astype(np.float64)  # Drone firmware expects float64
        assert action.shape == self.action_space.shape, f"Invalid action shape: {action.shape}"
        pos, vel, acc, yaw, rpy_rate = action[:3], action[3:6], action[6:9], action[9], action[10:]
        self.sim.drone.full_state_cmd(pos, vel, acc, yaw, rpy_rate)
        collision = self._inner_step_loop()
        terminated = self.terminated or collision
        return self.obs, self.reward, terminated, False, self.info

    def _inner_step_loop(self) -> bool:
        """Run the desired action for multiple simulation sub-steps.

        The outer controller is called at a lower frequency than the firmware loop. Each environment
        step therefore consists of multiple simulation steps. At each simulation step, the
        lower-level controller is called to compute the thrust and attitude commands.

        Returns:
            True if a collision occured at any point during the simulation steps, else False.
        """
        thrust = self.sim.drone.desired_thrust
        collision = False
        while (
            self.sim.drone.tick / self.sim.drone.firmware_freq
            < (self._steps + 1) / self.config.env.freq
        ):
            self.sim.step(thrust)
            self.target_gate += self.gate_passed()
            if self.target_gate == self.sim.n_gates:
                self.target_gate = -1
            collision |= bool(self.sim.collisions)
            pos, rpy, vel = self.sim.drone.pos, self.sim.drone.rpy, self.sim.drone.vel
            thrust = self.sim.drone.step_controller(pos, rpy, vel)[::-1]
        self.sim.drone.desired_thrust[:] = thrust
        self._last_drone_pos[:] = self.sim.drone.pos
        self._steps += 1
        return collision

    @property
    def obs(self) -> dict[str, NDArray[np.floating]]:
        obs = {
            "pos": self.sim.drone.pos.astype(np.float32),
            "rpy": self.sim.drone.rpy.astype(np.float32),
            "vel": self.sim.drone.vel.astype(np.float32),
            "ang_vel": self.sim.drone.ang_vel.astype(np.float32),
        }
        obs["ang_vel"][:] = R.from_euler("XYZ", obs["rpy"]).inv().apply(obs["ang_vel"])
        if "observation" in self.sim.disturbances:
            obs = self.sim.disturbances["observation"].apply(obs)
        return obs

    @property
    def reward(self) -> float:
        return -1.0

    @property
    def terminated(self) -> bool:
        state = {k: getattr(self.sim.drone, k).copy() for k in ("pos", "rpy", "vel", "ang_vel")}
        state["ang_vel"] = R.from_euler("XYZ", state["rpy"]).as_matrix().T @ state["ang_vel"]
        if state not in self.sim.state_space:
            return True  # Drone is out of bounds
        if self.sim.collisions:
            return True
        if self.target_gate == -1:  # Drone has passed all gates
            return True
        return False

    @property
    def info(self):
        info = {}
        info["collisions"] = self.sim.collisions
        gates = self.sim.gates
        info["target_gate"] = self.target_gate if self.target_gate < len(gates) else -1
        info["drone.pos"] = self.sim.drone.pos.copy()
        # Add the gate and obstacle poses to the info. If gates or obstacles are in sensor range,
        # use the actual pose, otherwise use the nominal pose.
        in_range = self.sim.in_range(gates, self.sim.drone, self.config.env.sensor_range)
        gates_pos = np.stack([g["nominal.pos"] for g in gates.values()])
        gates_pos[in_range] = np.stack([g["pos"] for g in gates.values()])[in_range]
        gates_rpy = np.stack([g["nominal.rpy"] for g in gates.values()])
        gates_rpy[in_range] = np.stack([g["rpy"] for g in gates.values()])[in_range]
        info["gates.pos"] = gates_pos
        info["gates.rpy"] = gates_rpy
        info["gates.in_range"] = in_range

        obstacles = self.sim.obstacles
        in_range = self.sim.in_range(obstacles, self.sim.drone, self.config.env.sensor_range)
        obstacles_pos = np.stack([o["nominal.pos"] for o in obstacles.values()])
        obstacles_pos[in_range] = np.stack([o["pos"] for o in obstacles.values()])[in_range]
        info["obstacles.pos"] = obstacles_pos
        info["obstacles.in_range"] = in_range
        info["symbolic.model"] = self.symbolic
        info["pyb_client"] = self.sim.pyb_client # As a way to access the pybullet client from inside the controller to draw debug lines for example.
        return info

    def gate_passed(self) -> bool:
        if self.sim.n_gates > 0 and self.target_gate < self.sim.n_gates and self.target_gate != -1:
            gate_pos = self.sim.gates[self.target_gate]["pos"]
            gate_rot = R.from_euler("xyz", self.sim.gates[self.target_gate]["rpy"])
            drone_pos = self.sim.drone.pos
            last_drone_pos = self._last_drone_pos
            gate_size = (0.45, 0.45)
            return check_gate_pass(gate_pos, gate_rot, gate_size, drone_pos, last_drone_pos)
        return False

    def render(self):
        self.sim.render()

    def close(self):
        self.sim.close()


class DroneRacingThrustEnv(DroneRacingEnv):
    def __init__(self, config: dict):
        super().__init__(config)
        bounds = np.array([1, np.pi, np.pi, np.pi], dtype=np.float32)
        self.action_space = spaces.Box(low=-bounds, high=bounds)

    def step(self, action: NDArray[np.floating]):
        """Step the drone racing environment with a thrust command.

        TODO: Clarify units of rpy and thrust.

        Args:
            action: Thrust command [roll, pitch, yaw, thrust].
        """
        assert action.shape == self.action_space.shape, f"Invalid action shape: {action.shape}"
        action = action.astype(np.float64)
        cmd_thrust = action[0]
        cmd_rpy = action[1:]
        self.sim.drone.collective_thrust_cmd(cmd_thrust, cmd_rpy)
        collision = self._inner_step_loop()
        terminated = self.terminated or collision
        return self.obs, self.reward, terminated, False, self.info
