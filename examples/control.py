from __future__ import annotations  # Python 3.10 type hints

import numpy as np
import numpy.typing as npt

from munch import munchify
import yaml

from examples.planner import Planner, FilePlanner, LinearPlanner, PolynomialPlanner, MinsnapPlanner
from examples.mpc_controller import MPC
from examples.model import Model
from examples.constraints import obstacle_constraints, gate_constraints


class Control:
    def __init__(self, initial_obs: npt.NDArray[np.floating], initial_info: dict):
        """Initialization of the controller.

        INSTRUCTIONS:
            The controller's constructor has access the initial state `initial_obs` and the a priori
            infromation contained in dictionary `initial_info`. Use this method to initialize
            constants, counters, pre-plan trajectories, etc.

        Args:
            initial_obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            initial_info: Additional environment information from the reset.
        """
        # Save environment and control parameters.
        self.CTRL_TIMESTEP = initial_info["ctrl_timestep"]
        self.CTRL_FREQ = initial_info["ctrl_freq"]
        self.initial_obs = initial_obs
        self.initial_info = initial_info

        if 'gate_idx' not in initial_info:
            initial_info['gate_idx'] = 0

        path = "config/mpc.yaml"
        with open(path, "r") as file:
            self.config = munchify(yaml.safe_load(file))

        # initialize planner
        self.planner = MinsnapPlanner(initial_info=initial_info, CTRL_FREQ=self.CTRL_FREQ)
        self.ref = self.planner.plan(initial_obs=self.initial_obs, speed=1.5, gate_index=initial_info['gate_idx'])

        # Get model and constraints
        self.model = Model(info=None)
        self.model.input_constraints_soft += [lambda u: 0.03 - u, lambda u: u - 0.145] # 0.03 <= thrust <= 0.145
        self.model.state_constraints_soft += [lambda x: 0.05 - x[2]]
        # self.model.state_constraints_soft += [lambda x: -3.0 - x[1], lambda x: x[1] - 3.0]
        # self.model.state_constraints_soft += [lambda x: -3.0 - x[0], lambda x: x[0] - 3.0]

        for obstacle_pos in self.initial_info['obstacles.pos']:
            self.model.state_constraints_soft += obstacle_constraints(obstacle_pos, r=0.17)

        for gate_pos, gate_rpy in zip(self.initial_info['gates.pos'], self.initial_info['gates.rpy']):
            self.model.state_constraints_soft += gate_constraints(gate_pos, gate_rpy[2], r=0.15)

        self.ctrl = MPC(model=self.model, horizon=int(self.config.mpc.horizon_sec * self.CTRL_FREQ),
                        q_mpc=self.config.mpc.q, r_mpc=self.config.mpc.r, soft_penalty=1e5)

        self.state = None

    def compute_control(
        self, state: npt.NDArray[np.floating], info: dict | None = None
    ) -> npt.NDarray[np.floating]:
        """Compute the next desired position and orientation of the drone.

        INSTRUCTIONS:
            Re-implement this method to return the target pose to be sent from Crazyswarm to the
            Crazyflie using the `cmdFullState` call.

        Args:
            state: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The drone pose [x_des, y_des, z_des, yaw_des] as a numpy array.
        """

        # Slice trajectory for horizon steps, if not long enough, repeat last state.
        start = info['step']
        end = min(start + self.ctrl.T + 1, self.ref.shape[-1])
        remain = max(0, self.ctrl.T + 1 - (end - start))
        remaining_ref = np.concatenate([
            self.ref[:, start:end],
            np.tile(self.ref[:, -1:], (1, remain))
        ], -1)

        info['q'] = np.outer(self.config.mpc.q, remaining_ref[13])

        inputs, next_state, outputs = self.ctrl.select_action(obs=state,
                                                     ref=remaining_ref[:12],
                                                     info=info)

        return inputs, next_state, outputs