from __future__ import annotations  # Python 3.10 type hints

# import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from examples.planner import MinsnapPlanner
from examples.mpc_controller import MPC
from examples.model import Model
from examples.constraints import obstacle_constraints, gate_constraints, to_rbf_potential


class Control:
    def __init__(self, initial_obs: npt.NDArray[np.floating], initial_info: dict, config: dict):
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

        self.config = config

        # initialize planner
        self.planner = MinsnapPlanner(initial_info=initial_info,
                                      initial_obs=self.initial_obs,
                                      speed=1.5,
                                      gate_index=initial_info['gate_idx'],
        )

        # Get model and constraints
        self.model = Model(info=None)

        self.model.state_constraints += [lambda x: 0.03 - x[12:16], lambda x: x[12:16] - 0.145] # 0.03 <= thrust <= 0.145
        self.model.state_constraints_soft += [lambda x: -84 / 180 * np.pi - x[6:8], lambda x: x[6:8] - 84 / 180 * np.pi] # max roll and pitch
        self.model.state_constraints_soft += [lambda x: 0.05 - x[2]]
        # self.model.state_constraints_soft += [lambda x: -3.0 - x[1], lambda x: x[1] - 3.0]
        # self.model.state_constraints_soft += [lambda x: -3.0 - x[0], lambda x: x[0] - 3.0]

        ellipsoid_constraints = list()
        for obstacle_pos in self.initial_info['obstacles.pos']:
            ellipsoid_constraints += obstacle_constraints(obstacle_pos, r=0.14)

        for gate_pos, gate_rpy in zip(self.initial_info['gates.pos'], self.initial_info['gates.rpy']):
            ellipsoid_constraints += gate_constraints(gate_pos, gate_rpy[2], r=0.12)

        self.model.state_constraints_soft += [to_rbf_potential(ellipsoid_constraints)]

        # f = to_rbf_potential(ellipsoid_constraints)
        # h = 0.525
        # t = np.linspace(-3, 3, 600)
        # z = np.array([[i, j, h] for j in t for i in t]).T
        # z = np.array(f(z))
        # X, Y = np.meshgrid(t, t)
        # Z = z.reshape(600, 600)
        # plt.contourf(X, Y, Z, levels=[-0.001, 0.001])
        # plt.show()

        self.ctrl = MPC(model=self.model,
                        horizon=int(self.config['horizon_sec'] * self.CTRL_FREQ),
                        q_mpc=self.config['q'], r_mpc=self.config['r'],
                        soft_penalty=1e5,
                        err_on_fail=False,
                        max_iter=1000,
        )

        self.forces = initial_info['init_thrusts'] if 'init_thrusts' in initial_info and initial_info[
            'init_thrusts'] is not None else self.ctrl.U_EQ


        self.state = None
        self.planning_step = 0


    def re_plan(self, initial_obs: npt.NDArray[np.floating], initial_info: dict):
        self.planning_step = initial_info['step']
        raise NotImplementedError


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

        state = np.concatenate([state, self.forces])
        step = info['step'] - self.planning_step

        # Slice trajectory for horizon steps, if not long enough, repeat last state.
        remaining_ref = self.to_horizon(self.planner.ref, step, self.ctrl.T + 1)
        gate_prox = self.to_horizon(self.planner.gate_prox, step, self.ctrl.T + 1)

        q_pos = np.zeros_like(self.config['q'])
        q_pos[0:3] = self.config['q'][0:3]
        info['q'] = np.array(self.config['q'])[:, np.newaxis] + 4.0 * np.outer(q_pos, gate_prox)
        # info['r'] = np.array(self.config.mpc.r)[:, np.newaxis] + 3.0 * np.outer(self.config.mpc.r, gate_prox)
        info['u_guess'] = np.tile(np.expand_dims(np.zeros(4), axis=1), (1, self.ctrl.T))

        try:
            inputs, next_state, outputs = self.ctrl.select_action(obs=state,
                                                                  ref=remaining_ref,
                                                                  info=info,
                                                                  err_on_fail=True)
        except RuntimeError: # re-plan on fail
            inputs, next_state, outputs = self.ctrl.select_action(obs=state,
                                                                  ref=remaining_ref,
                                                                  info=info,
                                                                  err_on_fail=False)
            # (1) re-plan and warm start with new reference
            # (2) add noise to reference

        self.forces = next_state[12:16]

        return inputs, next_state, outputs


    @staticmethod
    def to_horizon(series, step, horizon):
        series = np.atleast_2d(series)

        start = step
        end = min(start + horizon, series.shape[-1])
        remain = max(0, horizon - (end - start))

        reference = series[..., start:end]
        repeated = np.tile(series[..., -1:], (1, remain))

        if remain:
            if remain - horizon:
                remaining_series = np.concatenate([reference, repeated], -1)
            else:
                remaining_series = repeated
        else:
            remaining_series = reference

        return remaining_series
