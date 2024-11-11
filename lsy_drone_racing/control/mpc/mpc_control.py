from __future__ import annotations  # Python 3.10 type hints

import numpy as np
import numpy.typing as npt

from lsy_drone_racing.control.mpc.mpc import MPC
from lsy_drone_racing.control.mpc.model import DeltaModel as Model
from lsy_drone_racing.control.mpc.constraints import obstacle_constraints, gate_constraints, to_rbf_potential


class MPCControl:
    def __init__(self, initial_info: dict, initial_obs: dict, config: dict):
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
        self.CTRL_FREQ = initial_info["env_freq"]
        self.CTRL_TIMESTEP = 1 / self.CTRL_FREQ

        self.initial_info = initial_info
        self.initial_obs = initial_obs

        self.config = config
        # self.multi_starts = self.config['multi_starts']

        # Get model and constraints
        self.model = Model(info=self.initial_info)

        self.model.state_constraints += [lambda x: 0.03 - x[12:16], lambda x: x[12:16] - 0.145] # 0.03 <= thrust <= 0.145
        self.model.state_constraints_soft += [lambda x: -80 / 180 * np.pi - x[6:8], lambda x: x[6:8] - 80 / 180 * np.pi] # max roll and pitch
        self.model.state_constraints_soft += [lambda x: 0.05 - x[2]]

        self.model.input_constraints_soft += [lambda u: -10 * 0.145 / self.CTRL_FREQ - u, lambda u: u - 10 * 0.145 / self.CTRL_FREQ]
        # self.model.state_constraints_soft += [lambda x: -3.0 - x[1], lambda x: x[1] - 3.0]
        # self.model.state_constraints_soft += [lambda x: -3.0 - x[0], lambda x: x[0] - 3.0]

        ellipsoid_constraints = list()
        for obstacle_pos in self.initial_obs['obstacles_pos']:
            ellipsoid_constraints += obstacle_constraints(obstacle_pos, r=0.17) # r = 0.14

        for gate_pos, gate_rpy in zip(self.initial_obs['gates_pos'], self.initial_obs['gates_rpy']):
            ellipsoid_constraints += gate_constraints(gate_pos, gate_rpy[2], r=0.13) # r = 0.12

        self.model.state_constraints_soft += [to_rbf_potential(ellipsoid_constraints)]

        self.ctrl = MPC(model=self.model,
                        horizon=int(self.config['horizon_sec'] * self.CTRL_FREQ),
                        q_mpc=self.config['q'], r_mpc=self.config['r'],
                        soft_penalty=5e3,
                        err_on_fail=False,
                        horizon_skip=config['ratio'],
        )

        self.forces = initial_info['init_thrusts'] if 'init_thrusts' in initial_info and initial_info[
            'init_thrusts'] is not None else self.ctrl.X_EQ[-4:]

        self.state = None


    def compute_control(
        self, state, ref, info: dict,
    ) -> npt.NDarray[np.floating]:
        """Compute the next desired position and orientation of the drone.

        INSTRUCTIONS:
            Re-implement this method to return the target pose to be sent from Crazyswarm to the
            Crazyflie using the `cmdFullState` call.

        Args:
            state: The current observation of the environment. See the environment's observation space
                for details.
            ref: fullstate reference
            info: Optional additional information as a dictionary.

        Returns:
            The drone pose [x_des, y_des, z_des, yaw_des] as a numpy array.
        """


        state = np.concatenate([state, self.forces], axis=0)
        step = info['step']

        # Slice trajectory for horizon steps, if not long enough, repeat last state.
        remaining_ref = self.to_horizon(ref, step, self.ctrl.T + 1)
        gate_prox = self.to_horizon(info['gate_prox'], step, self.ctrl.T + 1)

        q_pos = np.zeros_like(self.config['q'])
        q_pos[0:3] = self.config['q'][0:3]
        info['q'] = np.array(self.config['q'])[:, np.newaxis] + 5.0 * np.outer(self.config['q'], gate_prox) # 4.0 * np.outer(q_pos, gate_prox)
        # try:
        #     inputs, states, outputs = self.ctrl.select_action(obs=state,
        #                                                           ref=remaining_ref,
        #                                                           info=info,
        #                                                           err_on_fail=True)
        #
        # except RuntimeError:  # use reference warm start on fail
        #     print('[WARN] First attempt failed, trying again with reference warm-start.')
        #     info['x_guess'] = None
        #     inputs, states, outputs = self.ctrl.select_action(obs=state,
        #                                                           ref=remaining_ref,
        #                                                           info=info,
        #                                                           err_on_fail=False,
        #                                                           force_warm_start=True)

        inputs, states, outputs = self.ctrl.select_action(obs=state,
                                                          ref=remaining_ref,
                                                          info=info)


        self.forces = states[12:16, 1]

        return inputs, states, outputs

    def reset(self):
        # clear warm start and result dict
        self.ctrl.reset()

    @staticmethod
    def to_horizon(series, step, horizon):
        series = np.atleast_2d(series)

        start = step
        end = min(start + horizon, series.shape[-1])
        remain = np.clip(horizon - (end - start), 0, horizon)

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
