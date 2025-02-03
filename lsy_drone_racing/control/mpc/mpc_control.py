from __future__ import annotations  # Python 3.10 type hints

import numpy as np
import numpy.typing as npt

from lsy_drone_racing.control.mpc.mpc import MPC
from lsy_drone_racing.control.mpc.model import DeltaModel as Model
from lsy_drone_racing.control.mpc.constraints import obstacle_constraints, gate_constraints, to_rbf_potential

import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter


class MPCControl:
    def __init__(self, initial_info: dict, initial_obs: dict):
        """Initialization of the controller.

        INSTRUCTIONS:
            The controller's constructor has access the initial state `initial_obs` and the a priori
            information contained in dictionary `initial_info`. Use this method to initialize
            constants, counters, pre-plan trajectories, etc.

        Args:
            initial_obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            initial_info: Additional environment information from the reset.
            config: MPC configuration
        """
        self.config = initial_info['config']
        mpc_config = self.config['mpc']

        initial_info['config'] = mpc_config

        # Get model and constraints
        self.model = Model(info=initial_info)

        constraint_config = mpc_config['constraints']

        self.model.state_constraints += [lambda x: constraint_config['min_thrust'] - x[-4:],
                                              lambda x: x[-4:] - constraint_config['max_thrust']]
        self.model.state_constraints += [lambda x: -constraint_config['max_tilt'] / 180 * np.pi - x[6:8],
                                         lambda x: x[6:8] - constraint_config['max_tilt'] / 180 * np.pi]
        self.model.state_constraints_soft += [lambda x: constraint_config['min_z'] - x[2]]
        self.model.input_constraints += [lambda u: -u - constraint_config['max_thrust_change'] * constraint_config['max_thrust'] * 0.8,
                                              lambda u: u - constraint_config['max_thrust_change'] * constraint_config['max_thrust']]

        ellipsoid_constraints = list()
        for obstacle_pos in initial_obs['obstacles_pos']:
            ellipsoid_constraints += obstacle_constraints(obstacle_pos, r=constraint_config['obstacle_r'], s=1.4)  # 1.3

        for gate_pos, gate_rpy in zip(initial_obs['gates_pos'], initial_obs['gates_rpy']):
            ellipsoid_constraints += gate_constraints(gate_pos, gate_rpy[2], r=constraint_config['gate_r'], s=1.75)  # 1.6

        self.model.state_constraints_soft += [to_rbf_potential(ellipsoid_constraints)]

        ## testing only
        # def figsize(scale, height=1.0):
        #     fig_width_pt = 418.25368  # Get this from LaTeX using \the\textwidth
        #     inches_per_pt = 1.0 / 72.27  # Convert pt to inch
        #     golden_mean = (np.sqrt(5.0) - 1.0) / 2.0 * height  # Aesthetic ratio (you could change this) (I am)
        #     fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
        #     fig_height = fig_width * golden_mean  # height in inches
        #     fig_size = [fig_width, fig_height]
        #     return fig_size
        #
        # mpl.use('pgf')
        #
        # pgf_with_latex = {  # setup matplotlib to use latex for output
        #     "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
        #     "text.usetex": True,  # use LaTeX to write all text
        #     "font.family": "serif",
        #     "font.serif": [],  # blank entries should cause plots to inherit fonts from the document
        #     "font.sans-serif": [],
        #     "font.monospace": [],
        #     "axes.labelsize": 10,  # LaTeX default is 10pt font.
        #     "font.size": 10,
        #     "legend.fontsize": 8,  # Make the legend/label fonts a little smaller
        #     "xtick.labelsize": 8,
        #     "ytick.labelsize": 8,
        #     "figure.figsize": figsize(1.1),  # default fig size of 0.9 textwidth
        #     "pgf.preamble": r"\usepackage[utf8]{inputenc} \usepackage[T1]{fontenc} \usepackage{siunitx}",
        #     # use utf8 fonts becasue your computer can handle it :)
        # }
        # mpl.rcParams.update(pgf_with_latex)
        # import matplotlib.pyplot as plt
        #
        #
        # f = to_rbf_potential(ellipsoid_constraints)
        #
        # x, y = np.meshgrid(np.linspace(2, -2, 400),
        #                    np.linspace(0.5, 1.5, 100))
        #
        # z = np.empty_like(x)
        # for i, (x_i, y_i) in enumerate(zip(x, y)):
        #     for j, (x_j, y_j) in enumerate(zip(x_i, y_i)):
        #         z[i, j] = f(np.array([x_j, y_j, 0.35])[:, None])
        #
        #
        # fig, ax = plt.subplots()
        # img = ax.contourf(-x + 1.5, -y + 1, z, 10, vmin=-0.5, vmax=1.1)
        # ax.set_aspect('equal', 'box')
        # ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f m'))
        # ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f m'))
        # cbar = fig.colorbar(img, fraction=0.015, pad=0.05)
        # img.cmap.set_under('w')
        # img.set_clim(0.0)
        #
        # fig.savefig('output/plots/{}.pgf'.format("potential_function"), dpi=300, bbox_inches='tight')
        # fig.savefig('output/plots/{}.pdf'.format("potential_function"), dpi=300, bbox_inches='tight')



        self.ctrl = MPC(model=self.model,
                        horizon=int(mpc_config['horizon_sec'] * initial_info['env_freq']),
                        q_mpc=mpc_config['q'], r_mpc=mpc_config['r'],
                        soft_penalty=mpc_config['soft_penalty'],
                        err_on_fail=False,
                        horizon_skip=self.config['n_actions'],
                        max_wall_time=mpc_config['max_wall_time']*self.config['n_actions']/initial_info['env_freq'],
                        max_iter=mpc_config['max_iter'],
                        logs=self.config['logs'],
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
        q = self.config['mpc']['q']

        # Slice trajectory for horizon steps, if not long enough, repeat last state.
        remaining_ref = self.to_horizon(ref, step, self.ctrl.T + 1)
        gate_prox = self.to_horizon(info['gate_prox'], step, self.ctrl.T + 1)

        q_pos = np.zeros_like(q)
        q_pos[0:3] = q[0:3]
        info['q'] = np.array(q)[:, np.newaxis] + self.config['mpc']['gate_prioritization'] * np.outer(q, gate_prox)

        horizons = self.ctrl.select_action(obs=state, ref=remaining_ref, info=info)

        self.forces = horizons['states'][-4:, self.config['n_actions']]

        return horizons

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

    @property
    def unwrapped(self):
        return self.ctrl
