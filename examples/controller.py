"""Write your control strategy.

Then run:

    $ python scripts/sim --config config/getting_started.yaml

Tips:
    Search for strings `INSTRUCTIONS:` and `REPLACE THIS (START)` in this file.

    Change the code between the 5 blocks starting with
        #########################
        # REPLACE THIS (START) ##
        #########################
    and ending with
        #########################
        # REPLACE THIS (END) ####
        #########################
    with your own code.

    They are in methods:
        1) __init__
        2) compute_control
        3) step_learn (optional)
        4) episode_learn (optional)

"""

from __future__ import annotations  # Python 3.10 type hints

import numpy as np
import numpy.typing as npt

from munch import munchify
import yaml

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from lsy_drone_racing.controller import BaseController

from examples.planner import Planner, FilePlanner, LinearPlanner, PointPlanner
from examples.mpc_controller import MPC
from examples.model import Model
from examples.constraints import obstacle_constraints, gate_constraints


class Controller(BaseController):
    """Template controller class."""

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
        super().__init__(initial_obs, initial_info)

        # Save environment and control parameters.
        self.CTRL_TIMESTEP = initial_info["ctrl_timestep"]
        self.CTRL_FREQ = initial_info["ctrl_freq"]
        self.initial_obs = initial_obs
        self.initial_info = initial_info

        self.episode_reset()

        path = "config/mpc.yaml"
        with open(path, "r") as file:
            config = munchify(yaml.safe_load(file))

        # initialize planner
        self.planner = LinearPlanner(initial_info=initial_info, CTRL_FREQ=self.CTRL_FREQ)
        self.ref = self.planner.plan(initial_obs=self.initial_obs, gates=None, speed=2.0)

        # Get model and constraints
        self.model = Model(info=None)
        self.model.input_constraints += [lambda u: 0.03 - u, lambda u: u - 0.145] # 0.03 <= thrust <= 0.145
        self.model.state_constraints += [lambda x: 0.04 - x[2]]

        for obstacle_pos in self.initial_info['obstacles.pos']:
            self.model.state_constraints_soft += obstacle_constraints(obstacle_pos, r=0.125)

        for gate_pos, gate_rpy in zip(self.initial_info['gates.pos'], self.initial_info['gates.rpy']):
            self.model.state_constraints_soft += gate_constraints(gate_pos, gate_rpy[2], r=0.125)

        self.ctrl = MPC(model=self.model, horizon=int(config.mpc.horizon_sec * self.CTRL_FREQ),
                                   q_mpc=config.mpc.q, r_mpc=config.mpc.r)

        self.state = None
        self.state_history = []
        self.action_history = []

        # TODO: draw reference

    def compute_control(
        self, obs: npt.NDArray[np.floating], info: dict | None = None
    ) -> npt.NDarray[np.floating]:
        """Compute the next desired position and orientation of the drone.

        INSTRUCTIONS:
            Re-implement this method to return the target pose to be sent from Crazyswarm to the
            Crazyflie using the `cmdFullState` call.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The drone pose [x_des, y_des, z_des, yaw_des] as a numpy array.
        """

        remaining_ref = self.ref[:, info['step']:]
        pos = obs[0:3]
        rpy = obs[3:6]
        vel = obs[6:9]
        body_rates = obs[9:12]
        self.state = np.concatenate([pos, vel, rpy, body_rates])

        action, next_state = self.ctrl.select_action(obs=self.state, info={"ref": remaining_ref})
        target_pos = next_state[:3]
        target_vel = next_state[3:6]

        target_yaw = next_state[8]

        # calculate target acc
        y = np.array(self.model.symbolic.g_func(x=next_state, u=action)['g']).flatten()
        target_acc = y[6:9]
        target_rpy_rates = y[12:15]

        self.state_history.append(self.state)
        self.action_history.append(action)

        action = np.hstack([target_pos, target_vel, target_acc, target_yaw, target_rpy_rates])
        return action

    def episode_reset(self):
        self.action_history = []
        self.state_history = []

    def episode_learn(self):
        mpc_plot_horizon = 4

        mpc_states = np.swapaxes(np.array(self.ctrl.results_dict['horizon_states'])[:, :, :mpc_plot_horizon], 0, 1)
        mpc_inputs = np.swapaxes(np.array(self.ctrl.results_dict['horizon_inputs'])[:, :, 0], 0, 1)
        state_history = np.array(self.state_history).transpose()
        action_history = np.array(self.action_history).transpose()

        plot_length = np.min([np.shape(self.ref)[1], np.shape(state_history)[1]])
        times = np.linspace(0, self.CTRL_TIMESTEP * plot_length, plot_length)

        # Plot states
        index_list = [0, 1, 2]

        # compute MSE
        mpc_error = ((mpc_states[index_list, 0:plot_length, 1] - self.ref[index_list, 0:plot_length]) ** 2).mean()
        lowlevel_error = ((np.array(state_history)[index_list, 1:plot_length] - mpc_states[index_list,
                                                                                0:plot_length - 1, 1]) ** 2).mean()

        fig, axs = plt.subplots(len(index_list))
        mpc_label = "mpc"
        for axs_i, state_i in enumerate(index_list):
            axs[axs_i].plot(times, state_history[state_i, 0:plot_length], label='actual')
            axs[axs_i].plot(times, self.ref[state_i, 0:plot_length], color='r', label='desired')

            # iterate mpc plot horizon
            for timestep_i in range(len(times)):
                if not timestep_i % mpc_plot_horizon and timestep_i + mpc_plot_horizon < len(times):
                    axs[axs_i].plot(times[timestep_i:timestep_i + mpc_plot_horizon],
                                    mpc_states[state_i, timestep_i], color='y',
                                    label=mpc_label)
                    mpc_label = "_nolegend_"

            axs[axs_i].set(ylabel=self.model.STATE_LABELS[state_i] + f'\n[{self.model.STATE_UNITS[state_i]}]')
            axs[axs_i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            if axs_i != len(index_list) - 1:
                axs[axs_i].set_xticks([])

        axs[0].set_title(f'State Trajectories | MPC MSE: {mpc_error:.4E} | LL MSE: {lowlevel_error:.4E}')
        axs[-1].legend(ncol=3, bbox_transform=fig.transFigure, bbox_to_anchor=(1, 0), loc='lower right')
        axs[-1].set(xlabel='time (sec)')

        index_list = [3, 4, 5, 6, 7, 8, 9, 10, 11]
        fig, axs = plt.subplots(len(index_list))
        mpc_label = "mpc"
        for axs_i, state_i in enumerate(index_list):
            axs[axs_i].plot(times, state_history[state_i, 0:plot_length], label='actual')
            axs[axs_i].plot(times, self.ref[state_i, 0:plot_length], color='r', label='desired')

            # iterate mpc plot horizon
            for timestep_i in range(len(times)):
                if not timestep_i % mpc_plot_horizon and timestep_i + mpc_plot_horizon < len(times):
                    axs[axs_i].plot(times[timestep_i:timestep_i + mpc_plot_horizon],
                                    mpc_states[state_i, timestep_i], color='y',
                                    label=mpc_label)
                    mpc_label = "_nolegend_"

            axs[axs_i].set(ylabel=self.model.STATE_LABELS[state_i] + f'\n[{self.model.STATE_UNITS[state_i]}]')
            axs[axs_i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            if axs_i != len(index_list) - 1:
                axs[axs_i].set_xticks([])

        axs[0].set_title(f'State Trajectories | MPC MSE: {mpc_error:.4E} | LL MSE: {lowlevel_error:.4E}')
        axs[-1].legend(ncol=3, bbox_transform=fig.transFigure, bbox_to_anchor=(1, 0), loc='lower right')
        axs[-1].set(xlabel='time (sec)')

        index_list = range(4)
        fig, axs = plt.subplots(len(index_list))
        for axs_i, state_i in enumerate(index_list):
            axs[axs_i].plot(times, action_history[state_i, 0:plot_length], label='Low-Level Controller')
            axs[axs_i].plot(times, mpc_inputs[state_i, 0:plot_length], color='r', label='MPC')

            axs[axs_i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            if axs_i != len(index_list) - 1:
                axs[axs_i].set_xticks([])

        axs[0].set_title(f'Action Trajectories')
        axs[-1].legend(ncol=3, bbox_transform=fig.transFigure, bbox_to_anchor=(1, 0), loc='lower right')
        axs[-1].set(xlabel='time (sec)')

        plt.show()

    def reset(self):
        pass
