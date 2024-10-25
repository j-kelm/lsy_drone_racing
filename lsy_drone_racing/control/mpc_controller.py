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

# from mpl_toolkits import mplot3d
# import matplotlib.pyplot as plt
# from matplotlib.ticker import FormatStrFormatter

from munch import munchify
import yaml
from numpy._typing import NDArray

from lsy_drone_racing.control import BaseController
from lsy_drone_racing.utils.utils import draw_trajectory, draw_segment_of_traj

from lsy_drone_racing.control.mpc.planner import MinsnapPlanner
from lsy_drone_racing.control.mpc.mpc_control import MPCControl


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

        path = "config/mpc.yaml"
        with open(path, "r") as file:
            mpc_config = munchify(yaml.safe_load(file))

        # Save environment and control parameters.
        self.CTRL_FREQ = initial_info['env.freq']
        self.CTRL_TIMESTEP = 1 / self.CTRL_FREQ
        self.initial_obs = initial_obs
        self.initial_info = initial_info

        self.planner = MinsnapPlanner(initial_info=self.initial_info,
                                      initial_obs=self.initial_obs,
                                      speed=2.0,
                                      )
        self.ctrl = MPCControl(initial_info, mpc_config)

        self.episode_reset()

        self.line = draw_trajectory(initial_info, self.planner.waypoint_pos,
                        self.planner.ref[0], self.planner.ref[1], self.planner.ref[2],
                        num_plot_points=200)

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

        pos = obs['pos']
        rpy = obs['rpy']
        vel = obs['vel']
        body_rates = obs['ang_vel']
        state = np.concatenate([pos, vel, rpy, body_rates])

        if 'step' not in info:
            info['step'] = self._tick

        if len(self.state_history):
            draw_segment_of_traj(self.initial_info, self.state_history[-1][0:3], pos, [0, 1, 0, 1])

        info['gate_prox'] = self.planner.gate_prox

        inputs, next_state, outputs = self.ctrl.compute_control(state, self.planner.ref[:, 1:], info)

        target_pos = next_state[:3]
        target_vel = next_state[3:6]
        target_yaw = next_state[8]
        target_acc = outputs[6:9]
        target_rpy_rates = outputs[12:15]

        action = np.hstack([target_pos, target_vel, target_acc, target_yaw, target_rpy_rates])

        self.state_history.append(state)
        self.action_history.append(action)

        return action


    def episode_reset(self):
        self.action_history = []
        self.state_history = []
        self._tick = 0

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: NDArray[np.floating],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ):
        self._tick += 1

    def save_episode(self, file):
        mpc_states = np.swapaxes(np.array(self.ctrl.ctrl.results_dict['horizon_states']), 0, 1)
        mpc_inputs = np.swapaxes(np.array(self.ctrl.ctrl.results_dict['horizon_inputs']), 0, 1)

        np.savez(file, mpc_states=mpc_states, mpc_inputs=mpc_inputs, mpc_reference=self.ctrl.planner.ref)

    def episode_learn(self):
        # use this function to plot episode data instead of learning
        file_path = "output/states.npz"
        # self.save_episode(file_path)
        # history = np.load(file_path)
        # plot_3d(history)

    def reset(self):
        pass


def plot_all(history, model, dt, mpc_plot_horizon=4):
    # (batch x state x timestep x horizon)

    mpc_states = history['mpc_states'][:, :, :mpc_plot_horizon]
    mpc_inputs = history['mpc_inputs'][:, :, 0]

    state_history = mpc_states[:, :, 0]
    ref = history['mpc_reference']

    plot_length = np.min([np.shape(ref)[1], np.shape(state_history)[1]])
    times = np.linspace(0, dt * plot_length, plot_length)

    # Plot states
    index_list = [0, 1, 2]

    # compute MSE
    mpc_error = ((mpc_states[index_list, 0:plot_length, 1] - ref[index_list, 0:plot_length]) ** 2).mean()
    lowlevel_error = ((np.array(state_history)[index_list, 1:plot_length] - mpc_states[index_list,
                                                                            0:plot_length - 1, 1]) ** 2).mean()

    fig, axs = plt.subplots(len(index_list))
    mpc_label = "mpc"
    for axs_i, state_i in enumerate(index_list):
        axs[axs_i].plot(times, state_history[state_i, 0:plot_length], label='actual')
        axs[axs_i].plot(times, ref[state_i, 0:plot_length], color='r', label='desired')

        # iterate mpc plot horizon
        for timestep_i in range(len(times)):
            if not timestep_i % mpc_plot_horizon and timestep_i + mpc_plot_horizon < len(times):
                axs[axs_i].plot(times[timestep_i:timestep_i + mpc_plot_horizon],
                                mpc_states[state_i, timestep_i], color='y',
                                label=mpc_label)
                mpc_label = "_nolegend_"

        axs[axs_i].set(ylabel=model.STATE_LABELS[state_i] + f'\n[{model.STATE_UNITS[state_i]}]')
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
        axs[axs_i].plot(times, ref[state_i, 0:plot_length], color='r', label='desired')

        # iterate mpc plot horizon
        for timestep_i in range(len(times)):
            if not timestep_i % mpc_plot_horizon and timestep_i + mpc_plot_horizon < len(times):
                axs[axs_i].plot(times[timestep_i:timestep_i + mpc_plot_horizon],
                                mpc_states[state_i, timestep_i], color='y',
                                label=mpc_label)
                mpc_label = "_nolegend_"

        axs[axs_i].set(ylabel=model.STATE_LABELS[state_i] + f'\n[{model.STATE_UNITS[state_i]}]')
        axs[axs_i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        if axs_i != len(index_list) - 1:
            axs[axs_i].set_xticks([])

    axs[0].set_title(f'State Trajectories | MPC MSE: {mpc_error:.4E} | LL MSE: {lowlevel_error:.4E}')
    axs[-1].legend(ncol=3, bbox_transform=fig.transFigure, bbox_to_anchor=(1, 0), loc='lower right')
    axs[-1].set(xlabel='time (sec)')

    index_list = range(4)
    fig, axs = plt.subplots(len(index_list))
    for axs_i, state_i in enumerate(index_list):
        axs[axs_i].plot(times, mpc_inputs[state_i, 0:plot_length], color='r', label='MPC')

        axs[axs_i].set(ylabel=model.INPUT_LABELS[state_i] + f'\n[{model.INPUT_UNITS[state_i]}]')
        axs[axs_i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        if axs_i != len(index_list) - 1:
            axs[axs_i].set_xticks([])

    axs[0].set_title(f'Action Trajectories')
    axs[-1].legend(ncol=3, bbox_transform=fig.transFigure, bbox_to_anchor=(1, 0), loc='lower right')
    axs[-1].set(xlabel='time (sec)')

    plt.show()


def plot_3d(history, mpc_plot_horizon=4):
    # ((TODO: batch) x state x timestep x horizon)
    state_history = history['mpc_states'][0:3 , :, 0]
    ref = history['mpc_reference'][0:3, :]
    ax = plt.axes(projection="3d")

    ax.plot3D(state_history[0], state_history[1], state_history[2])
    ax.plot3D(ref[0], ref[1], ref[2], color="red")
    plt.show()