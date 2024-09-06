"""Test the mpc controller on its own system model.

"""

from __future__ import annotations  # Python 3.10 type hints

import numpy as np

from munch import munchify
import yaml

from examples.planner import Planner, PointPlanner, FilePlanner, LinearPlanner
from examples.mpc_controller import MPC_Controller
from examples.model import Model

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

SAVE_HISTORY = True


# create mock initial info
INFO = {
    'nominal_physical_parameters': {
        'quadrotor_mass': 0.03454,
        'quadrotor_ixx_inertia': 1.4e-05,
        'quadrotor_iyy_inertia': 1.4e-05,
        'quadrotor_izz_inertia': 2.17e-05
    },
    'x_reference': np.array([ 0. ,  0. , -2. ,  0. ,  0.5,  0. ,  0. ,  0. ,  0. ,  0. ,  0. , 0. ]),
    'u_reference': np.array([0.084623, 0.084623, 0.084623, 0.084623]),
    'ctrl_timestep': 0.03333333333333333,
    'ctrl_freq': 30,
    'episode_len_sec': 33,
    'quadrotor_kf': 3.16e-10,
    'quadrotor_km': 7.94e-12,
    'gate_dimensions': {
        'tall': {'shape': 'square', 'height': 1.0, 'edge': 0.45},
        'low': {'shape': 'square', 'height': 0.525, 'edge': 0.45}
    },
    'obstacle_dimensions': {'shape': 'cylinder', 'height': 1.05, 'radius': 0.05},
    'nominal_gates_pos_and_type': [
        [-0.5, -0.5, 0, 0, 0, 3.14, 0]
    ],
    'nominal_obstacles_pos': [],
}

if __name__ == "__main__":
    state = np.array([1.0, 0.0,
                      1.0, 0.0,
                      0.3, 0.0,
                      0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0])

    CTRL_FREQ = 33
    dt = 1.0 / CTRL_FREQ

    # initialize planner
    path = "config/planner_config.yaml"
    with open(path, "r") as file:
        config = munchify(yaml.safe_load(file))

    planner = LinearPlanner(initial_info=INFO, CTRL_FREQ=CTRL_FREQ)
    ref = planner.plan(initial_obs=state, gates=None, duration=config.planner.duration, speed=1.5)

    # initialize mpc controller
    model = Model(info=None)
    ctrl = MPC_Controller(model=model, horizon=int(config.mpc.horizon_sec * CTRL_FREQ), q_mpc=config.mpc.q, r_mpc=config.mpc.r)

    state_history = []

    # loop over time steps
    for step in range(np.shape(ref)[1]):
        state_history.append(state)
        remaining_ref = ref[:, step:]
        action, state = ctrl.select_action(obs=state, info={"ref": remaining_ref})


    # plot results
    mpc_data = ctrl.results_dict['horizon_states']
    mpc_array = np.array(mpc_data)[:, :, 1].transpose()
    state_history = np.array(state_history).transpose()

    # save state history to file
    if SAVE_HISTORY:
        with open('examples/state_history.csv', 'wb') as f:
            np.savetxt(f, state_history, delimiter=',')

    plot_length = np.min([np.shape(ref)[1], np.shape(state_history)[1]])
    times = np.linspace(0, dt * plot_length, plot_length)

    # Plot states
    index_list = [0, 2, 4]

    # compute MSE
    mpc_error = ((mpc_array[index_list, 0:plot_length] - ref[index_list, 0:plot_length]) ** 2).mean()
    lowlevel_error = ((np.array(state_history)[index_list, 1:plot_length] - mpc_array[index_list,
                                                                            0:plot_length - 1]) ** 2).mean()

    fig, axs = plt.subplots(len(index_list))
    for axs_i, state_i in enumerate(index_list):
        axs[axs_i].plot(times, np.array(state_history)[state_i, 0:plot_length], label='actual')
        axs[axs_i].plot(times, ref[state_i, 0:plot_length], color='r', label='desired')
        axs[axs_i].plot(times + dt, mpc_array[state_i, 0:plot_length], color='y', label='mpc')

        axs[axs_i].set(ylabel=model.STATE_LABELS[state_i] + f'\n[{model.STATE_UNITS[state_i]}]')
        axs[axs_i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        if axs_i != len(index_list) - 1:
            axs[axs_i].set_xticks([])

    axs[0].set_title(f'State Trajectories | MPC MSE: {mpc_error:.4E} | LL MSE: {lowlevel_error:.4E}')
    axs[-1].legend(ncol=3, bbox_transform=fig.transFigure, bbox_to_anchor=(1, 0), loc='lower right')
    axs[-1].set(xlabel='time (sec)')

    index_list = [1, 3, 5, 6, 7, 8, 9, 10, 11]
    fig, axs = plt.subplots(len(index_list))
    for axs_i, state_i in enumerate(index_list):

        axs[axs_i].plot(times, np.array(state_history)[state_i, 0:plot_length], label='actual')
        axs[axs_i].plot(times, ref[state_i, 0:plot_length], color='r', label='desired')
        axs[axs_i].plot(times + dt, mpc_array[state_i, 0:plot_length], color='y', label='mpc')

        axs[axs_i].set(ylabel=model.STATE_LABELS[state_i] + f'\n[{model.STATE_UNITS[state_i]}]')
        axs[axs_i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        if axs_i != len(index_list) - 1:
            axs[axs_i].set_xticks([])
    axs[0].set_title('State Trajectories')
    axs[-1].legend(ncol=3, bbox_transform=fig.transFigure, bbox_to_anchor=(1, 0), loc='lower right')
    axs[-1].set(xlabel='time (sec)')

    plt.show()
