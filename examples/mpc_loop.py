"""Test the mpc controller on its own system model.

"""

from __future__ import annotations  # Python 3.10 type hints

import numpy as np

from munch import munchify
import yaml
import toml

from examples.planner import Planner, PointPlanner, FilePlanner, LinearPlanner
from examples.mpc import MPC
from examples.model import Model

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

SAVE_HISTORY = False


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
    'gates.pos': [
        [0.45, -1.0, 0.525],
        [1.0, -1.55, 1.0],
        [0.0, 0.5, 0.525],
        [-0.5, -0.5, 1.0],
    ],
    'gates.rpy': [
        [0.0, 0.0, 2.35],
        [0.0, 0.0, -0.78],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 3.14],
    ],
}

NUM_RUNS = 15


if __name__ == "__main__":

    CTRL_FREQ = 33
    dt = 1.0 / CTRL_FREQ

    # initialize planner
    path = "config/multi_modality.toml"
    with open(path, "r") as file:
        config = toml.load(file)

    # initialize planner
    path = "config/mpc.yaml"
    with open(path, "r") as file:
        mpc_config = munchify(yaml.safe_load(file))

    drone = config['env']['track']['drone']
    nominal_state = np.array(drone['pos'] + drone['vel'] + drone['rpy'] + drone['ang_vel'])

    info = dict()
    info['gates.pos'] = []
    info['gates.rpy'] = []
    gates = config['env']['track']['gates']
    for gate in gates:
        info['gates.pos'].append(np.array(gate['pos']))
        info['gates.rpy'].append(np.array(gate['rpy']))
    info['gates.pos'] = np.array(info['gates.pos'])
    info['gates.rpy'] = np.array(info['gates.rpy'])


    planner = LinearPlanner(initial_info=info, CTRL_FREQ=CTRL_FREQ)
    ref = planner.plan(initial_obs=nominal_state, gates=None, speed=2.0)

    # initialize mpc controller
    model = Model(info=None)
    model.input_constraints += [lambda u: 0.03 - u, lambda u: u - 0.145]  # 0.03 <= thrust <= 0.145
    model.state_constraints += [lambda x: 0.04 - x[2]]
    model.state_constraints += obstacle_constraint([2.0, 1.0, 1.05])
    ctrl = MPC(model=model, horizon=int(mpc_config.mpc.horizon_sec * CTRL_FREQ), q_mpc=mpc_config.mpc.q, r_mpc=mpc_config.mpc.r)

    states = []
    actions = []

    for n in range(NUM_RUNS):

        # randomize initial state
        noise = np.random.uniform(-0.01, 0.01, nominal_state.size)
        state = nominal_state + noise

        ctrl.reset()

        state_history = []
        action_history = []

        # loop over time steps
        for step in range(np.shape(ref)[1]):
            state_history.append(state)
            remaining_ref = ref[:, step:]
            action, state = ctrl.select_action(obs=state, info={"ref": remaining_ref})
            action_history.append(action)

        states.append(state_history)
        actions.append(action_history)


    # plot results
    states = np.swapaxes(np.array(states), 1, 2)
    actions = np.swapaxes(np.array(actions), 1, 2)

    # save state history to file
    if SAVE_HISTORY:
        with open('examples/state_history.csv', 'wb') as f:
            np.savetxt(f, states, delimiter=',')

    plot_length = np.min([np.shape(ref)[1], np.shape(states)[2]])
    times = np.linspace(0, dt * plot_length, plot_length)

    # Plot states
    index_list = [0, 1, 2]

    fig, axs = plt.subplots(len(index_list))
    for axs_i, state_i in enumerate(index_list):
        axs[axs_i].plot(times, ref[state_i, 0:plot_length], color='r', label='desired')

        for run_states in states:
            axs[axs_i].plot(times, np.array(run_states)[state_i, 0:plot_length])


        axs[axs_i].set(ylabel=model.STATE_LABELS[state_i] + f'\n[{model.STATE_UNITS[state_i]}]')
        axs[axs_i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        if axs_i != len(index_list) - 1:
            axs[axs_i].set_xticks([])

    axs[0].set_title(f'State Trajectories')
    axs[-1].legend(ncol=3, bbox_transform=fig.transFigure, bbox_to_anchor=(1, 0), loc='lower right')
    axs[-1].set(xlabel='time (sec)')

    index_list = [3, 4, 5, 6, 7, 8, 9, 10, 11]
    fig, axs = plt.subplots(len(index_list))
    for axs_i, state_i in enumerate(index_list):
        axs[axs_i].plot(times, ref[state_i, 0:plot_length], color='r', label='desired')

        for run_states in states:
            axs[axs_i].plot(times, np.array(run_states)[state_i, 0:plot_length])

        axs[axs_i].set(ylabel=model.STATE_LABELS[state_i] + f'\n[{model.STATE_UNITS[state_i]}]')
        axs[axs_i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        if axs_i != len(index_list) - 1:
            axs[axs_i].set_xticks([])

    axs[0].set_title(f'State Trajectories')
    axs[-1].legend(ncol=3, bbox_transform=fig.transFigure, bbox_to_anchor=(1, 0), loc='lower right')
    axs[-1].set(xlabel='time (sec)')

    index_list = range(4)
    fig, axs = plt.subplots(len(index_list))
    for axs_i, state_i in enumerate(index_list):
        for run_actions in actions:
            axs[axs_i].plot(times, np.array(run_actions)[state_i, 0:plot_length])

        axs[axs_i].set(ylabel=f'T{state_i+1} ' + f'\n[N]')
        axs[axs_i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        if axs_i != len(index_list) - 1:
            axs[axs_i].set_xticks([])

    axs[0].set_title(f'Action Trajectories')
    # axs[-1].legend(ncol=3, bbox_transform=fig.transFigure, bbox_to_anchor=(1, 0), loc='lower right')
    axs[-1].set(xlabel='time (sec)')

    plt.show()
