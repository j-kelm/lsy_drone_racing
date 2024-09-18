"""Test the mpc controller on its own system model.

"""

from __future__ import annotations  # Python 3.10 type hints

import numpy as np

from munch import munchify
import yaml
import toml

from examples.planner import Planner, PointPlanner, FilePlanner, LinearPlanner
from examples.mpc_controller import MPC
from examples.model import Model
from examples.constraints import obstacle_constraints, gate_constraints

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

NUM_RUNS = 5

def data_for_cylinder_along_z(center_x,center_y,radius,height_z):
    z = np.linspace(0, height_z, 20)
    theta = np.linspace(0, 2*np.pi, 20)
    theta_grid, z_grid=np.meshgrid(theta, z)
    x_grid = radius*np.cos(theta_grid) + center_x
    y_grid = radius*np.sin(theta_grid) + center_y
    return x_grid,y_grid,z_grid


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
    ref = planner.plan(initial_obs=nominal_state, gates=None, speed=2.5, acc=8.0)

    # initialize mpc controller
    model = Model(info=None)
    model.input_constraints_soft += [lambda u: 0.03 - u, lambda u: u - 0.145]  # 0.03 <= thrust <= 0.145
    model.state_constraints_soft += [lambda x: 0.04 - x[2]]

    model.state_constraints_soft += obstacle_constraints([-2.0, 1.0, 1.05], r=0.2)
    model.state_constraints_soft += gate_constraints([-1.5, 1.0, 0.525], gate_yaw=-np.pi/2, r=0.125)
    ctrl = MPC(model=model, horizon=int(mpc_config.mpc.horizon_sec * CTRL_FREQ), q_mpc=mpc_config.mpc.q, r_mpc=mpc_config.mpc.r)

    states = []
    actions = []

    for n in range(NUM_RUNS):
        # randomize initial state
        noise = np.random.uniform(-2.5e-2, 2.5e-2, nominal_state.size)
        state = nominal_state + noise

        # ref = planner.plan(initial_obs=state, gates=None, speed=1.0)

        ctrl.reset()

        action, _ = ctrl.select_action(obs=state, info={"ref": ref})
        states.append(ctrl.results_dict['horizon_states'][0])
        actions.append(ctrl.results_dict['horizon_inputs'][0])


    # plot results
    states = np.array(states)
    actions = np.array(actions)

    # save state history to file
    if SAVE_HISTORY:
        mpc_states = states
        mpc_inputs = actions
        np.savez("output/multi_modality.npz", mpc_states=mpc_states, mpc_inputs=mpc_inputs, mpc_reference=ref)

    states = np.load("output/nn.npy")

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    # ax.plot3D(ref[0], ref[1], ref[2], color="red")
    Xc, Yc, Zc = data_for_cylinder_along_z(-2.0, 1, 0.18, 1.0)
    ax.plot_surface(Xc, Yc, Zc, alpha=0.1)
    for run_states in states:
        ax.plot3D(run_states[0], run_states[1], run_states[2])

    #ax.set_xlim([-2.6, -1.])
    ax.set_aspect('equal', 'box')
    fig.tight_layout()
    plt.show()

    plot_length = np.min([np.shape(actions)[2], np.shape(states)[2]])
    times = np.linspace(0, dt * plot_length, plot_length)

