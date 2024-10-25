import numpy as np
import h5py

from munch import munchify
import yaml
import toml

from lsy_drone_racing.control.mpc.mpc_control import MPCControl
from lsy_drone_racing.control.mpc.planner import MinsnapPlanner

NUM_TRACKS = 1
CTRL_FREQ = 30
hdf_path = "output/mm.hdf5"

def dict_to_group(root, name: str, data: dict):
    grp = root.create_group(name)
    for key in data:
        grp[key] = data[key]

if __name__ == "__main__":
    path = "config/mpc.yaml"
    with open(path, "r") as file:
        mpc_config = munchify(yaml.safe_load(file))

    mpc_config['ctrl_timestep'] = 1 / CTRL_FREQ
    mpc_config['env.freq'] = CTRL_FREQ

    f = h5py.File(hdf_path, 'w', libver='latest')

    dict_to_group(f, 'config', mpc_config)

    for track_i in range(NUM_TRACKS):
        # TODO: Randomize gate, obstacle and starting positions
        with open('config/cluttered_new.toml', "r") as file:
            track_config = munchify(toml.load(file))

        gates = track_config.env.track.gates
        gates_pos = [gate.pos for gate in gates]
        gates_rpy = [gate.rpy for gate in gates]
        obstacles_pos = [obstacle.pos for obstacle in track_config.env.track.obstacles]

        initial_info = {
            'gates.pos': gates_pos,
            'gates.rpy': gates_rpy,
            'obstacles.pos': obstacles_pos,
            'env.freq': CTRL_FREQ,
        }

        planner = MinsnapPlanner(initial_info=initial_info,
                                      initial_obs=track_config.env.track.drone,
                                      speed=1.5,
                                      )

        state = track_config.env.track.drone
        state = np.concatenate([state['pos'], state['vel'], state['rpy'], state['ang_vel']])

        ctrl = MPCControl(initial_info, mpc_config)

        # loop mpc through entire track
        for step in range(np.shape(planner.ref)[1]):
            info = {
                'step': step,
                'gate_prox': planner.gate_prox
            }
            inputs, next_state, outputs = ctrl.compute_control(state, planner.ref, info)
            state = next_state[:12]

        result = {
            'gates.pos': gates_pos,  # ndarray (gates, 3)
            'gates.rpy': gates_rpy,  # ndarray (gates, 3)
            'obstacles.pos': obstacles_pos,  # ndarray (obstacles, 3)
            'track_reference': planner.ref,  # ndarray (states, steps)
            'next_gate': planner.next_gate_idx,  # ndarray (1, steps)
            'gate_prox': planner.gate_prox,  # ndarray (1, steps)
            'x_horizons': np.array(ctrl.ctrl.results_dict['horizon_states']), # ndarray (steps, states, horizon + 1)
            'y_horizons': np.array(ctrl.ctrl.results_dict['horizon_outputs']),  # ndarray (steps, outputs, horizon)
            'u_horizons': np.array(ctrl.ctrl.results_dict['horizon_inputs']),  # ndarray (steps, inputs, horizon)
            'ref_horizons': np.array(ctrl.ctrl.results_dict['horizon_references']), # ndarray (steps, states, horizon + 1)
            'initial_states': np.array(ctrl.ctrl.results_dict['horizon_states'])[:, :, 0],  # ndarray (states, steps)
            't_wall': np.array(ctrl.ctrl.results_dict['t_wall']),
            'iter_count': np.array(ctrl.ctrl.results_dict['iter_count']),
            'solution_found': np.array(ctrl.ctrl.results_dict['solution_found']),
            'objective': np.array(ctrl.ctrl.results_dict['obj']),
        }

        track_grp = f.create_group(f'track_{track_i}')
        dict_to_group(track_grp, 'config', result)












