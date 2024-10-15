from __future__ import annotations  # Python 3.10 type hints

import numpy as np
import h5py

from examples.control import Control

TRACK_INDEX = 0
RUNS_PER_POINT = 1
STEPS_PER_RUN = 5
CTRL_FREQ = 30
SEED = 42

hdf_path = "output/track_data.hdf5"

def dict_to_group(root, name: str | None, data: dict):
    grp = root.create_group(name) if name else root
    for key in data:
        grp[key] = data[key]

def to_dict(grp):
    temp = dict()
    for key in grp:
        temp[key] = np.array(grp[key])
    return temp

if __name__ == "__main__":
    f = h5py.File(hdf_path, 'r+', libver='latest')
    mpc_config = to_dict(f['config'])
    track_grp = f[f'track_{TRACK_INDEX}']
    track_config = track_grp['config']

    # get random generator and seed
    rng = np.random.default_rng(seed=SEED)  # TODO: use index here
    randomizer_range = np.array([0.25, 0.25, 0.25, 0.5, 0.5, 0.5, np.pi/4, np.pi/4, np.pi/4, np.pi/2, np.pi/2, np.pi/2, 0.01, 0.01, 0.01, 0.01])

    worker_grp = track_grp.require_group(f'worker_{SEED}')
    worker_config = {
        'seed': SEED,
        'randomizer_range': randomizer_range,
    }

    gates_pos = track_config['gates.pos']
    gates_rpy = track_config['gates.rpy']
    obstacles_pos = track_config['obstacles.pos']
    initial_info = {
        'gates.pos': gates_pos,
        'gates.rpy': gates_rpy,
        'obstacles.pos': obstacles_pos,
        'ctrl_timestep': 1 / CTRL_FREQ,
        'ctrl_freq': CTRL_FREQ,
    }

    for init_i, (initial_state, next_gate_idx, x_guess, u_guess) in enumerate(zip(track_config['initial_states'],
                                                              track_config['next_gate'],
                                                              track_config['x_horizons'],
                                                              track_config['u_horizons'])):

        state_grp = worker_grp.require_group(f'point_{init_i}')

        for run in range(RUNS_PER_POINT):
            # randomize initial state
            noise = rng.uniform(-randomizer_range, randomizer_range, initial_state.size)
            state = initial_state + noise

            initial_info['gate_index'] = next_gate_idx
            initial_info['init_thrusts'] = state[12:16]
            ctrl = Control(initial_obs=np.array(state[:12]), initial_info=initial_info, config=mpc_config)

            snippet_grp = state_grp.create_group(f'snippet_{run}')
            snippet_config = {
                'init_state': state,
                'reference': ctrl.planner.ref,
                'next_gate': ctrl.planner.next_gate_idx,
                'gate_prox': ctrl.planner.gate_prox,
            }
            dict_to_group(snippet_grp, 'config', snippet_config)


            # sample a few steps per initial point
            for step in range(STEPS_PER_RUN):

                inputs, next_state, outputs = ctrl.compute_control(state=state[:12], info={
                    'step': step,
                    'x_guess': x_guess,
                    'u_guess': u_guess,
                })
                state = next_state[:12]

            result = {
                'x_horizons': np.array(ctrl.ctrl.results_dict['horizon_states']),
                # ndarray (steps, states, horizon + 1)
                'y_horizons': np.array(ctrl.ctrl.results_dict['horizon_outputs']),  # ndarray (steps, outputs, horizon)
                'u_horizons': np.array(ctrl.ctrl.results_dict['horizon_inputs']),  # ndarray (steps, inputs, horizon)
                'ref_horizons': np.array(ctrl.ctrl.results_dict['horizon_references']),
                # ndarray (steps, states, horizon + 1)
                'initial_states': np.array(ctrl.ctrl.results_dict['horizon_states'])[:, :, 0],  # ndarray (states, steps)
                't_wall': np.array(ctrl.ctrl.results_dict['t_wall']),
                'iter_count': np.array(ctrl.ctrl.results_dict['iter_count']),
                'solution_found': np.array(ctrl.ctrl.results_dict['solution_found']),
            }

            dict_to_group(snippet_grp, None, result)




