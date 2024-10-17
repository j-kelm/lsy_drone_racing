from __future__ import annotations  # Python 3.10 type hints

import numpy as np
import h5py

from examples.mpc_control import MPCControl

TRACK_INDEX = 0
RUNS_PER_POINT = 3
STEPS_PER_RUN = 5
CTRL_FREQ = 30
SEED = 33232

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
    rng = np.random.default_rng(seed=SEED)
    randomizer_range = np.array([0.25, 0.25, 0.25, 0.5, 0.5, 0.5, np.pi/4, np.pi/4, np.pi/4, np.pi/2, np.pi/2, np.pi/2, 0.01, 0.01, 0.01, 0.01])
    # randomizer_range = np.array([0.05, 0.05, 0.05, 0.01, 0.01, 0.01, 0, 0, 0, 0, 0, 0, 0.01, 0.01, 0.01, 0.01])
    lower_state_bound = np.array(
        [-3.0, -3.0, 0.0, -2.5, -2.5, -2.5, -np.pi/2, -np.pi/2, -np.inf, -10.0, -10.0, -10.0, 0.03, 0.03, 0.03, 0.03])
    upper_state_bound = np.array(
        [3.0, 3.0, 2.5, 2.5, 2.5, 2.5, np.pi/2, np.pi/2, np.inf, 10.0, 10.0, 10.0, 0.145, 0.145, 0.145, 0.145])



    gates_pos = track_config['gates.pos']
    gates_rpy = track_config['gates.rpy']
    obstacles_pos = track_config['obstacles.pos']
    initial_info = {
        'gates.pos': gates_pos,
        'gates.rpy': gates_rpy,
        'obstacles.pos': obstacles_pos,
        'env.timestep': 1 / CTRL_FREQ,
        'env.freq': CTRL_FREQ,
    }

    ref = track_config['track_reference']
    gate_prox = track_config['gate_prox']
    next_gate_idx = track_config['next_gate']
    ctrl = MPCControl(initial_info, mpc_config)

    worker_config = {
        'seed': SEED,
        'randomizer_range': randomizer_range,
        'lower_state_bound': lower_state_bound,
        'upper_state_bound': upper_state_bound,
    }
    worker_grp = track_grp.require_group(f'worker_{SEED}')
    dict_to_group(worker_grp, 'config', worker_config)


    for init_i, (initial_state, x_guess, u_guess) in enumerate(zip(track_config['initial_states'],
                                                              track_config['x_horizons'],
                                                              track_config['u_horizons'])):


        state_config = {
            'next_gate': ctrl.to_horizon(next_gate_idx, init_i, STEPS_PER_RUN)
        }
        state_grp = worker_grp.require_group(f'point_{init_i}')
        dict_to_group(state_grp, 'config', state_config)

        for run in range(RUNS_PER_POINT):
            # randomize initial state
            lower_bound = np.clip(initial_state - randomizer_range, lower_state_bound, upper_state_bound)
            upper_bound = np.clip(initial_state + randomizer_range, lower_state_bound, upper_state_bound)

            state = rng.uniform(lower_bound, upper_bound, initial_state.size)

            initial_info['next_gate'] = next_gate_idx
            initial_info['init_thrusts'] = state[12:16]

            # clear old warm start and result dict
            ctrl.reset()

            # sample a few steps per initial point
            for step in range(STEPS_PER_RUN):

                inputs, next_state, outputs = ctrl.compute_control(state=state[:12], ref=ref, info={
                    'step': init_i + step,
                    'gate_prox': gate_prox,
                    'x_guess': x_guess,
                    'u_guess': u_guess,
                })
                state = next_state[:12]

                if not ctrl.ctrl.results_dict['solution_found'][-1]:
                    break

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
                'objective': np.array(ctrl.ctrl.results_dict['obj']),
            }

            snippet_grp = state_grp.create_group(f'snippet_{run}')
            dict_to_group(snippet_grp, None, result)




