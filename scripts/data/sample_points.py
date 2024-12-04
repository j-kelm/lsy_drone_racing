from __future__ import annotations  # Python 3.10 type hints

import numpy as np
import h5py
import os
import argparse

from lsy_drone_racing.control.mpc.mpc_control import MPCControl

def dict_to_group(root, name: str, data: dict):
    grp = root.create_group(name)
    for key in data:
        if isinstance(data[key], dict):
            dict_to_group(grp, key, data[key])
        else:
            grp[key] = data[key]

def to_dict(grp):
    temp = dict()
    for key in grp:
        if isinstance(grp[key], h5py.Dataset):
            temp[key] = np.array(grp[key])
        elif isinstance(grp[key], h5py.Group):
            temp[key] = to_dict(grp[key])
    return temp

if __name__ == "__main__":
    parser = argparse.ArgumentParser("sample_points")
    parser.add_argument("--track", "-t", help="Index of the track to be used for sampling.", type=int)
    parser.add_argument("--seed", "-s", help="Seed used for sampling randomized initial states.", type=int)
    parser.add_argument("--list", "-i", help="Input file containing config and tracks.", type=str)
    parser.add_argument("--out", "-o", help="Output path.", type=str)
    parser.add_argument("--runs", "-r", help="Amount of randomized initial states sampled for each reference step.", type=int, default=1)
    parser.add_argument("--steps" "-n", help="Steps to sample from each randomized initial state.", type=int, default=10)
    # parser.add_argument("--goal", "-g", help="Index of the last goal to be reached", type=int, default=-1)

    args = parser.parse_args()

    # copy tracklist/config to own worker file
    with h5py.File(os.path.join(args.out, f'{args.track}_{args.seed}.hdf5'), 'w-', libver='latest') as out_file:
        with h5py.File(args.list, 'r', libver='latest') as in_file:
            out_file.copy(source=in_file['config'], dest=out_file)
            out_file.copy(source=in_file[f'track_{args.track}'], dest=f'track_{args.track}')

        mpc_config = to_dict(out_file['config'])
        track_grp = out_file[f'track_{args.track}']
        track_config = track_grp['config']

        # get random generator and seed
        rng = np.random.default_rng(seed=args.seed)
        randomizer_range = np.array([
            0.25, 0.25, 0.25,
            0.25, 0.25, 0.25,
            np.pi/6, np.pi/6, np.pi/2,
            np.pi/8, np.pi/8, np.pi/8,
            0.02, 0.02, 0.02, 0.02])
        lower_state_bound = np.array([
            -3.0, -3.0, -mpc_config['constraints']['min_z'],
            -2.0, -2.0, -2.0,
            -np.pi, -np.pi, -np.inf,
            -10.0, -10.0, -10.0,
            *[mpc_config['constraints']['min_thrust']]*4])
        upper_state_bound = np.array([
            3.0, 3.0, 2.5,
            3.0, 3.0, 3.0,
            np.pi, np.pi, np.inf,
            10.0, 10.0, 10.0,
            *[mpc_config['constraints']['max_thrust']]*4])

        initial_info = {
            'env_freq': mpc_config['env_freq'],
            'nominal_physical_parameters': mpc_config['drone_params'],
        }

        initial_obs = track_config

        ref = track_config['track_reference']
        gate_prox = track_config['gate_prox']
        next_gate_idx = track_config['next_gate']

        ctrl = MPCControl(initial_info, initial_obs=initial_obs, config=mpc_config)

        worker_config = {
            'seed': args.seed,
            'randomizer_range': randomizer_range,
            'lower_state_bound': lower_state_bound,
            'upper_state_bound': upper_state_bound,
        }
        worker_grp = track_grp.create_group(f'worker_{args.seed}')
        dict_to_group(worker_grp, 'config', worker_config)


        for init_i, (initial_state, x_guess, u_guess) in enumerate(zip(track_config['initial_states'],
                                                                  track_config['x_horizons'],
                                                                  track_config['u_horizons'])):

            state_config = {
                'next_gate': ctrl.to_horizon(next_gate_idx, init_i, args.steps_n)
            }
            state_grp = worker_grp.require_group(f'point_{init_i}')
            dict_to_group(state_grp, 'config', state_config)

            print(f'[WORKER{args.seed}] Step {init_i}')

            for run in range(args.runs):
                # randomize initial state
                lower_bound = np.clip(initial_state - randomizer_range, lower_state_bound, upper_state_bound)
                upper_bound = np.clip(initial_state + randomizer_range, lower_state_bound, upper_state_bound)

                state = rng.uniform(lower_bound, upper_bound, initial_state.size)

                initial_info['next_gate'] = next_gate_idx
                initial_info['init_thrusts'] = state[12:16]

                # clear old warm start and result dict
                ctrl.reset()

                # sample a few steps per initial point, automatic warm start from reference
                for step in range(args.steps_n):

                    inputs, states, outputs = ctrl.compute_control(state=state[:12], ref=ref, info={
                        'step': init_i + step,
                        'gate_prox': gate_prox,
                        #'x_guess': x_guess,
                        #'u_guess': u_guess,
                    })
                    state = states[:12, 1]

                    if not ctrl.ctrl.results_dict['solution_found'][-1]:
                        break

                result = {
                    'x_horizons': np.array(ctrl.ctrl.results_dict['horizon_states']),
                    # ndarray (steps, states, horizon + 1)
                    'y_horizons': np.array(ctrl.ctrl.results_dict['horizon_outputs']),  # ndarray (steps, outputs, horizon)
                    'u_horizons': np.array(ctrl.ctrl.results_dict['horizon_inputs']),  # ndarray (steps, inputs, horizon)
                    'state_slack': np.array(ctrl.ctrl.results_dict['horizon_state_slack']),
                    'input_slack': np.array(ctrl.ctrl.results_dict['horizon_input_slack']),
                    'ref_horizons': np.array(ctrl.ctrl.results_dict['horizon_references']),
                    # ndarray (steps, states, horizon + 1)
                    'initial_states': np.array(ctrl.ctrl.results_dict['horizon_states'])[:, :, 0],  # ndarray (states, steps)
                    't_wall': np.array(ctrl.ctrl.results_dict['t_wall']),
                    'iter_count': np.array(ctrl.ctrl.results_dict['iter_count']),
                    'solution_found': np.array(ctrl.ctrl.results_dict['solution_found']),
                    'objective': np.array(ctrl.ctrl.results_dict['obj']),
                }

                dict_to_group(state_grp, f'snippet_{run}', result)

    print(f'[WORKER{args.seed}] DONE')




