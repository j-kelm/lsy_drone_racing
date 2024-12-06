import h5py
import numpy as np

from lsy_drone_racing.control.mpc.mpc_utils import states_for_obs, outputs_for_actions
from lsy_drone_racing.control.utils import to_local_obs, to_local_action

hdf_path = "output/merged.hdf5"
output_path = "output/race_data.npz"

PREDICTION_HORIZON = 32
LAST_GATE_INDEX = 0
MAX_SNIPPET_LENGTH = 30

pos_i = slice(0, 3)
vel_i = slice(3, 6)
rpy_i = slice(6, 9)
ang_vel_i = slice(9, 12)

if __name__ == '__main__':
    in_file = h5py.File(hdf_path, 'r', libver='latest')

    # input: state (12,) |  gate_idx (1,) | obstacles (3 x N,) | gates (6 x M,)
    # output: actions (13, T)
    obs = list()
    actions = list()

    local_obs = list()
    local_actions = list()

    # loop over hdf5 leafs (very "efficient")
    for track_key in in_file:
        if 'track_' in track_key:
            track_grp = in_file[track_key]

            obstacles_pos = np.array(track_grp['config/obstacles_pos']).T
            gates_pos = np.array(track_grp['config/gates_pos']).T
            gates_rpy = np.array(track_grp['config/gates_rpy']).T

            for worker_key in track_grp:
                if 'worker_' in worker_key:
                    worker_grp = track_grp[worker_key]
                    for point_key in worker_grp:
                        if 'point_' in point_key:
                            point_grp = worker_grp[point_key]

                            gate_index = np.array(point_grp['config/next_gate']).T[:MAX_SNIPPET_LENGTH]

                            if gate_index[0] > LAST_GATE_INDEX:
                                continue

                            for snippet_key in point_grp:
                                if 'snippet_' in snippet_key:
                                    snippet = point_grp[snippet_key]
                                    # filter bad snippets
                                    if not np.array(snippet['solution_found']).all():
                                        print('No solution')
                                        continue
                                    elif (np.array(snippet['objective']) > 1e3).any():
                                        print('Bad objective')
                                        continue
                                    elif (np.array(snippet['state_slack']) > 5e-2).any():
                                        print('Bad state slack')
                                        continue
                                    elif (np.array(snippet['input_slack']) > 5e-2).any():
                                        print('Bad input slack')
                                        continue
                                    else:
                                        # fetch snippet


                                        init_states = np.array(snippet['initial_states'])[:MAX_SNIPPET_LENGTH, states_for_obs]
                                        horizon_outputs = np.array(snippet['y_horizons'])[:MAX_SNIPPET_LENGTH, :, :PREDICTION_HORIZON]
                                        positions = np.array(snippet['initial_states'])[:MAX_SNIPPET_LENGTH, pos_i]
                                        vels = np.array(snippet['initial_states'])[:MAX_SNIPPET_LENGTH, vel_i]
                                        rpys = np.array(snippet['initial_states'])[:MAX_SNIPPET_LENGTH, rpy_i]
                                        ang_vels = np.array(snippet['initial_states'])[:MAX_SNIPPET_LENGTH, ang_vel_i]

                                        snippet_length = init_states.shape[0]

                                        ## local frame
                                        # transform actions
                                        local_action = to_local_action(horizon_outputs, rpys, positions)
                                        local_actions.append(local_action)

                                        # transform observations
                                        local_ob = to_local_obs(pos=positions,
                                                                 vel=vels,
                                                                 rpy=rpys,
                                                                 ang_vel=ang_vels,
                                                                 obstacles_pos=obstacles_pos,
                                                                 gates_pos=gates_pos,
                                                                 gates_rpy=gates_rpy,
                                                                 target_gate=gate_index,
                                                                 )
                                        local_obs.append(local_ob)

                                        ## global frame
                                        track = np.repeat(np.hstack([obstacles_pos.reshape((1, -1)),
                                                                     gates_pos.reshape((1, -1)),
                                                                     gates_rpy.reshape((1, -1))]),
                                                          snippet_length, axis=0)

                                        obs.append(np.hstack([init_states, gate_index, track]))
                                        actions.append(horizon_outputs[:, outputs_for_actions, :])

    assert len(obs) == len(actions) == len(local_obs) == len(local_actions)

    obs = np.concatenate(obs, axis=0)
    actions = np.concatenate(actions, axis=0)
    local_obs = np.concatenate(local_obs, axis=0)
    local_actions = np.concatenate(local_actions, axis=0)

    np.savez_compressed(output_path, obs=obs, local_obs=local_obs, actions=actions, local_actions=local_actions)



