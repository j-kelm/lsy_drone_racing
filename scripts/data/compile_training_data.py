import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R

hdf_path = "output/race_data.hdf5"
output_path = "output/race_data.npz"

PREDICTION_HORIZON = 32 # steps, must be bigger than horizon from MPC
EGO_PERSPECTIVE = True

if __name__ == '__main__':
    in_file = h5py.File(hdf_path, 'r', libver='latest')

    # input: state (12,) |  gate_idx (1,) | obstacles (3 x N,) | gates (6 x M,)
    # output: actions (13, T)
    inputs = list()
    outputs = list()

    states_for_input = range(12)
    states_for_output = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14]

    # loop over hdf5 leafs in a very "efficient" manner
    for track_key in in_file:
        if 'track_' in track_key:
            track_grp = in_file[track_key]

            obstacles_pos = np.array(track_grp['config/obstacles.pos'])
            gates_pos = np.array(track_grp['config/gates.pos'])
            gates_rpy = np.array(track_grp['config/gates.rpy'])

            for worker_key in track_grp:
                if 'worker_' in worker_key:
                    worker_grp = track_grp[worker_key]
                    for point_key in worker_grp:
                        if 'point_' in point_key:
                            point_grp = worker_grp[point_key]

                            gate_index = np.array(point_grp['config/next_gate'])

                            for snippet_key in point_grp:
                                if 'snippet_' in snippet_key:
                                    snippet = point_grp[snippet_key]
                                    # filter bad snippets
                                    if np.array(snippet['solution_found']).all() and (np.array(snippet['objective']) < 1.0e2).all():
                                        # fetch snippet length
                                        snippet_length = np.array(snippet['solution_found']).shape[0]
                                        init_states = np.array(snippet['initial_states'])[:, states_for_input]

                                        if EGO_PERSPECTIVE:
                                            positions = np.array(snippet['initial_states'])[:, 0:3]
                                            rpys = np.array(snippet['initial_states'])[:, 6:9]
                                            vels = np.array(snippet['initial_states'])[:, 3:6]

                                            r = R.from_euler('zyx', rpys, degrees=True).as_matrix()

                                            # (N, 3, 3) x (N, 4, 3) -> (N, 4, 3)
                                            obstacles_offset = obstacles_pos[None, :, :] - positions[:, None, :]
                                            gates_offset = gates_pos[None, :, :] - positions[:, None, :]
                                            for i in range(snippet_length):
                                                obstacles_offset[i, :, :] = obstacles_offset[i, :, :] @ r[i].T
                                                gates_offset[i, :, :] = gates_offset[i, :, :] @ r[i].T

                                            gates_rpy_obs = np.tensordot(r, gates_rpy, axes=([2,], [1,])).swapaxes(1, 2)

                                            # obstacles_pos_obs = np.tensordot(r, obstacles_pos_obs, axes=([0, 2,], [2, 0,])).swapaxes(1, 2)
                                            obstacles_pos_obs = obstacles_offset.reshape((snippet_length, -1))
                                            gates_pos_obs = gates_offset.reshape((snippet_length, -1))
                                            gates_rpy_obs = gates_rpy_obs.reshape((snippet_length, -1))
                                            obs_states = np.hstack([])
                                            inputs.append(np.hstack([obs_states, gate_index.T, obstacles_pos_obs, gates_pos_obs, gates_rpy_obs]))

                                        else:
                                            init_states = np.array(snippet['initial_states'])[:, states_for_input]
                                            gate_index = np.array(point_grp['config/next_gate'])
                                            obstacles_pos = np.array(track_grp['config/obstacles.pos']).reshape((1, -1))
                                            gates_pos = np.array(track_grp['config/gates.pos']).reshape((1, -1))
                                            gates_rpy = np.array(track_grp['config/gates.rpy']).reshape((1, -1))

                                            track = np.repeat(np.hstack([obstacles_pos, gates_pos, gates_rpy]),
                                                              snippet_length, axis=0)
                                            inputs.append(np.hstack([init_states, gate_index.T, track]))

                                        horizon_outputs = np.array(snippet['y_horizons'])
                                        outputs.append(horizon_outputs[:, states_for_output, :PREDICTION_HORIZON])


    inputs = np.concatenate(inputs, axis=0)
    outputs = np.concatenate(outputs, axis=0)
    np.savez_compressed(output_path, obs=inputs, action=outputs)

                                        



