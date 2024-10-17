import h5py
import numpy as np

hdf_path = "output/track_data.hdf5"
output_path = "output/training_data.hdf5"

PREDICTION_HORIZON = 15 # steps, must be bigger than horizon from MPC

if __name__ == '__main__':
    in_file = h5py.File(hdf_path, 'r', libver='latest')
    out_file = h5py.File(output_path, 'w', libver='latest')

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
            for worker_key in track_grp:
                if 'worker_' in worker_key:
                    worker_grp = track_grp[worker_key]
                    for point_key in worker_grp:
                        if 'point_' in point_key:
                            point_grp = worker_grp[point_key]
                            for snippet_key in point_grp:
                                if 'snippet_' in snippet_key:
                                    snippet = point_grp[snippet_key]
                                    if np.array(snippet['solution_found']).all():
                                        # fetch snippet length
                                        snippet_length = np.array(snippet['solution_found']).shape[0]

                                        init_states = np.array(snippet['initial_states'])[:, states_for_input]
                                        gate_index = snippet['config/next_gate'][:snippet_length]
                                        obstacles_pos = np.array(track_grp['config/obstacles.pos']).reshape((1, -1))
                                        gates_pos = np.array(track_grp['config/gates.pos']).reshape((1, -1))
                                        gates_rpy = np.array(track_grp['config/gates.rpy']).reshape((1, -1))
                                        track = np.repeat(np.hstack([obstacles_pos, gates_pos, gates_rpy]), snippet_length, axis=0)
                                        inputs.append(np.hstack([init_states, gate_index[:, None], track]))

                                        horizon_outputs = np.array(snippet['y_horizons'])
                                        outputs.append(horizon_outputs[:, states_for_output, :PREDICTION_HORIZON])


    inputs = np.concatenate(inputs, axis=0)
    outputs = np.concatenate(outputs, axis=0)
    np.savez_compressed("output/training.npz", inputs=inputs, outputs=outputs)

                                        



