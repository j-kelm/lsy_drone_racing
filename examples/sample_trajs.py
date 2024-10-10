
import numpy as np
import pandas as pd

from munch import munchify
import yaml

from examples.control import Control

NUM_TRAJS = 1
CTRL_FREQ = 30

if __name__ == "__main__":
    # set up results list
    # track_index | gates_pos gates_rpy obstacles_pos | full_reference state_history action_history gate_index
    #      1      |    3x4       3x4         3x4      |     12xN           12xN           4xN           1xN
    results = list()

    for traj in range(NUM_TRAJS):
        # TODO: Randomize gate, obstacle and starting positions
        gates_pos = np.array([
            [0.45, -1.0, 0.525],
            # [1.0, -1.55, 1.0],
            # [0.0, 0.5, 0.525],
            # [-0.5, -0.5, 1.0],
        ])

        gates_rpy = np.array([
            [0.0, 0.0, 2.35],
            # [0.0, 0.0, -0.78],
            # [0.0, 0.0, 0.0],
            # [0.0, 0.0, 3.14],
        ])

        obstacles_pos = np.array([
            [1.0, -0.5, 1.05],
            [0.5, -1.5, 1.05],
            [-0.5, 0.0, 1.05],
            [0.0, 1.0, 1.05],
        ])

        state = [
            1.0, 1.0, 0.05,
            0, 0, 0,
            0, 0, 0,
            0, 0, 0,
        ]

        initial_info = {
            'gates.pos': gates_pos,
            'gates.rpy': gates_rpy,
            'obstacles.pos': obstacles_pos,
            'ctrl_timestep': 1/CTRL_FREQ,
            'ctrl_freq': CTRL_FREQ,
        }

        # track_df = pd.DataFrame(initial_info)

        #       | gate_pos gate_rpy (obstacle_pos)
        # index | x y z    r p y    (x y z)

        ctrl = Control(initial_obs=np.array(state), initial_info=initial_info)

        # loop mpc through entire track
        for step in range(np.shape(ctrl.planner.ref)[1]):
            info = {
                'step': step,
            }
            inputs, next_state, outputs = ctrl.compute_control(state[:12], info)

        result = {
            'gates.pos': gates_pos,  # ndarray (gates, 3)
            'gates.rpy': gates_rpy,  # ndarray (gates, 3)
            'obstacles.pos': obstacles_pos,  # ndarray (obstacles, 3)
            'track_reference': ctrl.planner.ref,  # ndarray (states, steps)
            'next_gate': ctrl.planner.next_gate_idx,  # ndarray (1, steps)
            'gate_prox': ctrl.planner.gate_prox,  # ndarray (1, steps)
            'x_horizons': np.array(ctrl.ctrl.results_dict['horizon_states']),  # ndarray (steps, states, horizon + 1)
            'y_horizons': np.array(ctrl.ctrl.results_dict['horizon_outputs']),  # ndarray (steps, outputs, horizon)
            'u_horizons': np.array(ctrl.ctrl.results_dict['horizon_inputs']),  # ndarray (steps, inputs, horizon)
            'ref_horizons': np.array(ctrl.ctrl.results_dict['horizon_references']),  # ndarray (steps, states, horizon + 1)
            'initial_states': np.array(ctrl.ctrl.results_dict['horizon_states'])[:, :, 0], # ndarray (states, steps)
        }

        results.append(result)

    df = pd.DataFrame(results)
    df.to_hdf("output/track_data.hdf5", key="track_data", mode='w')
    print(df.head)









