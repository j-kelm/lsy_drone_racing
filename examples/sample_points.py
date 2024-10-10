
import numpy as np
import pandas as pd

from examples.control import Control

TRACK_INDEX = 0
RUNS_PER_POINT = 2
STEPS_PER_RUN = 3
CTRL_FREQ = 30

# fully shared mpc config file
# track file with tracks, containing obstacles & references

if __name__ == "__main__":
    # get traj data
    traj_data = pd.read_hdf("output/track_data.hdf5").loc[TRACK_INDEX]

    # get run index
    index = 42

    # get random generator and seed
    rng = np.random.default_rng(seed=index)  # TODO: use index here
    randomizer_range = np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25, np.pi/4, np.pi/4, np.pi/4, np.pi, np.pi, np.pi, 0.01, 0.01, 0.01, 0.01])

    gates_pos = traj_data['gates.pos']
    gates_rpy = traj_data['gates.rpy']
    obstacles_pos = traj_data['obstacles.pos']

    initial_info = {
        'gates.pos': gates_pos,
        'gates.rpy': gates_rpy,
        'obstacles.pos': obstacles_pos,
        'ctrl_timestep': 1 / CTRL_FREQ,
        'ctrl_freq': CTRL_FREQ,
    }

    results = list()

    for initial_state, next_gate_idx, x_guess, u_guess in zip(traj_data['initial_states'],
                                                              traj_data['next_gate'],
                                                              traj_data['x_horizons'],
                                                              traj_data['u_horizons']):
        for run in range(RUNS_PER_POINT):
            # randomize initial state
            noise = rng.uniform(-randomizer_range, randomizer_range, initial_state.size)
            state = initial_state + noise

            initial_info['gate_index'] = next_gate_idx
            initial_info['init_thrusts'] = state[12:16]

            ctrl = Control(initial_obs=np.array(state[:12]), initial_info=initial_info)

            # sample a few steps per initial point
            for step in range(STEPS_PER_RUN):
                inputs, next_state, outputs = ctrl.compute_control(state=state[:12], info={
                    'step': step,
                    'x_guess': x_guess,
                    'u_guess': u_guess,
                })

            result = {
                'ctrl_timestep': 1 / CTRL_FREQ,
                'ctrl_freq': CTRL_FREQ,
                'gates.pos': gates_pos,  # ndarray (gates, 3)
                'gates.rpy': gates_rpy,  # ndarray (gates, 3)
                'obstacles.pos': obstacles_pos,  # ndarray (obstacles, 3)
                'track_reference': ctrl.planner.ref,  # ndarray (states, steps)
                'next_gate': ctrl.planner.next_gate_idx,  # ndarray (1, steps)
                'gate_prox': ctrl.planner.gate_prox,  # ndarray (1, steps)
                'x_horizons': np.array(ctrl.ctrl.results_dict['horizon_states']),
                # ndarray (steps, states, horizon + 1)
                'y_horizons': np.array(ctrl.ctrl.results_dict['horizon_outputs']),  # ndarray (steps, outputs, horizon)
                'u_horizons': np.array(ctrl.ctrl.results_dict['horizon_inputs']),  # ndarray (steps, inputs, horizon)
                'ref_horizons': np.array(ctrl.ctrl.results_dict['horizon_references']),
                # ndarray (steps, states, horizon + 1)
                'initial_states': np.array(ctrl.ctrl.results_dict['horizon_states'])[:, :, 0],  # ndarray (states, steps)
            }

            results.append(result)

    df = pd.DataFrame(results)
    df.to_hdf("output/track_data.hdf5", key=f"sampled_data_{index}", mode='w')

    print(df.head())



