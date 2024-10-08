
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

    # load state history
    traj_states = traj_data['state_history']

    # get run index
    index = 42

    # get random generator and seed
    rng = np.random.default_rng(seed=index)  # TODO: use index here
    randomizer_range = np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25, np.pi/4, np.pi/4, np.pi/4, np.pi, np.pi, np.pi])/10

    horizon_states = list()
    horizon_actions = list()
    horizon_refs = list()
    initial_states = list()

    initial_info = {
        'gates.pos': traj_data['gates.pos'],
        'gates.rpy': traj_data['gates.rpy'],
        'obstacles.pos': traj_data['obstacles.pos'],
        'ctrl_timestep': 1 / CTRL_FREQ,
        'ctrl_freq': CTRL_FREQ,
    }

    for i, initial_state in enumerate(traj_states.T):
        for run in range(RUNS_PER_POINT):
            # randomize initial state
            noise = rng.uniform(-randomizer_range, randomizer_range, initial_state.size)
            state = initial_state + noise

            initial_info['gate_index'] = traj_data['next_gate'][i]

            ctrl = Control(initial_obs=np.array(state), initial_info=initial_info)

            for step in range(STEPS_PER_RUN):
                inputs, next_state, outputs = ctrl.compute_control(state=state, info={
                    'step': step,
                    'x_guess': None,
                })

                horizon_states.append(ctrl.ctrl.results_dict['horizon_states'])
                horizon_actions.append(ctrl.ctrl.results_dict['horizon_inputs'])
                horizon_refs.append(ctrl.ctrl.results_dict['goal_states'])
                initial_states.append(ctrl.ctrl.results_dict['horizon_states'][0])

    output = pd.DataFrame(
        {
            'horizon_states': horizon_states,
            'horizon_actions': horizon_actions,
            'horizon_reference': horizon_refs,
            'initial_state': initial_states,
        }
    )

    print(output.head())



