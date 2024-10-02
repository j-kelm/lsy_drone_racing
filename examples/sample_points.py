
import numpy as np
import pandas as pd

from munch import munchify
import yaml

from examples.model import Model
from examples.mpc_controller import MPC
from examples.constraints import obstacle_constraints, gate_constraints

TRACK_INDEX = 0
RUNS_PER_POINT = 2
STEPS_PER_RUN = 3
CTRL_FREQ = 30

# fully shared mpc config file
# track file with tracks, containing obstacles & references

if __name__ == "__main__":
    # get mpc config
    path = "config/mpc.yaml"
    with open(path, "r") as file:
        mpc_config = munchify(yaml.safe_load(file))

    # get traj data
    traj_data = pd.read_hdf("output/track_data.hdf5").loc[TRACK_INDEX]

    # initialize mpc controller
    model = Model(info=None)
    model.input_constraints_soft += [lambda u: 0.03 - u, lambda u: u - 0.145]  # 0.03 <= thrust <= 0.145
    model.state_constraints_soft += [lambda x: 0.04 - x[2]]

    for obstacle_pos in traj_data['obstacles.pos']:
        model.state_constraints_soft += obstacle_constraints(obstacle_pos, r=0.125)

    for gate_pos, gate_rpy in zip(traj_data['gates.pos'], traj_data['gates.rpy']):
        model.state_constraints_soft += gate_constraints(gate_pos, gate_rpy[2], r=0.11)

    ctrl = MPC(model=model, horizon=int(mpc_config.mpc.horizon_sec * CTRL_FREQ), q_mpc=mpc_config.mpc.q, r_mpc=mpc_config.mpc.r)

    # load reference
    traj_ref = traj_data['full_reference']
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

    for i, initial_state in enumerate(traj_states.T):
        for run in range(RUNS_PER_POINT):
            # randomize initial state
            noise = rng.uniform(-randomizer_range, randomizer_range, initial_state.size)
            state = initial_state + noise

            for step in range(STEPS_PER_RUN):
                action, state = ctrl.select_action(obs=state, info={"ref": traj_ref[:, step:]})

                horizon_states.append(ctrl.results_dict['horizon_states'])
                horizon_actions.append(ctrl.results_dict['horizon_inputs'])
                horizon_refs.append(ctrl.results_dict['goal_states'])
                initial_states.append(ctrl.results_dict['horizon_states'][0])

            # reset MPC to clear warmstart
            ctrl.reset()

    output = pd.DataFrame(
        {
            'horizon_states': horizon_states,
            'horizon_actions': horizon_actions,
            'horizon_reference': horizon_refs,
            'initial_state': initial_states,
        }
    )

    print(output.head())



