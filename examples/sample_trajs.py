
import numpy as np
import pandas as pd

from munch import munchify
import yaml

from examples.model import Model
from examples.mpc_controller import MPC
from examples.constraints import obstacle_constraints, gate_constraints
from examples.planner import LinearPlanner

NUM_TRAJS = 2
CTRL_FREQ = 30

if __name__ == "__main__":
    # get mpc config
    path = "config/mpc.yaml"
    with open(path, "r") as file:
        mpc_config = munchify(yaml.safe_load(file))

    # set up results list
    # track_index | gates_pos gates_rpy obstacles_pos | full_reference state_history gate_index
    #      1      |    3x4       3x4         3x4      |     12xN           12xN         1xN
    results = list()

    for traj in range(NUM_TRAJS):
        # TODO: Randomize gate, obstacle and starting positions
        gates_pos = np.array([
            [0.45, -1.0, 0.525],
            [1.0, -1.55, 1.0],
            [0.0, 0.5, 0.525],
            [-0.5, -0.5, 1.0],
        ])

        gates_rpy = np.array([
            [0.0, 0.0, 2.35],
            [0.0, 0.0, -0.78],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 3.14],
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
        }

        # track_df = pd.DataFrame(initial_info)

        #       | gate_pos gate_rpy (obstacle_pos)
        # index | x y z    r p y    (x y z)

        # initialize planner
        planner = LinearPlanner(initial_info=initial_info, CTRL_FREQ=CTRL_FREQ)
        ref, next_gate = planner.plan(initial_obs=state, gates=None, speed=2.5, acc=8)

        # initialize mpc controller
        model = Model(info=None)
        model.input_constraints_soft += [lambda u: 0.03 - u, lambda u: u - 0.145]  # 0.03 <= thrust <= 0.145
        model.state_constraints_soft += [lambda x: 0.04 - x[2]]

        for obstacle_pos in initial_info['obstacles.pos']:
            model.state_constraints_soft += obstacle_constraints(obstacle_pos, r=0.125)

        for gate_pos, gate_rpy in zip(initial_info['gates.pos'], initial_info['gates.rpy']):
            model.state_constraints_soft += gate_constraints(gate_pos, gate_rpy[2], r=0.11)

        ctrl = MPC(model=model, horizon=int(mpc_config.mpc.horizon_sec * CTRL_FREQ), q_mpc=mpc_config.mpc.q,
                   r_mpc=mpc_config.mpc.r)


        state_history = list()
        action_history = list()

        # loop mpc through entire track
        for step in range(np.shape(ref)[1]):
            state_history.append(state)
            remaining_ref = ref[:, step:]
            action, state = ctrl.select_action(obs=state, info={"ref": remaining_ref})

        result = {
            'gates.pos': gates_pos,
            'gates.rpy': gates_rpy,
            'obstacles.pos': obstacles_pos,
            'full_reference': ref,
            'state_history': np.array(state_history).T,
            'next_gate': next_gate,
        }

        results.append(result)

    df = pd.DataFrame(results)
    df.to_hdf("output/track_data.hdf5", key="track_data", mode='w')
    print(df.head)









