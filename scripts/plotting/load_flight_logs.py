from lsy_drone_racing.control.mpc.mpc_utils import outputs_for_actions
from lsy_drone_racing.utils.plotting import *

flight_data = np.load("output/logs/mm/diff_a=2_s=100_i=5/diff_01.npz", allow_pickle=True)

FREQ = 50 if 'env_freq' not in flight_data else flight_data['env_freq']

n_actions = flight_data['n_actions'] if 'n_actions' in flight_data else 1
offset = flight_data['offset'] if 'offset' in flight_data else 0

# plot states
states = np.atleast_3d(flight_data['horizon_states'])
timesteps = np.linspace(start=0, stop=len(states)/FREQ * n_actions, num=len(states))
if states.shape[1] == 15:
    state_groups.append(('Thrust', (12, 13, 14, 15), '$N$'))

plot_groups(states, timesteps, state_groups, state_labels, "States")
plot_trajectories3d([states,], state_groups)

# plot inputs (if possible)
if 'horizon_inputs' in flight_data:
    inputs = np.atleast_3d(flight_data['horizon_inputs'])
    plot_groups(inputs, timesteps, input_groups, input_labels, "Inputs")

if not 'horizon_actions' in flight_data and 'horizon_outputs' in flight_data:
    actions = flight_data['horizon_outputs'][:, outputs_for_actions, :]
else:
    actions = flight_data['horizon_actions']

actions = np.atleast_3d(actions)[:, :, offset:offset+n_actions].swapaxes(1, 2).reshape((-1, 13, 1), order='C')
timesteps = np.linspace(start=0, stop=len(actions)/FREQ, num=len(actions))
plot_groups(actions, timesteps, action_groups, action_labels, "Actions")

# plot timings
if 't_wall' in flight_data or 't_solver' in flight_data:
    fig, ax = plt.subplots(figsize=(20, 15))
    fig.suptitle(f'Solution Times')
    ax.set_xlabel('s')
    ax.set_ylabel('$ms$', rotation=0)
    fig.legend(loc='lower right')

    timesteps = np.linspace(start=0, stop=len(states) / FREQ * n_actions, num=len(states))
    
    if 't_wall' in flight_data:
        timings = flight_data['t_wall'] * 1000
        ax.plot(timesteps[1:-1], timings[1:-1], color='b', label='total')
    
    if 't_solver' in flight_data:
        timings = flight_data['t_solver'] * 1000
        ax.plot(timesteps[1:-1], timings[1:-1], color='r', label='solver')

plt.show()

## MPC
# horizon_inputs: (T, 4, H)
# horizon_outputs: (T, 22, H)

# horizon_states: (T, 16, H + 1)

## Diffusion
# horizon_actions: (T, 13, H)
# horizon_samples: (T, S, 13, H)

# horizon_states: (T, 12, 1)

