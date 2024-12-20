import numpy as np
from matplotlib import pyplot as plt

from lsy_drone_racing.control.mpc.mpc_utils import outputs_for_actions

FREQ = 50

action_labels = ['$x$', '$y$', '$z$',
                 '$v_x$', '$v_y$', '$v_z$',
                 '$a_x$', '$a_y$', '$a_z$',
                 '$\\psi$',
                 '$p$', '$q$', '$r$']

action_units = [*['$m$']*3,
                *['$\\frac{m}{s}$']*3,
                *['$\\frac{m}{s^2}$']*3,
                'rad',
                *['$\\frac{rad}{s}$']*3]

state_labels = ['$x$', '$y$', '$z$',
                '$v_x$', '$v_y$', '$v_z$',
                '$\\phi$', '$\\theta$', '$\\psi$',
                '$p$', '$q$', '$r$',
                '$F_1$', '$F_2$', '$F_2$', '$F_4$',
                ]

state_units = [*['$m$']*3,
               *['$\\frac{m}{s}$']*3,
               *['rad']*3,
               *['$\\frac{rad}{s}$']*3,
               *['$N$']*4]

input_labels = ['$\\dot{F}_1$', '$\\dot{F}_2$', '$\\dot{F}_3$', '$\\dot{F}_4$']
input_units = [*['$\\frac{N}{s}$']*4]

flight_data = np.load("output/logs/diffusion_run.npz", allow_pickle=True)

n_actions = flight_data['n_actions']
offset = flight_data['offset']

# plot states
states = flight_data['horizon_states']

timesteps = np.linspace(start=0, stop=len(states)/FREQ * n_actions, num=len(states))

fig, axs = plt.subplots(states.shape[1], sharex=True, figsize=(30, 20))
fig.suptitle(f'States')
for i, ax in enumerate(axs):
    ax.set_title(state_labels[i])
    ax.set_ylabel(state_units[i], rotation=0)
    ax.plot(timesteps, states[:, i, 0], color='g', label=None if i else 'MPC')

axs[-1].set_xlabel('s')
fig.legend(loc='lower right')

# plot inputs (if possible)
if 'horizon_inputs' in flight_data:
    inputs = flight_data['horizon_inputs']
    fig, axs = plt.subplots(inputs.shape[1], sharex=True, figsize=(30, 20))
    fig.suptitle(f'Inputs')
    for i, ax in enumerate(axs):
        ax.set_title(input_labels[i])
        ax.set_ylabel(input_units[i], rotation=0)
        ax.plot(timesteps, inputs[:, i, 0], color='g', label=None if i else 'MPC')

    axs[-1].set_xlabel('s')
    fig.legend(loc='lower right')

if not 'horizon_actions' in flight_data and 'horizon_outputs' in flight_data:
    actions = flight_data['horizon_outputs'][:, outputs_for_actions, :]
else:
    actions = flight_data['horizon_actions']

actions = actions[:, :, offset:offset+n_actions].swapaxes(1, 2).reshape((-1, 13, 1), order='C')
timesteps = np.linspace(start=0, stop=len(actions)/FREQ * n_actions, num=len(actions))
fig, axs = plt.subplots(actions.shape[1], sharex=True, figsize=(30, 20))
fig.suptitle(f'Actions')
for i, ax in enumerate(axs):
    ax.set_title(action_labels[i])
    ax.set_ylabel(action_units[i], rotation=0)
    ax.plot(timesteps, actions[:, i, 0], color='g', label=None if i else 'MPC')

axs[-1].set_xlabel('s')
fig.legend(loc='lower right')

plt.show()

## MPC
# horizon_inputs: (T, 4, H)
# horizon_outputs: (T, 22, H)

# horizon_states: (T, 16, H + 1)

## Diffusion
# horizon_actions: (T, 13, H)
# horizon_samples: (T, S, 13, H)

# horizon_states: (T, 12, 1)

