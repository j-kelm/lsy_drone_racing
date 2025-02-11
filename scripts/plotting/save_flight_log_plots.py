""" Another messy plot export script that plots all states and actions of a single log file.
"""

import numpy as np
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter, StrMethodFormatter

from lsy_drone_racing.control.mpc.mpc_utils import outputs_for_actions
from lsy_drone_racing.utils.plotting import state_groups, state_labels, action_groups, action_labels

flight_data = np.load("output/logs/mm/diff_a=1_s=1_i=2/diff_01.npz", allow_pickle=True)
name = "diff_a=1_s=25_rate"

action_groups = [action_groups[4],]
state_groups = [state_groups[3],]

def figsize(scale, height=1.0):
    fig_width_pt = 418.25368                        # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0 * height           # Aesthetic ratio (you could change this) (I am)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*golden_mean              # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size


mpl.use('pgf')
pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 10,               # LaTeX default is 10pt font.
    "font.size": 10,
    "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": figsize(1.1),     # default fig size of 0.9 textwidth
    "pgf.preamble": r"\usepackage[utf8]{inputenc} \usepackage[T1]{fontenc} \usepackage{siunitx} \usepackage{amsmath}",    # use utf8 fonts because your computer can handle it :)
    }
mpl.rcParams.update(pgf_with_latex)
import matplotlib.pyplot as plt

FREQ = 50 if 'env_freq' not in flight_data else flight_data['env_freq']

n_actions = flight_data['n_actions'] if 'n_actions' in flight_data else 1
offset = flight_data['offset'] if 'offset' in flight_data else 0

# plot states
states = np.atleast_3d(flight_data['horizon_states'])
timesteps = np.linspace(start=0, stop=len(states)/FREQ * n_actions, num=len(states))
if states.shape[1] == 16:
    state_groups.append(('Thrust', (12, 13, 14, 15), '$N$'))

fig, axs = plt.subplots(len(state_groups), sharex=True, squeeze=False, figsize=figsize(1.1, 0.4*len(state_groups)))
axs = axs.squeeze(1)

for ax, (group_label, index_group, group_units) in zip(axs, state_groups):
    if group_label == 'Position':
        group_label += ' $\mathbf{p}$'
        ax.set_ylabel(r'\si{\meter}')  # , rotation=0
    elif group_label == 'Velocity':
        group_label += ' $\dot{\mathbf{p}}$'
        ax.set_ylabel(r'\si{\meter\per\second}')  # , rotation=0
    elif group_label == 'Attitude':
        group_label += r' $\boldsymbol{\varphi}_{OB}$'
        ax.set_ylabel(r'\si{\radian}')  # , rotation=0
    elif group_label == 'Rate':
        group_label += r' $\boldsymbol{\omega}_B$'
        ax.set_ylabel(r'\si{\radian\per\second}')  # , rotation=0
    else:
        group_label += r' $\boldsymbol{\omega}_B$'
        ax.set_ylabel(r'\si{\newton}')  # , rotation=0

    # ax.set_title(group_label)
    for i in index_group:
        ax.plot(timesteps, states[:, i, 0], label=state_labels[i])
        ax.legend(loc='lower right')
axs[-1].xaxis.set_major_formatter(StrMethodFormatter('{x} s'))

fig.savefig('output/plots/commands/{}_states.pgf'.format(name), dpi=300, bbox_inches='tight')
fig.savefig('output/plots/commands/{}_states.pdf'.format(name), dpi=300, bbox_inches='tight')





if not 'horizon_actions' in flight_data and 'horizon_outputs' in flight_data:
    actions = flight_data['horizon_outputs'][:, outputs_for_actions, :]
else:
    actions = flight_data['horizon_actions']

actions = np.atleast_3d(actions)[:, :, offset:offset+n_actions].swapaxes(1, 2).reshape((-1, 13, 1), order='C')
timesteps = np.linspace(start=0, stop=len(actions)/FREQ, num=len(actions))

fig, axs = plt.subplots(len(action_groups), sharex=True, squeeze=False, figsize=figsize(1.1, 0.4*len(action_groups)) )
axs = axs.squeeze(1)

for ax, (group_label, index_group, group_units) in zip(axs, action_groups):
    if group_label == 'Position':
        group_label += ' $\mathbf{p}$'
        ax.set_ylabel(r'\si{\meter}')  # , rotation=0
    elif group_label == 'Velocity':
        group_label += ' $\dot{\mathbf{p}}$'
        ax.set_ylabel(r'\si{\meter\per\second}')  # , rotation=0
    elif group_label == 'Acceleration':
        group_label += ' $\ddot{\mathbf{p}}$'
        ax.set_ylabel(r'\si{\meter\per\square\second}')  # , rotation=0
    elif group_label == 'Yaw':
        group_label += ' $\psi$'
        ax.set_ylabel(r'\si{\radian}')  # , rotation=0
    else:
        group_label += r' $\boldsymbol{\omega}_B$'
        ax.set_ylabel(r'\si{\radian\per\second}')  # , rotation=0

    # ax.set_title(group_label)
    for i in index_group:
        ax.plot(timesteps, actions[:, i, 0], label=action_labels[i])
        ax.legend(loc='lower right')
axs[-1].xaxis.set_major_formatter(StrMethodFormatter('{x} s'))

fig.savefig('output/plots/commands/{}_actions.pgf'.format(name), dpi=300, bbox_inches='tight')
fig.savefig('output/plots/commands/{}_actions.pdf'.format(name), dpi=300, bbox_inches='tight')

## MPC
# horizon_inputs: (T, 4, H)
# horizon_outputs: (T, 22, H)

# horizon_states: (T, 16, H + 1)

## Diffusion
# horizon_actions: (T, 13, H)
# horizon_samples: (T, S, 13, H)

# horizon_states: (T, 12, 1)

