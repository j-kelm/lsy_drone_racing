""" I HATE PLOTS

"""

import os
import numpy as np
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter

folder = "output/logs/racing"
name = 'speed'
SAVE = True

data_list = list()
flight_data = None
for file in os.listdir(folder):
    if file.endswith(".npz"):
        flight_data = np.load(os.path.join(folder, file), allow_pickle=True)
        data_list.append(np.atleast_3d(flight_data['horizon_states']))

def figsize(scale, height=1.0):
    fig_width_pt = 418.25368                        # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0 * height           # Aesthetic ratio (you could change this) (I am)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*golden_mean              # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

if SAVE:
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
        "figure.figsize": figsize(1.1, 1.0),     # default fig size of 0.9 textwidth
        "pgf.preamble": r"\usepackage[utf8]{inputenc} \usepackage[T1]{fontenc} \usepackage{siunitx}",    # use utf8 fonts becasue your computer can handle it :)
        }
    mpl.rcParams.update(pgf_with_latex)
    import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection="3d")

# plot continuous lines
for i, episode in enumerate(data_list):
    # ax.plot(episode[:, 0, 0], episode[:, 1, 0], episode[:, 2, 0], c='gray', alpha=0.5)

    data_list[i] = episode[..., 0:6, 0].reshape(-1, 6)

state_data = np.concatenate(data_list)

img = ax.scatter(state_data[:, 0], state_data[:, 1], state_data[:, 2], c=np.linalg.norm(state_data[:, 3:6], axis=1),
                 cmap='turbo', s=2.5, rasterized=True)  # , alpha=0.5)
ax.set_aspect('equal', 'box')
ax.view_init(15, 45, 0)
ax.xaxis.set_major_locator(plt.MaxNLocator(4))
cbar = fig.colorbar(img, fraction=0.015, pad=0.05)
cbar.ax.yaxis.set_major_formatter(FormatStrFormatter(r'\SI{%.1f}{\meter\per\second}'))
# fig.tight_layout()

if SAVE:
    plt.savefig('output/plots/trajectories/{}.pgf'.format(name), dpi=300, bbox_inches='tight')
    plt.savefig('output/plots/trajectories/{}.pdf'.format(name), dpi=300, bbox_inches='tight')
else:
    plt.show()

## MPC
# horizon_inputs: (T, 4, H)
# horizon_outputs: (T, 22, H)

# horizon_states: (T, 16, H + 1)

## Diffusion
# horizon_actions: (T, 13, H)
# horizon_samples: (T, S, 13, H)

# horizon_states: (T, 12, 1)

