import os
import matplotlib as mpl

from lsy_drone_racing.utils.plotting import *

folder = "output/logs/diff/"
SAVE = True

def figsize(scale):
    fig_width_pt = 418.25368                         # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
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
        "figure.figsize": figsize(1.0),     # default fig size of 0.9 textwidth
        "pgf.preamble": r"\usepackage[utf8]{inputenc} \usepackage[T1]{fontenc}",    # use utf8 fonts becasue your computer can handle it :)
        }
    mpl.rcParams.update(pgf_with_latex)
    import matplotlib.pyplot as plt

data_list = list()
flight_data = None
for file in os.listdir(folder):
    if file.endswith(".npz"):
        flight_data = np.load(os.path.join(folder, file), allow_pickle=True)
        data_list.append(np.atleast_3d(flight_data['horizon_states']))

cmap = mpl.colormaps['turbo']
normalizer = mpl.colors.Normalize(0, 3.0)
im = mpl.cm.ScalarMappable(norm=normalizer)

fig = plt.figure(dpi=100)
axes = fig.subplots(nrows=3, ncols=1, sharex=False)
for ax in axes:
    state_data = data_list.copy()
    state_merged = data_list.copy()
    # plot continuous lines
    for i, episode in enumerate(state_data):
        # ax.plot(-episode[:, 0, 0], episode[:, 1, 0], c='gray', alpha=0.25)
        state_merged[i] = episode[..., 0:6, 0].reshape(-1, 6)

    state_merged = np.concatenate(state_merged)
    img = ax.scatter(-state_merged[:, 0], state_merged[:, 1], c=np.linalg.norm(state_merged[:, 3:6], axis=1), cmap=cmap, norm=normalizer,
                     s=1)  # , alpha=0.5)
    ax.set_aspect('equal', 'box')
    ax.set_title("MPC")

cbar = fig.colorbar(img, ax=axes, location='right')
cbar.ax.set_ylabel(state_groups[1][2], rotation=0)
#fig.tight_layout()

if SAVE:
    plt.savefig('{}.pgf'.format("plot"), bbox_inches='tight')
    plt.savefig('{}.pdf'.format("plot"), bbox_inches='tight')
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

