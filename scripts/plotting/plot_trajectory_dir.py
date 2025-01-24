import os
import matplotlib as mpl

from lsy_drone_racing.utils.plotting import *

folder = "output/logs/diff/"
SAVE = False

data_list = list()
flight_data = None
for file in os.listdir(folder):
    if file.endswith(".npz"):
        flight_data = np.load(os.path.join(folder, file), allow_pickle=True)
        data_list.append(np.atleast_3d(flight_data['horizon_states']))

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
        "figure.figsize": figsize(0.9),     # default fig size of 0.9 textwidth
        "pgf.preamble": r"\usepackage[utf8]{inputenc} \usepackage[T1]{fontenc}",    # use utf8 fonts becasue your computer can handle it :)
        }
    mpl.rcParams.update(pgf_with_latex)
    import matplotlib.pyplot as plt

plot_trajectories2d(data_list, state_groups)

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

