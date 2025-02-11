"""Plot export script for the thesis that generates a 2D-view of actions within a dataset.
"""

import numpy as np
from numpy.random import default_rng
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter

HORIZON = 8

states = np.load("output/training_data_racing.npz", allow_pickle=True)
outputs = states['actions']

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
    "figure.figsize": figsize(1.1, 1.0),     # default fig size of 0.9 textwidth
    "pgf.preamble": r"\usepackage[utf8]{inputenc} \usepackage[T1]{fontenc} \usepackage{siunitx}",    # use utf8 fonts becasue your computer can handle it :)
    }
mpl.rcParams.update(pgf_with_latex)
import matplotlib.pyplot as plt

# select random samples from training data
rng = default_rng(seed=45)
numbers = rng.choice(len(outputs), size=min(250, len(outputs)), replace=False)

fig, ax = plt.subplots()

ax.add_patch(plt.Circle((1, 0), 0.12, color="gray", alpha=0.5))
ax.add_patch(plt.Circle((2, 0), 0.12, color="gray", alpha=0.5))

for index in numbers:
    snippet = outputs[index]
    ax.plot(-snippet[0, :HORIZON] + 1.5, -snippet[1, :HORIZON] + 1)

ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f m'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f m'))

ax.set_aspect('equal', 'box')

fig.savefig('output/plots/{}.pgf'.format("dataset"), dpi=300, bbox_inches='tight')
fig.savefig('output/plots/{}.pdf'.format("dataset"), dpi=300, bbox_inches='tight')